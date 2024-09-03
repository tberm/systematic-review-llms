import re
import os
from pathlib import Path

import yaml
import torch
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
from transformers.integrations import WandbCallback
from lora_plus import LoraPlusTrainer, LoraPlusTrainingArguments
from peft import LoraConfig, IA3Config, PeftModel, prepare_model_for_kbit_training, get_peft_model
from sklearn.metrics import average_precision_score
import wandb

from global_config import CONFIG
from models import load_model_and_tokenizer, load_adapters
from run_common import (
    preprocess_row, load_dataset, load_reviews, init_config, init_wandb,
    get_label_tokens, check_disk_space, log_prompt
)
from evaluate import evaluate


class CustomTrainer(Trainer):
    """
    Override loss computation of trainer to calculate loss on the answer token only
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            #task_ids=torch.tensor([[0] * len(inputs['input_ids'])]).long(),
        )
        pred_logits = outputs.logits[:, -1, :]
        #ce_loss = CrossEntropyLoss(reduce=False)
        ce_loss = CrossEntropyLoss()
        loss = ce_loss(
            target=inputs['labels'],
            input=pred_logits,
        )
        return (loss, outputs) if return_outputs else loss

    #def training_step(self, model, inputs):
    #    model.train()
    #    inputs = self._prepare_inputs(inputs)

    #    with self.compute_loss_context_manager():
    #        losses = self.compute_loss(model, inputs)

    #    del inputs

    #    for loss_id, loss in enumerate(losses):
    #        # if we weren't running again for full loss
    #        #retain_graph = loss_id < len(losses) - 1
    #        self.accelerator.backward(loss, retain_graph=True)
    #        grad_vec = self._get_grad_vector()

    #    return losses.mean().detach() / self.args.gradient_accumulation_steps

    def _get_grad_vector(self):
        return torch.concat([
            param.grad
            for group in self.optimizer.param_groups
            for layer in group['params']
            for param in layer.flatten()
        ])

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p, dtype=torch.int8).to(p.device))
                else:
                    shape.append(p.grad.shape)
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p, dtype=torch.int8).to(p.device))
        grad_flatten = self._flatten_grad(grad, shape)
        has_grad_flatten = self._flatten_grad(has_grad, shape)
        return grad_flatten, shape, has_grad_flatten


class CustomLoraPlusTrainer(CustomTrainer, LoraPlusTrainer):
    pass


class GradConflictMonitor(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_step_begin(self, args, state, control, **kwargs):
        ...


def accuracy(preds, labels):
    acc = (labels == preds).mean()
    return {'accuracy': acc}


def drop_irrelevant_logits(label_tokens, logits, labels):
    return logits[:, -1, label_tokens]


class BatchedMetricsCalculator:
    """
    Calculates metrics iteratively over batches rather than over all examples at once.
    Use with `batch_eval_metrics=True` Trainer argument to reduce system RAM use
    """
    def __init__(self, label_tokens):
        self.label_tokens = label_tokens
        self.true_pos_count = 0
        self.false_pos_count = 0
        self.true_neg_count = 0
        self.false_neg_count = 0
        self.total_count = 0

    def handle_preds(self, eval_preds, compute_result=False):
        """
        Callback which receives model predictions for a batch and updates running counts
        accordingly.
        On the last batch we receive compute_result=True and we should return the
        final metrics
        """
        label_scores = eval_preds.predictions
        pred_labels = label_scores.argmax(1)
        # turn the label_ids, which are token IDs, into {0,1} labels
        labels = (eval_preds.label_ids == self.label_tokens[1]).to(int)

        self.true_pos_count += sum((labels == 1) & (pred_labels == 1)).cpu().item()
        self.false_pos_count += sum((labels == 0) & (pred_labels == 1)).cpu().item()
        self.true_neg_count += sum((labels == 0) & (pred_labels == 0)).cpu().item()
        self.false_neg_count += sum((labels == 1) & (pred_labels == 0)).cpu().item()
        self.total_count += len(pred_labels)

        if compute_result:
            out = self._compute_final_metrics()
            print(out)
            return out

    def _compute_final_metrics(self):
        acc = (self.true_pos_count + self.true_neg_count) / self.total_count

        pos_preds = self.true_pos_count + self.false_pos_count
        prec = (self.true_pos_count / pos_preds) if pos_preds > 0 else 0

        pos_labels = self.true_pos_count + self.false_neg_count
        rec = (self.true_pos_count / pos_labels) if pos_labels > 0 else 0

        f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
        return {'accuracy': acc, 'prec': prec, 'rec': rec, 'f1': f1}


def compute_metrics(eval_preds, label_tokens):
    label_scores = eval_preds.predictions
    pred_labels = label_scores.argmax(1)
    # turn the label_ids, which are token IDs, into {0,1} labels
    labels = (eval_preds.label_ids == label_tokens[1]).astype(int)

    acc = (pred_labels == labels).mean()
    label_probs = torch.tensor(label_scores).softmax(1)
    ap = average_precision_score(labels, label_probs[:, 1])
    tp = sum((labels == 1) & (pred_labels == 1))
    tn = sum((labels != 1) & (pred_labels != 1))
    fp = sum((labels != 1) & (pred_labels == 1))
    fn = sum((labels == 1) & (pred_labels != 1))
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
    neg_recall = tn / (tn + fp) if tn + fp > 0 else 0
    pos_rate = pred_labels.mean()
    return {
        'accuracy': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'neg_recall': neg_recall,
        'pos_rate': pos_rate,
        'average_precision': ap,
    }


def count_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return 100 * trainable_params / all_param, all_param


class PartialRuns:
    def __init__(self, models_dir):
        self.yaml_path = Path(models_dir) / 'partials.yaml'
        if self.yaml_path.exists():
            with self.yaml_path.open() as yaml_file:
                self.data = yaml.safe_load(yaml_file) or {}
        else:
            self.data = {}

    def _save(self):
        with self.yaml_path.open('w') as yaml_file:
            yaml.safe_dump(self.data, yaml_file)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        self._save()

    def __delitem__(self, key):
        del self.data[key]
        self._save()

    def __contains__(self, key):
        return key in self.data


class CustomWandbCallback(WandbCallback):
    def __init__(self, disk_check_dir=None):
        self._disk_check_dir = disk_check_dir
        super().__init__()

    def on_log(self, *args, **kwargs):
        if self._disk_check_dir is not None:
            metric_name = f'% Space used ({self._disk_check_dir})'
            value = check_disk_space(self._disk_check_dir)
            if value is not None:
                self._wandb.log({metric_name: value})
                if value >= 87:
                    self._wandb.alert(
                        title="NFS almost full",
                        text=f"NFS is {value}% full",
                        wait_duration=600,
                    )

        super().on_log(*args, **kwargs)


def main():
    models_path = Path(CONFIG.train_config['models_dir'])
    if not models_path.exists():
        models_path.mkdir()

    partials = PartialRuns(CONFIG.train_config['models_dir'])
    partial_run = None
    if CONFIG.train_config.get('resume'):
        if experiment_name:
            print('experiment name:', experiment_name)
            if experiment_name in partials:
                partial_run = partials[experiment_name]

        if partial_run is not None:
            CONFIG.start_ts = partial_run['start_ts']
            CONFIG.wandb_config['run_id'] = partial_run.get('wandb_run_id')
            CONFIG.wandb_config['run_name'] = partial_run.get('wandb_run_name')

    run = None
    if CONFIG.log_to_wandb:
        run = init_wandb(partial_run=partial_run, run_name=experiment_name)

    if partial_run is None and experiment_name is not None:
        # create a new partial
        partials[experiment_name] = {
            'start_ts': CONFIG.start_ts,
            'wandb_run_id': CONFIG.wandb_config.get('run_id'),
            'wandb_run_name': CONFIG.wandb_config.get('run_name'),
        }

    eval_ds = load_dataset('eval')
    if CONFIG.run_config.get('trunc_dataset_while_training'):
        eval_ds = eval_ds.train_test_split(
            test_size=CONFIG.run_config['trunc_dataset_while_training']
        )['test']

    train_ds = load_dataset('train')
    reviews = load_reviews()
    peft_model_name = (
        'NouRed/BioMed-Tuned-Llama-3-8b' if CONFIG.use_biomed_ft
        else None
    )
    model, tokenizer = load_model_and_tokenizer(
        CONFIG.model, CONFIG.device_map, CONFIG.use_qlora_config, CONFIG.full_prec, peft_model_name
    )

    adapters_dict = None
    if CONFIG.run_config.get('load_adapter'):
        if CONFIG.run_config.get('load_multi_adapters'):
            raise RuntimeError('Cannot set both `load_adapter` and `load_multi_adapters`')

        adapters_dict = {CONFIG.run_config['load_adapter']: 1}

    elif CONFIG.run_config.get('load_multi_adapters'):
        adapters_dict = CONFIG.run_config['load_multi_adapters']

    if adapters_dict is not None:
        model = load_adapters(model, adapters_dict, run)
        model = model.merge_and_unload()
        print('Adapter(s) merged into model')


    # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
    model.gradient_checkpointing_enable()
    if not CONFIG.full_prec:
        prepare_model_for_kbit_training(model)

    if CONFIG.train_config.get('load_wandb_model'):
        artifact = run.use_artifact(CONFIG.train_config['load_wandb_model'], type='model')
        artifact_dir = artifact.download()
        print('Loading pre-trained adapter from', artifact_dir)
        peft_model_name = artifact_dir
        if CONFIG.full_prec:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name,
                is_trainable=True,
            )
        else:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name,
                torch_dtype=torch.float16,
                is_trainable=True,
            )
    elif CONFIG.train_config['peft_method'] == 'lora':
        peft_config = LoraConfig(
            r=CONFIG.train_config.get('lora_r', 8),
            lora_alpha=CONFIG.train_config.get('lora_alpha', 32),
            lora_dropout=CONFIG.train_config.get('lora_dropout', 0.05),
            task_type="SEQ_CLS",
        )
        if CONFIG.train_config.get('seed') is not None:
            torch.manual_seed(CONFIG.train_config['seed'])

        model = get_peft_model(model, peft_config)

    elif CONFIG.train_config['peft_method'] == 'ia3':
        peft_config = IA3Config(task_type="SEQ_CLS")

        if CONFIG.train_config.get('seed') is not None:
            torch.manual_seed(CONFIG.train_config['seed'])

        model = get_peft_model(model, peft_config)

    pc_trainable, tot_params = count_trainable_parameters(model)
    print(f'Model has {tot_params} parameters, of which {pc_trainable:.2f}% trainable')
    if CONFIG.log_to_wandb:
        wandb.log({'Total parameters': tot_params, 'Trainable parameters %': pc_trainable})

    label_tokens = get_label_tokens(tokenizer)
    label_tokens_gpu = torch.tensor(label_tokens).to(CONFIG.device)

    def preproc_fn(row):
        return preprocess_row(reviews, tokenizer, label_tokens_gpu, row, tokenize=True)

    train_ds = train_ds.map(preproc_fn)
    eval_ds = eval_ds.map(preproc_fn)

    # log an example of the prompts we're using
    if run is not None:
        log_prompt(train_ds[0]['prompt'], run)

    save_every = CONFIG.train_config['save_every']
    save_strategy = 'no' if save_every == 0 else 'steps'
    if 'eval_every' in CONFIG.train_config:
        eval_every = CONFIG.train_config['eval_every']
    else:
        eval_every = save_every
    eval_strategy = 'no' if eval_every == 0 else 'steps'

    args_dict = {
        'output_dir': CONFIG.checkpoints_dir,
        'per_device_train_batch_size': CONFIG.train_config['batch_size'],
        'per_device_eval_batch_size': CONFIG.run_config['batch_size'],
        'learning_rate': CONFIG.train_config.get('init_lr'),
        'lr_scheduler_type': CONFIG.train_config.get('lr_schedule', 'linear'),
        'report_to': 'wandb' if CONFIG.log_to_wandb else 'none',
        'run_name': CONFIG.wandb_config['run_name'] if CONFIG.log_to_wandb else None,
        'num_train_epochs': CONFIG.train_config.get('num_epochs', 3),
        'eval_strategy': eval_strategy,
        'save_strategy': save_strategy,
        'eval_steps': eval_every,
        'save_steps': save_every,
        'logging_strategy': 'steps',
        'logging_steps': CONFIG.train_config['log_every'],
        'eval_delay': CONFIG.train_config.get('eval_delay', 0),
        'save_total_limit': 1,
        'load_best_model_at_end': False,
        'metric_for_best_model': 'average_precision',
        'overwrite_output_dir': True,
        'gradient_accumulation_steps': CONFIG.train_config.get('gradient_accumulation_steps', 1),
        #'remove_unused_columns': not bool(CONFIG.train_config.get('use_poly_with_experts')),
    }
    if (
        CONFIG.train_config['peft_method'] == 'lora'
        and CONFIG.train_config.get('use_lora_plus')
    ):
        trainer_cls = CustomLoraPlusTrainer
        args_cls = LoraPlusTrainingArguments
        args_dict['loraplus_lr_ratio'] = CONFIG.train_config['lora_plus_ratio']
    else:
        trainer_cls = CustomTrainer
        args_cls = TrainingArguments

    #metrics_calculator = BatchedMetricsCalculator(label_tokens_gpu)

    train_args = args_cls(**args_dict)
    trainer = trainer_cls(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=lambda preds: compute_metrics(preds, label_tokens),
        #compute_metrics=metrics_calculator.handle_preds,
        data_collator=DataCollatorWithPadding(tokenizer),
        # avoid memory cost of storing all logits produced by model for evaluation
        preprocess_logits_for_metrics=(lambda log, lab:
                                       drop_irrelevant_logits(label_tokens, log, lab)),
    )
    if CONFIG.train_config.get('disk_check_path'):
        wandb_callback = CustomWandbCallback(CONFIG.train_config['disk_check_path'])
        trainer.add_callback(wandb_callback)

    #trainer.add_callback(GradConflictMonitor())

    #if CONFIG.train_config.get('use_poly_with_experts'):
    #    # HF trainer fails to pick up the correct model forward() signature, so supply it manually
    #    trainer._signature_columns = ['input_ids', 'task_ids', 'attention_mask', 'position_ids',
    #    'inputs_embeds', 'head_mask', 'past_key_values', 'labels', 'use_cache',
    #    'output_attentions', 'output_hidden_states', 'return_dict', 'label_ids',
    #    'label', 'labels']

    resume = False
    if partial_run is not None:
        checkpoints = list(CONFIG.checkpoints_dir.glob('*checkpoint*'))
        if checkpoints:
            resume = True
            print(f'Resuming training from checkpoint at {CONFIG.checkpoints_dir}')
        else:
            print(
                f"No checkpoints found at {CONFIG.checkpoints_dir}"
                "- starting from scratch'"
            )

    trainer.train(resume_from_checkpoint=resume)

    print('Training complete!')
    if CONFIG.train_config.get('save_model_local', True):
        print(f'Saving final model to {CONFIG.final_model_dir}')
        trainer.save_model(CONFIG.final_model_dir)

    # run is complete, so remove from partials
    if experiment_name in partials:
        del partials[experiment_name]

    print('Evaluating final model...')
    eval_df = load_dataset('eval', return_df=True)
    results_df, metrics = evaluate(model, tokenizer, eval_df, reviews, run)
    if CONFIG.log_to_wandb:
        wandb.log(metrics)
        save_cols = ['index', 'label', 'predicted_label', 'label_probs', 'total_label_probs']
        table = wandb.Table(dataframe=results_df[save_cols])
        wandb.log({"Eval results": table})

        run.finish()


if __name__ == '__main__':
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    init_config(experiment_name=experiment_name)
    print(CONFIG)

    if CONFIG.train_config.get('cross_validate'):
        # train/eval set should be given as review dir
        base_dir_train = CONFIG.train_config['train_set'].strip('/')
        assert re.search(r'review_\d{6}$', base_dir_train) is not None
        base_dir_eval = CONFIG.run_config['eval_set'].strip('/')
        assert re.search(r'review_\d{6}$', base_dir_eval) is not None

        prefix = CONFIG.train_config.get('fold_files_prefix', '')

        for fold in 'ABCDE':
            CONFIG.train_config['fold'] = fold
            CONFIG.train_config['train_set'] = base_dir_train + f'/{prefix}train-fold-{fold}.txt'
            CONFIG.run_config['eval_set'] = base_dir_eval + f'/{prefix}val-fold-{fold}.txt'
            print('Starting fold', fold)
            print('Train set:', CONFIG.train_config['train_set'])
            print('Eval set:', CONFIG.run_config['eval_set'])
            main()
            torch.cuda.empty_cache()

    else:
        main()
