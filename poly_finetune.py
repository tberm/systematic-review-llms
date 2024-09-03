import os

import wandb
import torch
from torch.nn import CrossEntropyLoss
from peft import prepare_model_for_kbit_training
from transformers import (
    default_data_collator,
    TrainingArguments,
    Trainer
)

from models import load_model_and_tokenizer, make_poly_model, load_poly_model_skills
from run_common import (
    load_reviews, load_dataset, init_config, get_label_tokens, preprocess_row,
    init_wandb, get_wandb_adapter
)
from global_config import CONFIG
from finetune import compute_metrics, drop_irrelevant_logits
from evaluate import evaluate


class CustomTrainer(Trainer):
    """
    Override loss computation of trainer to calculate loss on the answer token only
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            task_ids=inputs.get('task_ids'),
        )
        pred_logits = outputs.logits[:, -1, :]
        #ce_loss = CrossEntropyLoss(reduce=False)
        ce_loss = CrossEntropyLoss()
        loss = ce_loss(
            target=inputs['labels'],
            input=pred_logits,
        )
        return (loss, outputs) if return_outputs else loss


def main(experiment_name=None):
    init_config(experiment_name=experiment_name)

    run = None
    if CONFIG.log_to_wandb:
        run = init_wandb(run_name=experiment_name)

    model, tokenizer = load_model_and_tokenizer(
        CONFIG.model, CONFIG.device_map, CONFIG.use_qlora_config
    )

    # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
    # currently this breaks poly (task_ids is None, I guess some layers get replaced or sth)
    #model.gradient_checkpointing_enable()
    #prepare_model_for_kbit_training(model)
    
    expert_paths = CONFIG.train_config['use_poly_with_experts']
    local_expert_paths = []
    for full_path in expert_paths:
        try:
            scheme, adapter_path = full_path.split('://', maxsplit=1)
        except ValueError:
            scheme = 'wandb'
            adapter_path = full_path

        if scheme == 'wandb':
            adapter_dir, _ = get_wandb_adapter(adapter_path, run)
        else:
            adapter_dir = adapter_path

        local_expert_paths.append(adapter_dir)

    model = make_poly_model(
        model,
        rank=CONFIG.train_config['poly_rank'],
        n_skills=len(expert_paths)
    )
    load_poly_model_skills(model, local_expert_paths, device=CONFIG.device, freeze='skills')

    print('Loaded skills into Poly model. Parameters to train:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    reviews = load_reviews()

    label_tokens = get_label_tokens(tokenizer)
    label_tokens_gpu = torch.tensor(label_tokens).to(CONFIG.device)

    def preproc_fn(row):
        out = preprocess_row(reviews, tokenizer, label_tokens_gpu, row, tokenize=True)
        out['task_ids'] = torch.tensor([0]).long()
        return out

    eval_ds = load_dataset('eval')
    eval_ds = eval_ds.map(preproc_fn).select_columns(
        ['input_ids', 'attention_mask', 'label', 'task_ids']
    )
    train_ds = load_dataset('train')
    train_ds = train_ds.map(preproc_fn).select_columns(
        ['input_ids', 'attention_mask', 'label', 'task_ids']
    )

    save_every = CONFIG.train_config['save_every']
    save_strategy = 'no' if save_every == 0 else 'steps'
    if 'eval_every' in CONFIG.train_config:
        eval_every = CONFIG.train_config['eval_every']
    else:
        eval_every = save_every
    eval_strategy = 'no' if eval_every == 0 else 'steps'

    training_args = TrainingArguments(
        output_dir=CONFIG.checkpoints_dir,
        per_device_train_batch_size=CONFIG.train_config['batch_size'],
        per_device_eval_batch_size=CONFIG.run_config['batch_size'],
        learning_rate=CONFIG.train_config.get('init_lr'),
        lr_scheduler_type=CONFIG.train_config.get('lr_schedule', 'linear'),
        warmup_ratio=CONFIG.train_config.get('lr_warmup_ratio', 0.0),
        weight_decay=CONFIG.train_config.get('weight_decay', 0.0),
        report_to='wandb' if CONFIG.log_to_wandb else 'none',
        run_name=CONFIG.wandb_config['run_name'] if CONFIG.log_to_wandb else None,
        num_train_epochs=CONFIG.train_config.get('num_epochs', 3),
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_every,
        save_steps=save_every,
        logging_strategy='steps',
        logging_steps=CONFIG.train_config['log_every'],
        eval_delay=CONFIG.train_config.get('eval_delay', 0),
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model='average_precision',
        overwrite_output_dir=True,
        gradient_accumulation_steps=CONFIG.train_config.get('gradient_accumulation_steps', 1),
        remove_unused_columns=False,
        label_names=['labels'],
    )

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        compute_metrics=lambda preds: compute_metrics(preds, label_tokens),
        preprocess_logits_for_metrics=(lambda log, lab:
                                        drop_irrelevant_logits(label_tokens, log, lab)),
    )

    trainer.train()

    print('Training complete!')
    if CONFIG.train_config.get('save_model_local', True):
        print(f'Saving final model to {CONFIG.final_model_dir}')
        trainer.save_model(CONFIG.final_model_dir)

    print('Evaluating final model...')
    eval_df = load_dataset('eval', return_df=True)
    results_df, metrics = evaluate(model, tokenizer, eval_df, reviews)

    if CONFIG.log_to_wandb:
        wandb.log(metrics)
        save_cols = ['index', 'label', 'predicted_label', 'label_probs', 'total_label_probs']
        table = wandb.Table(dataframe=results_df[save_cols])
        wandb.log({"Eval results": table})

        run.finish()


if __name__ == '__main__':
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    main(experiment_name=experiment_name)
