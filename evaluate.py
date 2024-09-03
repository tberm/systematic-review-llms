from ast import literal_eval
from argparse import ArgumentParser
import re
from pathlib import Path
import os
import subprocess
import sys
from shutil import rmtree

from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import wandb
from peft import PeftModel

from run_common import (
    init_config, init_wandb, load_reviews, preprocess_row, log_prompt,
    load_dataset, get_label_tokens, get_wandb_adapter, calculate_threshold_scores
)
from prompt_utils import LABELS
from models import load_model_and_tokenizer, load_poly_model_skills, make_poly_model, load_adapters
from global_config import CONFIG


normed_labels = {val.strip().lower(): idx for idx, val in enumerate(LABELS)}
def text_to_label(text):
    return normed_labels.get(text.strip().lower(), -1)


def check_repo_is_clean():
    res = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked=no'],
        check=True, capture_output=True,
    )
    output = res.stdout.decode().strip('\n')
    if (not output) or (output == ' M experiments/config.yaml'):
        return
    
    print('There are uncommitted changes in the Git repo:')
    print(output)
    cont = input('Continue anyway? (y/[n]) ')
    if cont.strip().lower() != 'y':
        print('Aborting')
        sys.exit(1)


def compute_metrics(df):
    acc = (df.predicted_label == df.label).mean()
    ap = average_precision_score(df.label, df.pos_prob)
    auroc = roc_auc_score(df.label, df.pos_prob)
    threshold_scores = calculate_threshold_scores(df.label, df.pos_prob)
    tp = sum((df.label == 1) & (df.predicted_label == 1))
    fp = sum((df.label != 1) & (df.predicted_label == 1))
    tn = sum((df.label != 1) & (df.predicted_label != 1))
    fn = sum((df.label == 1) & (df.predicted_label != 1))
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
    neg_recall = tn / (tn + fp) if tn + fp > 0 else 0
    pos_rate = df.predicted_label.mean()
    out = {
        'acc': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'neg_recall': neg_recall,
        'pos_rate': pos_rate,
        'average_precision': ap,
        'auroc': auroc,
    }
    out.update(threshold_scores)
    return out


def main(experiment_name=None, example_idx=None, model=None, tokenizer=None):
    if CONFIG.use_gpu:
        # in case we are chaining multiple runs
        torch.cuda.empty_cache()

    init_config(experiment_name=experiment_name)

    if example_idx is not None:
        CONFIG.run_config['batch_size'] = 1
        CONFIG.run_config['trunc_dataset'] = None
        CONFIG.wandb_config['log'] = False

    print(CONFIG)

    if not CONFIG.skip_repo_check:
        check_repo_is_clean()

    if CONFIG.wandb_config.get('overwrite_run'):
        wandb_run = init_wandb(
            {'wandb_run_id': CONFIG.wandb_config['overwrite_run']},
             run_name=experiment_name
        )
        CONFIG.wandb_config['run_id'] = wandb_run.id
        CONFIG.wandb_config['run_name'] = wandb_run.name
    else:
        wandb_run = init_wandb(run_name=experiment_name)

    if (model is None) != (tokenizer is None):
        raise RuntimeError('Must provide both of `model` and `tokenizer` or neither')

    if model is None:
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

    elif CONFIG.run_config.get('use_poly_with_experts'):
        expert_paths = CONFIG.run_config['use_poly_with_experts']
        local_expert_paths = []
        for full_path in expert_paths:
            try:
                scheme, adapter_path = full_path.split('://', maxsplit=1)
            except ValueError:
                scheme = 'wandb'
                adapter_path = full_path

            if scheme == 'wandb':
                adapter_dir, _ = get_wandb_adapter(adapter_path, wandb_run)
            else:
                adapter_dir = adapter_path

            local_expert_paths.append(adapter_dir)

        model = make_poly_model(
            model,
            rank=CONFIG.run_config['poly_rank'],
            n_skills=len(expert_paths)
        )
        load_poly_model_skills(model, local_expert_paths, device=CONFIG.device, freeze='all')

    elif CONFIG.run_config.get('load_multi_adapters'):
        adapters_dict = CONFIG.run_config['load_multi_adapters']

    if adapters_dict is not None:
        model = load_adapters(model, adapters_dict, wandb_run)


    eval_df = load_dataset('eval', return_df=True)
    reviews = load_reviews()

    results_df, metrics = evaluate(model, tokenizer, eval_df, reviews, wandb_run, example_idx)

    if CONFIG.log_to_wandb:
        wandb.log(metrics)
        save_cols = ['index', 'label', 'predicted_label', 'label_probs', 'total_label_probs']
        res_table = wandb.Table(dataframe=results_df[save_cols])
        wandb.log({"Eval results": res_table})
        wandb_run.finish()

    save_results(results_df)
    return model, tokenizer, metrics


def evaluate(model, tokenizer, df, reviews, wandb_run=None, example_idx=None):
    model.eval()
    df = df.copy()
    # get tokenized labels, which we compare with output for each example
    label_tokens = torch.tensor(get_label_tokens(tokenizer)).to(CONFIG.device)

    single_example = example_idx is not None
    if single_example:
        df = df.loc[[example_idx]]

    df['label_probs'] = pd.Series(dtype=str)
    df['total_label_probs'] = pd.Series(dtype=float)
    df['predicted_label'] = pd.Series(dtype=int)

    # so that the index value is available in loader batches
    df['index'] = df.index
    # leave out the label_probs resul column which the data loader does not like
    cols = ['title', 'abstract', 'label', 'review_id', 'predicted_label', 'index']
    ds = Dataset.from_dict(df[cols])

    preproc_fn = lambda row: preprocess_row(reviews, tokenizer, label_tokens, row)
    ds = ds.map(preproc_fn)
    dl = DataLoader(ds, CONFIG.run_config['batch_size'])

    # log an example of the prompts we're using
    if wandb_run is not None:
        log_prompt(ds[0]['prompt'], wandb_run)

    print('Evaluating...')

    all_label_probs = []
    batches = dl if single_example else tqdm(dl)
    for batch in batches:
        inputs = tokenizer(batch['prompt'], return_tensors='pt', padding=True).to(CONFIG.device)

        if single_example:
            print(batch['prompt'][0])
            print('\n' + '-' * 100 + '\n')

        with torch.no_grad():
            if CONFIG.run_config.get('use_poly_with_experts') or CONFIG.train_config.get('use_poly_with_experts'):
                output = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    task_ids=torch.tensor([[0]] * len(inputs['input_ids'])).long(),
                )
            else:
                output = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                )
        # NOTE: we calculate loss in the usual way over all logits, but in order to make
        # the classification prediction we take the softmax over the yes/no labels only.
        # This is mainly for the sake of zero-shot evaluation, where other irrelevant
        # tokens would otherwise sometimes take a share of the probability mass

        # logits shape (batch, sequence, vocab)
        next_token_logits = output.logits[:, -1, :]
        next_token_probs = output.logits[:, -1, :].softmax(1)
        # token id of each label, repeated over batch dimension
        this_batch_size = len(batch['label'])
        label_tokens_batch = label_tokens.tile(this_batch_size, 1)  # (batch, labels)
        # logits/probs corresponding to the labels
        label_logits = torch.gather(next_token_logits, 1, label_tokens_batch)  # (batch, labels)
        label_probs = torch.gather(next_token_probs, 1, label_tokens_batch)  # (batch, labels)
        normed_label_probs = label_logits.softmax(1)
        all_label_probs += label_probs.tolist()
        preds = normed_label_probs.argmax(1)

        # When the index has 2 levels (several reviews), we get this as two tensors.
        # Need to convert tensors to lists and invert the nesting order so we can use it
        # for dataframe indexing
        nparr = np.array([tnsr.numpy() for tnsr in batch['index']])
        index = nparr.T.tolist()

        df.loc[index, 'predicted_label'] = preds.cpu().numpy()
        df.loc[index, 'label_probs'] = [
            str(arr) for arr in label_probs.cpu().tolist()
        ]
        df.loc[index, 'total_label_probs'] = label_probs.sum(dim=1).cpu().numpy()
        df.loc[index, 'normed_label_probs'] = [
            str(arr) for arr in normed_label_probs.cpu().tolist()
        ]

        if single_example:
            print('Model predictions (probabilities):')
            for i, prob in enumerate(label_probs[0]):
                print(f'    {LABELS[i]}:  {prob.cpu().item()}')

            print('\nTrue label:', LABELS[batch['label'][0]]) 
            return df, None

        del inputs
        del output


    normed_probs_tensor = torch.tensor(df.normed_label_probs.apply(literal_eval).tolist())
    df['pos_prob'] = normed_probs_tensor[:, 1]
    metrics = compute_metrics(df)

    print('Accuracy:', metrics['acc'])
    print('F1:', metrics['f1'])
    print('Average precision:', metrics['average_precision'])

    bs_iters = CONFIG.run_config.get('num_bootstrap', 1000)
    if bs_iters > 0:
        print(f'Computing bootstrapped metrics... ({bs_iters} datasets)')
        bs_metrics = []
        for i in range(bs_iters):
            bs_df = df.sample(frac=1, replace=True)
            bs_metrics.append(compute_metrics(bs_df))

        bs_results = pd.DataFrame(bs_metrics)
        metrics.update({
            'acc_bs_mean': bs_results.acc.mean(),
            'acc_bs_stdev': bs_results.acc.std(),
            'f1_bs_mean': bs_results.f1.mean(),
            'f1_bs_stdev': bs_results.f1.std(),
            'av_prec_bs_mean': bs_results.average_precision.mean(),
            'av_prec_bs_stdev': bs_results.average_precision.std(),
        })

        acc_stderr = metrics['acc_bs_stdev'] / np.sqrt(bs_iters)
        print(f'Bootstrapped accuracy: {metrics["acc_bs_mean"]} +- {acc_stderr})')
        f1_stderr = metrics['f1_bs_stdev'] / np.sqrt(bs_iters)
        print(f'Bootstrapped f1: {metrics["f1_bs_mean"]} +- {f1_stderr})')
        ap_stderr = metrics['av_prec_bs_stdev'] / np.sqrt(bs_iters)
        print(f'Bootstrapped AP: {metrics["av_prec_bs_mean"]} +- {ap_stderr})')
    
    return df, metrics


def save_results(df, partial=False):
    here_dir = Path(__file__).parent
    results_dir = CONFIG.run_config['results_dir']
    results_dir = Path(results_dir) if results_dir[0] == '/' else here_dir / results_dir
    path = results_dir / f'{CONFIG.run_id}{"-partial" if partial else ""}.csv'

    print('Saving results to', str(path.resolve()))
    df[['index', 'label', 'predicted_label', 'label_probs']].to_csv(path, index=False)


def parse_str_tensor(string):
    """
    much faster str->arr conversion than ast.literal_eval
    """
    return torch.tensor(np.fromstring(string[1:-1], sep=',', dtype=np.float32))


def get_gpu_memory():
    """
    Get insufficient permissions on EIDF
    """
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    print(memory_free_info)
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return sum(memory_free_values)


def get_gpu_memory_temp():
    lines = subprocess.check_output('nvidia-smi').decode('ascii').split('\n')
    values = parse_smi(lines)
    if 'MIG' in values:
        return values['MIG']
    elif 'NVIDIA' in values:
        return values['NVIDIA']
    

def parse_smi(out):
    """
    Grotesque hack to get used mem from standard nvidia-smi
    """
    parsed = {}
    last_was_gap = False
    last_was_top_border = False
    section = None
 
    for line in out:
        if last_was_top_border:
            try:
                section = re.match(r'^\| (\w+)', line).groups()[0]
            except AttributeError:
                pass
        is_hr = re.match(r'\+\-+\+', line)
        match = re.search(r'(\d*)MiB', line)
        if match is not None and section not in parsed:
            parsed[section] = int(match.groups()[0])
        last_was_top_border = last_was_gap and is_hr
        last_was_gap = line == ''

    return parsed



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--single-example', '-e', type=int)
    args = parser.parse_args()
    experiment_name = os.environ.get('EXPERIMENT_NAME')

    if experiment_name and " " in experiment_name:
        experiments = experiment_name.split()
        print('Re-using model for chain of experiments:')
        print(', '.join(experiments))
        model = None
        tokenizer = None
        for i, exp in enumerate(experiments):
            print(f'\nStarting experiment {i+1} of {len(experiments)}...')
            model, tokenizer, _ = main(
                exp, args.single_example,
                model=model, tokenizer=tokenizer,
            )

    else:
        main(experiment_name, args.single_example)
