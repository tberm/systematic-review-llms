from pathlib import Path
import os
import subprocess
import re

from datasets import Dataset
import pandas as pd
import numpy as np
import wandb
from sklearn.metrics import (
    precision_recall_curve, roc_curve
)

from global_config import CONFIG
from prompt_utils import assemble_prompt, assemble_prompt_chat_model, LABELS


def preprocess_row(reviews, tokenizer, label_tokens, row, tokenize=False):
    review = reviews.loc[row['review_id']]
    if CONFIG.use_instruct_format:
        prompt = assemble_prompt_chat_model(
            tokenizer, review, row['title'], row['abstract'],
            num_examples=CONFIG.run_config['num_shots'],
            prompt_format=CONFIG.run_config.get('prompt_format', 'chat_history'),
            model_class=CONFIG.model_class,
            reverse=CONFIG.run_config.get('prompt_reverse', False),
            remind=CONFIG.run_config.get('prompt_remind', False),
            short_criteria=CONFIG.run_config.get('prompt_short_criteria', False),
            incl_title=CONFIG.run_config.get('prompt_include_title', True),
        )
    else:
        prompt = assemble_prompt(
            review['title'], review['criteria'], row['title'], row['abstract'],
            num_examples=CONFIG.run_config['num_shots'],
        )

    if not tokenize:
        return {'prompt': prompt}

    tokens = tokenizer(prompt)
    label = label_tokens[row['label']]
    return dict(tokens, prompt=prompt, label=label)


def get_label_tokens(tokenizer):
    return np.array([tokenizer.encode(lbl)[-1] for lbl in LABELS])


def load_dataset(split, return_df=False):
    here_dir = Path(__file__).parent
    if split == 'train':
        path = Path(CONFIG.train_config['train_set'])
    elif split == 'eval':
        path = Path(CONFIG.run_config['eval_set'])
    else:
        raise ValueError('Invalid split name:', split)

    if not path.is_absolute():
        path = (here_dir / path).resolve()

    config_dict = CONFIG.train_config if split == 'train' else CONFIG.run_config

    if config_dict.get('multi_review_dataset'):
        split_df = pd.read_csv(path)
        reviews = split_df.review_id.unique()
        full_df = pd.concat([
            pd.read_csv(here_dir / f'data/covidence/review_{rev_id}/full.csv', index_col=0)
            for rev_id in reviews
        ])
        full_df = full_df.set_index([full_df.index, full_df.review_id]) 
        indexer = [(row.df_index, row.review_id) for _, row in split_df.iterrows()]
        df = full_df.loc[indexer]

    else:
        if path.suffix == '.txt':
            csv_path = path.parent / 'full.csv'
            split_idx = np.genfromtxt(path)
            rng = np.random.default_rng(24)
            rng.shuffle(split_idx)
        else:
            csv_path = path
            split_idx = None

        df = pd.read_csv(csv_path, index_col=0)
        if split_idx is not None:
            df = df.loc[split_idx]

    df = df.fillna({'title': ''})

    if config_dict.get('trunc_dataset'):
        target_size = config_dict['trunc_dataset']
        if config_dict.get('random_sample_data'):
            if config_dict.get('trunc_balanced'):
                df = pd.concat([
                    df[df.label == 1].sample(int(target_size / 2)),
                    df[df.label == 0].sample(target_size - int(target_size / 2)),
                ]).sample(frac=1)
            else:
                df = df.sample(target_size)
        else:
            if config_dict.get('trunc_balanced'):
                df = pd.concat([
                    df[df.label == 1][:int(target_size / 2)],
                    df[df.label == 0][:int(target_size / 2)],
                ]).sample(frac=1)
            else:
                df = df[:target_size]

    print(f'{split} split has {len(df)} items ({df.label.sum()} positive)')

    if CONFIG.run_config.get('override_review_criteria'):
        rev_id = CONFIG.run_config['override_review_criteria']
        print(f'Overriding review ID to {rev_id} for all papers')
        # we assume that papers are never relevant to a different review than the one
        # they belong to
        df.loc[df.review_id != rev_id, 'label'] = 0
        df.review_id = rev_id

    # so that the index value is available in loader batches
    df['index'] = df.index
    cols = ['title', 'abstract', 'label', 'review_id', 'index']
    if return_df:
        return df

    ds = Dataset.from_dict(df[cols])
    return ds


def load_reviews():
    here_dir = Path(__file__).parent
    reviews = pd.read_csv(here_dir / 'data/covidence/reviews.csv', index_col='index')
    if CONFIG.run_config.get('override_criteria'):
        with open(CONFIG.run_config['override_criteria']) as crit_file:
            reviews['criteria'] = crit_file.read()
    return reviews


def init_config(experiment_name=None):
    """
    For running multiple experiments at once, can nest each config dict under a key
    matching the experiment name
    """
    config_path = os.environ.get('SCRIPT_CONFIG')
    here_dir = Path(__file__).parent
    if config_path:
        config_path = Path(config_path)
    else:
        config_path = here_dir / 'configs'

    if experiment_name:
        conf_file = config_path / f'{experiment_name}.yaml'
    else:
        conf_file = config_path / 'local.yaml'

    conf_file.resolve(strict=True)  # check for existence

    print(f'Using config from {conf_file}')
    CONFIG.set_config(conf_file)

    return CONFIG


def init_wandb(partial_run=None, run_name=None):
    if not CONFIG.log_to_wandb:
        return

    if CONFIG.wandb_config.get('save_model_wandb', True):
        os.environ["WANDB_LOG_MODEL"] = "end"
    else:
        os.environ["WANDB_LOG_MODEL"] = ""

    env_key = os.environ.get('WANDB_KEY')
    if env_key:
        wandb.login(key=env_key)

    entity = CONFIG.wandb_config['entity']
    project = CONFIG.wandb_config['project']
    config_dict = CONFIG.__dict__
    if partial_run is None:
        # NOTE the config passed here will overwrite any previous config for the run
        # in wandb if it is different
        run = wandb.init(
            entity=entity,
            project=project,
            config=config_dict,
            name=run_name,
            dir=CONFIG.wandb_config.get('log_dir', 'wandb'),
        )
        CONFIG.wandb_config['run_name'] = run.name
        CONFIG.wandb_config['run_id'] = run.id
    else:
        run = wandb.init(
            entity=entity,
            project=project,
            config=CONFIG.__dict__,
            id=partial_run['wandb_run_id'],
            resume='must',
        )

    if CONFIG.wandb_config.get('tags'):
        run.tags += tuple(CONFIG.wandb_config['tags'])

    return run


def check_disk_space(dir):
    out = subprocess.run(['df', '-h'], capture_output=True)
    match = re.search(r'(\d+)%\s+' + dir, out.stdout.decode())
    try:
        return float(match.groups()[0])
    except (ValueError, IndexError, AttributeError) as exc:
        print('WARNING: could not determine available disk space due to this exception:')
        print(exc)
    

def log_prompt(prompt, run):
    """
    Log an example of a prompt from this run to wandb
    """
    path = CONFIG.wandb_config['log_dir'] + '/prompt_example.txt'
    with open(path, 'w') as file:
        file.write(prompt)
    
    artifact = wandb.Artifact(name='prompt_example', type='prompt')
    artifact.add_file(path)
    run.log_artifact(artifact)

def get_wandb_adapter(wandb_name, wandb_run=None):
    try:
        adapter_name = wandb_name.split('-', maxsplit=1)[1].rsplit(':', maxsplit=1)[0]
    except IndexError:
        adapter_name = wandb_name

    if wandb_run is None:
        env_key = os.environ.get('WANDB_KEY')
        if env_key:
            wandb.login(key=env_key)
        api = wandb.Api()
        artifact = api.artifact(wandb_name)
    else:
        artifact = wandb_run.use_artifact(wandb_name, type='model')

    download_dir = wandb_name.split('/')[-1]
    artifacts_dir = os.environ.get('WANDB_ARTIFACT_DIR')
    if artifacts_dir and (Path(artifacts_dir) / download_dir).is_dir():
        print('Using pre-downloaded wandb model for', download_dir)
        return str(Path(artifacts_dir) / download_dir), adapter_name

    return artifact.download(), adapter_name

def calculate_threshold_scores(labels, probs):
    fprs, tprs, _ = roc_curve(
        labels, probs, drop_intermediate=True
    )
    out = {}
    for fpr, tpr in zip(fprs, tprs):
        if tpr < 0.95:
            continue
        if 'specificity@95' not in out:
            out['specificity@95'] = 1 - fpr
        if tpr >= 0.98 and 'specificity@98' not in out:
            out['specificity@98'] = 1 - fpr
        if tpr >= 0.99:
            out['specificity@99'] = 1 - fpr
            break

    precs, recs, _ = precision_recall_curve(
        labels, probs, drop_intermediate=True
    )
    revved = (reversed(precs), reversed(recs))
    for prec, rec in zip(*revved):
        if rec < 0.95:
            continue
        if 'precision@95' not in out:
            out['precision@95'] = prec
        if rec >= 0.98 and 'precision@98' not in out:
            out['precision@98'] = prec
        if rec >= 0.99:
            out['precision@99'] = prec
            break

    return out


def parse_adapter_paths(adapter_paths, wandb_run=None):
    local_adapter_paths = []
    adapter_names = []
    for qual_path in adapter_paths:
        try:
            scheme, adapter_path = qual_path.split('://', maxsplit=1)
        except ValueError:
            scheme = 'wandb'
            adapter_path = qual_path

        if scheme == 'wandb':
            adapter_dir, adapter_name = get_wandb_adapter(adapter_path, wandb_run)
        else:
            adapter_name = adapter_path.split('/')[-1]
            adapter_dir = adapter_path

        local_adapter_paths.append(adapter_dir)
        adapter_names.append(adapter_name)
    
    return local_adapter_paths, adapter_names
