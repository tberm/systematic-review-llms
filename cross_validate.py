import os
from pathlib import Path

import pandas as pd
import numpy as np

from run_common import init_config, init_wandb, load_reviews
from models import load_model_and_tokenizer
from global_config import CONFIG
from evaluate import evaluate


if __name__ == '__main__':
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    init_config(experiment_name=experiment_name)
    if CONFIG.run_config.get('num_bootstrap', 0) > 0:
        raise Exception("Shouldn't use bootstrapping for cross-validation script!")

    CONFIG.run_config['num_bootstrap'] = 0

    wandb_run = init_wandb(run_name=experiment_name)

    model, tokenizer = load_model_and_tokenizer(
        CONFIG.model, CONFIG.device_map, CONFIG.use_qlora_config
    )

    dataset_dir = Path(CONFIG.run_config['dataset_dir'])
    full_df = pd.read_csv(dataset_dir / 'full.csv', index_col=0)
    full_df['index'] = full_df.index
    val_split = full_df.loc[np.genfromtxt(dataset_dir / 'val_split.txt')]
    num_pos = val_split.label.sum()
    num_neg = len(val_split) - num_pos

    if CONFIG.run_config.get('load_wandb_model'):
        artifact = wandb_run.use_artifact(CONFIG.run_config['load_wandb_model'], type='model')
        artifact_dir = artifact.download()
        print('Loading pre-trained adapter from', artifact_dir)
        model.load_adapter(artifact_dir)

    reviews = load_reviews()

    calc_metrics = ['f1', 'acc', 'average_precision', 'precision', 'recall']
    metrics_list = []
    for i in range(CONFIG.run_config['num_splits']):
        split_df = pd.concat([
            full_df[full_df.label == 1].sample(num_pos),
            full_df[full_df.label == 0].sample(num_neg),
        ]).sample(frac=1)

        _, metrics = evaluate(model, tokenizer, split_df, reviews, wandb_run)
        metrics_list.append({
            metric: metrics[metric] for metric in calc_metrics
        })

    agg_metrics = {}
    for metric in calc_metrics:
        values = np.array([dict[metric] for dict in metrics_list])
        agg_metrics[f'{metric}_mean'] = values.mean()
        agg_metrics[f'{metric}_std'] = values.std()

    _, full_metrics = evaluate(model, tokenizer, full_df, reviews, wandb_run)

    for metric in calc_metrics:
        agg_metrics[f'{metric}_full'] = full_metrics[metric]

    wandb_run.log(agg_metrics)
    wandb_run.finish()