import os
from itertools import combinations

import wandb
from peft import PeftModel

from run_common import (
    init_config, init_wandb, load_dataset, load_reviews, parse_adapter_paths
)
from global_config import CONFIG
from models import load_model_and_tokenizer
from evaluate import evaluate


def main(experiment_name=None):
    init_config(experiment_name=experiment_name)

    adapter_paths = CONFIG.adapter_config['adapters']
    num_active = CONFIG.adapter_config['choose_all_n_combos']


    local_adapter_paths, adapter_names = parse_adapter_paths(adapter_paths)

    model, tokenizer = load_model_and_tokenizer(
        CONFIG.model, CONFIG.device_map, CONFIG.use_qlora_config
    )

    print('\nLoading pre-trained adapters...\n')
    for adapter_dir, adapter_name in zip(local_adapter_paths, adapter_names):
        print(f'{adapter_dir} ({adapter_name})')
        # we turn model into a PeftModel when loading the first adapter
        if not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(
                model, adapter_dir, adapter_name=adapter_name
            )
        else:
            model.load_adapter(adapter_dir, adapter_name=adapter_name)

    merge_method = CONFIG.adapter_config.get('merge_method', 'linear')
    print(f'\nMerging adapters using {merge_method} method...\n')
    adapter_combos = []
    for i, adapters in enumerate(combinations(adapter_names, num_active)):
        adapter_combos.append(adapters)
        if merge_method == 'linear':
            weights = [1/num_active for _ in adapters]  # uniform weighting
            model.add_weighted_adapter(
                adapters, weights, f'combined-{i}', merge_method
            )
        elif merge_method == 'ties':
            # recommended in https://huggingface.co/docs/peft/en/developer_guides/model_merging?merge-method=TIES
            weights = [1.0 for _ in adapters]
            density = CONFIG.adapter_config.get('ties_density', 0.2)
            model.add_weighted_adapter(
                adapters, weights, f'combined-{i}', merge_method, density=density
            )

    print('\nRunning...\n')
    for i, adapters in enumerate(adapter_combos):
        wandb_run = None
        if CONFIG.log_to_wandb:
            wandb_run = init_wandb(run_name=f'{experiment_name}-{i}')
            wandb_run.config['active_adapters'] = adapters

        model.set_adapter(f'combined-{i}')

        eval_df = load_dataset('eval', return_df=True)
        reviews = load_reviews()

        results_df, metrics = evaluate(model, tokenizer, eval_df, reviews, wandb_run)

        if CONFIG.log_to_wandb:
            wandb.log(metrics)
            save_cols = ['index', 'label', 'predicted_label', 'label_probs', 'total_label_probs']
            res_table = wandb.Table(dataframe=results_df[save_cols])
            wandb.log({"Eval results": res_table})
            wandb_run.finish()



if __name__ == '__main__':
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    main(experiment_name)
