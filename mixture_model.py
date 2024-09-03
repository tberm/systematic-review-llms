import re

import torch
import wandb
from bitsandbytes.nn.modules import Linear4bit
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.bnb import Linear4bit
from peft import PeftConfig, get_peft_model
from safetensors import safe_open

from global_config import CONFIG
from run_common import (
    parse_adapter_paths, init_config, init_wandb, load_dataset, load_reviews
)
from models import load_model_and_tokenizer
from compare_loras import get_layerwise_sim_weightings
from evaluate import evaluate


def get_mixture_model(model, wandb_run=None):
    config = CONFIG.adapter_config
    local_adapter_paths, adapter_names = parse_adapter_paths(config['adapters'], wandb_run)
    peft_config = PeftConfig.from_pretrained(local_adapter_paths[0])
    mod_map = {
        torch.nn.modules.linear.Linear: LoraRouter,
        Linear4bit: LoraRouter4bit,
    } 
    peft_config._register_custom_module(mod_map)
    model = get_peft_model(model, peft_config) 
    for path, name in zip(local_adapter_paths, adapter_names):
        model.load_adapter(path, adapter_name=name)

    return model, adapter_names


def set_router_weights(model, weights, layerwise=False):
    for name, module in model.named_modules():
        if not isinstance(module, (LoraRouter, LoraRouter4bit)):
            continue

        if layerwise:
            layer_match = re.search(r'layers\.(\d+)\.', name)
            layer = int(layer_match.groups()[0])
            module.set_adapter_weights(weights[layer])
        else:
            module.set_adapter_weights(weights)


class LoraRouter(LoraLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_weights = {}

    def set_adapter_weights(self, weights):
        self.adapter_weights = weights


    def forward(self, x, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError('Mixed batch forward not implemented for LoraRouter')
        elif self.merged:
            raise RuntimeError('self.merged should not be True on a LoraRouter')
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            # Calculate weighted average of lora_A and lora_B
            weighted_lora_A = None
            weighted_lora_B = None
            total_weight = sum(self.adapter_weights.values())

            for active_adapter, weight in self.adapter_weights.items():
                if active_adapter not in self.lora_A.keys():
                    continue

                if weighted_lora_A is None:
                    weighted_lora_A = self.lora_A[active_adapter].weight * (weight / total_weight)
                    weighted_lora_B = self.lora_B[active_adapter].weight * (weight / total_weight)
                else:
                    weighted_lora_A += self.lora_A[active_adapter].weight * (weight / total_weight)
                    weighted_lora_B += self.lora_B[active_adapter].weight * (weight / total_weight)

            if weighted_lora_A is not None and weighted_lora_B is not None:
                x = x.to(weighted_lora_A.dtype)
                # Use dropout from first adapter
                dropout = self.lora_dropout[list(self.adapter_weights.keys())[0]]
                # Merge the scaling factors
                scaling = sum(
                    self.scaling[adapter] * weight
                    for adapter, weight in self.adapter_weights.items()
                ) / total_weight

                result = result + torch.nn.functional.linear(dropout(x), weighted_lora_B @ weighted_lora_A) * scaling

            result = result.to(torch_result_dtype)

        return result


class LoraRouter4bit(Linear4bit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_weights = {}

    def set_adapter_weights(self, weights):
        self.adapter_weights = weights


    def forward(self, x, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError('Mixed batch forward not implemented for LoraRouter')
        elif self.merged:
            raise RuntimeError('self.merged should not be True on a LoraRouter')
        else:
            result = self.base_layer(x, *args, **kwargs)
            result = result.clone()

            # Calculate weighted average of lora_A and lora_B
            weighted_lora_A = None
            weighted_lora_B = None
            total_weight = sum(self.adapter_weights.values())

            for active_adapter, weight in self.adapter_weights.items():
                if active_adapter not in self.lora_A.keys():
                    continue

                if weighted_lora_A is None:
                    weighted_lora_A = self.lora_A[active_adapter].weight * (weight / total_weight)
                    weighted_lora_B = self.lora_B[active_adapter].weight * (weight / total_weight)
                else:
                    weighted_lora_A += self.lora_A[active_adapter].weight * (weight / total_weight)
                    weighted_lora_B += self.lora_B[active_adapter].weight * (weight / total_weight)

            if weighted_lora_A is not None and weighted_lora_B is not None:
                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(weighted_lora_A.dtype)

                # Use dropout from first adapter
                dropout = self.lora_dropout[list(self.adapter_weights.keys())[0]]
                # Merge the scaling factors
                scaling = sum(
                    self.scaling[adapter] * weight
                    for adapter, weight in self.adapter_weights.items()
                ) / total_weight

                result = result + torch.nn.functional.linear(dropout(x), weighted_lora_B @ weighted_lora_A) * scaling

            if requires_conversion:
                output = output.to(expected_dtype)

            result = result + output

        return result


def main(experiment_name=None):
    init_config(experiment_name=experiment_name)
    print(CONFIG)
    wandb_run = init_wandb(run_name=experiment_name)

    model, tokenizer = load_model_and_tokenizer(
        CONFIG.model, CONFIG.device_map, CONFIG.use_qlora_config,
    )

    model, adapter_names = get_mixture_model(model, wandb_run)

    if CONFIG.adapter_config['adapter_routing_method'] == 'uniform':
        weights = {name: 1 for name in adapter_names}
        set_router_weights(model, weights)

    elif CONFIG.adapter_config['adapter_routing_method'] == 'layerwise_lora_similarity':
        weights = get_layerwise_sim_weightings(
            CONFIG.adapter_config['adapters'],
            CONFIG.adapter_config['few_shot_adapter'],
            device=CONFIG.device,
        )
        if CONFIG.log_to_wandb:
            wandb.config['adapter_weights'] = weights
        set_router_weights(model, weights, layerwise=True)
    else:
        raise ValueError(
            'Invalid adapter routing method: ' +
            CONFIG.adapter_config['adapter_routing_method']
        )


    eval_df = load_dataset('eval', return_df=True)
    reviews = load_reviews()

    results_df, metrics = evaluate(model, tokenizer, eval_df, reviews, wandb_run)

    if CONFIG.log_to_wandb:
        wandb.log(metrics)
        save_cols = ['index', 'label', 'predicted_label', 'label_probs', 'total_label_probs']
        res_table = wandb.Table(dataframe=results_df[save_cols])
        wandb.log({"Eval results": res_table})
        wandb_run.finish()

    return model, tokenizer, metrics


if __name__ == '__main__':
    import os
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    main(experiment_name)
