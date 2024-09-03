import os
import re
import pickle
from collections import defaultdict
import argparse

import wandb
import torch
from safetensors import safe_open
import numpy as np
from torch.nn.functional import cosine_similarity

from run_common import parse_adapter_paths


def get_param_vector_from_file(params_file, device):
    with safe_open(params_file, framework='pt', device=device) as file:
        return torch.cat([
            file.get_tensor(key).view(-1) for key in file.keys()
            if 'self_attn' in key
        ]) 


def safetensors_to_layer_dict(tensors_path, device):
    layer_pat = re.compile(r'layers\.(\d+)\.')
    layer_vecs = defaultdict(lambda: torch.tensor([], device=device))

    with safe_open(tensors_path, framework='pt', device=device) as file:
        for key in file.keys():
            try:
                layer = int(layer_pat.search(key).groups()[0])
            except AttributeError:
                continue
            layer_vecs[layer] = torch.cat([
                layer_vecs[layer], file.get_tensor(key).view(-1)
            ])

    return layer_vecs


def get_avg_param_vector_from_file(params_file, device):
    """
    Average across layers rather than concatenating
    """
    layer_vecs = safetensors_to_layer_dict(params_file, device)
    return torch.stack(list(layer_vecs.values()), dim=0).mean(dim=0)


def get_layerwise_sim_weightings(source_models, target_model, device='cuda'):
    source_model_dirs, source_model_names = parse_adapter_paths(source_models)
    target_model_dirs, _ = parse_adapter_paths([target_model])
    target_model_dir = target_model_dirs[0]

    target_tensors_path = target_model_dir + '/adapter_model.safetensors'
    target_layer_vecs = safetensors_to_layer_dict(target_tensors_path, device)

    # out[layer][model_name] = similarity
    out = {layer: {} for layer in target_layer_vecs.keys()}

    for model_dir, model_name in zip(source_model_dirs, source_model_names):
        tensors_path = model_dir + '/adapter_model.safetensors'
        source_layer_vecs = safetensors_to_layer_dict(tensors_path, device)
        for layer in source_layer_vecs:
            sim = cosine_similarity(source_layer_vecs[layer], target_layer_vecs[layer], dim=0)
            # take exponent -- we don't want negative weightings
            out[layer][model_name] = torch.exp(sim)

    return out


def compare_signs(models_file, target_models_file=None, device='cuda'):
    try:
        save_dir = os.environ['WANDB_ARTIFACT_DIR']
    except KeyError as exc:
        print('WANDB_ARTIFACT_DIR must be set so we can save the results array')
        raise exc

    env_key = os.environ.get('WANDB_KEY')
    if env_key:
        wandb.login(key=env_key)

    run = wandb.init(entity='tberm-org', project='adapter-comparison')

    with open(models_file) as file:
        source_model_paths = [line.strip() for line in file.readlines()]
        run.log({'adapters': source_model_paths})
        print('Comparing models:\n ', '\n  '.join(source_model_paths))

    if target_models_file is None:
        target_model_paths = source_model_paths
        symmetrical = True
    else: 
        symmetrical = False
        with open(target_models_file) as file:
            target_model_paths = [line.strip() for line in file.readlines()]

        run.log({'target_adapters': target_model_paths})
        print('To:\n ', '\n  '.join(source_model_paths))

    source_model_dirs, _ = parse_adapter_paths(source_model_paths)
    target_model_dirs, _ = parse_adapter_paths(target_model_paths)

    # init results to -1
    results = - np.ones((len(source_model_dirs), len(target_model_dirs)))
    for i, i_model_dir in enumerate(source_model_dirs):
        for j, j_model_dir in enumerate(target_model_dirs):
            if symmetrical and results[j, i] >= 0:
                # already have a result for this pair
                results[i, j] = results[j, i]
                continue

            i_params = get_param_vector_from_file(
                i_model_dir + '/adapter_model.safetensors',
                device=device
            )
            j_params = get_param_vector_from_file(
                j_model_dir + '/adapter_model.safetensors',
                device=device
            )



def avg_layerwise_sims(models_file, target_models_file=None, device='cuda'):
    try:
        save_dir = os.environ['WANDB_ARTIFACT_DIR']
    except KeyError as exc:
        print('WANDB_ARTIFACT_DIR must be set so we can save the results array')
        raise exc

    env_key = os.environ.get('WANDB_KEY')
    if env_key:
        wandb.login(key=env_key)

    run = wandb.init(entity='tberm-org', project='adapter-comparison')

    with open(models_file) as file:
        source_model_paths = [line.strip() for line in file.readlines()]
        run.log({'adapters': source_model_paths})
        print('Comparing models:\n ', '\n  '.join(source_model_paths))

    if target_models_file is None:
        target_model_paths = source_model_paths
        symmetrical = True
    else: 
        symmetrical = False
        with open(target_models_file) as file:
            target_model_paths = [line.strip() for line in file.readlines()]

        run.log({'target_adapters': target_model_paths})
        print('To:\n ', '\n  '.join(target_model_paths))

    source_model_dirs, _ = parse_adapter_paths(source_model_paths)
    target_model_dirs, _ = parse_adapter_paths(target_model_paths)

    # init results to -1
    results = - np.ones((len(source_model_dirs), len(target_model_dirs)))
    for i, i_model_dir in enumerate(source_model_dirs):
        for j, j_model_dir in enumerate(target_model_dirs):
            if symmetrical and results[j, i] >= 0:
                # already have a result for this pair
                results[i, j] = results[j, i]
                continue

            layer_dict_i = safetensors_to_layer_dict(
                i_model_dir + '/adapter_model.safetensors', device=device
            )
            layer_dict_j = safetensors_to_layer_dict(
                j_model_dir + '/adapter_model.safetensors', device=device
            )

            sims = []
            for key in layer_dict_i.keys():
                sim = cosine_similarity(layer_dict_i[key], layer_dict_j[j], dim=0)
                sims.append(sim)

            result = torch.tensor(sims).mean().numpy()
            results[i, j] = result

    save_file = save_dir + '/cosine_similarities.txt'
    np.savetxt(save_file, results)
    artifact = wandb.Artifact('cosine_similarities', 'results')
    artifact.add_file(save_file)
    run.log_artifact(artifact)
 

def main(models_file, target_models_file=None, device='cuda', avg_layers=False, init_adapter=None):
    try:
        save_dir = os.environ['WANDB_ARTIFACT_DIR']
    except KeyError as exc:
        print('WANDB_ARTIFACT_DIR must be set so we can save the results array')
        raise exc

    env_key = os.environ.get('WANDB_KEY')
    if env_key:
        wandb.login(key=env_key)

    run = wandb.init(entity='tberm-org', project='adapter-comparison')

    with open(models_file) as file:
        source_model_paths = [line.strip() for line in file.readlines()]
        run.log({'adapters': source_model_paths})
        print('Comparing models:\n ', '\n  '.join(source_model_paths))

    if target_models_file is None:
        target_model_paths = source_model_paths
        symmetrical = True
    else: 
        symmetrical = False
        with open(target_models_file) as file:
            target_model_paths = [line.strip() for line in file.readlines()]

        run.log({'target_adapters': target_model_paths})
        print('To:\n ', '\n  '.join(target_model_paths))

    source_model_dirs, _ = parse_adapter_paths(source_model_paths)
    target_model_dirs, _ = parse_adapter_paths(target_model_paths)

    get_vector = get_avg_param_vector_from_file if avg_layers else get_param_vector_from_file

    if init_adapter is not None:
        init_adapter_dirs, _ = parse_adapter_paths([init_adapter])
        init_adapter_dir = init_adapter_dirs[0]
        init_params = get_vector(init_adapter_dir + '/adapter_model.safetensors', device)

    # init results to -1
    results = - np.ones((len(source_model_dirs), len(target_model_dirs)))
    for i, i_model_dir in enumerate(source_model_dirs):
        for j, j_model_dir in enumerate(target_model_dirs):
            if symmetrical and results[j, i] >= 0:
                # already have a result for this pair
                results[i, j] = results[j, i]
                continue

            param_vecs = [
                get_vector(model_dir + '/adapter_model.safetensors', device)
                for model_dir in (i_model_dir, j_model_dir)
            ]

            if init_adapter is not None:
                param_vecs = [pv - init_params for pv in param_vecs]

            sim = cosine_similarity(*param_vecs, dim=0)
            results[i, j] = sim

    save_file = save_dir + '/cosine_similarities.txt'
    np.savetxt(save_file, results)
    artifact = wandb.Artifact('cosine_similarities', 'results')
    artifact.add_file(save_file)
    run.log_artifact(artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('adapters_list_file')
    parser.add_argument('--target-adapters-file', help='Provide to get asymmetrical similarities')
    parser.add_argument('--device', '-d', default='cuda')
    parser.add_argument('--avg-layers', action='store_true')
    parser.add_argument('--layerwise', action='store_true')
    parser.add_argument('--get-weightings-for')
    parser.add_argument('--init-adapter', help='path to blank init adapter to subtract from trained ones')
    args = parser.parse_args()

    if sum([args.avg_layers, args.layerwise, (args.get_weightings_for is not None)]) > 1:
        raise ValueError('Cannot have more than one of `--avg-layers`, `--layerwise`, `--get-weightings-for')
    if args.init_adapter is not None and (args.layerwise or args.get_weightings_for is not None):
        raise NotImplementedError('`--init-adapter` only implemented for standard comparison mode')

    if args.layerwise:
        avg_layerwise_sims(args.adapters_list_file, args.target_adapters_file, args.device)
    elif args.get_weightings_for is not None:
        with open(args.adapters_list_file) as file:
            source_model_paths = [line.strip() for line in file.readlines()]
        get_layerwise_sim_weightings(source_model_paths, args.get_weightings_for, args.device)
    else:
        main(args.adapters_list_file, args.target_adapters_file, args.device, args.avg_layers, args.init_adapter)