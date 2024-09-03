import os
import re
import torch
from transformers import (
  GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel, PolyConfig, PolyModel, get_peft_model, TaskType
from peft.tuners.poly.layer import Linear as PolyLinear
from safetensors.torch import load_file as safe_load_file

from run_common import get_wandb_adapter
from global_config import CONFIG


#std_model_forward = PolyModel.forward
#
#def patched_model_forward(self, *args, **kwargs):
#    input_ids = kwargs['input_ids']
#    return std_model_forward(self, *args, task_ids=torch.tensor([[0] * len(input_ids)]).long(), **kwargs)
#
#PolyModel.forward = patched_model_forward
#
#
#std_linear_forward = PolyLinear.forward
#
#def patched_linear_forward(self, *args, **kwargs):
#    #return std_linear_forward(self, *args, task_ids=torch.tensor([[0] * len(input_ids)]).long(), **kwargs)
#    return std_linear_forward(self, *args, **kwargs)
#
#PolyLinear.forward = patched_linear_forward



def load_adapters(model, adapters_dict, wandb_run):
    weight_total = sum([weight for weight in adapters_dict.values()])
    adapter_names = []
    adapter_weights = []
    for full_path, weight in adapters_dict.items():
        try:
            scheme, adapter_path = full_path.split('://', maxsplit=1)
        except ValueError:
            scheme = 'wandb'
            adapter_path = full_path

        if scheme == 'wandb':
            adapter_dir, adapter_name = get_wandb_adapter(adapter_path, wandb_run)
        else:
            adapter_name = adapter_path.split('/')[-1]
            adapter_dir = adapter_path

        adapter_names.append(adapter_name)
        adapter_weights.append(weight / weight_total)

        # we turn model into a PeftModel when loading the first adapter
        if not isinstance(model, PeftModel):
            print('Loading pre-trained adapter from', adapter_dir)
            model = PeftModel.from_pretrained(
                model, adapter_dir, adapter_name=adapter_name
            )
        # adapter could have been loaded in a previous run already
        elif adapter_name not in model.peft_config:
            print('Loading pre-trained adapter from', adapter_dir)
            model.load_adapter(adapter_dir, adapter_name=adapter_name)
        else:
            print(f'{adapter_name} is already loaded')

    if len(adapters_dict) == 1:
        model.set_adapter(adapter_names[0])
    else:
        # merge the adapters if they haven't been merged already
        merge_method = CONFIG.run_config.get('adapter_merge_method', 'linear')
        combined_name = '--'.join(adapter_names)
        if combined_name not in model.peft_config:
            print(f'Adding merged adapter {combined_name}')
            model.add_weighted_adapter(
                adapter_names, adapter_weights, combined_name, merge_method
            )
        else:
            print(f'{combined_name} is already loaded')
        model.set_adapter(combined_name)

    print('Active adapters:', model.active_adapters)
    return model




def load_model_and_tokenizer(
    model_name, device_map='auto', use_qlora_config=True, full_prec=False, peft_model_name=None
):
    """
    Receives dict of run config, creates the required HF model and tokenizer
    """
    if use_qlora_config:
        assert not full_prec
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        ) 
        print(f'Loading {model_name} on device {device_map} in 4bit NormalFloat')
    elif full_prec:
        bnb_config=None
        print(f'Loading {model_name} on device {device_map} with full precision')

    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print(f'Loading {model_name} on device {device_map} in 4bit')

    if 'pythia' in model_name.lower():
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/{model_name}", device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    elif 'meta-llama' in model_name.lower():
        from huggingface_hub import login
        env_token = os.environ.get('HF_TOKEN')
        if env_token:
            login(token=env_token)
        else:
            login()

        if CONFIG.use_flash_attention:
            model = AutoModelForCausalLM.from_pretrained(
                f"meta-llama/{model_name}", device_map=device_map,
                quantization_config=bnb_config, attn_implementation='flash_attention_2'
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                f"meta-llama/{model_name}", device_map=device_map,
                quantization_config=bnb_config,
            )
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}")
        # As recommended in https://huggingface.co/docs/transformers/main/en/model_doc/llama3
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    elif 'mistral' in model_name.lower():
        from huggingface_hub import login
        env_token = os.environ.get('HF_TOKEN')
        if env_token:
            login(token=env_token)
        else:
            login()

        model = AutoModelForCausalLM.from_pretrained(
            f"mistralai/{model_name}", device_map=device_map,
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(f"mistralai/{model_name}")
        tokenizer.pad_token_id = tokenizer.bos_token_id

    else:
        raise ValueError(f"Model name invalid: {model_name}")
 
    tokenizer.padding_side = 'left'

    if peft_model_name is not None:
        model = PeftModel.from_pretrained(
            model,
            peft_model_name,
            torch_dtype=torch.float16,
        )

    return model, tokenizer


def make_poly_model(model, n_skills, rank=8, adapter_name='poly-adapter'):
    poly_config = PolyConfig(
        #task_type=TaskType.SEQ_2_SEQ_LM,
        poly_type='poly',
        r=rank,
        n_skills=n_skills,
        n_tasks=1,
        n_splits=1,
    )
    #model = PolyModel(model, poly_config, adapter_name)
    return get_peft_model(model, poly_config, adapter_name=adapter_name)


def load_poly_model_skills(
    model, expert_paths, adapter_name='poly-adapter', device='cpu',
    freeze=None,
):
    if isinstance(model, PeftModel):
        base_model = model.base_model
    else:
        base_model = model

    state_dicts = [
        safe_load_file(expert + '/adapter_model.safetensors', device=device)
        for expert in expert_paths
    ]

    assert base_model.peft_config['poly-adapter'].n_skills == len(state_dicts)
    layer_pat = re.compile(r'\.layers\.(\d+)\.')
    attn_matrix_pat = re.compile(r'self_attn\.(v|q)_proj')

    print('Loading Poly expert modules from:\n' + '    \n'.join(expert_paths))
    for name, module in base_model.named_modules():
        if name.endswith('poly-adapter'):
            module.load_state_dict(
                {'module_logits': torch.tensor([[0.5] * len(expert_paths)]).to(device)}
            )
            continue

        if not 'poly_lora' in name:
            continue

        layer = layer_pat.search(name).groups()[0]
        attn_matrix_name = attn_matrix_pat.search(name).group()
        lora_matrix_name = name[-6:]
        assert lora_matrix_name in ('lora_A', 'lora_B')
        tensors_to_load = []
        for state_dict in state_dicts:
            matches = [
                key for key in state_dict
                if f'layers.{layer}.' in key
                and attn_matrix_name in key
                and lora_matrix_name in key
            ]
            assert len(matches) == 1
            key = matches[0]
            # seems dims in poly module are reversed compared to saved lora!
            tensor = state_dict[key].T
            tensors_to_load.append(tensor)
        # there is a first dim of size 1, not sure why
        new_tensor = torch.stack(tensors_to_load, dim=0).unsqueeze(0)
        module.load_state_dict({adapter_name: new_tensor})

        if freeze == 'skills':
            for param in module.parameters():
                param.requires_grad = False


    if freeze == 'all':
        for param in model.parameters():
            param.requires_grad = False

    return model
