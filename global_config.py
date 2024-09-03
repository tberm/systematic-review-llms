from copy import deepcopy
from pathlib import Path
from datetime import datetime
import yaml


class GlobalConfig:
    """
    Globally available configuration.
    We assume we will never be running more than one experiment per process so this can
    contain run config
    """
    # default params
    run_config = None
    train_config = None
    wandb_config = None
    adapter_config = None
    skip_repo_check = False
    model = None
    config_source = None
    use_gpu = False
    use_qlora_config = True
    full_prec = False
    use_biomed_ft = False
    use_flash_attention = False
    max_title_chars = 500
    max_abstract_chars = 8000

    def __init__(self):
        self.start_ts = datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')

    def set_config(self, yaml_path):
        self.run_config = self.run_config or {}
        self.train_config = self.train_config or {}
        self.wandb_config = self.wandb_config or {}
        self.config_source = yaml_path.name
        with yaml_path.open(encoding='utf-8') as config_file:
            config = yaml.load(config_file, Loader=yaml.SafeLoader)
            for attr in config:
                if hasattr(self, attr) and isinstance(getattr(self, attr), dict):
                    current_dict = getattr(self, attr)
                    current_dict.update(config[attr])
                else:
                    setattr(self, attr, config[attr])

    @property
    def model_class(self):
        if 'llama' in self.model.lower():
            return 'llama'
        if 'mistral' in self.model.lower():
            return 'mistral'
        return 'unknown'

    @property
    def include_space_in_labels(self):
        """
        Depending on how prompts are formatted, we may or may not expect a space before
        the model answer
        """
        return 'pythia' in self.model

    @property
    def device_map(self):
        if self.use_gpu:
            return 'auto'
        return 'cpu'

    @property
    def device(self):
        if self.use_gpu:
            return 'cuda'
        return 'cpu'

    @property
    def log_to_wandb(self):
        return hasattr(self, 'wandb_config') and self.wandb_config.get('log') is True

    @property
    def use_instruct_format(self):
        return 'instruct' in self.model.lower()

    @property
    def run_id(self):
        if self.wandb_config.get('run_id'):
            return self.wandb_config['run_id']
        return self.start_ts

    @property
    def run_name(self):
        if self.wandb_config.get('run_name'):
            return self.wandb_config['run_name']
        return self.run_id

    @property
    def checkpoints_dir(self):
        return Path(self.train_config['models_dir']) / (self.run_name + '-checkpoints')

    @property
    def final_model_dir(self):
        return Path(self.train_config['models_dir']) / (self.run_name + '-final')

    def __repr__(self):
        return f'GlobalConfig({vars(self)})'


CONFIG = GlobalConfig()
