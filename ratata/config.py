import yaml
from easydict import EasyDict as edict

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)
