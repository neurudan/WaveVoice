import wandb
import json

from utils.path_handler import get_config_path

class Config():
    def __init__(self, path):
        wandb.init()
        if path is None:
            path = get_config_path(wandb.config.get('config_name'))
        else:
            path = get_config_path(path)
        dic = json.load(open(path, 'r'))
        wandb.config.update(dic)
        self.store = {}

    def get(self, key):
        try:
            val = self.store
            for t in key.split('.'):
                val = val[t]
            return val
        except:
            pass
        val = wandb.config.get('SWEEP.' + key)
        if val is None:
            keys = key.split('.')
            val = wandb.config.get(keys[0])
            for key in keys[1:]:
                val = val[key]
        return val
        
    def set(self, key, val):
        dic = self.store
        keys = key.split('.')
        for key in keys[:-1]:
            if key not in dic:
                dic[key] = {}
            dic = dic[key]
        dic[keys[-1]] = val