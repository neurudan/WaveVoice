import wandb
import json

from utils.path_handler import get_base_config_path
from utils.data_handler import get_speaker_list


class Config():
    def __init__(self, file_name):
        self.store = {}
        self.run = wandb.init()
        if file_name is None:
            file_name = wandb.config.get('config_name')
        self.file_name = file_name

        path = get_base_config_path(self.file_name)

        dic = json.load(open(path, 'r'))
        wandb.config.update(dic)

    def get(self, key):
        try:
            val = self.store
            for t in key.split('.'):
                val = val[t]
            return val
        except:
            pass
        val = wandb.config.get('01.SWEEP.' + key)
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