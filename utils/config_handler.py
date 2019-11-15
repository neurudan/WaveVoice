import wandb
import json
import mlflow

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

    def set_mlflow_params(self):
        keys = ['TRAINING', 'MODEL', 'DATASET', 'ANGULAR_LOSS']

        for key in keys:
            dic = wandb.config.get(key)
            sub_keys = list(dic.keys())
            if key in self.store:
                sub_keys.extend(list(self.store[key].keys()))
            
            for sub_key in sub_keys:
                val = self.get(key + '.' + sub_key)
                mlflow.log_param(key + '.' + sub_key, val)

    def get(self, key):
        try:
            val = self.store
            for t in key.split('.'):
                val = val[t]
            return val
        except:
            pass
        val = wandb.config.get('01;SWEEP;' + key.replace('.',';'))
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