import wandb
import json
import mlflow
import os

from utils.path_handler import get_base_config_path
from data_handlers.train_data_handler import get_speaker_list


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
        keys = ['TRAINING', 'MODEL', 'DATASET', 'ANGULAR_LOSS', 'HYPEREPOCH']

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
    
    def update_run_name(self):
        run_name = '_'.join(self.file_name.split('.')[:-1])
        try:
            name_parts = self.get('run_name').split('+')
            name = ''
            for part in name_parts:
                try:
                    p = self.get(part)
                    if p is None:
                        name += part
                    else:
                        if type(p) is int:
                            name += '%03d'%p
                        else:
                            name += str(p)
                except:
                    name += part
            run_name = name
        except:
            pass
        run_id = self.run.id
        
        api = wandb.Api()
        runs = api.runs(path=os.environ['WANDB_ENTITY']+"/"+os.environ['WANDB_PROJECT'])
        for run in runs:
            if run.id == run_id:
                run.name = run_name
                run.update()
        return run_name

    def store_required_variables(self):
        dilation_base = self.get('MODEL.dilation_base')
        dilation_depth = self.get('MODEL.dilation_depth')
        filter_size = self.get('MODEL.filter_size')

        receptive_field = (dilation_base ** dilation_depth) * (filter_size - dilation_base + 1)
        if dilation_base == filter_size:
            receptive_field = filter_size ** dilation_depth
        
        self.set('MODEL.receptive_field', receptive_field)        

        label = self.get('DATASET.label')

        train_speakers = get_speaker_list(self)
        num_speakers = len(train_speakers)

        if self.get('TRAINING.setting') == 'hyperepochs':
            num_speakers = self.get('HYPEREPOCH.num_speakers')

        if label in ['single_timestep', 'all_timesteps']:
            self.set('MODEL.output_bins', 256)
        elif label == 'speaker':
            self.set('MODEL.output_bins', num_speakers)
        self.set('DATASET.num_speakers', num_speakers)