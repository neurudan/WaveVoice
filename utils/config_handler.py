from utils.path_handler import get_config_path, create_result_dir

import configparser
import json


class Config():
    def __init__(self, file_name):
        path = get_config_path(file_name)
        self.file_name = file_name
        self.config = configparser.ConfigParser()
        self.config.read_file(open(path))
        self.result_dir = create_result_dir(self)
        self.store = {}

    def get(self, topic, key, subkey=None, store=False):
        if store:
            return self.store[topic][key]
        else:
            res = json.loads(self.config.get(topic, key))
            if subkey is not None:
                if type(subkey) is list:
                    for k in subkey:
                        res = res[k]
                else:
                    res = res[subkey]
            return res
        
    def set(self, topic, key, val, store=False):
        if store:
            if topic not in self.store:
                self.store[topic] = {}
            self.store[topic][key] = val
        else:
            val = json.dumps(val)
            try:
                self.config.set(topic, key, val)
            except configparser.NoSectionError:
                self.config.add_section(topic)
                self.config.set(topic, key, val)


    def get_result_dir(self):
        return self.result_dir

    def get_file_name(self):
        return self.file_name