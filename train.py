from utils.path_handler import get_config_path
from utils.config_handler import Config
from training.NormalSetting import train_normal
from training.HyperepochSetting import train_hyperepochs

import os

import json
import mlflow
import mlflow.keras
import atexit


@atexit.register
def terminate_subprocesses():
    os.system('kill -9 $(pgrep -f "python main.py")')


def train(config_name=None, project_name=None):
    # Initialize Config Handler
    config = Config(config_name)
    config.store_required_variables()
    run_name = config.update_run_name()

    # Start MlFlow log
    path = get_config_path() + 'global.json'
    global_settings = json.load(open(path, 'r'))

    if global_settings['mlflow']:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("/Users/neurudan@students.zhaw.ch/" + project_name)
        
        mlflow.start_run(run_name=run_name)
        config.set_mlflow_params()
        mlflow.keras.autolog()

    # Normal training setting
    if config.get('TRAINING.setting') == 'normal':
        train_normal(config)
    # Hyperepoch training setting
    elif config.get('TRAINING.setting') == 'hyperepochs':
        train_hyperepochs(config)

    # Finish mlFlow run
    if global_settings['mlflow']:
        mlflow.end_run(status='FINISHED')
