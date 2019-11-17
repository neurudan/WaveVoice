from wandb.util import set_api_key
from wandb.apis import InternalApi
from wandb import wandb_dir, util
from utils.path_handler import get_sweep_config_path, get_config_path
from shutil import copyfile
from train import train
from time import sleep
from databricks_cli.configure.provider import DatabricksConfig, update_and_persist_config

import wandb
import sys
import os
import os.path
import json
import subprocess
import fcntl


if __name__ == '__main__':
    project_name = 'no_project_666!'
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    if not os.path.isfile(get_sweep_config_path(project_name + '.json')):
        print('create dat damn file or learn typin\'!!!')
    else:
        path = get_config_path() + 'credentials.json'
        credentials = json.load(open(path, 'r'))

        # Setup wandb and create wandb project if it doesn't exist
        entity = credentials['wandb_entity']

        api = InternalApi()
        set_api_key(api, credentials['wandb_api_key'])
        viewer = api.viewer()

        project = None
        projects = api.list_projects(entity=entity)
        for cproject in projects:
            if project_name.lower() == cproject['name'].lower():
                project = cproject
        if project is None:
            project = api.upsert_project(project_name, entity=entity)["name"]

        api.set_setting('entity', entity)
        api.set_setting('project', project_name.lower())
        api.set_setting('base_url', api.settings().get('base_url'))

        util.mkdir_exists_ok(wandb_dir())

        os.environ['WANDB_ENTITY'] = 'bratwolf'
        os.environ['WANDB_AGENT_REPORT_INTERVAL'] = '0'
        os.environ['WANDB_PROJECT'] = project_name.lower()

        # Log in MlFlow and create experiment if not existing
        path = get_config_path() + 'global.json'
        global_settings = json.load(open(path, 'r'))

        if global_settings['mlflow']:
            config = DatabricksConfig.empty()
            new_config = DatabricksConfig.from_password(credentials['mlflow_host'], 
                                                        credentials['mlflow_username'], 
                                                        credentials['mlflow_password'])
            update_and_persist_config(None, new_config)

            os.system('export MLFLOW_TRACKING_URI=databricks')
            subprocess.run('mlflow experiments create -n ' + credentials['mlflow_experiment_base'] + project_name, shell=True, stderr=subprocess.PIPE)


        path = get_sweep_config_path(project_name + '.json')
        sweep_config = json.load(open(path, 'r'))
        sweep_id = wandb.sweep(sweep_config)
        wandb.agent(sweep_id, function=train)

