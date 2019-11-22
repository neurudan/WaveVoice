from utils.data_handler import DataGenerator
from utils.path_handler import get_sweep_config_path, get_config_path
from utils.config_handler import Config
from utils.preprocessing import setup_datasets 

from model.WaveNet import build_WaveNet

from keras.engine import Input, Model
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.metrics import categorical_accuracy

from wandb.keras import WandbCallback

from clustering import ClusterCallback

import wandb
import json
import os
import argparse

import mlflow
import mlflow.keras


def update_run_name(config):
    run_name = '_'.join(config.file_name.split('.')[:-1])
    try:
        name_parts = config.get('run_name').split('+')
        name = ''
        for part in name_parts:
            try:
                p = config.get(part)
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
    run_id = config.run.id
    
    api = wandb.Api()
    runs = api.runs(path=os.environ['WANDB_ENTITY']+"/"+os.environ['WANDB_PROJECT'])
    for run in runs:
        if run.id == run_id:
            run.name = run_name
            run.update()
    return run_name

def store_required_variables(config):
    dilation_base = config.get('MODEL.dilation_base')
    dilation_depth = config.get('MODEL.dilation_depth')
    filter_size = config.get('MODEL.filter_size')

    receptive_field = (dilation_base ** dilation_depth) * (filter_size - dilation_base + 1)
    if dilation_base == filter_size:
        receptive_field = filter_size ** dilation_depth
    
    config.set('MODEL.receptive_field', receptive_field)


def setup_optimizer(config):
    lr = config.get('OPTIMIZER.lr')
    optimizers = {
        'sgd': SGD(learning_rate=lr, clipnorm=1.),
        'rmsprop': RMSprop(learning_rate=lr, clipnorm=1.),
        'adagrad': Adagrad(learning_rate=lr, clipnorm=1.),
        'adadelta': Adadelta(learning_rate=lr, clipnorm=1.),
        'adam': Adam(learning_rate=lr, clipnorm=1.),
        'adamax': Adamax(learning_rate=lr, clipnorm=1.),
        'nadam': Nadam(learning_rate=lr, clipnorm=1.)
    }
    return optimizers[config.get('OPTIMIZER.type')]


def train(config_name=None, project_name=None):
    config = Config(config_name)

    num_epochs = config.get('TRAINING.num_epochs')
    batch_type = config.get('DATASET.batch_type')

    store_required_variables(config)


    # Setup Data-Generator
    data_generator = DataGenerator(config)
    
    train_generator = data_generator.get_generator('train')
    train_steps = data_generator.steps_per_epoch

    val_generator = None
    val_steps = None
    if batch_type == 'real':
        val_set = config.get('DATASET.val_set')
        val_generator = data_generator.get_generator('val')
        val_steps = int(train_steps * val_set / (1 - val_set))


    # Setup Optimizer
    optimizer = setup_optimizer(config)


    # Setup Model
    model, loss = build_WaveNet(config)
    model.summary()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    run_name = update_run_name(config)


    # Setup Callback
    wandb_cb = WandbCallback()
    cb = ClusterCallback(config, model, data_generator)


    # Start MlFlow log
    path = get_config_path() + 'global.json'
    global_settings = json.load(open(path, 'r'))

    if global_settings['mlflow']:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("/Users/neurudan@students.zhaw.ch/" + project_name)
        
        mlflow.start_run(run_name=run_name)
        config.set_mlflow_params()
        mlflow.keras.autolog()

    # Train Model
    model.fit_generator(train_generator,
                        steps_per_epoch=train_steps,
                        validation_data=val_generator,
                        validation_steps=val_steps,
                        epochs=num_epochs,
                        callbacks=[wandb_cb, cb])

    # Terminate enqueueing process
    data_generator.terminate_enqueuer()
    if global_settings['mlflow']:
        mlflow.end_run(status='FINISHED')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for the WaveVoice Project')
    parser.add_argument('--project', '-p', dest='project', default=None,
                        help='The name of the project on wandb. (define first using "wandb init")')
    parser.add_argument('--sweep', '-s', dest='sweep', action='store_true',
                        help='If passed, uses the wandb sweep function for gridsearch.')
    parser.add_argument('--config', '-c', dest='config', default='default.json',
                        help='The config file to use - is only used, if --sweep has not been passed. (default is "default.json")')
    parser.add_argument('--sweep-config', '-sc', dest='sweep_config', default='sweep_default.json',
                        help='The sweep config file to use - is only used, if --sweep has been passed aswell. (default is "sweep_default.json")')
    parser.add_argument('--setup', dest='setup', action='store_true',
                        help='If passed, the datasets are being set up.')

    args = parser.parse_args()
    if args.setup:
        setup_datasets()
    else:
        project = args.project
        if project == None:
            project = args.sweep_config.split('.')[0]

        os.environ['WANDB_ENTITY'] = 'bratwolf'
        os.environ['WANDB_AGENT_REPORT_INTERVAL'] = '0'
        os.environ['WANDB_PROJECT'] = project

        if args.sweep:
            path = get_sweep_config_path(args.sweep_config)
            sweep_config = json.load(open(path, 'r'))
            sweep_id = wandb.sweep(sweep_config)
            wandb.agent(sweep_id, function=train)
        else:
            train(config_name=args.config)
