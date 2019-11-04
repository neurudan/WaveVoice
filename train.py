from utils.data_handler import DataGenerator
from utils.path_handler import get_config_path
from utils.config_handler import Config

from model.WaveNet import build_WaveNet

from keras.engine import Input, Model
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.metrics import categorical_accuracy
from keras.callbacks import CSVLogger, ModelCheckpoint

from wandb.keras import WandbCallback

import wandb
import json
import os
import argparse


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
        'sgd': SGD(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr),
        'adagrad': Adagrad(learning_rate=lr),
        'adadelta': Adadelta(learning_rate=lr),
        'adam': Adam(learning_rate=lr),
        'adamax': Adamax(learning_rate=lr),
        'nadam': Nadam(learning_rate=lr)
    }
    return optimizers[config.get('OPTIMIZER.type')]

def train(config_name=None):
    config = Config(config_name)

    steps_per_epoch = config.get('TRAINING.steps_per_epoch')
    num_epochs = config.get('TRAINING.num_epochs')

    store_required_variables(config)

    # Setup Data-Generator
    data_generator = DataGenerator(config)
    
    train_generator = data_generator.get_generator('train')
    val_generator = data_generator.get_generator('val')

    # Setup Optimizer
    optimizer = setup_optimizer(config)

    # Setup Callback
    (x, y) = val_generator.__next__()
    wandb_cb = WandbCallback(training_data=(x, y), log_weights=True, log_gradients=True)

    # Setup Model
    model = build_WaveNet(config)
    model.summary()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train Model
    model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_generator,
                        validation_steps=int(steps_per_epoch / 10),
                        epochs=num_epochs,
                        callbacks=[wandb_cb])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for the WaveVoice Project')
    parser.add_argument('--project', '-p', dest='project', default='SV',
                        help='The name of the project on wandb. (define first using "wandb init")')
    parser.add_argument('--sweep', '-s', dest='sweep', action='store_true',
                        help='If passed, uses the wandb sweep function for gridsearch.')
    parser.add_argument('--config', '-c', dest='config', default='default.json',
                        help='The config file to use - is only used, if --sweep has not been passed. (default is "default.json")')
    parser.add_argument('--sweep-config', '-sc', dest='sweep_config', default='sweep_default.json',
                        help='The sweep config file to use - is only used, if --sweep has been passed aswell. (default is "sweep_default.json")')

    args = parser.parse_args()
    
    os.environ['WANDB_ENTITY'] = 'bratwolf'
    os.environ['WANDB_AGENT_REPORT_INTERVAL'] = '0'
    os.environ['WANDB_PROJECT'] = args.project

    if args.sweep:
        path = get_config_path(args.sweep_config)
        sweep_config = json.load(open(path, 'r'))
        sweep_id = wandb.sweep(sweep_config)
        wandb.agent(sweep_id, function=train)
    else:
        train(config_name=args.config)
