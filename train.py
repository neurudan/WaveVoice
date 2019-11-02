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

os.environ['WANDB_ENTITY'] = "bratwolf"
os.environ['WANDB_PROJECT'] = "SV"


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

def train():
    config = Config()

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
    x, y = val_generator.__next__()
    print(x.shape)
    print(y.shape)
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
    path = get_config_path('sweep.json')
    sweep_config = json.load(open(path, 'r'))
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train)
