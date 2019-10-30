from utils.data_handler import DataGenerator, get_speakers
from utils.path_handler import create_result_dir
from utils.config_handler import Config

from model.WaveNet import build_WaveNet

from keras.engine import Input, Model
from keras.optimizers import Adadelta, Adam
from keras.metrics import categorical_accuracy
from keras.callbacks import CSVLogger, ModelCheckpoint

import argparse

DEFAULT_CONFIG = 'default.cfg'


def store_required_variables(config, section):
    dilation_base = config.get(section, 'dilation_base')
    dilation_depth = config.get(section, 'dilation_depth')
    filter_size = config.get(section, 'filter_size')

    receptive_field = (dilation_base ** dilation_depth) * (filter_size - dilation_base + 1)
    if dilation_base == filter_size:
        receptive_field = filter_size ** dilation_depth
    
    config.set(section, 'receptive_field', receptive_field)


def setup_optimizer(config):
    adam_lr = config.get('TRAINING', 'adam', 'lr')
    adam_beta_1 = config.get('TRAINING', 'adam', 'beta_1')
    adam_beta_2 = config.get('TRAINING', 'adam', 'beta_2')
    
    optimizers = {'adadelta': Adadelta(),
                  'adam': Adam(adam_lr, adam_beta_1, adam_beta_2)}

    return optimizers


def create_callbacks(config):
    csv_logger = CSVLogger(config.result_dir + 'logs.csv')
    net_saver_best = ModelCheckpoint(config.result_dir + 'best.h5', monitor='accuracy', save_best_only=True)
    net_saver = ModelCheckpoint(config.result_dir + 'final.h5')
    return [csv_logger, net_saver_best, net_saver]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training for speaker recognition')
    parser.add_argument('-c', dest='config_name', default=DEFAULT_CONFIG,
                        help='The config to use for training')

    args = parser.parse_args()
    config = Config(args.config_name)

    used_model = config.get('GENERAL', 'model')

    steps_per_epoch = config.get('TRAINING', 'steps_per_epoch')
    num_epochs = config.get('TRAINING', 'num_epochs')

    store_required_variables(config, used_model)

    # Setup Data-Generator
    data_generator = DataGenerator(config, used_model)

    train_generator = data_generator.get_generator('train')
    val_generator = data_generator.get_generator('val')

    # Setup Optimizer
    optimizer = setup_optimizer(config)[config.get('TRAINING', 'used_optimizer')]

    # Setup Model
    model = build_WaveNet(config, used_model)
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
                        callbacks=create_callbacks(config))
