from utils.data_handler import DataGenerator, get_speakers
from utils.path_handler import get_config_path, create_result_dir
from model.WaveNet import *

from keras.engine import Input, Model
from keras.optimizers import Adadelta, Adam
from keras.metrics import categorical_accuracy
from keras.callbacks import CSVLogger, ModelCheckpoint

import argparse
import configparser

DEFAULT_CONFIG = 'default.cfg'

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read_file(open(config_path))
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training for speaker recognition')
    parser.add_argument('-c', dest='config_name', default=DEFAULT_CONFIG,
                        help='The config to use for training')

    args = parser.parse_args()

    config_name = args.config_name
    config = load_config(get_config_path() + config_name)

    dataset = config['DATASET']['base']
    speaker_list = config['DATASET']['speaker_list']
    use_ulaw = bool(config['DATASET']['use_ulaw'])

    dilation_base = int(config['WAVENET']['dilation_base'])
    dilation_depth = int(config['WAVENET']['dilation_depth'])
    filter_size = int(config['WAVENET']['filter_size'])
    num_stacks = int(config['WAVENET']['num_stacks'])
    num_filters = int(config['WAVENET']['num_filters'])
    used_output = config['WAVENET']['used_output']
    used_resblock = config['WAVENET']['used_resblock']
    used_optimizer = config['WAVENET']['used_optimizer']

    dense_n_hidden = int(config['DENSE_OUT']['n_hidden'])

    conv_v2_filter_size = int(config['CONV_OUT_V2']['filter_size'])

    conv_v3_downsample_factor = int(config['CONV_OUT_V3']['downsample_factor'])

    adam_lr = float(config['ADAM']['adam_lr'])
    adam_beta_1 = float(config['ADAM']['adam_beta_1'])
    adam_beta_2 = float(config['ADAM']['adam_beta_2'])

    batch_size = int(config['TRAINING']['batch_size'])
    num_epochs = int(config['TRAINING']['num_epochs'])
    steps_per_epoch = int(config['TRAINING']['steps_per_epoch'])

    utterance_length = (dilation_base ** dilation_depth) * (filter_size - dilation_base + 1)
    if dilation_base == filter_size:
        utterance_length = filter_size ** dilation_depth

    speakers = get_speakers(dataset, speaker_list)
    num_output_bins = len(speakers)


    optimizers = {'adadelta': Adadelta(),
                  'adam': Adam(adam_lr, adam_beta_1, adam_beta_2)}
    optimizer = optimizers[used_optimizer]


    dg = DataGenerator(dataset, utterance_length, batch_size, speakers, use_ulaw)
    bg = dg.batch_generator()
    bg.__next__()


    input = Input(shape=(utterance_length,1), name='input')
    if use_ulaw:
        input = Input(shape=(utterance_length,256), name='input')


    resblocks = {'v1': residual_block_v1,
                 'v2': residual_block_v2}
    wavenet = wavenet_base(input, num_filters, num_stacks, dilation_base, dilation_depth, resblocks[used_resblock])

    output = None
    if used_output == 'dense':
        output = final_output_dense(wavenet, dense_n_hidden, num_output_bins)
    elif used_output == 'conv_v1':
        output = final_output_conv_v1(wavenet, num_output_bins, utterance_length)
    elif used_output == 'conv_v2':
        output = final_output_conv_v2(wavenet, conv_v2_filter_size, num_output_bins, utterance_length)
    elif used_output == 'conv_v3':
        output = final_output_conv_v3(wavenet, filter_size, conv_v3_downsample_factor, num_output_bins, utterance_length)

    result_folder = create_result_dir(config_name)

    csv_logger = CSVLogger(result_folder + 'logs.csv')
    net_saver = ModelCheckpoint(result_folder + 'best.h5', monitor='accuracy', save_best_only=True)

    model = Model(input, output)
    model.summary()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(bg, steps_per_epoch=100, epochs=num_epochs, callbacks=[csv_logger, net_saver])
    model.save(result_folder + 'final.h5')
