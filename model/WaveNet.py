from keras.layers import Conv1D, Lambda
from keras.engine import Model, Input
from model.WaveNet_utils import RESIDUAL_BLOCKS, CONNECTION_BLOCKS, OUTPUT_BLOCKS, EMBEDDING_BLOCKS, get_input, SincConv1D
from model.losses import AngularLoss

import keras.backend as K

def build_WaveNet(config):
    dilation_depth = config.get('MODEL.dilation_depth')
    filter_size = config.get('MODEL.filter_size')
    num_filters = config.get('MODEL.num_filters')
    reverse = config.get('MODEL.reverse')

    loss = config.get('MODEL.loss')
    if loss == 'angular_margin':
        loss = AngularLoss(config)
        config.set('loss', loss)
        loss = loss.angular_loss

    residual_block = RESIDUAL_BLOCKS[config.get('MODEL.residual_block')]
    connection_block = CONNECTION_BLOCKS[config.get('MODEL.connection_block')]
    embedding_block = EMBEDDING_BLOCKS[config.get('MODEL.output_block')]
    output_block = OUTPUT_BLOCKS[config.get('MODEL.output_block')]


    input = get_input(config)
    residual_connection = Conv1D(num_filters, filter_size, padding='same', name='Initial_Conv1D')(input[0])

    skip_connections = []
    residual_connections = []
    for dilation_rate in range(0, dilation_depth + 1):
        res_input = [residual_connection]
        res_input.extend(input[1:])
        residual_connection, skip_connection = residual_block(res_input, dilation_rate, False, config)
        skip_connections.append(skip_connection)

    residual_connections.append(residual_connection)
    if reverse:
        reverse_out = Lambda(lambda x: x[:,::-1,:], output_shape=input[0]._keras_shape[1:])(input[0])
        print(reverse_out._keras_shape)
        residual_connection = Conv1D(num_filters, filter_size, padding='same', name='Initial_Conv1D_reverse')(reverse_out)

        for dilation_rate in range(0, dilation_depth + 1):
            res_input = [residual_connection]
            res_input.extend(input[1:])
            residual_connection, skip_connection = residual_block(res_input, dilation_rate, reverse, config)
            skip_connections.append(skip_connection)
        
        residual_connections.append(residual_connection)

    
    output = connection_block(residual_connections, skip_connections, config)
    output = embedding_block(output, config)
    output = output_block(output, config)

    return Model(input, output), loss


def make_trainable(full_model):
    for layer in full_model.layers[:]:
        layer.trainable = True
    return full_model


def change_output_dense(full_model, config):
    output_block = OUTPUT_BLOCKS[config.get('MODEL.output_block')]

    input = full_model.input
    embedding = full_model.layers[-2].output
    output = output_block(embedding, config)
    model = Model(inputs=input, outputs=output)

    for layer in model.layers[:-1]:
        layer.trainable = False
    return model

