from keras.layers import Conv1D, Lambda
from keras.engine import Model
from model.WaveNet_utils import RESIDUAL_BLOCKS, CONNECTION_BLOCKS, OUTPUT_BLOCKS, get_input, SincConv1D
from model.losses import AngularLoss

import keras.backend as K

def build_WaveNet(config):
    dilation_depth = config.get('MODEL.dilation_depth')
    filter_size = config.get('MODEL.filter_size')
    num_filters = config.get('MODEL.num_filters')
    reverse = config.get('MODEL.reverse')

    loss = 'categorical_crossentropy'
    if config.get('MODEL.loss') == 'angular_margin':
        loss = AngularLoss(config)
        config.set('loss', loss)
        loss = loss.angular_loss

    residual_block = RESIDUAL_BLOCKS[config.get('MODEL.residual_block')]
    connection_block = CONNECTION_BLOCKS[config.get('MODEL.connection_block')]
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
    output = output_block(output, config)

    return Model(input, output), loss