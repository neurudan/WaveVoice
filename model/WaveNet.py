from keras.layers import Conv1D
from keras.engine import Model
from model.WaveNet_utils import RESIDUAL_BLOCKS, CONNECTION_BLOCKS, OUTPUT_BLOCKS, get_input, SincConv1D


def build_WaveNet(config):
    dilation_depth = config.get('MODEL.dilation_depth')
    filter_size = config.get('MODEL.filter_size')
    num_filters = config.get('MODEL.num_filters')

    residual_block = RESIDUAL_BLOCKS[config.get('MODEL.residual_block')]
    connection_block = CONNECTION_BLOCKS[config.get('MODEL.connection_block')]
    output_block = OUTPUT_BLOCKS[config.get('MODEL.output_block')]


    input = get_input(config)
    residual_connection = Conv1D(num_filters, filter_size, padding='same', name='Initial_Conv1D')(input[0])
    #residual_connection = SincConv1D(num_filters, 251, name='SincNet_Conv1D')(input[0])

    skip_connections = []
    for dilation_rate in range(0, dilation_depth + 1):
        res_input = [residual_connection]
        res_input.extend(input[1:])
        residual_connection, skip_connection = residual_block(res_input, dilation_rate, config)
        skip_connections.append(skip_connection)
    
    output = connection_block(residual_connection, skip_connections, config)
    output = output_block(output, config)

    return Model(input, output)