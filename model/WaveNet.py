from keras.layers import Conv1D
from keras.engine import Model
from model.WaveNet_utils import RESIDUAL_BLOCKS, CONNECTION_BLOCKS, OUTPUT_BLOCKS, get_input


def build_WaveNet(config, section):
    dilation_depth = config.get(section, 'dilation_depth')
    filter_size = config.get(section, 'filter_size')
    num_filters = config.get(section, 'num_filters')

    residual_block = RESIDUAL_BLOCKS[config.get(section, 'used_residual_block')]
    connection_block = CONNECTION_BLOCKS[config.get(section, 'used_connection_block')]
    output_block = OUTPUT_BLOCKS[config.get(section, 'used_output_block')]


    sample_input = get_input(config, section)
    full_input = sample_input
    if type(full_input) is list:
        sample_input = full_input[0]
        speaker_input = full_input[1]

    residual_connection = Conv1D(num_filters, filter_size, padding='same', name='Initial_Conv1D')(sample_input)

    skip_connections = []
    for dilation_rate in range(0, dilation_depth + 1):
        residual_connection, skip_connection = residual_block(residual_connection, dilation_rate, config, section)
        skip_connections.append(skip_connection)
        
    output = connection_block(residual_connection, skip_connections, config, section)
    output = output_block(output, config, section)

    return Model(full_input, output)