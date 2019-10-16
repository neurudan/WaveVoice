from keras.layers import *
from keras.engine import Input, Model


def residual_block_v1(input, num_filters, dilation_base, dilation_rate, stack,
                      filter_size=3, use_bias=True):

    # Gated Activation Unit
    # =====================
    output = Conv1D(num_filters, filter_size,
                           dilation_rate=dilation_base ** dilation_rate,
                           padding='same',
                           use_bias=use_bias,
                           activation='tanh',
                           name='Conv1D-stack%d_dilation%d'%(stack, dilation_rate))(input)

    # Filter:
    filter_output = Activation('tanh', name='filter-stack%d_dilation%d'%(stack, dilation_rate))(output)

    # Gate:
    gate_output = Activation('sigmoid', name='gate-stack%d_dilation%d'%(stack, dilation_rate))(output)

    output = Multiply(name='Gated_Activation_Unit-stack%d_dilation%d'%(stack, dilation_rate))([filter_output, gate_output])


    # Skip & Residual Connection Outputs
    # ==================================
    # Skip Connection
    skip_output = Conv1D(num_filters, filter_size,
                         padding='same',
                         use_bias=use_bias,
                         name='skip-stack%d_dilation%d'%(stack, dilation_rate))(output)

    # Residual Connection
    residual_output = Add(name='residual-stack%d_dilation%d'%(stack, dilation_rate))([input, skip_output])

    return residual_output, skip_output


def residual_block_v2(input, num_filters, dilation_base, dilation_rate, stack,
                      filter_size=3, use_bias=True):

    # Gated Activation Unit
    # =====================
    # Filter:
    filter_output = Conv1D(num_filters, filter_size,
                           dilation_rate=dilation_base ** dilation_rate,
                           padding='same',
                           use_bias=use_bias,
                           activation='tanh',
                           name='filter-stack%d_dilation%d'%(stack, dilation_rate))(input)

    # Gate:
    gate_output = Conv1D(num_filters, filter_size,
                         dilation_rate=dilation_base ** dilation_rate,
                         padding='same',
                         use_bias=use_bias,
                         activation='sigmoid',
                         name='gate-stack%d_dilation%d'%(stack, dilation_rate))(input)

    output = Multiply()([filter_output, gate_output])


    # Skip & Residual Connection Outputs
    # ==================================
    # Skip Connection
    skip_output = Conv1D(num_filters, filter_size,
                         padding='same',
                         use_bias=use_bias,
                         name='skip-stack%d_dilation%d'%(stack, dilation_rate))(output)

    # Residual Connection
    residual_output = Conv1D(num_filters, filter_size,
                             padding='same',
                             use_bias=use_bias,
                             name='residual_conv-stack%d_dilation%d'%(stack, dilation_rate))(output)
    residual_output = Add(name='residual_add-stack%d_dilation%d'%(stack, dilation_rate))([input, residual_output])

    return residual_output, skip_output


def wavenet_base(input, num_filters, num_stacks, dilation_base, dilation_depth, block_func,
                 filter_size=3, use_bias=True, used_connections='skip'):

    # Initial convolution - leave out?
    output = Conv1D(num_filters, filter_size,
                    padding='same',
                    use_bias=use_bias,
                    name='Initial_Conv1D')(input)

    # Residual Blocks:
    # ================
    skip_connections = []
    for stack in range(num_stacks):
        for dilation_rate in range(0, dilation_depth + 1):
            output, skip_output = block_func(output, num_filters, dilation_base, dilation_rate,
                                             stack, filter_size, use_bias)
            skip_connections.append(skip_output)

    # Which connections should be used? skip, residual or both (default is skip)
    if used_connections == 'skip':
        output = Add(name='add_skip_connections')(skip_connections)
    elif used_connections == 'both':
        skip_connections.append(output)
        output = Add(name='add-skip&residual_connections')(skip_connections)

    output = Activation('relu')(output)
    return output


def final_output_dense(input, num_filters, num_output_bins,
                       use_bias=True):
    output = Flatten()(input)
    if type(num_filters) is list:
        for i, num_filter in enumerate(num_filters):
            output = Dense(num_filter,
                           use_bias=use_bias,
                           activation='relu',
                           name='Dense_%d_final'%i)(output)
    else:
        output = Dense(num_filters,
                       use_bias=use_bias,
                       activation='relu',
                       name='Dense_1_final')(output)

    output = Dense(num_output_bins,
                   use_bias=use_bias,
                   activation='softmax',
                   name='Dense_final')(output)
    return output

# add the output method mentioned in the WaveNet Paper
def final_output_conv_v1(input, num_output_bins, utterance_length,
                         use_bias=True, output_filter_size=1):
    output = Conv1D(num_output_bins, output_filter_size,
                    padding='same',
                    use_bias=use_bias,
                    activation='relu',
                    name='Conv1D_1_final')(input)

    output = Conv1D(num_output_bins, output_filter_size,
                    padding='same',
                    use_bias=use_bias,
                    name='Conv1D_2_final')(output)

    middle_bin_id = int((utterance_length - 1) / 2)
    output = Lambda(lambda x: x[:,middle_bin_id,:],
                    output_shape=(output._keras_shape[-1],),
                    name='Select_only_bin_in_middle')(output)

    output = Activation('softmax', name='Softmax_final')(output)

    return output

def final_output_conv_v2(input, filter_size, num_output_bins, utterance_length,
                         use_bias=True):
    output = input
    i = 0
    while utterance_length > 1:
        output = Conv1D(num_output_bins, filter_size,
                        strides=filter_size,
                        padding='valid',
                        use_bias=use_bias,
                        activation='relu',
                        name='Conv1D_Downsample_%d'%i)(output)
        utterance_length /= filter_size
        i += 1
    output = Reshape((num_output_bins,))(output)
    output = Activation('softmax', name='Softmax_final')(output)

    return output


def final_output_conv_v3(input, filter_size, downsample_factor, num_output_bins, utterance_length,
                         use_bias=True):
    output = input
    i = 0
    while utterance_length > 1:
        output = Conv1D(num_output_bins, filter_size,
                        padding='same',
                        use_bias=use_bias,
                        activation='relu',
                        name='Conv1D_Downsample_%d'%i)(output)
        output = AveragePooling1D(downsample_factor,
                                  padding='same',
                                  name='AvgPooling_Downsample_%d'%i)(output)
        utterance_length /= downsample_factor
        i += 1
    output = Flatten()(output)
    output = Activation('softmax', name='Softmax_final')(output)

    return output
