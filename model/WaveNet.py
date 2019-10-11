from keras.layers import Conv1D, Multiply, Add, Activation, Lambda, Dense
from keras.engine import Input, Model

def residual_block(input, num_filters, dilation_rate,
                   filter_size=3, use_bias=True):

    # Gated Activation Unit
    # =====================
    # Filter:
    filter_output = Conv1D(num_filters, filter_size,
                           dilation_rate=dilation_rate,
                           padding='same',
                           use_bias=use_bias,
                           activation='tanh')(input)

    # Gate:
    gate_output = Conv1D(num_filters, filter_size,
                         dilation_rate=dilation_rate,
                         padding='same',
                         use_bias=use_bias,
                         activation='sigmoid')(input)

    output = Multiply()([filter_output, gate_output])

    # Skip & Residual Connection Outputs
    # ==================================
    # Skip Connection
    skip_output = Conv1D(num_filters, filter_size, padding='same', use_bias=use_bias)(output)

    # Residual Connection
    residual_output = Conv1D(num_filters, filter_size, padding='same', use_bias=use_bias)(output)
    residual_output = Add()([input, residual_output])

    return residual_output, skip_output


def wavenet_base(input, num_filters, num_stacks, dilation_depth,
                 filter_size=3, use_bias=True, used_connections='skip'):

    # Initial convolution - leave out?
    output = Conv1D(num_filters, filter_size, padding='same', use_bias=use_bias)(input)

    # Residual Blocks:
    # ================
    skip_connections = []
    for stack in range(num_stacks):
        for dilation_rate in range(0, dilation_depth + 1):
            output, skip_output = residual_block(output, num_filters, dilation_rate,
                                                 filter_size, use_bias)
            skip_connections.append(skip_output)

    # Which connections should be used? skip, residual or both (default is skip)
    if used_connections == 'skip':
        output = Add()(skip_connections)
    elif used_connections == 'both':
        skip_connections.append(output)
        output = Add()(skip_connections)

    return output

# add dense layers after the WaveNet base, needs to be tested, will most probably yield bad results (shape(6561, 256) => shape(512)? decreasement of factor 3280...)
def final_output_dense(input, num_filters, num_output_bins,
                       use_bias=True):
    output = Activation('relu')(input)
    output = Dense(num_filters, use_bias=use_bias)(output)

    output = Activation('relu')(output)
    output = Dense(num_output_bins, use_bias=use_bias)(output)
    output = Activation('softmax')(output)

    return output

# add the output method mentioned in the WaveNet Paper
def final_output_conv(input, num_output_bins, utterance_length,
                      use_bias=True, output_filter_size=1):
    output = Activation('relu')(input)
    output = Conv1D(num_output_bins, output_filter_size, padding='same', use_bias=use_bias)(output)

    output = Activation('relu')(output)
    output = Conv1D(num_output_bins, output_filter_size, padding='same', use_bias=use_bias)(output)

    middle_bin_id = int((utterance_length - 1) / 2)
    output = Lambda(lambda x: x[:,middle_bin_id,:], output_shape=(output._keras_shape[-1],))(output)
    output = Activation('softmax')(output)

    return output
