from keras.layers import Conv1D, Dense, Activation, Lambda, AveragePooling1D, Add, Multiply, RepeatVector, Flatten
from keras.engine import Input
from keras.utils.conv_utils import conv_output_length

import keras.backend as K
import tensorflow as tf

# ==============================================================================
# Causal Conv1D Layer
#
# (original: https://github.com/basveeling/wavenet/blob/master/wavenet_utils.py)
# ==============================================================================

def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    '''Pad the middle dimension of a 3D tensor
    with "left_pad" zeros left and "right_pad" right.
    '''
    pattern = [[0, 0], [left_pad, right_pad], [0, 0]]
    return tf.pad(x, pattern)

class CausalConv1D(Conv1D):
    def __init__(self, filters, kernel_size, 
                 init='glorot_uniform', activation=None, padding='valid', strides=1, dilation_rate=1, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_bias=True, causal=False, **kwargs):
        
        super(CausalConv1D, self).__init__(filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding='valid' if causal else padding,
                                           dilation_rate=dilation_rate,
                                           activation=activation,
                                           use_bias=use_bias,
                                           kernel_initializer=init,
                                           activity_regularizer=activity_regularizer,
                                           bias_regularizer=bias_regularizer,
                                           kernel_constraint=kernel_constraint,
                                           bias_constraint=bias_constraint,
                                           **kwargs)

        self.causal = causal

    def compute_output_shape(self, input_shape):
        input_length = input_shape[1]

        if self.causal:
            input_length += self.dilation_rate[0] * (self.kernel_size[0] - 1)

        length = conv_output_length(input_length,
                                    self.kernel_size[0],
                                    self.padding,
                                    self.strides[0],
                                    dilation=self.dilation_rate[0])

        return (input_shape[0], length, self.filters)

    def call(self, x):
        if self.causal:
            x = asymmetric_temporal_padding(x, self.dilation_rate[0] * (self.kernel_size[0] - 1), 0)
        return super(CausalConv1D, self).call(x)


# ==================
# Create Model Input
# ==================

def get_input(config):
    receptive_field = config.get('MODEL.receptive_field')
    use_ulaw = config.get('DATASET.use_ulaw')

    input = [Input(shape=(receptive_field,), name='Input')]
    if use_ulaw:
        input = [Input(shape=(receptive_field, 256), name='U-Law_Input')]
    
    if config.get('DATASET.condition') == 'speaker':
        num_speakers = config.get('DATASET.num_speakers')
        input.append(Input(shape=(num_speakers,), name='Speaker_Input'))
    return input


# ===============
# Residual Blocks
# ===============

def resblock_cond(input, dilation_rate, config):
    dilation_base = config.get('MODEL.dilation_base')
    filter_size = config.get('MODEL.filter_size')
    num_filters = config.get('MODEL.num_filters')
    causal = config.get('MODEL.causal')
    receptive_field = config.get('MODEL.receptive_field')

    suffix = '-dilation%d'%dilation_rate

    filter_conv_output = CausalConv1D(num_filters, filter_size,
                                 dilation_rate=dilation_base ** dilation_rate, causal=causal,
                                 padding='same', name='Filter_Conv1d'+suffix)(input[0])
    
    gate_conv_output = CausalConv1D(num_filters, filter_size,
                               dilation_rate=dilation_base ** dilation_rate, causal=causal,
                               padding='same', name='Gate_Conv1d'+suffix)(input[0])

    filter_condition = Dense(num_filters, name='Filter_Condition'+suffix)(input[1])
    gate_condition = Dense(num_filters, name='Gate_Condition'+suffix)(input[1])

    broadcasted_filter_condition = RepeatVector(receptive_field, name='Broadcast_Filter_Condition'+suffix)(filter_condition)
    broadcasted_gate_condition = RepeatVector(receptive_field, name='Broadcast_Gate_Condition'+suffix)(gate_condition)

    filter_add_output = Add(name='Filter_Output'+suffix)([filter_conv_output, broadcasted_filter_condition])
    gate_add_output = Add(name='Gate_Output'+suffix)([gate_conv_output, broadcasted_gate_condition])

    filter_output = Activation('tanh', name='Filter_Activation'+suffix)(filter_add_output)
    gate_output = Activation('sigmoid', name='Gate_Activation'+suffix)(gate_add_output)

    gau_output = Multiply(name='Gated_Activation_Unit'+suffix)([filter_output, gate_output])

    skip_output = Conv1D(num_filters, filter_size, padding='same', name='Skip_Connection'+suffix)(gau_output)
    residual_conv_output = Conv1D(num_filters, filter_size, padding='same', name='Residual_Conv1D'+suffix)(gau_output)
    residual_output = Add(name='Residual_Connection'+suffix)([input[0], residual_conv_output])

    return residual_output, skip_output


def resblock_orig(input, dilation_rate, config):
    dilation_base = config.get('MODEL.dilation_base')
    filter_size = config.get('MODEL.filter_size')
    num_filters = config.get('MODEL.num_filters')
    causal = config.get('MODEL.causal')

    suffix = '-dilation%d'%dilation_rate

    filter_output = CausalConv1D(num_filters, filter_size,
                                 dilation_rate=dilation_base ** dilation_rate, activation='tanh', causal=causal,
                                 padding='same', name='Filter'+suffix)(input[0])

    gate_output = CausalConv1D(num_filters, filter_size,
                               dilation_rate=dilation_base ** dilation_rate, activation='sigmoid', causal=causal,
                               padding='same', name='Gate'+suffix)(input[0])


    gau_output = Multiply(name='Gated_Activation_Unit'+suffix)([filter_output, gate_output])

    skip_output = Conv1D(num_filters, filter_size, padding='same', name='Skip_Connection'+suffix)(gau_output)
    residual_output = Conv1D(num_filters, filter_size, padding='same', name='Residual_Conv1D'+suffix)(gau_output)
    residual_output = Add(name='Residual_Connection'+suffix)([input[0], residual_output])

    return residual_output, skip_output


# =================
# Connection Blocks
# =================

def use_skip_connections(residual_connection, skip_connections, config):
    output = Add(name='Add_Skip_Connections')(skip_connections)
    output = Activation('relu')(output)
    return output


def use_residual_connections(residual_connection, skip_connections, config):
    output = Activation('relu')(residual_connection)
    return output


def use_both_connections(residual_connection, skip_connections, config):
    skip_connections.append(residual_connection)
    output = Add(name='Add_Skip_&_Residual_Connections')(skip_connections)
    output = Activation('relu')(output)
    return output


# =============
# Output Blocks
# =============

def output_dense(input, config):
    num_filters = config.get('OUTPUT_DENSE.num_filters')
    output_bins = config.get('DATASET.output_bins')

    output = input
    for i, num_filter in enumerate(num_filters):
        output = Dense(num_filter, activation='relu', name='Embeddings_%d'%i)(output)
    
    output = Dense(output_bins, activation='softmax', name='Output')(output)
    return output


def output_conv_orig(input, config):
    filter_size = config.get('OUTPUT_CONV_ORIG.filter_size')
    select_middle = config.get('OUTPUT_CONV_ORIG.select_middle')
    output_bins = config.get('DATASET.output_bins')
    receptive_field = config.get('MODEL.receptive_field')

    bin_id = -1
    if select_middle:
        bin_id = int((receptive_field - 1) / 2)


    output = Conv1D(output_bins, filter_size[0],
                    activation='relu', padding='same', name='Embeddings')(input[0])

    output = Conv1D(output_bins, filter_size[1],
                    padding='same', name='Conv1D_Output')(output)

    output = Lambda(lambda x: x[:,bin_id,:],
                    output_shape=(output._keras_shape[-1],), name='Select_single_bin')(output)

    output = Activation('softmax', name='Output')(output)
    return output

def output_conv_down(input, config):
    filter_sizes = config.get('OUTPUT_CONV_DOWN.filter_size')
    output_bins = config.get('DATASET.output_bins')

    output = input
    for i, filter_size in enumerate(filter_sizes):
        output = Conv1D(output_bins, filter_size, strides=filter_size, activation='relu',
                        name='Conv1D_Downsample_%d'%i)(output)
    
    output = Dense(output_bins, activation='softmax', name='Output')(output)
    return output


def output_pool_down(input, config):
    downsample_factors = config.get('OUTPUT_POOL_DOWN.downsample_factor')
    filter_sizes = config.get('OUTPUT_POOL_DOWN.filter_size')
    output_bins = config.get('DATASET.output_bins')

    output = input
    for i, [filter_size, downsample_factor] in enumerate(zip(filter_sizes, downsample_factors)):
        output = Conv1D(output_bins, filter_size,
                        padding='same', activation='relu', name='Conv1D_Downsample_%d'%i)(output)
        output = AveragePooling1D(downsample_factor,
                                  padding='same', name='AvgPool_Downsample_%d'%i)(output)
                                  
    output = Dense(output_bins, activation='softmax', name='Output')(output)
    return output


# =========
# Constants
# =========

RESIDUAL_BLOCKS = {'resblock_orig': resblock_orig,
                   'resblock_cond': resblock_cond}
                   

CONNECTION_BLOCKS = {'skip': use_skip_connections,
                     'residual': use_residual_connections,
                     'both': use_both_connections}


OUTPUT_BLOCKS = {'output_dense': output_dense,
                 'output_conv_orig': output_conv_orig,
                 'output_conv_down': output_conv_down,
                 'output_pool_down': output_pool_down}