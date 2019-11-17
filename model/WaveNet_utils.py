from keras.layers import Conv1D, Dense, Activation, Lambda, AveragePooling1D, Add, Multiply, RepeatVector, Flatten, Layer, Dropout
from keras.engine import Input
from keras.utils.conv_utils import conv_output_length

import keras.backend as K
import tensorflow as tf
import numpy as np

import math


# ===========================================================================
# Conv1D Layer with SincNet filters
# 
# (original: https://github.com/grausof/keras-sincnet/blob/master/sincnet.py)
# ===========================================================================


def debug_print(*objects):
    print(*objects)

def sinc(band, t_right):
    y_right = K.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = K.reverse(y_right, 0)
    y = K.concatenate([y_left, K.variable(K.ones(1)), y_right])
    return y

class SincConv1D(Layer):
    def __init__(self, num_filters, kernel_size, fs=16000,**kwargs):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.fs = fs

        super(SincConv1D, self).__init__(**kwargs)


    def build(self, input_shape):
        # The filters are trainable parameters.
        self.filt_b1 = self.add_weight(name='filt_b1', shape=(self.num_filters,), initializer='uniform', trainable=True)
        self.filt_band = self.add_weight(name='filt_band', shape=(self.num_filters,), initializer='uniform', trainable=True)

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))          # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.num_filters)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1))                       # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.freq_scale=self.fs * 1.0
        self.set_weights([b1/self.freq_scale, (b2-b1)/self.freq_scale])
        
        # Be sure to call this at the end
        super(SincConv1D, self).build(input_shape) 

    def call(self, x):

        debug_print("call")
        #filters = K.zeros(shape=(filters, kernel_size))

        # Get beginning and end frequencies of the filters.
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = K.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (K.abs(self.filt_band) + min_band / self.freq_scale)

        # Filter window (hamming).
        n = np.linspace(0, self.kernel_size, self.kernel_size)
        window = 0.54 - 0.46 * K.cos(2 * math.pi * n / self.kernel_size)
        window = K.cast(window, "float32")
        window = K.variable(window)
        debug_print("  window", window)

        # TODO what is this?
        t_right_linspace = np.linspace(1, (self.kernel_size - 1) / 2, int((self.kernel_size -1) / 2))
        t_right = K.variable(t_right_linspace / self.fs)
        debug_print("  t_right", t_right)

        # Compute the filters.
        output_list = []
        for i in range(self.num_filters):
            low_pass1 = 2 * filt_beg_freq[i] * sinc(filt_beg_freq[i] * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i] * sinc(filt_end_freq[i] * self.freq_scale, t_right)
            band_pass= (low_pass2 - low_pass1)
            band_pass = band_pass / K.max(band_pass)
            output_list.append(band_pass * window)
        filters = K.stack(output_list) #(80, 251)
        filters = K.transpose(filters) #(251, 80)
        
        #(251,1,80) in TF: (filter_width, in_channels, out_channels) in PyTorch (out_channels, in_channels, filter_width)
        filters = K.reshape(filters, (self.kernel_size, 1, self.num_filters))   
        '''
        Given an input tensor of shape [batch, in_width, in_channels] if data_format is "NWC", 
        or [batch, in_channels, in_width] if data_format is "NCW", and a filter / kernel tensor of shape [filter_width, in_channels, out_channels], 
        this op reshapes the arguments to pass them to conv2d to perform the equivalent convolution operation.
        Internally, this op reshapes the input tensors and invokes tf.nn.conv2d. For example, if data_format does not start with "NC", 
        a tensor of shape [batch, in_width, in_channels] is reshaped to [batch, 1, in_width, in_channels], and the filter is reshaped to 
        [1, filter_width, in_channels, out_channels]. The result is then reshaped back to [batch, out_width, out_channels] 
        (where out_width is a function of the stride and padding as in conv2d) and returned to the caller.
        '''

        # Do the convolution.
        debug_print("call")
        debug_print("  x", x)
        debug_print("  filters", filters)
        out = K.conv1d(x, kernel=filters, padding='same')
        debug_print("  out", out)

        return out

    def compute_output_shape(self, input_shape):
        new_size = conv_output_length(input_shape[1], self.kernel_size,
                                      padding="same", stride=1, dilation=1)
        return (input_shape[0],) + (new_size,) + (self.num_filters,)


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
    data_type = config.get('DATASET.data_type')

    input = None
    if data_type == 'original':
        input = [Input(shape=(receptive_field, 1), name='Input')]
    elif data_type == 'ulaw':
        input = [Input(shape=(receptive_field, 256), name='U-Law_Input')]
    elif data_type == 'mel':
        input = [Input(shape=(receptive_field, 128), name='Mel_Input')]
    
    if config.get('DATASET.condition') == 'speaker':
        num_speakers = config.get('DATASET.num_speakers')
        input.append(Input(shape=(num_speakers,), name='Speaker_Input'))
    return input


# ===============
# Residual Blocks
# ===============

def resblock_cond(input, dilation_rate, reverse, config):
    dilation_base = config.get('MODEL.dilation_base')
    filter_size = config.get('MODEL.filter_size')
    num_filters = config.get('MODEL.num_filters')
    causal = config.get('MODEL.causal')
    receptive_field = config.get('MODEL.receptive_field')
    res_drop_rate = config.get('MODEL.res_drop_rate')
    skip_drop_rate = config.get('MODEL.skip_drop_rate')

    suffix = '-dilation%d'%dilation_rate
    if reverse:
        suffix += '-reverse'

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
    skip_output = Dropout(skip_drop_rate)(skip_output)

    residual_conv_output = Conv1D(num_filters, filter_size, padding='same', name='Residual_Conv1D'+suffix)(gau_output)
    residual_conv_output = Dropout(res_drop_rate)(residual_conv_output)
    residual_output = Add(name='Residual_Connection'+suffix)([input[0], residual_conv_output])

    return residual_output, skip_output


def resblock_orig(input, dilation_rate, reverse, config):
    dilation_base = config.get('MODEL.dilation_base')
    filter_size = config.get('MODEL.filter_size')
    num_filters = config.get('MODEL.num_filters')
    causal = config.get('MODEL.causal')
    res_drop_rate = config.get('MODEL.res_drop_rate')
    skip_drop_rate = config.get('MODEL.skip_drop_rate')

    suffix = '-dilation%d'%dilation_rate
    if reverse:
        suffix += '-reverse'

    filter_output = CausalConv1D(num_filters, filter_size,
                                 dilation_rate=dilation_base ** dilation_rate, activation='tanh', causal=causal,
                                 padding='same', name='Filter'+suffix)(input[0])

    gate_output = CausalConv1D(num_filters, filter_size,
                               dilation_rate=dilation_base ** dilation_rate, activation='sigmoid', causal=causal,
                               padding='same', name='Gate'+suffix)(input[0])


    gau_output = Multiply(name='Gated_Activation_Unit'+suffix)([filter_output, gate_output])

    skip_output = Conv1D(num_filters, filter_size, padding='same', name='Skip_Connection'+suffix)(gau_output)
    skip_output = Dropout(skip_drop_rate)(skip_output)

    residual_output = Conv1D(num_filters, filter_size, padding='same', name='Residual_Conv1D'+suffix)(gau_output)
    residual_output = Dropout(res_drop_rate)(residual_output)
    residual_output = Add(name='Residual_Connection'+suffix)([input[0], residual_output])

    return residual_output, skip_output


# =================
# Connection Blocks
# =================

def use_skip_connections(residual_connections, skip_connections, config):
    output = Add(name='Add_Residual_Connections')(skip_connections)
    output = Activation('relu')(output)
    return output


def use_residual_connections(residual_connections, skip_connections, config):
    output = Add(name='Add_Skip_Connections')(skip_connections)
    output = Activation('relu')(output)
    return output


def use_both_connections(residual_connections, skip_connections, config):
    skip_connections.extend(residual_connections)
    output = Add(name='Add_Skip_&_Residual_Connections')(skip_connections)
    output = Activation('relu')(output)
    return output


# =============
# Output Blocks
# =============

# This should be the same as output_conv_orig

def output_dense(input, config):
    embedding_size = config.get('MODEL.embedding_size')
    select_middle = config.get('MODEL.select_middle')
    output_bins = config.get('MODEL.output_bins')
    receptive_field = config.get('MODEL.receptive_field')
    dense_drop_rate = config.get('MODEL.dense_drop_rate')

    bin_id = -1
    if select_middle:
        bin_id = int((receptive_field - 1) / 2)

    output = Lambda(lambda x: x[:,bin_id,:],
                    output_shape=(input._keras_shape[-1],), name='Select_single_bin')(input)

    output = Dense(embedding_size, activation='relu', name='Embeddings')(output)
    output = Dropout(dense_drop_rate)(output)
    
    if config.get('MODEL.loss') == 'angular_margin':
        dense = config.get('loss').get_dense()
        output = dense(name='Output')(output)
    else:
        output = Dense(output_bins, activation='softmax', name='Output')(output)
    return output


def output_conv(input, config):
    embedding_size = config.get('MODEL.embedding_size')
    select_middle = config.get('MODEL.select_middle')
    output_bins = config.get('MODEL.output_bins')
    receptive_field = config.get('MODEL.receptive_field')
    label = config.get('DATASET.label')

    output = Conv1D(embedding_size, 1, activation='relu', padding='same', name='Embeddings')(input)
    output = Conv1D(output_bins, 1, activation='softmax', padding='same', name='Conv1D_Output')(output)

    if label != 'all_timesteps':
        bin_id = -1
        if select_middle:
            bin_id = int((receptive_field - 1) / 2)
        output = Lambda(lambda x: x[:,bin_id,:],
                        output_shape=(output._keras_shape[-1],), name='Select_single_bin')(output)
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
                 'output_conv': output_conv}