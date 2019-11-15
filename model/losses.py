from keras.losses import categorical_crossentropy
from keras.layers import Layer, Dense
from keras import backend as K

import tensorflow as tf

class AngularLoss:
    def __init__(self, config):
        self.output_bins = config.get('MODEL.output_bins')
        self.margin_cosface = config.get('ANGULAR_LOSS.margin_cosface')
        self.margin_arcface = config.get('ANGULAR_LOSS.margin_arcface')
        self.margin_sphereface = config.get('ANGULAR_LOSS.margin_sphereface')
        self.scale = config.get('ANGULAR_LOSS.scale')

    def get_dense(self):
        output_bins = self.output_bins
        class AngularLossDense(Layer):
            def __init__(self, **kwargs):
                super(AngularLossDense, self).__init__(**kwargs)

            def build(self, input_shape):
                super(AngularLossDense, self).build(input_shape[0])
                self.W = self.add_weight(name='W',
                                         shape=(input_shape[-1], output_bins),
                                         initializer='glorot_uniform',
                                         trainable=True)

            def call(self, inputs):
                x = tf.nn.l2_normalize(inputs, axis=1)
                W = tf.nn.l2_normalize(self.W, axis=0)

                logits = x @ W
                return logits

            def compute_output_shape(self, input_shape):
                return (None, output_bins)
        return AngularLossDense

    def angular_loss(self, y_true, y_pred):
        logits = y_pred
        if self.margin_sphereface != 1.0 or self.margin_arcface != 0.0:
            y_pred = K.clip(y_pred, -1.0 + K.epsilon(), 1.0 - K.epsilon())
            theta = tf.acos(y_pred)
            if self.margin_sphereface != 1.0:
                theta = theta * self.margin_sphereface
            if self.margin_arcface != 0.0:
                theta = theta + self.margin_arcface
            y_pred = tf.cos(theta)
        target_logits = y_pred
        if self.margin_cosface != 0:
            target_logits = target_logits - self.margin_cosface

        logits = logits * (1 - y_true) + target_logits * y_true
        logits *= self.scale

        out = tf.nn.softmax(logits)
        loss = categorical_crossentropy(y_true, out)
        return loss