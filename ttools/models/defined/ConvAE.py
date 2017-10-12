from ttools.models.layers import *
from ttools.models.autoencoder import Autoencoder
import tensorflow as tf
import math

class ConvAE(Autoencoder):
    def __init__(self,
                 filter_sizes=[3, 3, 3, 3],
                 n_filters=[1, 10, 10, 10],
                 optimizer='adam',
                 lrn_rate=0.01,
                 **kwargs):
        super().__init__(filter_sizes=filter_sizes,
                         n_filters=n_filters,
                         optimizer=optimizer,
                         lrn_rate=lrn_rate,
                         **kwargs)

    def encode(self, x):
        self.encoder = []
        self.shapes = []
        corruption = False

        current_input = x
        if corruption:
            current_input = self._corrupt(current_input)

        for layer_i, n_output in enumerate(self.hps['n_filters'][1:]):
            n_input = current_input.get_shape().as_list()[3]
            self.shapes.append(current_input.get_shape().as_list())
            W = tf.Variable(
                tf.random_uniform([
                    self.hps['filter_sizes'][layer_i],
                    self.hps['filter_sizes'][layer_i],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)),
                name = 'W_enc_' + str(layer_i))
            b = tf.Variable(tf.zeros([n_output]),
                    name='b_enc_' + str(layer_i))
            self.encoder.append(W)
            output = lrelu(
                tf.add(tf.nn.conv2d(
                    current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        return current_input

    def decode(self, z):
        current_input = z
        self.encoder.reverse()
        self.shapes.reverse()

        for layer_i, shape in enumerate(self.shapes):
            W = self.encoder[layer_i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]),
                    name='b_dec_' + str(layer_i))
            output = lrelu(tf.add(
                tf.nn.conv2d_transpose(
                    current_input, W,
                    tf.stack([tf.shape(z)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        return current_input

    def _corrupt(self, x):
        return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                   minval=0,
                                                   maxval=2,
                                                   dtype=tf.int32), tf.float32))
