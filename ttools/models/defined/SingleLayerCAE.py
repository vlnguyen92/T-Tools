from ttools.models.layers import *
from ttools.models.autoencoder import Autoencoder
import random
import tensorflow as tf

class SingleLayerCAE(Autoencoder):
    def __init__(self,
                 initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                            mode='FAN_IN',
                                                                            uniform=False,
                                                                            seed=random.seed(),
                                                                            dtype=tf.float32),
                 filter_side=3,
                 filters_number=32,
                 optimizer='adam',
                 lrn_rate=0.01,
                 **kwargs):

        super().__init__(initializer=initializer,
                         filter_side=filter_side,
                         filters_number=filters_number,
                         optimizer=optimizer,
                         lrn_rate=lrn_rate,
                         **kwargs)


    def _pad(self, input_x):
        """
        pads input_x with the right amount of zeros.
        Args:
            input_x: 4-D tensor, [batch_side, widht, height, depth]
            filter_side: used to dynamically determine the padding amount
        Returns:
            input_x padded
        """
        # calculate the padding amount for each side
        amount = self.hps['filter_side'] - 1
        # pad the input on top, bottom, left, right, with amount zeros
        return tf.pad(input_x, [[0, 0], [amount, amount], [amount, amount],
                                [0, 0]])

    def encode(self, x):

        input_x = self._pad(x)
        self.shapes.append(input_x.get_shape())

        with tf.variable_scope("encode"):
            # the encoding convolutions is a [3 x 3 x input_depth] x 32 convolution
            # the activation function chosen is the tanh
            # 32 is the number of feature extracted. It's completely arbitrary as is
            # the side of the convolutional filter and the activation function used
            filter_side = self.hps['filter_side']
            filters_number = self.hps['filters_number']
            encoding = conv(input_x, 
                            shape=[filter_side, filter_side,
                                   self.shapes[0][3].value, filters_number],
                            stride=1,
                            padding='VALID',
                            activation=tf.nn.tanh,
                            initializer=self.hps['initializer'])
        return encoding

    def decode(self, z):
        with tf.variable_scope("decode"):
            # the decoding convolution is a [3 x 3 x 32] x input_depth convolution
            # the activation function chosen is the tanh
            # The dimenensions of the convolutional filter in the decoding convolution,
            # differently from the encoding, are constrained by the
            # choices made in the encoding layer
            # The only degree of freedom is the chose of the activation function.
            # We have to choose an activation function that constraints the outputs
            # to live in the same space of the input values.
            # Since the input values are between -1 and 1, we can use the tanh function
            # directly, or we could use the sigmoid and then scale the output
            filter_side = self.hps['filter_side']
            filters_number = self.hps['filters_number']
            output_x = conv(z, 
                            shape=[filter_side, filter_side, filters_number,
                                   self.shapes[0][3].value],
                            stride=1,
                            padding='VALID',
                            activation=tf.nn.tanh,
                            initializer=self.hps['initializer'])
        return output_x


    def compute_cost(self):
        x = self.input_data
        y = self.predictions
        mse = tf.divide(tf.reduce_mean(
            tf.square(tf.subtract(x, y))),
            2.0)
        self.cost = mse

        tf.summary.scalar('cost', self.cost)
