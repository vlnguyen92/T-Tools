from ttools.models.layers import *
from ttools.models.classifier import Classifier
import tensorflow as tf

class LeNet(Classifier):

    def __init__(self, optimizer='adam', lrn_rate=0.001, **kwargs):
        super().__init__(optimizer=optimizer,
                         lrn_rate=lrn_rate,
                         **kwargs)

    def _build_model(self):
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        dropout = tf.constant(0.75, dtype=tf.float32)

        # Convolution Layer
        with tf.variable_scope('conv1'):
            conv1 = conv(self.input_data,
                         shape=[5, 5, 1, 32],
                         stride=1,
                         activation=tf.nn.relu,
                         initializer=tf.random_normal_initializer())

        # Max Pooling (down-sampling)
        with tf.variable_scope('pool1'):
            conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1,
                         shape=[5, 5, 32, 64],
                         stride=1,
                         activation=tf.nn.relu,
                         initializer=tf.random_normal_initializer())

        # Max Pooling (down-sampling)
        with tf.variable_scope('pool2'):
            conv2 = maxpool2d(conv2, k=2)
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        with tf.variable_scope('fc1'):
            fc1 = tf.reshape(conv2, [-1, 7 * 7 * 64])
            fc1 = fc(fc1,
                     shape=[7 * 7 * 64, 1024],
                     activation=tf.nn.relu,
                     initializer=tf.random_normal_initializer())
        # Apply Dropout
        with tf.variable_scope('dropout'):
            fc1 = tf.nn.dropout(fc1, dropout)

        #TODO: Change this somehow
        num_classes = 10
        # Output, class prediction
        with tf.variable_scope('softmax_linear'):
            logits = fc(fc1, shape=[1024, num_classes], initializer=tf.random_normal_initializer())

        self.logits = logits
        self.predictions = tf.nn.softmax(logits)
