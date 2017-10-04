from ttools.models.layers import *
from ttools.models.classifier import Classifier
import tensorflow as tf
import os

class VGG(Classifier):

    def __init__(self):
        super().__init__(self.__class__.__name__)

    def infer(self, x):
        with tf.variable_scope(self.name()):
            self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
            initializer = tf.contrib.layers.variance_scaling_initializer(
                factor=2.0,
                mode='FAN_IN',
                uniform=False,
                seed=1234,
                dtype=tf.float32
            )

            # Convolution Layer
            with tf.variable_scope('conv1'):
                conv1 = conv(x,
                             shape=[3, 3, 3, 64],
                             stride=1,
                             activation=tf.nn.relu,
                             initializer=initializer)

            # Max Pooling (down-sampling)
            with tf.variable_scope('conv2'):
                conv2 = conv(conv1,
                             shape=[3, 3, 64, 64],
                             stride=1,
                             activation=tf.nn.relu,
                             initializer=initializer)

            with tf.variable_scope('pool1'):
                pool1 = maxpool2d(conv2, padding='VALID')

            
            with tf.variable_scope('conv3'):
                conv3 = conv(pool1,
                            shape=[3, 3, 64, 128],
                            stride=1,
                            activation=tf.nn.relu,
                            initializer=initializer)

            with tf.variable_scope('conv4'):
                conv4 = conv(conv3,
                             shape=[3,3,128,128],
                             stride=1,
                             activation=tf.nn.relu,
                             initializer=initializer)

            with tf.variable_scope('pool2'):
                pool2 = maxpool2d(conv4, padding='VALID')

            with tf.variable_scope('conv5'):
                conv5 = conv(pool2, 
                             shape=[3, 3, 128, 256],
                             stride=1,
                             activation=tf.nn.relu,
                             initializer=initializer)

            with tf.variable_scope('conv6'):
                conv6 = conv(conv5, 
                             shape=[3, 3, 256, 256],
                             stride=1,
                             activation=tf.nn.relu,
                             initializer=initializer)

            with tf.variable_scope('conv7'):
                conv7 = conv(conv6, 
                             shape=[3, 3, 256, 256],
                             stride=1,
                             activation=tf.nn.relu,
                             initializer=initializer)

            with tf.variable_scope('pool3'):
                pool3 = maxpool2d(conv7, padding='VALID')

            with tf.variable_scope('conv8'):
                conv8 = conv(pool3, 
                             shape=[3, 3, 256, 512],
                             stride=1,
                             activation=tf.nn.relu,
                             initializer=initializer)

            with tf.variable_scope('conv9'):
                conv9 = conv(conv8, 
                             shape=[3, 3, 512, 512],
                             stride=1,
                             activation=tf.nn.relu,
                             initializer=initializer)

            with tf.variable_scope('conv10'):
                conv10 = conv(conv9, 
                              shape=[3, 3, 512, 512],
                              stride=1,
                              activation=tf.nn.relu,
                              initializer=initializer)

            with tf.variable_scope('pool4'):
                pool4 = maxpool2d(conv10, padding='VALID')

            with tf.variable_scope('conv11'):
                conv11 = conv(pool4, 
                              shape=[3, 3, 512, 512],
                              stride=1,
                              activation=tf.nn.relu,
                              initializer=initializer)

            with tf.variable_scope('conv12'):
                conv12 = conv(conv11, 
                              shape=[3, 3, 512, 512],
                              stride=1,
                              activation=tf.nn.relu,
                              initializer=initializer)

            with tf.variable_scope('conv13'):
                conv13 = conv(conv12, 
                              shape=[3, 3, 512, 512],
                              stride=1,
                              activation=tf.nn.relu,
                              initializer=initializer)

            with tf.variable_scope('pool5'):
                pool5 = maxpool2d(conv13, padding='VALID')
                pool5 = tf.reshape(pool5, [-1, 512])

            with tf.variable_scope('fc'):
                fc1 = fc(pool5, 
                         shape=[512, 512],
                         activation=tf.nn.relu,
                         initializer=initializer)

            #TODO: Change this somehow
            num_classes = 10
            # Output, class prediction
            with tf.variable_scope('softmax_linear'):
                logits = fc(fc1, shape=[512, num_classes],
                            initializer=initializer)

            return logits

    def loss(self, logits, labels):
        labels = tf.argmax(labels, axis=1)
        labels = tf.cast(labels, tf.int64)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def accuracy(self, logits, labels):
        logits = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy
