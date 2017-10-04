import tensorflow as tf
from tensorflow.python.training import moving_averages

def weight(name, 
        shape,
        initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0,
            mode='FAN_IN',
            uniform=False,
            seed=None,
            dtype=tf.float32)):

     weights = tf.get_variable(name,
                    shape=shape,
                    initializer=initializer,
                    dtype=tf.float32)
     return weights

def bias(name,
        shape,
        initializer=tf.constant_initializer(value=0.0)):

    return weight(name, shape, initializer=initializer)

def conv(x,
         shape,
         stride,
         padding='SAME',
         activation=tf.identity,
         biases=True,
         initializer =
         tf.contrib.layers.variance_scaling_initializer(
             factor=2.0,
             mode='FAN_IN',
             uniform=False,
             seed=None,
             dtype=tf.float32)):

    W = weight("W", shape, initializer=initializer)

    result = tf.nn.conv2d(x, W, [1, stride, stride, 1], padding)
    if biases:
        b = bias("b", [shape[3]])
        result = tf.nn.bias_add(result, b)

    out = activation(result)
    return out


def fc(x,
       shape,
       activation=tf.identity,
       biases=True,
       initializer =
       tf.contrib.layers.variance_scaling_initializer(
           factor=2.0,
           mode='FAN_IN',
           uniform=False,
           seed=None,
           dtype=tf.float32)):

    W = weight("W", shape, initializer=initializer)

    result = tf.matmul(x, W)
    if biases:
        b = bias("b", [shape[1]])
        result = tf.nn.bias_add(result, b)

    out = activation(result)
    return out

def maxpool2d(x, k=2, padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
            padding=padding)

def lrelu(x, leak = 0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
