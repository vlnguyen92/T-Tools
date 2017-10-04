""" Processing functions for different datasets"""

# Borrow from Paolo Galeon's dynamic training bench

import tensorflow as tf

def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
      values: A scalar or list of values.
    Returns:
      a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
      values: A string.
    Returns:
      a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def float_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
      values: A string.
    Returns:
      a TF-Feature.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/class/label': int64_feature(class_id),
    }))

def get_all_trainable_variables():
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

def build_saver():
    return tf.train.Saver(var_list=get_all_trainable_variables())

def build_logger(path, graph):
    return tf.summary.FileWriter(path, graph=graph)

# Adapted from
# https://gist.github.com/kukuruza/03731dc494603ceab0c5#gistcomment-1879326
def put_kernels_on_grid(kernel, grid_side, pad=1):
    """Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
        kernel:    tensor of shape [Y, X, NumChannels, NumKernels]
        grid_side: side of the grid. Require: NumKernels == grid_side**2
        pad:       number of black pixels around each filter (between them)

    Returns:
        An image Tensor with shape [(Y+2*pad)*grid_side, (X+2*pad)*grid_side, NumChannels, 1].
    """

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(
        kernel1,
        tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]),
        mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2,
                    tf.stack(
                        values=[grid_side, Y * grid_side, X, channels],
                        axis=0))  #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4,
                    tf.stack(
                        values=[1, X * grid_side, Y * grid_side, channels],
                        axis=0))  #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype=tf.uint8)

def image_summary(name, inputs, outputs = None, grid_size=[5, 5]):
    with tf.variable_scope('visualization'):
        grid_side = grid_size[0]
        inputs = put_kernels_on_grid(
            tf.transpose(inputs, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
            grid_side)

        if outputs is None:
            return tf.summary.image('inputs', inputs, max_outputs=10)

        inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 10], [0, 0]])
        outputs = put_kernels_on_grid(
            tf.transpose(outputs, perm=(1, 2, 3, 0))[:, :, :, 0:grid_side**2],
            grid_side)
        return tf.summary.image(name,
                         tf.concat([inputs, outputs], axis=2),
                         max_outputs=10)




