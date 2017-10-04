import tensorflow as tf

def scale_image(image):
    """Returns the image tensor with values in [-1, 1].
    Args:
        image: [height, width, depth] tensor with values in [0,1]
    """
    image = tf.subtract(image, 0.5)
    # now image has values with zero mean in range [-0.5, 0.5]
    image = tf.multiply(image, 2.0)
    # now image has values with zero mean in range [-1, 1]
    return image

def cifar_aug(image):
    image_size=32
    image = tf.image.resize_image_with_crop_or_pad(
        image, image_size + 4, image_size + 4)
    image = tf.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
    # image = tf.image.random_brightness(image, max_delta=63. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)
    return image

def normalize(image):
    image = tf.divide(image, 255.0)
    image = scale_image(image)
    return image