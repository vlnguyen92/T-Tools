from ttools.pipelines import standard_input_pipeline
from ttools.utils.augmentations import normalize, cifar_aug
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'Cifar10', 'Choose your dataset')
tf.app.flags.DEFINE_string('model', 'ResNet', 'Choose your model')
tf.app.flags.DEFINE_integer('num_steps', 1000, 'How many train steps')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')

def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    if(FLAGS.dataset == 'Cifar10'):
        data_aug = cifar_aug
    elif(FLAGS.dataset == 'MNIST'):
        data_aug = normalize

    input_producer = standard_input_pipeline.StandardInputPipeline(FLAGS.dataset)
    train_images, train_labels = input_producer.provide_train_data_tensor(batch_size=FLAGS.batch_size,
                                                                          augmentation_fn=data_aug)
    test_images, test_labels = input_producer.provide_train_data_tensor(batch_size=FLAGS.batch_size,
                                                                       augmentation_fn=data_aug)

    with tf.device(dev):
        model_module = __import__('.'.join(['ttools', 'models', 'defined', FLAGS.model]), fromlist=[''])
        model_instance = getattr(model_module, FLAGS.model)(num_steps=FLAGS.num_steps)
        if FLAGS.mode == 'train':
            model_instance.train(input_data=train_images, input_labels=train_labels)
        elif FLAGS.mode == 'eval':
            model_instance.evaluate(test_images, test_labels)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
