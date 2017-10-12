from ttools.pipelines import standard_input_pipeline
from ttools.utils.augmentations import normalize, cifar_aug
from ttools.utils.utils import image_summary 
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'MNIST', 'Choose your dataset')
tf.app.flags.DEFINE_string('model', 'LeNet', 'Choose your model')
tf.app.flags.DEFINE_string('autoencoder', 'ConvAE', 'Choose your autoencoder')
tf.app.flags.DEFINE_string('mode', 'train', 'Train or eval')
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

        ae_module = __import__('.'.join(['ttools', 'models', 'defined', FLAGS.autoencoder]), fromlist=[''])
        ae_instance = getattr(ae_module, FLAGS.autoencoder)(num_steps=FLAGS.num_steps)
        
        recon = ae_instance.reconstruct(test_images)

        image_summary('test_attacked_input_output', inputs=test_images, outputs=recon, grid_size=[5, 5])
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./summary')

        if FLAGS.mode == 'train':
            model_instance.train(input_data=train_images, input_labels=train_labels)
        elif FLAGS.mode == 'eval':
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                step = 1
                avg_acc = 0.0
                try:
                    for step in range(10):
                        summary = sess.run(summary_op)
                        summary_writer.add_summary(summary, step)
                        # plt.imshow(train_imgs[1][:, :, 0], cmap='gray')
                        # plt.show()
                except Exception as e:
                    coord.request_stop(e)
                finally:
                    coord.request_stop()
                coord.join(threads=threads)
                sess.close()
                #model_instance.evaluate(recon, test_labels)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
