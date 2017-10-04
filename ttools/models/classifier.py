from ttools.models.layers import *
from ttools.core.model import Model
from ttools.utils.utils import image_summary
from ttools.trainers.Optimizer import Optimizer
import tensorflow as tf
import sys

class Classifier(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name()):
            self.global_step = tf.contrib.framework.get_or_create_global_step()
            self._build_model()
            self._compute_accuracy()
            if self.mode == 'train':
                self._build_train_op()
            self.summaries = tf.summary.merge_all()

    def _build_train_op(self):
        self.compute_cost()
        self.opt = Optimizer(self.hps['optimizer'],
                learning_rate=self.hps['lrn_rate'])
        self.train_op = self.opt.train_op(self.cost, global_step=self.global_step)

    def _compute_accuracy(self):
        truth = tf.argmax(self.input_labels, axis=1)
        predictions = tf.argmax(self.predictions, axis=1)
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
        tf.summary.scalar('Accuracy', self.accuracy)

    def compute_cost(self, *args):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                           labels=self.input_labels))

    def _build_model(self, *args):
        pass

    def _build_hooks(self, *args):
        summary_hook = tf.train.SummarySaverHook(
            save_steps=100,
            output_dir=self.model_path,
            summary_op=self.summaries)

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': self.global_step,
                     'loss': self.cost,
                     'precision': self.accuracy},
            every_n_iter=100)

        saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=self.model_path,
            save_steps=100
        )

        self.hooks = [logging_hook, saver_hook]
        self.chief_only_hooks = [summary_hook]
    
    def train(self, input_data, input_labels, **kwargs):
        self.mode = 'train'
        self.input_data = input_data
        self.input_labels = input_labels
        image_summary('input', inputs=input_data)
        self.build_graph()
        self._build_hooks()
        self._build_sess()
        self._print_model_info()

        for _ in range(1, self.hps['num_steps'] + 2):
                self.sess.run(self.train_op)

    def evaluate(self, input_data, input_labels, **kwargs):
        self.mode = 'eval'
        self.input_data = input_data
        self.input_labels = input_labels
        self.build_graph()
        self.restore_from_checkpoint()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            step = 1
            avg_acc = 0.0
            try:
                while not coord.should_stop():
                    print(step, sess.run(self.accuracy))
                    step += 1
                    # plt.imshow(train_imgs[1][:, :, 0], cmap='gray')
                    # plt.show()
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
            coord.join(threads=threads)
            sess.close()
