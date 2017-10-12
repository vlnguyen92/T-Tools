from ttools.core.model import Model
from ttools.utils.utils import image_summary
from ttools.trainers.Optimizer import Optimizer
import tensorflow as tf
import sys

class Autoencoder(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_graph(self):
        with tf.variable_scope(self.name()):
            self.encode_decode()
            image_summary('input_output', inputs=self.input_data,
                          outputs=tf.reshape(self.predictions, self.input_data.get_shape()))
            tf.summary.scalar('cost', self.cost)
            if self.mode == 'train':
                self._build_train_op()
            self.summaries = tf.summary.merge_all()

    def encode(self, x):
        return

    def decode(self, z):
        return

    def encode_decode(self):
        self.shapes = []
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        z = self.encode(self.input_data)
        output = self.decode(z)

        self.predictions = output
        self.compute_cost()

    def _build_train_op(self):
        self.opt = Optimizer(self.hps['optimizer'],
                learning_rate=self.hps['lrn_rate'])

        self.train_op = self.opt.train_op(self.cost, global_step=self.global_step)

    def compute_cost(self, *args):
        self.cost = tf.reduce_sum(tf.square(self.predictions - self.input_data))

    def _build_hooks(self, *args):
        summary_hook = tf.train.SummarySaverHook(
            save_steps=100,
            output_dir=self.model_path,
            summary_op=self.summaries)

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': self.global_step,
                     'loss': self.cost},
            every_n_iter=100)

        saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=self.model_path,
            save_steps=100
        )

        self.hooks = [logging_hook, saver_hook]
        self.chief_only_hooks = [summary_hook]

    def train(self, input_data, **kwargs):
        self.mode = 'train'
        self.input_data = input_data
        self.build_graph()
        self._build_hooks()
        self._build_sess()
        self._print_model_info()

        for _ in range(1, self.hps['num_steps'] + 2):
            self.sess.run(self.train_op)



