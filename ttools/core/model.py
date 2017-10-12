from ttools import ROOT_DIR
import tensorflow as tf
import sys
import os

class Model(object):

    def __init__(self, **kwargs):
        """Constructor of a model.

        :param name: name of the model, should agree with filename
            string
        """
        self._name = (self.__class__.__name__)
        self.saver = None
        self.model_path = os.path.join(ROOT_DIR, '..', 'log', self.name())
        self.global_step = None #tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.vars = []
        self.train_op = None
        self.opt = None
        self.input_data = None
        self.input_labels = None
        self.tf_session = None
        self.train_op = None
        self.summaries = None
        self.predictions = None
        self.logits = None
        self.loss = None
        self.mode = None
        self._construct_hyperparams(**kwargs)

    def build_graph(self, *args):
        """
            Build the entire training graph
        """
        pass
    
    def _build_train_op(self, *args):
        """
        :return:
        """
        pass

    def compute_cost(self, *args):
        """Loss function
        :return: a loss that should be minimized
        """
        pass

    def train(self, input_data, **kwargs):
        """

        :param input_data:
        :param input_labels:
        :return:
        """
        pass

    def evaluate(self, input_data, *args):
        pass

    def name(self):
        return self._name

    def _build_sess(self):
        self.sess = tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.model_path,
                hooks=self.hooks,
                chief_only_hooks=self.chief_only_hooks,
                # Since we provide a SummarySaverHook, we need to disable default
                # SummarySaverHook. To do that we set save_summaries_steps to 0.
                save_summaries_steps=0,
                config=tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True))

    def _print_model_info(self):
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.
                TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    def _construct_hyperparams(self, **kwargs):
        hps = {}
        for key, val in kwargs.items():
            hps[key] = val
        self.hps = hps

    def restore_from_checkpoint(self, exclude_scope='', model_path=None):
        curr_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                scope=(exclude_scope + self.name()))
        if model_path is not None:
            model_path = model_path
        else:
            model_path = self.model_path

        prev_vars = [var_name for var_name, _ in tf.contrib.framework.list_variables(model_path)]
        restore_map = {var.op.name.replace(exclude_scope, ''): var for var in curr_vars
                       if var.op.name.replace(exclude_scope, '') in prev_vars}
        import pdb
        pdb.set_trace()

        tf.contrib.framework.init_from_checkpoint(model_path, restore_map)

        return curr_vars
