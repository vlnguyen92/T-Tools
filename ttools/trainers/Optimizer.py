"""Trainers module."""

import tensorflow as tf


class Optimizer(object):
    """Wrapper of Tensorflow Optimizers."""

    def __init__(self, optimizer_name, **kw):
        """Constructor.
        Parameters
        ----------
        optimizer : string
            Which optimizer to use. Possible values are ["sgd", "adagrad",
            "adam", "momentum"]
        kw :
            the following arguments should be provided:
                * sgd: learning_rate (float)
                * adagrad: learning_rate (float), initial_accumulator_value
                (float, default=0.1)
                * adam: learning_rate (float, default=0.001), beta1 (float,
                default=0.9), beta2 (float, default=0.999), epsilon (float,
                default 1e-08)
                * momentum: learning_rate (float), use_nesterov (bool)
        """
        assert optimizer_name in ["sgd", "adagrad", "adam", "momentum"]

        def d(k, other=None):
            if other is not None:
                return kw[k] if k in kw else other
            else:
                return kw[k]

        if optimizer_name == "sgd":
            self.opt_ = tf.train.GradientDescentOptimizer(d("learning_rate"))

        elif optimizer_name == "adagrad":
            self.opt_ = tf.train.AdagradOptimizer(
                d("learning_rate"), d("initial_accumulator_value", 0.1))

        elif optimizer_name == "adam":
            self.opt_ = tf.train.AdamOptimizer(d("learning_rate", 0.001),
                                               d("beta1", 0.9),
                                               d("beta2", 0.9),
                                               d("epsilon", 1e-08))

        elif optimizer_name == "momentum":
            self.opt_ = tf.train.MomentumOptimizer(
                d("learning_rate"), d("momentum"),
                use_nesterov=d("use_nesterov", False))

    def train_op(self, cost, name_scope="train", global_step=None, var_list=None):
        """The most basic train_op for any model, minimizing a loss function.
        Parameters
        ----------
        cost : Tensor
            A Tensor containing the value to minimize.
        name_scope : str , optional (default="train")
            Optional name scope for the optimizer graph ops.
        """
        with tf.name_scope(name_scope):
            return self.opt_.minimize(cost, global_step=global_step, var_list=var_list)
