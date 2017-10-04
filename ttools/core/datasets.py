""" Interfaces for all the datasets"""


import os
import tensorflow as tf
from ttools import ROOT_DIR

class Dataset(object):
    """ Class representing a dataset"""

    def __init__(self, name):
        """ Construct a dataset"""
        self._name = name
        self._data_dir = os.path.join(ROOT_DIR, '..', 'data', self._name)
        if not tf.gfile.Exists(self._data_dir):
            tf.gfile.MakeDirs(self._data_dir)

    def run(self):
        pass

    def _process(self):
        pass

    def _download_data(self, *args):
        pass

    def _clean_temp_files(self, *args):
        pass

    def get_data_dir(self):
        return self._data_dir

    def name(self):
        return self._name
