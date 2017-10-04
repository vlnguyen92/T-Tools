""" pipeline data provider """

import os
import sys
import tensorflow as tf
import glob
from ttools import ROOT_DIR

class InputPipeline(object):

    def __init__(self, dataset_name):
        dataset_module = __import__('.'.join(['ttools', 'datasets', dataset_name]),
                                    fromlist=[''])

        self._data_set = getattr(dataset_module, dataset_name)()
        self._keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string),
            'image/format': tf.FixedLenFeature((), tf.string),
            'image/class/label': tf.FixedLenFeature([1], tf.int64)}
        self._reader = tf.TFRecordReader()

        new_data = self._is_new_data(self._data_set)
        if new_data:
            print('Data not found. Downloading and processing')
            self._data_set.run()
        else:
            print('Data Downloaded')

    def _construct_filenames_queue(self, split_name):
        all_files = glob.glob(self._data_set.get_data_dir() + '/*_90_50_*' + split_name + '.tfrecord')
        return tf.train.string_input_producer(all_files)

    def _construct_single_filename_queue(self, split_name):
        all_files = glob.glob(self._data_set.get_data_dir() + '/*' + split_name + '.tfrecord')
        return tf.train.string_input_producer(all_files)

    def _is_new_data(self, dataset):
        # TODO: Change this instead of hard-coded path
        all_files = list(map(os.path.splitext, os.listdir(os.path.join(ROOT_DIR, 'datasets'))))
        all_datasets = [file[0] for file in all_files][2:]

        data_path = self._data_set.get_data_dir()

        if not (dataset.name() in all_datasets):
            print("Dataset is not implemented, make sure the name is accurate")
            sys.exit()

        if self._data_set.new_data:
            return False

        for file in all_datasets:
            if dataset.name() == file and not os.listdir(data_path):
                return True

        return False

    def _read_tf_records(self):
        return

    def _provide_data_tensor(self, phase_name, batch_size, num_threads):
        return

    def provide_train_data_tensor(self, batch_size=64, num_threads=4, augmentation_fn=None):
        return self._provide_data_tensor('train',
                                         batch_size=batch_size,
                                         num_threads=num_threads,
                                         augmentation_fn=augmentation_fn)

    def provide_test_data_tensor(self, batch_size=64, num_threads=4, augmentation_fn=None):
        return self._provide_data_tensor('test',
                                         batch_size=batch_size,
                                         num_threads=num_threads,
                                         augmentation_fn=augmentation_fn)

    def provide_val_data_tensor(self, batch_size=64, num_threads=4, augmentation_fn=None):
        return self._provide_data_tensor('val',
                                         batch_size=batch_size,
                                         num_threads=num_threads,
                                         augmentation_fn=augmentation_fn)
