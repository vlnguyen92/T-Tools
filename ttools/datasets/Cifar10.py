""" Defining the MNSIT dataset"""

from __future__ import absolute_import

import _pickle as cPickle
import os
import sys
import tarfile
import urllib.request

import numpy as np
import tensorflow as tf

from ttools.core.datasets import Dataset
from ttools.utils.utils import image_to_tfexample

class Cifar10(Dataset):

    # Initing dataset
    def __init__(self, name='Cifar10'):
        super().__init__(name)
        self._data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self._image_size = 32 
        self._num_channels = 3
        self.new_data = tf.gfile.Exists(self._get_output_filename(self.get_data_dir(), 'train')) \
                        and tf.gfile.Exists(self._get_output_filename(self.get_data_dir(), 'test'))

    def _download_data(self):
        """Download and uncompress Cifar10 locally."""
        filename = self._data_url.split('/')[-1]
        filepath = os.path.join(self._data_dir, filename)

        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(self._data_url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(self._data_dir)

    def _add_to_tfrecord(self, filename, tfrecord_writer, offset=0):
        """Loads data from the cifar10 pickle files and writes files to a TFRecord.
        Args:
          filename: The filename of the cifar10 pickle file.
          tfrecord_writer: The TFRecord writer to use for writing.
          offset: An offset into the absolute number of images previously written.
        Returns:
          The new offset.
        """
        with open(filename, 'rb') as f:
            if sys.version_info < (3,):
                data = cPickle.load(f)
            else:
                data = cPickle.load(f, encoding='bytes')

        images = data[b'data']
        num_images = images.shape[0]

        # Depth * width * height
        images = images.reshape((num_images, 3, 32, 32))
        labels = data[b'labels']
        shape = self.get_data_shape()

        with tf.Graph().as_default():
            image_placeholder = tf.placeholder(dtype=tf.uint8, shape=shape)
            float_image = tf.cast(image_placeholder, dtype=tf.float32)

            with tf.Session('') as sess:
                for j in range(num_images):
                    sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                        filename, offset + j + 1, offset + num_images))
                    sys.stdout.flush()

                    # Height * width * depth
                    image = np.squeeze(images[j]).transpose((1, 2, 0))
                    label = labels[j]

                    img = sess.run(float_image,
                                          feed_dict={image_placeholder: image})

                    example = image_to_tfexample(
                        tf.compat.as_bytes(img.tostring()), b'png', self._image_size, self._image_size, label)

                    tfrecord_writer.write(example.SerializeToString())

        return offset + num_images

    def _process(self):
        """
            Process data end-to-end: 
            download,extract, then write to TFRecords
        """
        training_filename = self._get_output_filename(self._data_dir, 'train')
        testing_filename = self._get_output_filename(self._data_dir, 'test')

        if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
            print('Dataset files already exist. Exiting without re-creating them.')
            return

        self._download_data()
        all_files_downloaded = []
        num_train_files = 5

        with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
            offset = 0
            for i in range(num_train_files):
                filename = os.path.join(self._data_dir,
                                        'cifar-10-batches-py',
                                        'data_batch_%d' % (i + 1))  # 1-indexed.
                all_files_downloaded.append(filename)
                offset = self._add_to_tfrecord(filename, tfrecord_writer, offset)

        # Next, process the testing data:
        with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
            filename = os.path.join(self._data_dir,
                                    'cifar-10-batches-py',
                                    'test_batch')
            all_files_downloaded.append(filename)
            self._add_to_tfrecord(filename, tfrecord_writer)
        
        return all_files_downloaded

    def _get_output_filename(self, data_dir, split_name):
        return '%s/cifar10_%s.tfrecord' % (data_dir, split_name)

    def _clean_temp_files(self, filenames):
        for filename in filenames:
            tf.gfile.Remove(filename)

    def get_data_shape(self):
        return (self._image_size, self._image_size, self._num_channels)

    def run(self):
        downloaded_files = self._process()
        all_files = downloaded_files + [os.path.join(self._data_dir, 'cifar-10-python.tar.gz')]
        self._clean_temp_files(all_files)

        print("\n\nAll data downloaded and converted to TFRecords.")
