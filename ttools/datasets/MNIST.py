""" Defining the MNSIT dataset"""

from __future__ import absolute_import

import gzip
import os
import sys
import urllib.request

import numpy as np
import tensorflow as tf

from ttools.core.datasets import Dataset
from ttools.utils.utils import image_to_tfexample

class MNIST(Dataset):

    # Initing dataset
    def __init__(self, name='MNIST'):
        super().__init__(name)
        self._data_url = 'http://yann.lecun.com/exdb/mnist/'
        self._image_size = 28
        self._num_channels = 1
        self.new_data = tf.gfile.Exists(self._get_output_filename(self.get_data_dir(), 'train')) \
                        and tf.gfile.Exists(self._get_output_filename(self.get_data_dir(), 'test'))

    def _extract_labels(self, filename, num_labels):
        """Extract the labels into a vector of int64 label IDs.
        Args:
          filename: The path to an MNIST labels file.
          num_labels: The number of labels in the file.
        Returns:
          A numpy array of shape [number_of_labels]
        """
        print('Extracting labels from: ', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_labels)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

    def _extract_images(self, filename, num_images):
        """Extract the images into a numpy array.
        Args:
            filename: The path to an MNIST images file.
            num_images: The number of images in the file.
        Returns:
            A numpy array of shape [number_of_images, height, width, channels].
        """

        print('Extracting images from: ', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(
                self._image_size * self._image_size * num_images * self._num_channels)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, self._image_size, self._image_size, self._num_channels)
        return data

    def _download_data(self, files_to_download):

        for filename in files_to_download:
            filepath = os.path.join(self._data_dir, filename)

            if not os.path.exists(filepath):
                print('Downloading file %s ...', filename)

                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %.1f%%' % (
                        float(count * block_size) / float(total_size) * 100.0))
                    sys.stdout.flush()

                filepath, _ = urllib.request.urlretrieve(self._data_url+filename,
                                                     filepath,
                                                     _progress)
                print()

                with tf.gfile.GFile(filepath) as f:
                    size = f.size()
                    print('Downloaded', filename, size, 'bytes.')
            else:
                print('File %s Exists.', filename)

    def _process(self):
        train_data_filename = 'train-images-idx3-ubyte.gz'
        train_labels_filename = 'train-labels-idx1-ubyte.gz'
        test_data_filename = 't10k-images-idx3-ubyte.gz'
        test_labels_filename = 't10k-labels-idx1-ubyte.gz'

        all_files_to_download = [train_data_filename, train_labels_filename,
                                 test_data_filename, test_labels_filename]

        all_files_downloaded = all_files_to_download

        training_filename = self._get_output_filename(self._data_dir, 'train')
        testing_filename = self._get_output_filename(self._data_dir, 'test')

        if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
            print('Dataset files already exist. Exiting without re-creating them.')
            return

        self._download_data(all_files_to_download)
        num_train_images = 60000
        num_test_images = 10000

        self._write_to_tfrecords(train_data_filename, train_labels_filename,
                                   training_filename, num_train_images)

        self._write_to_tfrecords(test_data_filename, test_labels_filename,
                                   testing_filename, num_test_images)

        return all_files_downloaded

    def _get_output_filename(self, data_dir, split_name):
        return '%s/mnist_%s.tfrecord' % (data_dir, split_name)
    
    def _add_to_tfrecords(self, images, labels,
                          num_datapoints, tfrecord_writer):
        """Loads data from the binary MNIST files and writes files to a TFRecord.
         Args:
           data_filename: The filename of the MNIST images.
           labels_filename: The filename of the MNIST labels.
           num_images: The number of images in the dataset.
           tfrecord_writer: The TFRecord writer to use for writing.
         """

        shape = (self._image_size, self._image_size, self._num_channels)
        with tf.Graph().as_default():
            image = tf.placeholder(dtype=tf.uint8, shape=shape)
            float_image = tf.cast(image, dtype=tf.float32)
    
            with tf.Session('') as sess:
                for j in range(num_datapoints):
                    sys.stdout.write('\r>> Converting image %d/%d with size' % (j + 1, num_datapoints))
                    sys.stdout.flush()
    
                    img = sess.run(float_image, feed_dict={image: images[j]})

                    example = image_to_tfexample(
                        tf.compat.as_bytes(img.tostring()), 'png'.encode(), 
                                           self._image_size, self._image_size,
                                           int(labels[j]))

                    tfrecord_writer.write(example.SerializeToString())
    
    def _write_to_tfrecords(self, data_filename,
                                  labels_filename,
                                  output_filename,
                                  num_points):

        # If the tfrecord already exists, return
        if os.path.exists(output_filename):
            return

        data = self._extract_images(os.path.join(self._data_dir, data_filename),
                                            num_points)
        labels = self._extract_labels(os.path.join(self._data_dir, labels_filename),
                                            num_points)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            self._add_to_tfrecords(data, labels, num_points, tfrecord_writer)

    def _clean_temp_files(self, filenames):
        for filename in filenames:
            filepath = os.path.join(self._data_dir, filename)
            tf.gfile.Remove(filepath)

    def get_data_shape(self):
        return (self._image_size, self._image_size, self._num_channels)

    def run(self):
        downloaded_files = self._process()
        self._clean_temp_files(downloaded_files)

        print("\n\nAll data downloaded and converted to TFRecords.")
