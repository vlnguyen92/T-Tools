from ttools.core.input_pipeline import InputPipeline
import tensorflow as tf

class StandardInputPipeline(InputPipeline):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def _read_tf_records(self, filename_queue):
        _, serialized_example = self._reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features=self._keys_to_features)

        shape = self._data_set.get_data_shape()
        image = tf.reshape(tf.decode_raw(features['image/encoded'], tf.float32), shape=shape)

        # TODO: Change this to use info from data_set
        label_index = tf.cast(features['image/class/label'], tf.int32)
        label = tf.one_hot(label_index, 10, axis=0, dtype=tf.float32)
        return image, label


    def _provide_data_tensor(self, phase, batch_size, num_threads, augmentation_fn = None):
        #filename_queue = self._construct_filenames_queue(phase, num_epochs=num_epochs)
        filename_queue = self._construct_single_filename_queue(phase)

        img, label = self._read_tf_records(filename_queue)
        if augmentation_fn is not None:
            img = augmentation_fn(img)

        images, labels = tf.train.shuffle_batch([img, label],
                                                batch_size=batch_size,
                                                num_threads=num_threads,
                                                capacity=1000,
                                                min_after_dequeue=1)

        return images, tf.squeeze(labels)
