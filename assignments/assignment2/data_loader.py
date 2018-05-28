import os
import random
import tensorflow as tf
from model import UnetModel

class DataLoader(object):

    def __init__(self, config):
        self.config = config
        self._get_paths()
        self._build_dataset()

    def _get_paths(self):

        training_dir = os.path.join(self.config.data_path, 'training')

        # Training data
        basenames = [os.path.splitext(os.path.basename(fname))[0] for fname in os.listdir(os.path.join(training_dir, 'images'))]
        
        random.seed(self.config.random_state)
        random.shuffle(basenames)

        split = int(len(basenames) * self.config.validation_size)
        training_names = basenames[split:]
        validation_names = basenames[:split]

        training_imgs = [os.path.join(training_dir, 'images', basename + '.jpg') for basename in training_names]
        training_labels = [os.path.join(training_dir, 'labels_plain', basename + '.png') for basename in training_names]
        validation_imgs = [os.path.join(training_dir, 'images', basename + '.jpg') for basename in validation_names]
        validation_labels = [os.path.join(training_dir, 'labels_plain', basename + '.png') for basename in validation_names]

        assert all(map(os.path.isfile, training_imgs))
        assert all(map(os.path.isfile, training_labels))
        assert all(map(os.path.isfile, validation_imgs))
        assert all(map(os.path.isfile, validation_labels))

        self.training_imgs = tf.constant(training_imgs, dtype=tf.string)
        self.training_labels = tf.constant(training_labels, dtype=tf.string)
        self.validation_imgs = tf.constant(validation_imgs, dtype=tf.string)
        self.validation_labels = tf.constant(validation_labels, dtype=tf.string)

        # Test data
        if self.config.use_test:
            test_dir = os.path.join(self.config.data_path, 'test')
            
            test_imgs = [os.path.join(test_dir, 'images', fname) for fname in os.listdir(os.path.join(test_dir, 'images'))]
            assert all(map(os.path.isfile, test_imgs))
            
            self.test_imgs = tf.constant(test_imgs)
        
        else:
            self.test_imgs = None


    def _parse_img_label(self, img_path, label_path):

        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=3)

        label_file = tf.read_file(label_path)
        label_decoded = tf.image.decode_png(label_file, channels=1)

        return img_decoded, label_decoded

    def _parse_img(self, img_path):
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=3)
        return img_decoded

    def _transform_img_tr(self, img, label):

        # Concatenate img and label before transformations
        img_ch = tf.shape(img)[-1]
        img = tf.concat([img, label], axis=2)

        img = tf.image.resize_images(img, (self.config.input_size, self.config.input_size))
        
        if "horizontal_flip" in self.config.augmentations:
            img = tf.image.random_flip_left_right(img)

        return img[:, :, :img_ch], img[:, :, img_ch:]

    def _transform_img_val(self, img, label):

        img_ch = tf.shape(img)[-1]

        img = tf.concat([img, label], axis=2)

        img = tf.expand_dims(img, 0)
        
        if "horizontal_flip" in self.config.augmentations:
            img = tf.concat([img, tf.image.flip_left_right(img)], 0)

        orig_imgs = img[:, :, :, :img_ch]
        orig_labels = img[:, :, :, img_ch:]

        img = tf.image.resize_images(img, (self.config.input_size, self.config.input_size))
        
        return orig_imgs, orig_labels, img[:, :, :, :img_ch], img[:, :, :, img_ch:]

    def _transform_img_test(self, img):
        
        img = tf.expand_dims(img, 0)
        new_img = tf.image.resize_images(img, (self.config.input_size, self.config.input_size))
        return img, new_img

    def _build_dataset(self):

        # Training
        tr_data = tf.data.Dataset.from_tensor_slices((self.training_imgs, self.training_labels))
        tr_data = tr_data.apply(tf.contrib.data.shuffle_and_repeat(self.config.batch_size * 2))
        tr_data = tr_data.map(self._parse_img_label, num_parallel_calls=self.config.num_parallel_calls)
        tr_data = tr_data.map(self._transform_img_tr, num_parallel_calls=self.config.num_parallel_calls)
        tr_data = tr_data.batch(self.config.batch_size)
        tr_data = tr_data.prefetch(1)

        tr_iterator = tr_data.make_one_shot_iterator()
        self.next_batch = tr_iterator.get_next()

        # Validation
        val_data = tf.data.Dataset.from_tensor_slices((self.validation_imgs, self.validation_labels))
        val_data = val_data.map(self._parse_img_label, num_parallel_calls=self.config.num_parallel_calls)
        val_data = val_data.map(self._transform_img_val, num_parallel_calls=self.config.num_parallel_calls)

        self.val_iterator = val_data.make_initializable_iterator()
        self.val_next = self.val_iterator.get_next()

        # Test
        self.test_iterator = None
        self.test_next = None
        if self.config.use_test:
            test_data = tf.data.Dataset.from_tensor_slices(self.test_imgs)
            test_data = test_data.map(self._parse_img, num_parallel_calls=self.config.num_parallel_calls)
            test_data = test_data.map(self._transform_img_test, num_parallel_calls=self.config.num_parallel_calls)

            self.test_iterator = test_data.make_initializable_iterator()
            self.test_next = self.test_iterator.get_next()
