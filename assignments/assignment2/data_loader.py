import os
import random
import tensorflow as tf
from model import UnetModel
from sklearn.model_selection import train_test_split

class DataLoader(object):

    def __init__(self, config):
        self.config = config
        self.get_training_validation_paths()
        self.build_dataset()

    def get_training_validation_paths(self):
        training_data = os.path.join(self.config.data_path, 'training')
        filenames = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(training_data, 'images'))]

        x_names, y_names = train_test_split(filenames, test_size=self.config.validation_size, random_state=self.config.random_state)

        train_img_paths = [os.path.join(training_data, 'images', filename + '.jpg') for filename in x_names]
        train_label_paths = [os.path.join(training_data, 'labels_plain', filename + '.png') for filename in x_names]
        val_img_paths = [os.path.join(training_data, 'images', filename + '.jpg') for filename in y_names]
        val_label_paths = [os.path.join(training_data, 'labels_plain', filename + '.png') for filename in y_names]

        assert all(map(os.path.isfile, train_img_paths))
        assert all(map(os.path.isfile, train_label_paths))
        assert all(map(os.path.isfile, val_img_paths))
        assert all(map(os.path.isfile, val_label_paths))

        self.train_imgs = tf.constant(train_img_paths)
        self.train_labels = tf.constant(train_label_paths)
        self.val_imgs = tf.constant(val_img_paths)
        self.val_labels = tf.constant(val_label_paths)

    def parse_img(self, img_path, label_path):
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)

        label_file = tf.read_file(label_path)
        label_decoded = tf.image.decode_png(label_file)

        return img_decoded, label_decoded

    def transform_img(self, img, label):
        size = [self.config.crop_size, self.config.crop_size]
        
        img_ch = tf.shape(img)[-1]
        label_ch = tf.shape(label)[-1]

        concatenated = tf.concat([img, label], axis=2)
        cropped = tf.random_crop(concatenated, size=tf.concat([size, [img_ch + label_ch]], axis=0))

        return cropped[:, :, :img_ch], cropped[:, :, img_ch:]

    def build_dataset(self):
        tr_data = tf.data.Dataset.from_tensor_slices((self.train_imgs, self.train_labels))
        tr_data = tr_data.apply(tf.contrib.data.shuffle_and_repeat(self.config.batch_size * 2))
        tr_data = tr_data.map(self.parse_img, num_parallel_calls=self.config.num_parallel_calls)
        tr_data = tr_data.map(self.transform_img, num_parallel_calls=self.config.num_parallel_calls)
        tr_data = tr_data.batch(self.config.batch_size)
        tr_data = tr_data.prefetch(1)

        iterator = tr_data.make_one_shot_iterator()

        self.next_batch = iterator.get_next()