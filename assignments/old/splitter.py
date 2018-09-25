import tensorflow as tf

class Splitter(object):
    def __init__(self, config):
        self.config = config
        self.build_splitter()

    def build_splitter(self):

        self.input = tf.placeholder(tf.uint8, shape=[None, None, None, 3])

        patch_height = self.config.crop_size
        patch_width = self.config.crop_size

        self.orig_shape = tf.shape(self.input)
        
        image_height = tf.cast(self.orig_shape[1], dtype=tf.float32)
        image_width = tf.cast(self.orig_shape[2], dtype=tf.float32)
        height = tf.cast(tf.ceil(image_height / (patch_height * 6)) * patch_height, dtype=tf.int32)
        width = tf.cast(tf.ceil(image_width / (patch_width * 6)) * patch_width, dtype=tf.int32)

        images = tf.image.resize_image_with_crop_or_pad(self.input, height, width)

        self.resized_shape = tf.shape(images)

        images = tf.reshape(images, [-1, patch_height, width, 3])
        images = tf.transpose(images, [0, 2, 1, 3])
        images = tf.reshape(images, [-1, patch_width, patch_height, 3])
        images = tf.transpose(images, [0, 2, 1, 3])

        self.patches = images


class Joiner(object):
    def __init__(self, config):
        self.config = config
        self.build_joiner()

    def build_joiner(self):

        self.input = tf.placeholder(tf.uint8, shape=[None, self.config.crop_size, self.config.crop_size, 1])
        self.resized_shape = tf.placeholder(tf.int32, shape=[4])
        self.orig_shape = tf.placeholder(tf.int32, shape=[4])

        patch_height = self.config.crop_size
        patch_height = self.config.crop_size

        images = tf.transpose(self.input, [0, 2, 1, 3])
        images = tf.reshape(images, [-1, self.resized_shape[2], patch_height, 1])
        images = tf.transpose(images, [0, 2, 1, 3])
        images = tf.reshape(images, [-1, self.resized_shape[1], self.resized_shape[2], 1])
        images = tf.image.resize_image_with_crop_or_pad(images, self.orig_shape[1], self.orig_shape[2])

        self.images = tf.cast(images, dtype=tf.uint8)
