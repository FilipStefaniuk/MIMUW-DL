import tensorflow as tf

class Validator(object):

    def __init__(self):
        self.build_validator()

    def build_validator(self):
        self.x = tf.placeholder(tf.int32, shape=[None, None, None, 1])
        self.y = tf.placeholder(tf.int32, shape=[None, None, None, 1])

        self.acc = tf.metrics.accuracy(labels=self.x, predictions=self.y)