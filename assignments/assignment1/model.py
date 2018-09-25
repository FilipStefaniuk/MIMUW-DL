import tensorflow as tf
from batch_normalization import batch_normalization
import os

class CNNModel(object):
    
    def __init__(self, name):
        self.name = name
        self.build_model()
        self.init_saver()

    def build_model(self):

        self.is_training = tf.placeholder(tf.bool, name="is_training")
        
        with tf.name_scope("Input"):
            self.x = tf.placeholder(tf.float32, [None, 784], name="x")
            self.y_target = tf.placeholder(tf.float32, [None, 10], name="y_target")
            self.input = tf.reshape(self.x, [-1, 28, 28, 1])

        self.conv1 = tf.layers.conv2d(
            inputs=self.input,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )

        self.bn1 = batch_normalization(self.conv1, is_conv=True)

        self.conv2 = tf.layers.conv2d(
            inputs=self.bn1,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )

        self.bn2 = batch_normalization(self.conv2, is_conv=True)

        self.pool1 = tf.layers.max_pooling2d(
            inputs=self.bn2,
            pool_size=[2, 2],
            strides=2
        )

        self.dropout1 = tf.layers.dropout(
            inputs=self.pool1,
            rate=0.25, 
            training=self.is_training
        )

        self.conv3 = tf.layers.conv2d(
            inputs=self.dropout1,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )

        self.bn3 = batch_normalization(self.conv3, is_conv=True)

        self.conv4 = tf.layers.conv2d(
            inputs=self.bn3,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )

        self.bn4 = batch_normalization(self.conv4, is_conv=True)

        self.pool2 = tf.layers.max_pooling2d(
            inputs=self.bn4,
            pool_size=[2, 2],
            strides=2
        )

        self.dropout2 = tf.layers.dropout(
            inputs=self.pool2,
            rate=0.25,
            training=self.is_training
        )

        with tf.name_scope("Flatten"):
            self.flattened = tf.reshape(self.dropout2, [-1, 7 * 7 * 32])

        self.fc1 = tf.layers.dense(
            inputs=self.flattened,
            units=512,
            activation=tf.nn.relu
        )

        self.bn5 = batch_normalization(self.fc1)

        self.dropout3 = tf.layers.dropout(
            inputs=self.bn5,
            rate=0.5,
            training=self.is_training
        )

        self.fc2 = tf.layers.dense(
            inputs=self.dropout3,
            units=1024,
            activation=tf.nn.relu
        )

        self.bn6 = batch_normalization(self.fc2)

        self.dropout4 = tf.layers.dropout(
            inputs=self.bn6,
            rate=0.5,
            training=self.is_training
        )

        self.logits = tf.layers.dense(
            inputs=self.dropout4,
            units=10
        )

        with tf.name_scope("Loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_target))
        
        with tf.name_scope("Accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(self.logits, axis=1)), tf.float32))

        with tf.name_scope("Training"):
            self.train_step = tf.train.RMSPropOptimizer(1e-4).minimize(self.loss)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=2)
    
    def save(self, sess, step):
        self.saver.save(sess, os.path.join('./models/', self.name), step)

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint('./models/')

        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)