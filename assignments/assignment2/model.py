import tensorflow as tf
import os

class UnetModel(object):

    def __init__(self, config):
        self.config = config
        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def conv2d_3x3(self, filters):
        return tf.layers.Conv2D(filters, kernel_size=(3,3), activation=tf.nn.relu, padding='same')

    def max_pool(self):
        return tf.layers.MaxPooling2D((2, 2), strides=2, padding='same')

    def conv2d_transpose_2x2(self, filters):
        return tf.layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')

    def concatenate(self, branches):
        return tf.concat(branches, axis=3)

    def batch_norm(self):
        return tf.layers.BatchNormalization()

    def build_model(self):

        self.x = tf.placeholder(tf.float32, [None, self.config.crop_size, self.config.crop_size, 3], name='x')
        self.y = tf.placeholder(tf.int32, [None, self.config.crop_size, self.config.crop_size, 1], name='y')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.labels = tf.reshape(self.y, shape=[-1, self.config.crop_size, self.config.crop_size], name='labels')

        c1 = self.conv2d_3x3(8) (self.x)        # 128x128x8
        b1 = self.batch_norm() (c1)
        p1 = self.max_pool() (b1)           

        c2 = self.conv2d_3x3(16) (p1)           # 64x64x16
        b2 = self.batch_norm() (c2)
        p2 = self.max_pool() (b2)

        c3 = self.conv2d_3x3(32) (p2)           # 32x32x32
        b3 = self.batch_norm() (c3)
        p3 = self.max_pool() (b3)

        c4 = self.conv2d_3x3(64) (p3)           # 16x14x64
        b4 = self.batch_norm() (c4)
        p4 = self.max_pool() (b4)

        c5 = self.conv2d_3x3(128) (p4)          # 8x8x128
        b5 = self.batch_norm() (c5)

        u6 = self.conv2d_transpose_2x2(64) (b5) # 16x16x64
        b6 = self.batch_norm() (u6)
        u6 = self.concatenate([b6, b4])         # 16x16x128
        c6 = self.conv2d_3x3(66) (u6)           # 16x16x66

        u7 = self.conv2d_transpose_2x2(66) (c6) # 32x32x66
        b7 = self.batch_norm() (u7)
        u7 = self.concatenate([b7, b3])         # 32x32x98
        c7 = self.conv2d_3x3(66) (u7)           # 32x32x66

        u8 = self.conv2d_transpose_2x2(66) (c7) # 64x64x66
        b8 = self.batch_norm() (u8)
        u8 = self.concatenate([b8, b2])         # 64x64x82
        c8 = self.conv2d_3x3(66) (u8)           # 64x64x66

        u9 = self.conv2d_transpose_2x2(66) (c8) # 128x128x66
        b9 = self.batch_norm() (u9)
        u9 = self.concatenate([b9, b1])         # 128x128x74
        c9 = self.conv2d_3x3(66) (u9)           # 128x128x66

        self.logits = tf.layers.Conv2D(66, (1, 1)) (c9)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.acc = tf.metrics.accuracy(labels=self.labels, predictions=tf.argmax(self.logits, axis=3))
        self.train_step = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(self.loss, global_step=self.global_step)
    
    def save(self, sess):
        model_name = self.saver.save(sess, os.path.join(self.config.checkpoint_dir, self.config.exp_name), self.global_step)
        
        if self.config.verbose:
            print('Model saved as {}'.format(model_name))

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            self.saver.restore(sess, latest_checkpoint)

            if self.config.verbose:
                print('Model loaded from checkpoint {}'.format(latest_checkpoint))