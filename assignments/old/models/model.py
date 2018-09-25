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

        self.x = tf.placeholder(tf.float32, [None, self.config.input_size, self.config.input_size, 3], name='x')
        self.y = tf.placeholder(tf.int32, [None, self.config.input_size, self.config.input_size, 1], name='y')
        
        self.orig_x = tf.placeholder(tf.float32, [None, None, None, 3], name='orig-x')
        self.orig_y = tf.placeholder(tf.int32, [None, None, None, 1], name='orig-y')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        c1 = self.conv2d_3x3(8) (self.x)        # 256x256x8
        c1 = self.batch_norm() (c1)
        # c1 = self.conv2d_3x3(8) (c1)
        # b1 = self.batch_norm() (c1)
        p1 = self.max_pool() (c1)           

        c2 = self.conv2d_3x3(16) (p1)           # 128x128x16
        c2 = self.batch_norm() (c2)
        # c2 = self.conv2d_3x3(16) (c2)
        # b2 = self.batch_norm() (c2)
        p2 = self.max_pool() (c2)

        c3 = self.conv2d_3x3(32) (p2)           # 64x64x32
        c3 = self.batch_norm() (c3)
        # c3 = self.conv2d_3x3(32) (c3)
        # b3 = self.batch_norm() (c3)
        p3 = self.max_pool() (c3)

        c4 = self.conv2d_3x3(64) (p3)           # 32x32x64
        c4 = self.batch_norm() (c4)
        # c4 = self.conv2d_3x3(64) (c4)
        # b4 = self.batch_norm() (c4)
        p4 = self.max_pool() (c4)

        c5 = self.conv2d_3x3(128) (p4)          # 16x16x128
        c5 = self.batch_norm() (c5)
        # c5 = self.conv2d_3x3(128) (c5)
        # b5 = self.batch_norm() (c5)

        u6 = self.conv2d_transpose_2x2(64) (c5) # 32x32x64
        u6 = self.batch_norm() (u6)
        # u6 = self.concatenate([u6, c4])         # 32x32x128
        # c6 = self.conv2d_3x3(66) (u6)           # 32x32x66
        # b6 = self.batch_norm() (c6)

        u7 = self.conv2d_transpose_2x2(66) (u6) # 64x64x66
        u7 = self.batch_norm() (u7)
        u7 = self.concatenate([u7, c3])         # 64x64x98
        # u7 = self.conv2d_3x3(66) (u7)           # 64x64x66
        # u7 = self.batch_norm() (u7)

        u8 = self.conv2d_transpose_2x2(66) (u7) # 128x128x66
        u8 = self.batch_norm() (u8)
        u8 = self.concatenate([u8, c2])         # 128x128x82
        # u8 = self.conv2d_3x3(66) (u8)           # 128x128x66
        # u8 = self.batch_norm() (u8)

        u9 = self.conv2d_transpose_2x2(66) (u8) # 256x256x66
        u9 = self.batch_norm() (u9)
        u9 = self.concatenate([u9, c1])         # 256x256x74
        # c9 = self.conv2d_3x3(66) (u9)           # 256x256x66
        # b9 = self.batch_norm() (c9)

        self.logits = tf.layers.Conv2D(66, (1, 1)) (u9)

        self.predictions = tf.cast(tf.expand_dims(tf.argmax(self.logits, axis=3), -1), tf.int32)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.y), logits=self.logits))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y, self.predictions), tf.float32))
        self.train_step = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(self.loss, global_step=self.global_step)

        orig_shape_x = tf.shape(self.orig_x)
        self.orig_predictions = tf.cast(tf.image.resize_images(self.predictions, (orig_shape_x[1], orig_shape_x[2])), tf.int32)
        self.orig_acc = tf.reduce_mean(tf.cast(tf.equal(self.orig_y, self.orig_predictions), tf.float32))
    
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
        