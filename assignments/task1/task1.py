import tensorflow as tf 
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data


logPath = "./tb_logs/"

class MnistTrainer(object):
    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.train_step, self.summaries, self.loss, self.accuracy],
                                feed_dict={self.x: batch_xs, self.y_target: batch_ys})
        return results[1], results[2:]

    def batch_normalization(self, x, filters, epsilon=1e-5):
        with tf.name_scope("batch_normalization"):
            mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True, name="mean")
            var = tf.reduce_mean(tf.squared_difference(x, tf.stop_gradient(mean)), [0, 1, 2], name="variance")
            
            scale = tf.Variable(tf.constant(1.0, shape=[filters]), name="scale")
            offset = tf.Variable(tf.constant(0.0, shape=[filters]), name="offset")

            inv = tf.rsqrt(var + epsilon) * scale

            return x * inv + (offset - mean * inv)
            
    def create_model(self):

        self.convActivations = []
        self.fieldOfView = []

        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x')
            self.y_target = tf.placeholder(tf.float32, [None, 10], name='y_target')
            input_layer = tf.reshape(self.x, [-1, 28, 28, 1])

        self.conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
        )

        self.convActivations.append(self.conv1)
        self.fieldOfView.append(3)

        # bn1 = tf.layers.batch_normalization(inputs=conv1)
        bn1 = self.batch_normalization(self.conv1, 16)
        relu1 = tf.nn.relu(bn1)

        conv2 = tf.layers.conv2d(
            inputs=relu1,
            filters=16,
            kernel_size=[3, 3],
            padding="same"
        )

        self.convActivations.append(conv2)
        self.fieldOfView.append(5)

        # bn2 = tf.layers.batch_normalization(inputs=conv2)
        bn2 = self.batch_normalization(conv2, 16)
        relu2 = tf.nn.relu(bn2)

        pool1 = tf.layers.max_pooling2d(inputs=relu2, pool_size=[2, 2], strides=2)
        dropout1 = tf.layers.dropout(inputs=pool1, rate=0.25)

        conv3 = tf.layers.conv2d(
            inputs=dropout1,
            filters=32,
            kernel_size=[3, 3],
            padding="same"
        )

        self.convActivations.append(conv3)
        self.fieldOfView.append(7)

        # bn3 = tf.layers.batch_normalization(inputs=conv3)
        bn3 = self.batch_normalization(conv3, 32)
        relu3 = tf.nn.relu(bn3)

        conv4 = tf.layers.conv2d(
            inputs=relu3,
            filters=32,
            kernel_size=[3, 3],
            padding="same"
        )

        self.convActivations.append(conv4)
        self.fieldOfView.append(9)

        # bn4 = tf.layers.batch_normalization(inputs=conv4)
        bn4 = self.batch_normalization(conv4, 32)
        relu4 = tf.nn.relu(bn4)

        pool2 = tf.layers.max_pooling2d(inputs=relu4, pool_size=[2, 2], strides=2)
        dropout2 = tf.layers.dropout(inputs=pool2, rate=0.25)

        flattened = tf.reshape(dropout2, [-1, 7 * 7 * 32])

        fc1 = tf.layers.dense(inputs=flattened, units=512, activation=tf.nn.relu)
        fc1_dropout = tf.layers.dropout(inputs=fc1, rate=0.5)

        fc2 = tf.layers.dense(inputs=fc1_dropout, units=1024, activation=tf.nn.relu)
        fc2_dropout = tf.layers.dropout(inputs=fc2, rate=0.5)

        self.logits = tf.layers.dense(inputs=fc2_dropout, units=10)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_target))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_target, axis=1), tf.argmax(self.logits, axis=1)), tf.float32))
  
        # self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.train_step = tf.train.RMSPropOptimizer(1e-4).minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)

    def visualize(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        self.create_model()
        self.saver = tf.train.Saver()

        with tf.Session() as self.sess:
            self.saver.restore(self.sess, 'tmp/model.ckp')

            # def getPatch(z, x, y, vield_of_view, image):
            #     d = vield_of_view / 2
            #     x_max, y_max = image.shape[1:3]
            #     print(x_max, y_max)
            #     x1 = max(0, x - d)
            #     y1 = max(0, y - d)
            #     x2 = min(x_max, x + d)
            #     y2 = min(y_max, y + d)
            #     return image[z, x1:x2, y1:y2, :]


            def vizLayer(layer, scale, fov, n=10):
                # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
                top_n = None
                for image in mnist.train.images[:1000]:
                    activation = self.sess.run(self.conv1, feed_dict={self.x: [image]})
                    images = image.reshape([-1, 28, 28, 1])
                    top_n = [[] for _ in range(activation.shape[-1])] if top_n is None else top_n
                    for c in range(activation.shape[-1]):
                        channel = activation[:, :, :, c]
                        xs, ys, zs = np.unravel_index(np.argsort(channel, axis=None)[:n], channel.shape)
                        top_n_vals = [(channel[x, y, z], image) for x, y, z in zip(xs, ys, zs)]
                        # top_n[c] = sorted(top_n[c] + top_n_vals, reverse=True)[:10]

                print(top_n)
                return top_n


            vizLayer(self.conv1, 1, 3)

    def train(self):

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.create_model()
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver()

        with tf.Session() as self.sess:

            tbTrainWriter = tf.summary.FileWriter(logPath + 'train', self.sess.graph)
            # tbTestWriter = tf.summary.FileWriter(logPath + 'test')
            
            tf.global_variables_initializer().run()
            
            # self.saver.restore(self.sess, 'tmp/model.ckp')

            batches_n = 10000
            mb_size = 32

            losses = []
            
            start_time = time.time()
            end_time = start_time

            try:
                for batch_idx in range(batches_n):
                    batch_xs, batch_ys = mnist.train.next_batch(mb_size)

                    summary, vloss = self.train_on_batch(batch_xs, batch_ys)

                    losses.append(vloss)

                    if batch_idx % 100 == 0:
                        
                        # t_summary, t_loss, t_accuracy = self.sess.run([self.summaries, self.loss, self.accuracy], 
                                                                        # feed_dict={self.x: mnist.test.images,
                                                                                #    self.y_target: mnist.test.labels})
                        
                        end_time = time.time()
                        print('Batch {batch_idx}: elapsed time {elapsed_time:.2f} seconds, mean_loss {mean_loss}'.format(
                            batch_idx=batch_idx, elapsed_time=end_time-start_time, mean_loss=np.mean(losses[-200:], axis=0))
                        )
                        
                        # print('Test results', [t_loss, t_accuracy])
                        
                        self.saver.save(self.sess, 'tmp/model.ckp')

                        tbTrainWriter.add_summary(summary, batch_idx)
                        # tbTestWriter.add_summary(t_summary, batch_idx)

            except KeyboardInterrupt:
                print('Stopping training!')
                pass

            print('Test results', self.sess.run([self.loss, self.accuracy], feed_dict={self.x: mnist.test.images,
                            self.y_target: mnist.test.labels}))
 
if __name__ == '__main__':
    trainer = MnistTrainer()
    # trainer.train()
    trainer.visualize()
