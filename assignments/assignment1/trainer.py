import tensorflow as tf
import numpy as np
import time
import os

class MnistTrainer(object):

    def __init__(self, sess, model, data):
        self.model = model
        self.sess = sess
        self.data = data

        # This way i can have test and train accuracy/loss on different graphs
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.training_loss_summary = tf.summary.scalar('training_loss', self.model.loss)
        self.training_accuracy_summary = tf.summary.scalar('training_accuracy', self.model.accuracy)
        self.test_loss_summary = tf.summary.scalar('test_loss', self.model.loss)
        self.test_accuracy_summary = tf.summary.scalar('test_accuracy', self.model.accuracy)

        self.tb_writer = tf.summary.FileWriter(os.path.join('./logs/', model.name), self.sess.graph)

        self.sess.run(self.init)

    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.model.train_step,
                                 self.training_loss_summary,
                                 self.training_accuracy_summary,
                                 self.model.loss, 
                                 self.model.accuracy],
                                 feed_dict={
                                     self.model.x: batch_xs,
                                     self.model.y_target: batch_ys,
                                     self.model.is_training: True
                                 })
        return results[1:]

    def evaluate(self, data_xs, data_ys):
        results = self.sess.run([self.test_loss_summary,
                                 self.test_accuracy_summary,
                                 self.model.loss,
                                 self.model.accuracy],
                                 feed_dict={
                                     self.model.x: data_xs,
                                     self.model.y_target: data_ys,
                                     self.model.is_training: False
                                 })
        return results


    def train(self, mb_size, batches_n, info_every=100, save_every=None, try_load=True):
        first_batch_idx = 0

        losses = []

        start_time = time.time()
        end_time = start_time

        try:
            for batch_idx in range(first_batch_idx, batches_n):
                
                batch_xs, batch_ys = self.data.train.next_batch(mb_size)

                results = self.train_on_batch(batch_xs, batch_ys)

                losses.append(results[2:])
                self.tb_writer.add_summary(results[0], batch_idx)
                self.tb_writer.add_summary(results[1], batch_idx)

                if batch_idx % info_every == 0:

                    training_loss, training_accuracy = np.mean(losses[-200:], axis=0)
                    
                    test_loss_summary, test_accuracy_summary, test_loss, test_accuracy = self.evaluate(self.data.test.images, self.data.test.labels)
                    
                    self.tb_writer.add_summary(test_loss_summary, batch_idx)
                    self.tb_writer.add_summary(test_accuracy_summary, batch_idx)

                    end_time = time.time()
                    
                    print('Batch {batch_idx:}:'.format(batch_idx=batch_idx))
                    print('Elapsed time: {time:.2f} s'.format(time=end_time-start_time))
                    print('Training results: loss: {mean_loss:.4f}, accuracy: {mean_accuracy:.4f}'.format(
                        mean_loss=training_loss, mean_accuracy=training_accuracy
                    ))

                    print('Test results: loss: {mean_loss:.4f}, accuracy: {mean_accuracy:.4f}'.format(
                        mean_loss = test_loss, mean_accuracy=test_accuracy
                    ))

                if save_every is not None and batch_idx % save_every == 0:
                    self.model.save(self.sess, batch_idx)

        except KeyboardInterrupt:
            print('Stopping training!')
            pass
        
        _, _, test_loss, test_accuracy = self.evaluate(self.data.test.images, self.data.test.labels)

        print('Final test results:')
        print('loss: {loss:.4f}'.format(loss=test_loss))
        print('accuracy: {accuracy:.4f}'.format(accuracy=test_accuracy))
