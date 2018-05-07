import tensorflow as tf
import numpy as np
import os

class MnistTrainer(object):

    def __init__(self, sess, model, data):
        self.model = model
        self.sess = sess
        self.data = data

        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.summary = tf.summary.merge_all()
        self.sess.run(self.init)

    def train_on_batch(self, batch_xs, batch_ys):
        results = self.sess.run([self.model.train_step,
                                #  self.summary, 
                                 self.model.loss, 
                                 self.model.accuracy],
                                 feed_dict={
                                     self.model.x: batch_xs,
                                     self.model.y_target: batch_ys,
                                     self.model.is_training: True
                                 })
        return results[1:]


    def train(self, mb_size, batches_n, logdir):
        losses = []

        tbTrainWriter = tf.summary.FileWriter(os.path.join(logdir, "train"), self.sess.graph)
        # tbTestWriter = tf.summary.FileWriter(os.path.join(logdir, "test"))

        try:
            for batch_idx in range(batches_n):
                batch_xs, batch_ys = self.data.train.next_batch(mb_size)

                results = self.train_on_batch(batch_xs, batch_ys)

                losses.append(results)

                if batch_idx % 100 == 0:

                    training_loss, training_accuracy = np.mean(losses[-200:], axis=0)
                    
                    test_loss, test_accuracy = self.sess.run([
                        # self.model.summary,
                        self.model.loss,
                        self.model.accuracy],
                        feed_dict={
                            self.model.x: self.data.test.images,
                            self.model.y_target: self.data.test.labels,
                            self.model.is_training: False
                        }
                    )
                    
                    print('Batch {batch_idx:}:'.format(batch_idx=batch_idx))
                    print('Training results: loss: {mean_loss}, accuracy: {mean_accuracy}'.format(
                        mean_loss=training_loss, mean_accuracy=training_accuracy
                    ))

                    print('Test results: loss: {mean_loss}, accuracy: {mean_accuracy}'.format(
                        mean_loss = test_loss, mean_accuracy=test_accuracy
                    ))

                    self.model.save(self.sess)

                    # tbTrainWriter.add_summary(results[0], batch_idx)
                    # tbTestWriter.add_summary(test_summary, batch_idx)

        except KeyboardInterrupt:
            print('Stopping training!')
            pass
        
        test_loss, test_accuracy = self.sess.run([
            # self.model.summary,
            self.model.loss,
            self.model.accuracy],
            feed_dict={
                self.model.x: self.data.test.images,
                self.model.y_target: self.data.test.labels,
                self.model.is_training:False
            }
        )

        print('Final test results:')
        print('loss: {loss}'.format(loss=test_loss))
        print('accuracy: {accuracy}'.format(accuracy=test_accuracy))
