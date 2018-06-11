from logger import Logger
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import brackets


class Trainer:

    def __init__(self, sess, model, config):
        self.sess = sess
        self.model = model
        self.config = config
        self.logger = Logger(sess, config)

    def train(self):

        accs = []
        losses = []
        loop = tqdm(range(self.config.steps), ncols=120)

        for _ in loop:

            step = self.sess.run(tf.train.get_global_step())
            data_x, data_y = brackets.next_batch(self.config.batch_size, self.config.max_brackets)

            feed_dict = {self.model.x: data_x, self.model.y: data_y}
            _, loss, acc = self.sess.run([
                self.model.train_step,
                self.model.loss,
                self.model.acc], feed_dict=feed_dict)

            accs.append(acc)
            losses.append(loss)

            if step % self.config.epoch == 0:
                mean_loss = np.mean(losses[-2*self.config.epoch:])
                mean_acc = np.mean(accs[-2*self.config.epoch:], axis=0)

                summaries_dict = {
                    'loss': mean_loss,
                    'acc_largest_num_open': mean_acc[0],
                    'acc_largest_num_consecutive': mean_acc[1],
                    'acc_largest_distance': mean_acc[2]
                }

                self.logger.summarize(step, summaries_dict=summaries_dict)

                if self.config.verbose:
                    print('Step {step}: mean_loss {loss}, accs {acc}'.format(
                        step=step, loss=mean_loss, acc=mean_acc
                    ))

            if step % self.config.save_every == 0 and step and self.config.save:
                self.model.save(self.sess)

        best_loss = np.min(losses)
        best_acc = np.max(accs, axis=0)

        summaries_dict = {
            'best_loss': best_loss,
            'best_acc_largest_num_open': best_acc[0],
            'best_acc_largest_num_consecutive': best_acc[1],
            'best_acc_largest_distance': best_acc[2]
        }

        self.logger.summarize(
            self.config.max_brackets,
            summarizer="best_score",
            summaries_dict=summaries_dict)

    def predict(self, data_x, data_y):

        feed_dict = {
            self.model.x: data_x,
            self.model.y: data_y
        }

        return self.sess.run([self.model.output, self.model.predictions], feed_dict=feed_dict)