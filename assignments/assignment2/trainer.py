import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


class UnetTrainer(object):
    
    def __init__(self, sess, model, data, config, logger):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        
        # Init variables
        self.sess.run(tf.group(
            tf.global_variables_initializer(), 
            tf.local_variables_initializer()
        ))

        # Create dirs for saving model and summary
        tf.gfile.MakeDirs(self.config.checkpoint_dir)
        tf.gfile.MakeDirs(self.config.summary_dir)
        
        if self.config.use_test:
            tf.gfile.MakeDirs(os.path.join(self.data.test_dir, 'predictions'))

        # Try to load model
        self.model.load(sess)

    def train_step(self):
        batch_x, batch_y = self.sess.run(self.data.next_batch)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.acc],
                                     feed_dict=feed_dict)
        return loss, acc

    def train_epoch(self, epoch_num):
        
        losses = []
        accs = []

        loop = tqdm(range(self.config.steps_in_epoch), ncols=120, desc='Epoch {0}/{1}'.format(epoch_num, self.config.epochs))

        for _ in loop:
            self.train_step()
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)

        loss = np.mean(losses)
        acc = np.mean(accs)

        summaries_dict = {
            'loss': loss,
            'acc': acc
        }

        global_step = tf.train.get_global_step().eval(session=self.sess)

        self.logger.summarize(global_step, summaries_dict=summaries_dict)

        if self.config.verbose:
            print('Global step: {}'.format(global_step))
            print('Training: loss {0}, accuracy {1}'.format(loss, acc))

        self.model.save(self.sess)
        self.validate()

    def train(self):
        for i in range(1, self.config.epochs + 1):
            self.train_epoch(i)

    def predict(self):

        if self.config.use_test:

            loop = tqdm(range(self.data.test_len), ncols=120, desc='Testing')
            self.sess.run(self.data.test_iterator.initializer)

            for _ in loop:
                path, img_orig, img = self.sess.run(self.data.test_next)
                path = path.decode('utf-8')

                feed_dict = {self.model.x: img, self.model.orig_x: img_orig}
                prediction = self.sess.run(self.model.orig_predictions, feed_dict=feed_dict)

                pred_img = Image.fromarray(np.uint8(np.squeeze(prediction)))
                basename = os.path.splitext(os.path.basename(path))[0]
                pred_img.save(os.path.join(self.config.predictions_dir, basename + 'png'))


        
        return prediction

    def validate(self):

        accs = []

        self.sess.run(self.data.val_iterator.initializer)

        loop = tqdm(range(self.data.validation_len), ncols=120, desc='Validation')
        for _ in loop:
            orig_x, orig_y, data_x, data_y = self.sess.run(self.data.val_next)

            feed_dict={
                self.model.x: data_x,
                self.model.y: data_y,
                self.model.orig_x: orig_x,
                self.model.orig_y: orig_y
            }

            acc = self.sess.run(self.model.orig_acc, feed_dict=feed_dict)
            accs.append(acc)

        acc = np.mean(accs)

        summaries_dict = {
            'test_acc': acc
        }

        global_step = tf.train.get_global_step().eval(session=self.sess)
        
        if self.config.verbose:
            print('Test: accuracy {0}'.format(acc))
            
            self.logger.summarize(global_step, summaries_dict=summaries_dict)
