import tensorflow as tf
import numpy as np
from tqdm import tqdm 

class UnetTrainer(object):
    
    def __init__(self, sess, model, data, config, logger):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        
        self.sess.run(tf.group(
            tf.global_variables_initializer(), 
            tf.local_variables_initializer()
        ))

        tf.gfile.MakeDirs(self.config.checkpoint_dir)
        tf.gfile.MakeDirs(self.config.summary_dir)

        self.model.load(sess)

    def train_step(self):
        batch_x, batch_y = self.sess.run(self.data.next_batch)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.acc],
                                     feed_dict=feed_dict)
        return loss, acc[0]

    def train_epoch(self, epoch_num):
        
        losses = []
        accs = []

        loop = tqdm(range(self.config.steps_in_epoch), ncols=120, desc='Epoch {0}/{1}'.format(epoch_num, self.config.epochs))

        for _ in loop:
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

    def train(self):
        for i in range(1, self.config.epochs + 1):
            self.train_epoch(i)