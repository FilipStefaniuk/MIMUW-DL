import tensorflow as tf

class UnetTrainer(object):
    
    def __init__(self, sess, model, data):
        self.sess = sess
        self.model = model
        self.data = data

    def train(self):
        pass