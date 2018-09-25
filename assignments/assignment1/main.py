import tensorflow as tf
from model import CNNModel
from trainer import MnistTrainer
from tensorflow.examples.tutorials.mnist import input_data

# model1: batch_norm, dropout, RMSProp
# model2: batch_norm, no dropout, RMSProp
# model3: no batch_norm, dropout, RMSProp
# model4: no batch_norm, no dropout, RMSProp

if __name__ == '__main__':

    data = input_data.read_data_sets('MNIST_data/', one_hot=True)
    model = CNNModel('model1')

    with tf.Session() as sess:
        trainer = MnistTrainer(sess, model, data)
        trainer.train(128, 10000, save_every=500)
