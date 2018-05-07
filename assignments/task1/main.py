import tensorflow as tf
from model import CNNModel
from trainer import MnistTrainer
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':

    data = input_data.read_data_sets('MNIST_data/', one_hot=True)

    model = CNNModel('./models/model1')

    with tf.Session() as sess:
        trainer = MnistTrainer(sess, model, data)
        trainer.train(128, 10000, './logs/model1')