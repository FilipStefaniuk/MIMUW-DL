import tensorflow as tf
from model import CNNModel
from visualizer import FilterVisualizer
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    model = CNNModel('./models/model1')
    data = input_data.read_data_sets('MNIST_data/')

    with tf.Session() as sess:
        viz = FilterVisualizer(sess, model, data)
        result = viz.visualizeLayer(model.conv3, 13)
        print([len(ch) for ch in result])