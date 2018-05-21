import tensorflow as tf

class UnetModel(object):

    def __init__(self):
        self.build_model()

    def build_model(self):
        
        self.x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.y_ = tf.placeholder(tf.float32, [None, 128, 128, 1])

        self.y = tf.layers.Conv2D(
            filters=66,
            kernel_size=(3, 3),
            padding='same'
        )

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.loss)