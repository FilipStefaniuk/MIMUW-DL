import tensorflow as tf

def batch_normalization(x, epsilon=1e-5):
    with tf.name_scope("batch_normalization"):
        filters = x.shape[-1]
        mean = tf.reduce_mean(x, [0, 1, 2], name="mean")
        var = tf.reduce_mean(tf.squared_difference(x, tf.stop_gradient(mean)), [0, 1, 2], name="variance")
        scale = tf.Variable(tf.constant(1.0, shape=[filters]), name="scale")
        offset = tf.Variable(tf.constant(0.0, shape=[filters]), name="offset")

        inv = tf.rsqrt(var + epsilon) * scale
        return x * inv + (offset - mean * inv)

