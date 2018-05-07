import tensorflow as tf
import numpy as np

class FilterVisualizer(object):
    def __init__(self, sess, model, data):
        self.model = model
        self.sess = sess
        self.data = data

        self.model.load(self.sess)

    def get_patch(self, x, y, img, scale, view):
        view = min(view, 28)
        r = view // 2

        x_min = max(0, x*scale - r)
        x_max = min(28, x*scale + r +1)

        y_min = max(0, y*scale-r)
        y_max = min(28, y*scale+r+1)

        img = img.reshape([28, 28])
        return img[x_min:x_max, y_min:y_max]

    def visualizeLayer(self, layer, view, n=10, scale=1, batch_size=1000):
        top_n = None

        # for i in range(0, self.data.train.images.shape[0], batch_size):
        for i in range(0, 1000, batch_size):
            batch_xs = self.data.train.images[i:i + batch_size]

            activations, _ = self.sess.run([layer, self.model.is_training], feed_dict={self.model.x:batch_xs, self.model.is_training: False})

            top_n = [[] for _ in range(activations.shape[-1])] if top_n is None else top_n
            images = batch_xs.reshape([-1, 28, 28, 1])
            
            for j in range(activations.shape[-1]):
                channel = activations[:, :, :, j]
                xs, ys, zs = np.unravel_index(np.argsort(channel, axis=None)[-n:], channel.shape)
                top_n_imgs = [(channel[x, y, z], self.get_patch(y, z, images[x], scale, view)) for x, y, z in zip(xs, ys, zs)]
                top_n[j] = sorted(top_n[j] + top_n_imgs, key=lambda x: x[0])[-n:]

        return [list(reversed([img for _, img in channel])) for channel in top_n]
