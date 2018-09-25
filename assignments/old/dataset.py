from bracket_generator import BracketGenerator
import tensorflow as tf


class Dataset(object):

    def __init__(self, **kwargs):
        self.gen = BracketGenerator(**kwargs)
        self._build_dataset()

    def _build_dataset(self):

        self.dataset = tf.data.Dataset.from_generator(
            self.gen,
            (tf.float32, tf.int64, tf.float32),
            (
                tf.TensorShape([self.gen.max_len * 2, 2]),
                tf.TensorShape([]),
                tf.TensorShape([3])
            )
        ).batch(32)

        self.next = self.dataset.make_one_shot_iterator().get_next()
