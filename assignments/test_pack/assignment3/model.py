import tensorflow as tf
import os


class RNNModel:

    def __init__(self, config):
        self.config = config
        self._get_cell_type()
        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _get_cell_type(self):
        if self.config.cell_type in ('rnn'):
            self.cell_type = tf.nn.rnn_cell.BasicRNNCell
        elif self.config.cell_type in ('lstm'):
            self.cell_type = tf.nn.rnn_cell.BasicLSTMCell
        elif self.config.cell_type in ('gru'):
            self.cell_type = tf.nn.rnn_cell.GRUCell
        else:
            raise ValueError('invalid cell type')

    def _build_model(self):

        with tf.name_scope("input"):
            self.x = tf.placeholder(tf.string, shape=[None])
            self.y = tf.placeholder(tf.int64, shape=[None, 3])

        with tf.name_scope('input_transformations'):
            table = tf.contrib.lookup.index_table_from_tensor(
                mapping=tf.constant(["(", ")"]),
                num_oov_buckets=0
            )

            x = tf.string_split(self.x, delimiter='')
            x = tf.sparse_to_dense(x.indices, x.dense_shape, x.values, default_value="")
            x = tf.one_hot(table.lookup(x), 2, dtype=tf.int64)
            x = tf.cast(x, tf.float32)

        sequence_length = tf.cast(tf.reduce_sum(x, [1, 2], name="sequence_length"), tf.int32)

        cells = [self.cell_type(size) for size in self.config.hidden_sizes]
        output, _ = tf.nn.dynamic_rnn(
            cell=tf.nn.rnn_cell.MultiRNNCell(cells),
            inputs=x,
            sequence_length=sequence_length,
            dtype=tf.float32
        )

        with tf.name_scope("get_last_output"):
            batch_size = tf.shape(output)[0]
            max_len = tf.shape(output)[1]

            output_size = self.config.hidden_sizes[-1]
            index = tf.range(0, batch_size) * max_len + (sequence_length - 1)
            output = tf.gather(tf.reshape(output, [-1, output_size]), index)

        with tf.name_scope("dense_layer"):
            weights = tf.Variable(tf.random_normal([output_size, 3]))
            biases = tf.Variable(tf.zeros([3]))
            self.output = tf.matmul(output, weights) + biases

        with tf.name_scope("loss"):
            self.loss = tf.losses.mean_squared_error(self.output, self.y)
            self.train_step = tf.train.RMSPropOptimizer(1e-4)\
                                .minimize(self.loss, global_step=tf.train.create_global_step())

        with tf.name_scope("accuracy"):
            self.predictions = tf.cast(tf.round(self.output), tf.int64)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.y), tf.float32), axis=0)

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
        else:
            print('Failed to load model')

    def save(self, sess):
        if self.config.verbose:
            print("Saving model...")
        name = os.path.join(self.config.checkpoint_dir, self.config.experiment_name)
        self.saver.save(sess, name, tf.train.get_global_step())
