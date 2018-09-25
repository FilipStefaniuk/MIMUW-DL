from bracket_generator import BracketGenerator
from dataset import Dataset
import tensorflow as tf

if __name__ == '__main__':
    gen = BracketGenerator(min_len=5, max_len=10)
    # next_batch = tf.data.Dataset.from_generator(
    #     gen,
    #     (tf.string, tf.int64),
    #     (tf.TensorShape([]), tf.TensorShape([3]))
    # ).batch(32).make_one_shot_iterator().get_next()

    # for i, x in enumerate(gen()):
    #     print(x)
        
    #     if i > 10:
    #         break


    # model = RNNModel()

    dataset = Dataset(max_len=10, min_len=1)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        x = sess.run(dataset.next)
        print(x)

