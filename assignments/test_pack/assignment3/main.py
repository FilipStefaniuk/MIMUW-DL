from trainer import Trainer
from model import RNNModel
from utils import get_args, get_config
import tensorflow as tf
import numpy as np
import brackets
import sys

if __name__ == '__main__':

    try:
        args = get_args()
        config = get_config(args.config)
    except:
        print("missing or invalid configuration file")
        exit(0)

    tf.gfile.MakeDirs(config.summary_dir)
    tf.gfile.MakeDirs(config.checkpoint_dir)

    model = RNNModel(config)

    with tf.Session() as sess:

        sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
            tf.tables_initializer()
        ])

        model.load(sess)
        trainer = Trainer(sess, model, config)

        if args.train:
            trainer.train()

        if args.test:
            for line in sys.stdin:
                try:
                    data_x = np.array([brackets.parse_string(line)])
                    data_y = np.array([brackets.statistics(b) for b in data_x])
                    _, preds, _, _ = trainer.predict(data_x, data_y)

                    for x, y, p in zip(data_x, data_y, preds):
                        print(x, ',', *y, ',', *p)
                except:
                    print("BAD INPUT")