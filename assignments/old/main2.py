from utils import process_config
from data_loader import DataLoader
from model import UnetModel
from trainer import UnetTrainer
from logger import Logger
from splitter import Splitter, Joiner
import tensorflow as tf

import numpy as np
from PIL import Image

if __name__ == '__main__':
    
    config = process_config('./configs/simple_config.json')
    
    data = DataLoader(config)
    model = UnetModel(config)
    splitter = Splitter(config)
    joiner = Joiner(config)

    with tf.Session() as sess:
        logger = Logger(sess, config)
        sess.run(data.val_iterator.initializer)
        
        sess.run(tf.group(
            tf.global_variables_initializer(), 
            tf.local_variables_initializer()
        ))
        
        data_x, data_y = sess.run(data.val_next)

        img = Image.fromarray(data_x[0])
        img.save('input.jpg')
        
        data_x, orig_shape, resized_shape = sess.run([splitter.patches, splitter.orig_shape, splitter.resized_shape], feed_dict={splitter.input: data_x})

        print(data_x.shape, orig_shape, resized_shape)

        data_x = sess.run(model.predictions, feed_dict={model.x: data_x})        

        print(data_x.shape)

        data_x = sess.run(joiner.images, feed_dict={
            joiner.input: data_x,
            joiner.resized_shape: resized_shape,
            joiner.orig_shape: orig_shape
        })

        print(data_x.shape)

        img2 = Image.fromarray(np.squeeze(data_x[0]), mode='L')
        img2.save('prediction.png')

        print(data_y.dtype)
        img3 = Image.fromarray(np.squeeze(data_y[0]), mode='L')
        img3.save('gold.png')

        # trainer = UnetTrainer(sess, model, data, config, logger)
        # trainer.train()