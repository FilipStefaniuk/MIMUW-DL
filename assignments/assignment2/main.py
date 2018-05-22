from utils import process_config
from data_loader import DataLoader
from model import UnetModel
from trainer import UnetTrainer
from logger import Logger
import tensorflow as tf


if __name__ == '__main__':
    
    config = process_config('./configs/simple_config.json')
    
    data = DataLoader(config)
    model = UnetModel(config)

    with tf.Session() as sess:
        logger = Logger(sess, config)
        trainer = UnetTrainer(sess, model, data, config, logger)
        trainer.train()