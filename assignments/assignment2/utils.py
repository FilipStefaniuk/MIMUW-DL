import os
import json
import argparse
from bunch import Bunch


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return Bunch(config_dict), config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join("./experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("./experiments", config.exp_name, "checkpoint/")
    return config

def get_args():
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    
    args = argparser.parse_args()
    
    return args