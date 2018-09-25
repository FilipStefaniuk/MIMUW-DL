from bunch import Bunch
import argparse
import json
import os


def get_config(json_file):

    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)
    config.summary_dir = os.path.join("./experiments", config.experiment_name, "summary/")
    config.checkpoint_dir = os.path.join("./experiments", config.experiment_name, "checkpoints/")

    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        help='The configuration file')

    parser.add_argument(
        '-n', '--max_brackets',
        default=1,
        type=int,
        help='maximum number of brackets (only relevant when trained)'
    )

    parser.add_argument(
        '--train',
        action='store_true'
    )

    parser.add_argument(
        '--test',
        action='store_true'
    )

    args = parser.parse_args()
    return args
