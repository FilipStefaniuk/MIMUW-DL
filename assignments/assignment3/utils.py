from bunch import Bunch
import argparse
import json
import os


def get_config(args):
    json_file = args.config
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)
    config.max_brackets = args.max_brackets
    config.best_summary_dir = os.path.join(
        "./experiments",
        config.experiment_name,
        "summary/",
        "best"
    )

    config.summary_dir = os.path.join(
        "./experiments",
        config.experiment_name,
        "summary/",
        'max-len-' + str(config.max_brackets))

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
        '--load',
        action='store_true'
    )

    parser.add_argument(
        '--predict',
        action='store_true'
    )

    args = parser.parse_args()
    return args
