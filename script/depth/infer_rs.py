import argparse

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import yaml
from src.trainer.predictor_base import Predictor

def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='large scale inference')

    # Configuration arguments
    parser.add_argument('--config', type=str, default='config/ls_mrg_pred.yaml',
                        help='Path to YAML configuration file')
    parser.add_argument('--rank', type=int, default=0)


    return parser.parse_args()


def main():

    args = parse_args()
    config = load_yaml_config(args.config)
    predictor = Predictor(config, rank = args.rank)
    predictor.build_model()
    predictor.load_files()
    out_dir_path = predictor.predict_all()
    print(f"PATH_OUTPUT: {out_dir_path}")


if __name__ == '__main__':
    main()