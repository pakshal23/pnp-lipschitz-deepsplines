import argparse
import json
import torch
from utils import trainer
import os


def main(config, device):
    
    # Set up directories for saving results
    exp_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    trainer_inst = trainer.Trainer(config, device)
    trainer_inst.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config.json', type=str,
                        help='Path to the config file')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='device to use')
    args = parser.parse_args()

    config = json.load(open(args.config))
    main(config, args.device)
