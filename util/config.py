'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/pointgroup_run2_partnet.yaml', help='path to config file')
    parser.add_argument('--category', type=str, default='chair', help='category to train')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
