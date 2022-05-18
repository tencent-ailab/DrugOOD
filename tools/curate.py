# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.

import argparse
import json
import os.path as osp
import random

import numpy as np
import tqdm
from mmcv import Config, print_log

from drugood.curators import GenericCurator
from drugood.apis import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument('config', help='curator config file path')
    parser.add_argument('--seed', type=int, default=12345, help='random seed (please not change it)')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    print_log(f'Curator Config:\n{cfg.pretty_text}''\n' + '-' * 60)

    # set random seed
    if args.seed is not None:
        print_log(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    curator = GenericCurator(cfg)
    # Processing Flow
    data = curator.data_loading()
    data = curator.noise_filtering(data)
    data = curator.uncertainty_processing(data)
    data = curator.classification_label_generating(data)
    data = curator.data_splitting(data)
    curator.data_saving(data)
    curator.statistics_reporting()


if __name__ == '__main__':
    main()
