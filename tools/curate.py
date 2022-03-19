# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.

import argparse
import json
import os.path as osp
import random

import numpy as np
import tqdm
from mmcv import Config, print_log

from drugood.curators import GenericCurator


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument('--cfg', help='curator config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    print_log(f'Curator Config:\n{cfg.pretty_text}''\n' + '-' * 60)
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
