# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import os
from functools import partial

import torch


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def move_to_device(obj, device=None):
    if (device is None):
        device = torch.device('cuda')
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, int):
        return obj
    else:
        return obj.to(device)


def make_dirs(dir):
    if (not os.path.exists(dir)):
        try:
            os.makedirs(dir)
        except FileNotFoundError as e:
            print(str(e))
