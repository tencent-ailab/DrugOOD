# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
from mmcv.utils import Registry

ALGORITHMS = Registry('algorithms')


def build_algorithm(cfg):
    return ALGORITHMS.build(cfg)
