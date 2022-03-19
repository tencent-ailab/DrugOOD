# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
CLASSIFIERS = MODELS

ATTENTION = Registry('attention', parent=MMCV_ATTENTION)
TASKERS = Registry('taskers')


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_losses(cfg):
    """Build loss."""
    if not isinstance(cfg, list):
        cfg = [cfg]
    return [LOSSES.build(_cfg) for _cfg in cfg]


def build_tasker(cfg):
    return TASKERS.build(cfg)


def build_model(cfg):
    """Build Models"""
    return MODELS.build(cfg)
