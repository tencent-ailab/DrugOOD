# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import warnings
from abc import ABCMeta

import torch
from mmcv.runner import BaseModule

from ..builder import TASKERS, build_backbone, build_head, build_neck


@TASKERS.register_module()
class Regressor(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 init_cfg=None):
        super(Regressor, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    def extract_feat(self, input):
        x = self.backbone(input)
        if self.with_neck:
            x = self.neck(x)
        return x


@TASKERS.register_module()
class MIRegressor(Regressor):
    def __init__(self,
                 aux_backbone,
                 aux_neck=None,
                 **kwargs
                 ):
        super(MIRegressor, self).__init__(**kwargs)
        self.aux_backbone = build_backbone(aux_backbone)
        if aux_neck is not None:
            self.aux_neck = build_neck(aux_neck)

    @property
    def with_aux_neck(self):
        return hasattr(self, 'aux_neck') and self.aux_neck is not None

    def extract_feat(self, input, aux_input, **kwargs):
        feats = super().extract_feat(input)
        aux_feats = self.aux_backbone(aux_input)
        feats = torch.cat([feats, aux_feats], dim=1)
        # if self.with_aux_neck:
        #     aux_feats = self.aux_neck(aux_feats)
        return feats
