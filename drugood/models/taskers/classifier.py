# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import warnings
from abc import ABCMeta

from mmcv.runner import BaseModule

from ..builder import TASKERS, build_backbone, build_head, build_neck


@TASKERS.register_module()
class Classifier(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 backbone,
                 aux_backbone=None,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(Classifier, self).__init__(init_cfg)

        if pretrained is not None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated \
                key, please consider using init_cfg')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        self.backbone = build_backbone(backbone)

        if aux_backbone is not None:
            self.aux_backbone = build_backbone(aux_backbone)

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

    @property
    def with_aux_backbone(self):
        return hasattr(self, 'aux_backbone') and self.aux_backbone is not None

    def extract_feat(self, input, aux_input=None, **kwargs):
        feats = self.backbone(input)

        if self.with_aux_backbone and aux_input is not None:
            feats = [feats, self.aux_backbone(aux_input)]

        if self.with_neck:
            feats = self.neck(feats)

        return feats
