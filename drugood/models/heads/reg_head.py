# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import torch
import torch.nn as nn

from drugood.models.losses import Error
from .base_head import BaseHead
from ..builder import HEADS, build_losses
from ..utils import is_tracing


@HEADS.register_module()
class RegHead(BaseHead):
    def __init__(self,
                 loss=None,
                 cal_metric=True,
                 init_cfg=None):
        if (loss is None):
            loss = dict(type='MeanSquaredLoss', loss_weight=1.0)
        super(RegHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, (dict, list))

        self.losses = build_losses(loss)
        self.compute_metric = Error(metric="mae")
        self.cal_metric = cal_metric

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        for _loss in self.losses:
            losses[_loss.__class__.__name__] = _loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_metric:
            # compute error
            err = self.compute_metric(cls_score, gt_label)
            losses['mae'] = err
        return losses

    def forward_test(self, score):
        if isinstance(score, tuple):
            score = score[-1]
        if isinstance(score, list):
            score = sum(score) / float(len(score))
        return score

    def forward_train(self, score, gt_label):
        if isinstance(score, tuple):
            score = score[-1]
        losses = self.loss(score, gt_label)
        return losses

    def simple_test(self, score):
        """Test without augmentation."""
        if isinstance(score, tuple):
            score = score[-1]
        if isinstance(score, list):
            score = sum(score) / float(len(score))
        return self.post_process(score)

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred


@HEADS.register_module()
class LinearRegHead(RegHead):
    """Linear classifier head.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 in_channels,
                 num_classes=1,
                 init_cfg=None,
                 dropout=0.0,
                 *args,
                 **kwargs):
        if (init_cfg is None):
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)
        super(LinearRegHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def simple_test(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        x = self.dropout(x)
        score = self.fc(x)
        if isinstance(score, list):
            score = sum(score) / float(len(score))
        return self.post_process(score)

    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            x = x[-1]
        score = self.fc(x)
        losses = self.loss(score, gt_label)
        return losses

    def forward_test(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        logits = self.fc(x)
        return logits

    def post_process(self, pred):
        pred = list(pred.detach().cpu().numpy())
        return pred
