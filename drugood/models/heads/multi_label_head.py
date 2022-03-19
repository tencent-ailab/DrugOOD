# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from .base_head import BaseHead
from ..builder import HEADS, build_losses
from ..utils import is_tracing


@HEADS.register_module()
class MultiLabelClsHead(BaseHead):
    """Classification head for multilabel tasks.

    Args:
        loss (dict): Config of classification loss.
    """

    def __init__(self,
                 loss=None,
                 init_cfg=None):
        if (loss is None):
            loss = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0)
        super(MultiLabelClsHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, dict)

        self.losses = build_losses(loss)

    def loss(self, cls_score, gt_label):
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()

        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        for _loss in self.losses:
            losses[_loss.__class__.__name__] = _loss(cls_score, gt_label, avg_factor=num_samples)
        return losses

    def forward_train(self, cls_score, gt_label):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        gt_label = gt_label.type_as(cls_score)
        losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        if isinstance(x, list):
            x = sum(x) / float(len(x))
        pred = F.sigmoid(x) if x is not None else None

        return self.post_process(pred)

    def post_process(self, pred):
        pred = F.sigmoid(pred)
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
