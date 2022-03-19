# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from .cls_head import ClsHead
from ..builder import HEADS


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        if (init_cfg is None):
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def simple_test(self, x):
        """Test without augmentation."""
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        return self.post_process(cls_score)

    def forward_train(self, x, gt_label):
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label)
        return losses

    def forward_test(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        logits = self.fc(x)
        return logits

    def post_process(self, logits):
        pred = F.softmax(logits, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return pred
