# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from drugood.models.losses import Accuracy
from .base_head import BaseHead
from ..builder import HEADS, build_losses
from ..utils import is_tracing


@HEADS.register_module()
class ClsHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use Mixup/CutMix or something like that during training,
            it is not reasonable to calculate accuracy. Defaults to False.
    """

    def __init__(self,
                 loss=None,
                 topk=(1,),
                 cal_acc=False,
                 init_cfg=None):
        if (loss is None):
            loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
        super(ClsHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(loss, (dict, list))
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk,)
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk
        self.losses = build_losses(loss)
        self.compute_accuracy = Accuracy(topk=self.topk)
        self.cal_acc = cal_acc

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        for _loss in self.losses:
            name = _loss.__class__.__name__.replace("Loss", "_loss").lower()
            losses[name] = _loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        return losses

    def forward_test(self, cls_score):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        return pred

    def forward_train(self, cls_score, gt_label):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        losses = self.loss(cls_score, gt_label)
        return losses

    def simple_test(self, cls_score):
        if isinstance(cls_score, tuple):
            cls_score = cls_score[-1]
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        return self.post_process(pred)

    def post_process(self, pred):
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred
