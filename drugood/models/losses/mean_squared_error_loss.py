# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES


def mean_squared_error(pred,
                       label,
                       weight=None,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       **kwargs):
    """Calculate the MeanSquared loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1)
        label (torch.Tensor): The gt label of the prediction (N, 1).
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        # TODO class weight may be used for solving long tail problem in regress problem
    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    if (pred.dim() == 2):
        pred = pred.squeeze()
    if (label.dim() == 2):
        label = label.squeeze()
    loss = F.mse_loss(pred, label, reduction='none')
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class MeanSquaredLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(MeanSquaredLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cls_criterion = mean_squared_error

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        cls_score = cls_score.to(torch.float32)
        label = label.to(torch.float32)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
