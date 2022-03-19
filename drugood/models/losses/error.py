# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch
import torch.nn as nn


def error_numpy(pred, target, metric="mae"):
    if metric == "mae":
        err = np.abs((pred - target)).mean()
    elif metric == "mse":
        err = np.abs(np.square(pred - target)).mean()
    else:
        raise TypeError(f"type should be mse or mae but got {metric}")
    return err


def error_torch(pred, target, metric="mae"):
    if metric == "mae":
        err = torch.abs(pred - target).mean().data
    elif metric == "mse":
        err = (torch.square(pred - target)).mean().data
    else:
        raise TypeError(f"type should be mse or mae but got {metric}")
    return err


def error(pred, target, metric="mae"):
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        res = error_torch(pred, target, metric)
    elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
        res = error_numpy(pred, target, metric)
    else:
        raise TypeError(
            f'pred and target should both be torch.Tensor or np.ndarray, '
            f'but got {type(pred)} and {type(target)}.')

    return res


class Error(nn.Module):
    def __init__(self, metric="mae"):
        """Module to calculate the error.

        Args:
            type (str): The criterion used to calculate the
                error. Defaults to "mae".
        """
        super().__init__()
        self.metric = metric

    def forward(self, pred, target):
        """Forward function to calculate error.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
           [float]: The error.
        """
        return error(pred, target, self.metric)
