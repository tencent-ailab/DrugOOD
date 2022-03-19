# Copyright (c) OpenMMLab. All rights reserved.
# Classification
from .cls_head import ClsHead
# Linear Classification
from .linear_head import LinearClsHead
# Multi-label Classification
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
# Regression and Linear Regression
from .reg_head import RegHead, LinearRegHead

__all__ = [
    'ClsHead', 'LinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead',
    'RegHead', 'LinearRegHead',
]
