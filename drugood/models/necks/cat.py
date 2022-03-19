# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class Concatenate(nn.Module):
    def __init__(self, dim=1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, inputs):
        assert isinstance(inputs, list)
        return torch.cat(inputs, dim=self.dim)
