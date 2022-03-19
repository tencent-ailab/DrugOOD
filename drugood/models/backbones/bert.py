# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import torch.nn as nn
from transformers import BertModel

from ..builder import BACKBONES
from ...core import move_to_device


@BACKBONES.register_module()
class Bert(nn.Module):
    def __init__(self, model="data/berts/bert-base-uncased"):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained(model)  # gradient_checkpointing = True

    def forward(self, input):
        input = move_to_device(input)
        feats = self.model(**input)
        return feats["pooler_output"]
