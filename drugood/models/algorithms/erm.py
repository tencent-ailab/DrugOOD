# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
from drugood.models.algorithms.base import BaseAlgorithm
from ..builder import MODELS, build_tasker


@MODELS.register_module()
class ERM(BaseAlgorithm):
    def __init__(self, tasker):
        super().__init__()
        self.tasker = build_tasker(tasker)

    def init_weights(self):
        pass

    def encode(self, input, **kwargs):
        feats = self.tasker.extract_feat(input, **kwargs)
        return feats

    def decode(self, feats, gt_label=None, return_loss=False):
        if return_loss:
            return self.tasker.head.forward_train(feats, gt_label)
        else:
            return self.tasker.head.forward_test(feats)

    def forward_train(self, input, group, gt_label, **kwargs):
        feats = self.encode(input, **kwargs)
        losses = self.decode(feats, gt_label, return_loss=True)
        return losses

    def simple_test(self, input, group, **kwargs):
        feats = self.encode(input, **kwargs)
        logits = self.decode(feats)
        preds = self.tasker.head.post_process(logits)
        return preds
