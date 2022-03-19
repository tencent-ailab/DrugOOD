# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
from drugood.models.algorithms.base import BaseAlgorithm
from ..builder import MODELS, build_tasker
from ..utils import Augments


@MODELS.register_module()
class MixUp(BaseAlgorithm):
    """
    Mixup
    Original paper:
        @article{zhang2017mixup,
          title={mixup: Beyond empirical risk minimization},
          author={Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David},
          journal={arXiv preprint arXiv:1710.09412},
          year={2017}
        }
    Note that we adopt the feature-level mixup strategy
    """

    def __init__(self, tasker, cfg=None):
        super().__init__()
        self.tasker = build_tasker(tasker)
        self.augment = None
        if cfg is not None:
            cfg['type'] = 'BatchMixup'
            if cfg.get('prob', None) is None:
                cfg['prob'] = 1.0
            self.augment = Augments(cfg)

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
        if self.augment is not None:
            feats, gt_label = self.augment(feats, gt_label)
        losses = self.decode(feats, gt_label, return_loss=True)
        return losses

    def simple_test(self, input, group, **kwargs):
        feats = self.encode(input, **kwargs)
        logits = self.decode(feats)
        preds = self.tasker.head.post_process(logits)
        return preds
