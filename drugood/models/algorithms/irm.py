# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import torch
from torch import autograd
from wilds.common.utils import split_into_groups

from drugood.models.algorithms.base import BaseAlgorithm
from ..builder import MODELS, build_tasker


@MODELS.register_module()
class IRM(BaseAlgorithm):
    """
    Invariant risk minimization.
    Original paper:
        @article{arjovsky2019invariant,
          title={Invariant risk minimization},
          author={Arjovsky, Martin and Bottou, L{\'e}on and Gulrajani, Ishaan and Lopez-Paz, David},
          journal={arXiv preprint arXiv:1907.02893},
          year={2019}
        }
    The IRM penalty function below is adapted from the code snippet
    provided in the above paper.
    """

    def __init__(self,
                 tasker,
                 irm_lambda=1,
                 irm_penalty_anneal_iters=500
                 ):
        super().__init__()
        self.tasker = build_tasker(tasker)
        # set IRM-specific variables
        self.irm_lambda = irm_lambda
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.scale = torch.nn.Parameter(torch.tensor(1.))
        self.update_count = 0

    def init_weights(self):
        pass

    def encode(self, input, group, **kwargs):
        feats = self.tasker.extract_feat(input, **kwargs)
        return feats

    def decode(self, feats, gt_label=None, return_loss=False):
        if return_loss:
            return self.tasker.head.forward_train(feats, gt_label)
        else:
            return self.tasker.head.forward_test(feats)

    def forward_train(self, input, group, gt_label, **kwargs):
        feats = self.encode(input, group, **kwargs)
        cls_score = self.tasker.head.fc(feats)
        _, group_indices, _ = split_into_groups(group)
        main_losses = []
        irm_penalty = []
        for i_group in group_indices:
            group_losses_dict = self.tasker.head.loss(self.scale * cls_score[i_group], gt_label[i_group])
            group_losses = sum(_value for _key, _value in group_losses_dict.items() if 'loss' in _key)
            if group_losses.numel() > 0:
                main_losses.append(group_losses.mean())
            irm_penalty.append(self.irm_penalty(group_losses))
        losses = {"main_loss": torch.vstack(main_losses), "irm_loss": torch.vstack(irm_penalty)}
        return losses

    def simple_test(self, input, group, **kwargs):
        feats = self.encode(input, group, **kwargs)
        logits = self.decode(feats)
        preds = self.tasker.head.post_process(logits)
        return preds

    def irm_penalty(self, losses):
        grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result
