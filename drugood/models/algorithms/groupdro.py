# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import torch
import torch_scatter

from drugood.models.algorithms.base import BaseAlgorithm
from ..builder import MODELS, build_tasker


@MODELS.register_module()
class GroupDRO(BaseAlgorithm):
    """
    Group distributionally robust optimization.
    Original paper:
        @inproceedings{sagawa2019distributionally,
          title={Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author={Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle={International Conference on Learning Representations},
          year={2019}
        }
    The GroupDRO implementation below is adapted from Wilds's implementation:
    https://github.com/p-lambda/wilds/blob/a7a452c80cad311cf0aabfd59af8348cba1b9861/examples/algorithms/groupDRO.py
    """

    def __init__(self,
                 tasker,
                 num_groups=44930,
                 group_dro_step_size=0.01,
                 ):
        super().__init__()
        self.tasker = build_tasker(tasker)
        self.num_groups = num_groups
        # set GroupDRO-specific variables
        self.group_weights_step_size = group_dro_step_size
        self.group_weights = torch.ones(num_groups)
        self.group_weights = self.group_weights / self.group_weights.sum()

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
        losses = self.decode(feats, gt_label, return_loss=True)
        losses = sum(_value for _key, _value in losses.items() if 'loss' in _key)
        #
        batch_idx = torch.where(~torch.isnan(gt_label))[0]
        group_idx = group[batch_idx]
        group_losses = torch_scatter.scatter(src=losses, index=group_idx, dim_size=self.num_groups,
                                             reduce='mean')
        #
        if self.group_weights.device != group_losses.device:
            self.group_weights = self.group_weights.to(device=group_losses.device)
        self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size * group_losses.data)
        self.group_weights = (self.group_weights / (self.group_weights.sum()))
        losses = {"groupdro_loss": group_losses @ self.group_weights}
        return losses

    def simple_test(self, input, group, **kwargs):
        feats = self.encode(input, group, **kwargs)
        logits = self.decode(feats)
        preds = self.tasker.head.post_process(logits)
        return preds
