# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.

import torch
from wilds.common.utils import split_into_groups

from drugood.models.algorithms.base import BaseAlgorithm
from ..builder import MODELS, build_tasker


@MODELS.register_module()
class CORAL(BaseAlgorithm):
    """
        Deep CORAL.
        This algorithm was originally proposed as an unsupervised domain adaptation algorithm.
        Original paper:
            @inproceedings{sun2016deep,
              title={Deep CORAL: Correlation alignment for deep domain adaptation},
              author={Sun, Baochen and Saenko, Kate},
              booktitle={European Conference on Computer Vision},
              pages={443--450},
              year={2016},
              organization={Springer}
            }
        The original CORAL loss is the distance between second-order statistics (covariances)
        of the source and target features.
        The CORAL implementation below is adapted from Wilds's implementation:
        https://github.com/p-lambda/wilds/blob/a7a452c80cad311cf0aabfd59af8348cba1b9861/examples/algorithms/deepCORAL.py
    """

    def __init__(self,
                 tasker,
                 coral_penalty_weight=0.1
                 ):
        super().__init__()
        self.tasker = build_tasker(tasker)
        # set IRM-specific variables
        self.coral_penalty_weight = coral_penalty_weight

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
        unique_groups, group_indices, _ = split_into_groups(group)
        coral_penalty = []
        n_groups_per_batch = unique_groups.numel()
        for i_group in range(n_groups_per_batch):
            for j_group in range(i_group + 1, n_groups_per_batch):
                coral_penalty.append(self.coral_penalty(feats[group_indices[i_group]], feats[group_indices[j_group]]))
        losses.update({"coral_loss": torch.vstack(coral_penalty) * self.coral_penalty_weight})
        return losses

    def simple_test(self, input, group, **kwargs):
        feats = self.encode(input, **kwargs)
        logits = self.decode(feats)
        preds = self.tasker.head.post_process(logits)
        return preds

    def coral_penalty(self, x, y):
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff
