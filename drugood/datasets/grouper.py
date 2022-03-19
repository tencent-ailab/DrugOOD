# This code is modified based on
# https://github.com/p-lambda/wilds/blob/a7a452c80cad311cf0aabfd59af8348cba1b9861/wilds/common/grouper.py
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

import warnings

import numpy as np
import torch
from wilds.common.utils import get_counts


class Grouper:
    """
    Groupers group data points together based on their metadata.
    They are used for training and evaluation,
    e.g., to measure the accuracies of different groups of data.
    """

    def __init__(self):
        raise NotImplementedError

    @property
    def n_groups(self):
        """
        The number of groups defined by this Grouper.
        """
        return self._n_groups

    def metadata_to_group(self, metadata, return_counts=False):
        """
        Args:
            - metadata (Tensor): An n x d matrix containing d metadata fields
                                 for n different points.
            - return_counts (bool): If True, return group counts as well.
        Output:
            - group (Tensor): An n-length vector of groups.
            - group_counts (Tensor): Optional, depending on return_counts.
                                     An n_group-length vector of integers containing the
                                     numbers of data points in each group in the metadata.
        """
        raise NotImplementedError

    def group_str(self, group):
        """
        Args:
            - group (int): A single integer representing a group.
        Output:
            - group_str (str): A string containing the pretty name of that group.
        """
        raise NotImplementedError

    def group_field_str(self, group):
        """
        Args:
            - group (int): A single integer representing a group.
        Output:
            - group_str (str): A string containing the name of that group.
        """
        raise NotImplementedError


class CombinatorialGrouper(Grouper):
    def __init__(self, dataset):
        grouped_metadata = dataset.groups
        if not isinstance(grouped_metadata, torch.LongTensor):
            grouped_metadata_long = grouped_metadata.long()
            if not torch.all(grouped_metadata == grouped_metadata_long):
                warnings.warn(f'CombinatorialGrouper: converting metadata into long')
            grouped_metadata = grouped_metadata_long

        self.cardinality = 1 + torch.max(
            grouped_metadata, dim=0)[0]
        cumprod = torch.cumprod(self.cardinality, dim=0)
        self._n_groups = cumprod[-1].item()
        self.factors_np = np.concatenate(([1], cumprod[:-1]))
        self.factors = torch.from_numpy(self.factors_np)

    def metadata_to_group(self, metadata, return_counts=False):
        groups = metadata[:, ].long() @ self.factors
        if return_counts:
            group_counts = get_counts(groups, self._n_groups)
            return groups, group_counts
        else:
            return groups

    def group_str(self, group):
        NotImplemented

    def group_field_str(self, group):
        return self.group_str(group).replace('=', ':').replace(',', '_').replace(' ', '')
