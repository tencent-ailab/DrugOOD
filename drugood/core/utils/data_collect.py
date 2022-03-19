# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
from collections.abc import Mapping, Sequence

import dgl
import torch
import torch.nn.functional as F
import torch.utils.data
from dgl import DGLGraph
from mmcv.parallel.data_container import DataContainer
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch


class Collater(object):
    def __init__(self, follow_batch=None, exclude_keys=None, convert_fn=lambda x: x):
        if (follow_batch is None):
            follow_batch = []
        if (exclude_keys is None):
            exclude_keys = []
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.conver_fn = convert_fn

    def collate(self, batch, samples_per_gpu=1):
        if not isinstance(batch, Sequence):
            raise TypeError(f'{batch.dtype} is not supported.')

        if isinstance(batch[0], DataContainer):
            stacked = []
            if batch[0].cpu_only:
                for i in range(0, len(batch), samples_per_gpu):
                    stacked.append(
                        [sample.data for sample in batch[i:i + samples_per_gpu]])
                return DataContainer(
                    stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
            elif batch[0].stack:
                for i in range(0, len(batch), samples_per_gpu):
                    assert isinstance(batch[i].data, torch.Tensor)

                    if batch[i].pad_dims is not None:
                        ndim = batch[i].dim()
                        assert ndim > batch[i].pad_dims
                        max_shape = [0 for _ in range(batch[i].pad_dims)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = batch[i].size(-dim)
                        for sample in batch[i:i + samples_per_gpu]:
                            for dim in range(0, ndim - batch[i].pad_dims):
                                assert batch[i].size(dim) == sample.size(dim)
                            for dim in range(1, batch[i].pad_dims + 1):
                                max_shape[dim - 1] = max(max_shape[dim - 1],
                                                         sample.size(-dim))
                        padded_samples = []
                        for sample in batch[i:i + samples_per_gpu]:
                            pad = [0 for _ in range(batch[i].pad_dims * 2)]
                            for dim in range(1, batch[i].pad_dims + 1):
                                pad[2 * dim -
                                    1] = max_shape[dim - 1] - sample.size(-dim)
                            padded_samples.append(
                                F.pad(
                                    sample.data, pad, value=sample.padding_value))
                        stacked.append(default_collate(padded_samples))
                    elif batch[i].pad_dims is None:
                        stacked.append(
                            default_collate([
                                sample.data
                                for sample in batch[i:i + samples_per_gpu]
                            ]))
                    else:
                        raise ValueError(
                            'pad_dims should be either None or integers (1-3)')

            else:
                for i in range(0, len(batch), samples_per_gpu):
                    stacked.append(
                        [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
        elif isinstance(batch[0], Sequence):
            if isinstance(batch[0], str):
                return self.conver_fn(batch)
            else:
                transposed = zip(*batch)
                return [self.collate(samples, samples_per_gpu) for samples in transposed]
        elif isinstance(batch[0], Mapping):
            return {
                key: self.collate([d[key] for d in batch], samples_per_gpu)
                for key in batch[0]
            }
        elif isinstance(batch[0], Data):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)

        elif isinstance(batch[0], DGLGraph):
            batched_graph = dgl.batch(batch)
            return batched_graph

        elif isinstance(batch[0], str):
            batched_token = self.conver_fn(batch)
            return batched_token
        else:
            return default_collate(batch)

    def __call__(self, batch, samples_per_gpu=1):
        return self.collate(batch, samples_per_gpu)
