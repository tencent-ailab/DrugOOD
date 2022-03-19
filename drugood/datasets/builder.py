# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
import platform
import random
from distutils.version import LooseVersion
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader, WeightedRandomSampler
from wilds.common.data_loaders import GroupSampler

from .samplers import DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import (ConcatDataset, RepeatDataset,
                                   ClassBalancedDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     round_up=True,
                     seed=None,
                     pin_memory=False,
                     persistent_workers=False,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        round_up (bool): Whether to round up the length of dataset by adding
            extra samples to make it evenly divisible. Default: True.
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default: True
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, round_up=round_up)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):
        kwargs['persistent_workers'] = persistent_workers

    collect_fn = dataset._collate if hasattr(dataset, '_collate') else collate

    if dataset.sample_mode == "group":
        sam_cfg = dataset.sample_config
        assert sam_cfg is not None, "Need to set sample config when sample grouply"
        if sam_cfg.uniform_over_groups is None:
            sam_cfg.uniform_over_groups = True

        assert sam_cfg.n_groups_per_batch is not None
        dataset_n_groups = len(dataset.groups.unique())
        if sam_cfg.n_groups_per_batch > dataset_n_groups:
            raise ValueError(f'n_groups_per_batch was set to {sam_cfg.n_groups_per_batch}'
                             f' but there are only {dataset_n_groups} groups specified.')

        batch_sampler = GroupSampler(
            group_ids=dataset.groups,
            batch_size=batch_size,
            n_groups_per_batch=sam_cfg.n_groups_per_batch,
            uniform_over_groups=sam_cfg.uniform_over_groups,
            distinct_groups=sam_cfg.distinct_groups)

        data_loader = DataLoader(dataset,
                                 sampler=None,
                                 num_workers=num_workers,
                                 collate_fn=partial(collect_fn, samples_per_gpu=samples_per_gpu),
                                 pin_memory=pin_memory,
                                 shuffle=False,
                                 worker_init_fn=init_fn,
                                 batch_sampler=batch_sampler,
                                 **kwargs)

    elif dataset.sample_mode == "weight":
        max_group_idx = 1 + torch.max(dataset.groups, dim=0)[0]
        unique_groups, unique_counts = torch.unique(dataset.groups, sorted=False, return_counts=True)

        counts = torch.zeros(max_group_idx, device=dataset.groups.device)
        counts[unique_groups] = unique_counts.float()

        group_weights = 1 / counts
        weights = group_weights[dataset.groups].squeeze(1)

        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        data_loader = DataLoader(dataset,
                                 sampler=sampler,
                                 num_workers=num_workers,
                                 batch_size=batch_size,
                                 collate_fn=partial(collect_fn, samples_per_gpu=samples_per_gpu),
                                 pin_memory=pin_memory,
                                 shuffle=False,
                                 worker_init_fn=init_fn,
                                 **kwargs)

    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collect_fn, samples_per_gpu=samples_per_gpu),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
