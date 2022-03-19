# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch_geometric
import transformers
from dgl import DGLGraph
from mmcv.runner import BaseModule


class BaseAlgorithm(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(BaseAlgorithm, self).__init__(init_cfg)

    @abstractmethod
    def forward_train(self, input, group, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def simple_test(self, input, group, **kwargs):
        """Placeholder for single case test."""
        pass

    def forward_test(self, input, group, **kwargs):
        return self.simple_test(input, group, **kwargs)

    def forward(self, input, group, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(input, group, **kwargs)
        else:
            return self.forward_test(input, group, **kwargs)

    def train_step(self, data_batch, optimizer):
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=self.get_batch_num(data_batch))

        return outputs

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def get_batch_num(self, batch):
        if isinstance(batch["input"], torch.Tensor):
            return len(batch["input"].data)
        elif isinstance(batch["input"], torch_geometric.data.Data):
            return batch["input"].num_graphs
        elif isinstance(batch['input'], DGLGraph):
            return batch['input'].batch_size
        elif isinstance(batch['input'], transformers.BatchEncoding):
            return len(batch['input'])
        else:
            raise NotImplementedError
