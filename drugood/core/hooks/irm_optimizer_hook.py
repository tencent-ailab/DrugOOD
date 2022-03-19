# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
import collections

from mmcv import print_log
from mmcv.runner import OptimizerHook
from mmcv.runner.hooks import HOOKS

from drugood.utils import get_root_logger


@HOOKS.register_module()
class IRMOptimizerHook(OptimizerHook):
    def after_train_iter(self, runner):
        if runner.model.module.update_count == runner.model.module.irm_penalty_anneal_iters:
            print_log("Hit IRM penalty anneal iters, Re-set optimizer", logger=get_root_logger())
            self.reset_optimizer(runner)

        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()
        runner.model.module.update_count += 1

    def reset_optimizer(self, runner):
        runner.optimizer.state = collections.defaultdict(dict)  # Reset state
