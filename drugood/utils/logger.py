# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2021 THL A29 Limited, a Tencent company.
# All rights reserved. All Tencent Modifications are Copyright (C) THL A29 Limited.
import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    return get_logger('drugood', log_file, log_level)
