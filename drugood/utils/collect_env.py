# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# All Tencent Modifications are Copyright (C) THL A29 Limited.

import dgl
import dgllife
import rdkit
import torch_geometric
import torch_scatter
import torch_sparse
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import drugood


def collect_drugood_env():
    """Collect the information of the environment related with graph and drug data .

    Returns:
        dict: The environment information. The following fields are contained.
            - DGL: DGL version
            - DGL Life: DGL Life version
            - Rdkit: Rdkit version
            - Torch Geometric: Torch Geometric version
            - Torch Sparse: Torch Sparse version
            - Torch Scatter: Torch Scatter version
    """
    env_info = {}
    env_info['DGL'] = dgl.__version__
    env_info['DGL Life'] = dgllife.__version__
    env_info['Rdkit'] = rdkit.__version__
    env_info['Torch Geometric'] = torch_geometric.__version__
    env_info['Torch Sparse'] = torch_sparse.__version__
    env_info['Torch Scatter'] = torch_scatter.__version__
    return env_info


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info.update(**collect_drugood_env())
    print(drugood.__version__)
    env_info['DrugOOD'] = drugood.__version__ + '+' + get_git_hash()[:7]
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
