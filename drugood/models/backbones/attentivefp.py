# The model implementation is adopted from the dgllife library
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
from dgllife.model import AttentiveFPGNN as AttentiveFPGNN_DGL
from dgllife.model import AttentiveFPReadout

from ..builder import BACKBONES
from ...core import move_to_device


@BACKBONES.register_module()
class AttentiveFPGNN(AttentiveFPGNN_DGL):
    def __init__(self, num_timesteps, get_node_weight=False, **kwargs):
        super(AttentiveFPGNN, self).__init__(**kwargs)
        self.get_node_weight = get_node_weight
        self.readout = AttentiveFPReadout(
            num_timesteps=num_timesteps,
            feat_size=kwargs.get("graph_feat_size"),
            dropout=kwargs.get("dropout"))

    def forward(self, input):
        input = move_to_device(input)
        node_feats = input.ndata["x"]
        edge_feats = input.edata["x"]
        node_feats = super().forward(input, node_feats, edge_feats)
        if self.get_node_weight:
            graph_feats, _ = self.readout(input, node_feats, self.get_node_weight)
        else:
            graph_feats = self.readout(input, node_feats, self.get_node_weight)
        return graph_feats
