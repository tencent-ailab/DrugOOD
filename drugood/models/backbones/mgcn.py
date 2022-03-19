# The model implementation is adopted from the dgllife library
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

import torch.nn as nn
from dgllife.model import MGCNGNN, MLPNodeReadout

from ..builder import BACKBONES
from ...core import move_to_device


@BACKBONES.register_module()
class MGCN(MGCNGNN):
    def __init__(self,
                 feats=128, n_layers=3, classifier_hidden_feats=64,
                 n_tasks=1, num_node_types=100, num_edge_types=3000,
                 cutoff=5.0, gap=1.0, predictor_hidden_feats=64):

        if predictor_hidden_feats == 64 and classifier_hidden_feats != 64:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        super(MGCNGNN, self).__init__(self, feats, n_layers, num_node_types,
                                      num_edge_types, cutoff, gap)

        self.readout = MLPNodeReadout(node_feats=(n_layers + 1) * feats,
                                      hidden_feats=predictor_hidden_feats,
                                      graph_feats=n_tasks,
                                      activation=nn.Softplus(beta=1, threshold=20))

    def forward(self, input):
        input = move_to_device(input)
        node_feats = input.ndata["x"]
        node_feats = super().forward(g=input, feats=node_feats)
        if self.get_node_weight:
            graph_feats, _ = self.readout(input, node_feats, self.get_node_weight)
        else:
            graph_feats = self.readout(input, node_feats, self.get_node_weight)
        return graph_feats
