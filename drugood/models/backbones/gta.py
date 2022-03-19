# The model implementation is adopted from the dgllife library
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

import torch.nn as nn
from dgllife.model import AttentiveFPGNN as AttentiveFPGNN_DGL
from dgllife.model import AttentiveFPReadout, WeightedSumAndMax, MLPNodeReadout
from dgllife.model import GAT as GAT_DGL
from dgllife.model import GCN as GCN_DGL
from dgllife.model import MGCNGNN as MGCNGNN_DGL

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
        node_feats = super().forward(g=input, node_feats=node_feats, edge_feats=edge_feats)
        if self.get_node_weight:
            graph_feats, _ = self.readout(input, node_feats, self.get_node_weight)
        else:
            graph_feats = self.readout(input, node_feats, self.get_node_weight)
        return graph_feats


@BACKBONES.register_module()
class GAT(GAT_DGL):
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 biases=None):
        super(GAT, self).__init__(in_feats, hidden_feats, num_heads, feat_drops,
                                  attn_drops, alphas, residuals, agg_modes, activations, biases)

        if self.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        # TODO design
        # self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
        #                             n_tasks, predictor_dropout)

    def forward(self, input):
        input = move_to_device(input)
        node_feats = input.ndata["x"]
        node_feats = super().forward(g=input, feats=node_feats)
        if self.get_node_weight:
            graph_feats, _ = self.readout(input, node_feats, self.get_node_weight)
        else:
            graph_feats = self.readout(input, node_feats, self.get_node_weight)
        return graph_feats


@BACKBONES.register_module()
class GCN(GCN_DGL):
    def __init__(self,
                 in_feats,
                 hidden_feats=None,
                 gnn_norm=None,
                 activation=None,
                 residual=None,
                 batchnorm=None,
                 dropout=None):

        super(GCN, self).__init__(self, in_feats, hidden_feats, gnn_norm, activation,
                                  residual, batchnorm, dropout)

        if self.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        # TODO design
        # self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
        #                             n_tasks, predictor_dropout)

    def forward(self, input):
        input = move_to_device(input)
        node_feats = input.ndata["x"]
        node_feats = super().forward(g=input, feats=node_feats)
        if self.get_node_weight:
            graph_feats, _ = self.readout(input, node_feats, self.get_node_weight)
        else:
            graph_feats = self.readout(input, node_feats, self.get_node_weight)
        return graph_feats


@BACKBONES.register_module()
class MGCNGNN(MGCNGNN_DGL):
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
