"""
From https://github.com/dmlc/dgl/tree/master/examples/pytorch/han
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class HAN(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()
        self.han = HANLayer(num_meta_paths, in_size, hidden_size, num_heads, dropout)
        self.predict = nn.Linear(hidden_size * num_heads, out_size)

    def forward(self, gs, h):
        h = self.han(gs, h)  # (N, K*d_hid)
        out = self.predict(h)  # (N, d_out)
        return out


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of meta_paths
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    gs : List[DGLGraph]
        The heterogeneous graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList(
            [
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
                for _ in range(num_meta_paths)
            ]
        )

        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)

    def forward(self, gs, h):
        zp = [gat(g, h).flatten(start_dim=1) for gat, g in zip(self.gat_layers, gs)]
        zp = torch.stack(zp, dim=1)  # (N, M, K*d_out)
        z = self.semantic_attention(zp)  # (N, K*d_out)
        return z
