"""Heterogeneous Graph Transformer (HGT)"""

import math

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


"""
Adapted from https://github.com/ZZy979/pytorch-tutorial/blob/master/gnn/hgt/model.py
"""


class HGT(nn.Module):

    def __init__(
        self,
        in_dims,
        hidden_dim,
        out_dim,
        num_heads,
        ntypes,
        etypes,
        category,
        num_layers,
        dropout=0.4,
        use_norm=True,
    ):
        """HGT

        :param in_dims: Dict[str, int]
        :param hidden_dim: int
        :param out_dim: int
        :param num_heads: int
        :param ntypes: List[str]
        :param etypes: List[(str, str, str)]
        :param predict_ntype: str
        :param num_layers: int
        :param dropout: dropout: float,
        :param use_norm: bool,
        """
        super().__init__()
        self.category = category
        self.adapt_fcs = nn.ModuleDict(
            {ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()}
        )
        self.layers = nn.ModuleList(
            [
                HGTLayer(
                    hidden_dim, hidden_dim, num_heads, ntypes, etypes, dropout, use_norm
                )
                for _ in range(num_layers)
            ]
        )
        self.predict = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, feats):
        """
        :param g: DGLGraph
        :param feats: Dict[str, tensor(N_i, d_in)]
        :return: tensor(N_i, d_out)
        """
        hs = {ntype: F.gelu(self.adapt_fcs[ntype](feats[ntype])) for ntype in feats}
        for layer in self.layers:
            hs = layer(g, hs)  # {ntype: tensor(N_i, d_hid)}
        out = self.predict(hs[self.category])  # tensor(N_i, d_out)
        return out


class HGTAttention(nn.Module):

    def __init__(
        self, out_dim, num_heads, k_linear, q_linear, v_linear, w_att, w_msg, mu
    ):
        """HGT

        :param out_dim: int
        :param num_heads: int
        :param k_linear: nn.Linear(d_in, d_out)
        :param q_linear: nn.Linear(d_in, d_out)
        :param v_linear: nn.Linear(d_in, d_out)
        :param w_att: tensor(K, d_out/K, d_out/K)
        :param w_msg: tensor(K, d_out/K, d_out/K)
        :param mu: tensor(1)
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.k_linear = k_linear
        self.q_linear = q_linear
        self.v_linear = v_linear
        self.w_att = w_att
        self.w_msg = w_msg
        self.mu = mu

    def forward(self, g, feat):
        """
        :param g: DGLGraph
        :param feat: tensor(N_src, d_in) or (tensor(N_src, d_in), tensor(N_dst, d_in))
        :return: tensor(N_dst, d_out)
        """
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, g)
            # (N_src, d_in) -> (N_src, d_out) -> (N_src, K, d_out/K)
            k = self.k_linear(feat_src).view(-1, self.num_heads, self.d_k)
            v = self.v_linear(feat_src).view(-1, self.num_heads, self.d_k)
            q = self.q_linear(feat_dst).view(-1, self.num_heads, self.d_k)

            # k[:, h] @= w_att[h] => k[n, h, j] = ∑(i) k[n, h, i] * w_att[h, i, j]
            k = torch.einsum("nhi,hij->nhj", k, self.w_att)
            v = torch.einsum("nhi,hij->nhj", v, self.w_msg)

            g.srcdata.update({"k": k, "v": v})
            g.dstdata["q"] = q
            g.apply_edges(fn.v_dot_u("q", "k", "t"))  # g.edata['t']: (E, K, 1)
            attn = g.edata.pop("t").squeeze(dim=-1) * self.mu / math.sqrt(self.d_k)
            attn = edge_softmax(g, attn)  # (E, K)
            g.edata["t"] = attn.unsqueeze(dim=-1)  # (E, K, 1)

            g.update_all(fn.u_mul_e("v", "t", "m"), fn.sum("m", "h"))
            out = g.dstdata["h"].view(-1, self.out_dim)  # (N_dst, d_out)
            return out


class HGTLayer(nn.Module):

    def __init__(
        self, in_dim, out_dim, num_heads, ntypes, etypes, dropout=0.2, use_norm=True
    ):
        """HGT层

        :param in_dim: int
        :param out_dim: int
        :param num_heads: int
        :param ntypes: List[str]
        :param etypes: List[(str, str, str)]
        :param dropout: dropout: float,
        :param use_norm: bool
        """
        super().__init__()
        d_k = out_dim // num_heads
        k_linear = {ntype: nn.Linear(in_dim, out_dim) for ntype in ntypes}
        q_linear = {ntype: nn.Linear(in_dim, out_dim) for ntype in ntypes}
        v_linear = {ntype: nn.Linear(in_dim, out_dim) for ntype in ntypes}
        w_att = {r[1]: nn.Parameter(torch.Tensor(num_heads, d_k, d_k)) for r in etypes}
        w_msg = {r[1]: nn.Parameter(torch.Tensor(num_heads, d_k, d_k)) for r in etypes}
        mu = {r[1]: nn.Parameter(torch.ones(num_heads)) for r in etypes}
        self.reset_parameters(w_att, w_msg)
        self.conv = HeteroGraphConv(
            {
                etype: HGTAttention(
                    out_dim,
                    num_heads,
                    k_linear[stype],
                    q_linear[dtype],
                    v_linear[stype],
                    w_att[etype],
                    w_msg[etype],
                    mu[etype],
                )
                for stype, etype, dtype in etypes
            },
            "mean",
        )

        self.a_linear = nn.ModuleDict(
            {ntype: nn.Linear(out_dim, out_dim) for ntype in ntypes}
        )
        self.skip = nn.ParameterDict(
            {ntype: nn.Parameter(torch.ones(1)) for ntype in ntypes}
        )
        self.drop = nn.Dropout(dropout)

        self.use_norm = use_norm
        if use_norm:
            self.norms = nn.ModuleDict(
                {ntype: nn.LayerNorm(out_dim) for ntype in ntypes}
            )

    def reset_parameters(self, w_att, w_msg):
        for etype in w_att:
            nn.init.xavier_uniform_(w_att[etype])
            nn.init.xavier_uniform_(w_msg[etype])

    def forward(self, g, feats):
        """
        :param g: DGLGraph
        :param feats: Dict[str, tensor(N_i, d_in)]
        :return: Dict[str, tensor(N_i, d_out)]
        """
        if g.is_block:
            feats_dst = {
                ntype: feats[ntype][: g.num_dst_nodes(ntype)] for ntype in feats
            }
        else:
            feats_dst = feats
        with g.local_scope():

            hs = self.conv(g, (feats, feats))  # {ntype: tensor(N_i, d_out)}

            out_feats = {}
            for ntype in g.dsttypes:
                if g.num_dst_nodes(ntype) == 0:
                    continue
                alpha = torch.sigmoid(self.skip[ntype])
                trans_out = self.drop(self.a_linear[ntype](hs[ntype]))
                out = alpha * trans_out + (1 - alpha) * feats_dst[ntype]
                out_feats[ntype] = self.norms[ntype](out) if self.use_norm else out
            return out_feats


class RelativeTemporalEncoding(nn.Module):

    def __init__(self, dim, t_max=240):
        r"""

        .. math::
          Base(\Delta T, 2i) = \sin(\Delta T / 10000^{2i/d}) \\
          Base(\Delta T, 2i+1) = \sin(\Delta T / 10000^{2i+1/d}) \\
          RTE(\Delta T) = T-Linear(Base(\Delta T))

        :param dim: int
        :param t_max: int ΔT∈[0, t_max)
        """
        super().__init__()
        t = torch.arange(t_max).unsqueeze(1)
        # 10000^(i/d) = e^((i/d)*ln 10000)
        denominator = torch.exp(torch.arange(dim) * math.log(10000.0) / dim)  #
        self.base = t / denominator
        self.base[:, 0::2] = torch.sin(self.base[:, 0::2])
        self.base[:, 1::2] = torch.cos(self.base[:, 1::2])
        self.t_linear = nn.Linear(dim, dim)

    def forward(self, delta_t):
        return self.t_linear(self.base[delta_t])
