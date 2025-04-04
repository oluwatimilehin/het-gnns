import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import EdgeWeightNorm
import dgl.function as fn


def transform_relation_graph_list(hg, category, identity=True):
    r"""
    extract subgraph :math:`G_i` from :math:`G` in which
    only edges whose type :math:`R_i` belongs to :math:`\mathcal{R}`

    Parameters
    ----------
        hg : dgl.heterograph
            Input heterogeneous graph
        category : string
            Type of predicted nodes.
        identity : bool
            If True, the identity matrix will be added to relation matrix set.
    """

    # get target category id
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    g = dgl.to_homogeneous(hg, ndata="h")
    # find out the target node ids in g
    loc = (g.ndata[dgl.NTYPE] == category_id).to("cpu")
    category_idx = torch.arange(g.num_nodes())[loc]

    edges = g.edges()
    etype = g.edata[dgl.ETYPE]
    ctx = g.device
    # g.edata['w'] = torch.ones(g.num_edges(), device=ctx)
    num_edge_type = torch.max(etype).item()

    # norm = EdgeWeightNorm(norm='right')
    # edata = norm(g.add_self_loop(), torch.ones(g.num_edges() + g.num_nodes(), device=ctx))
    graph_list = []
    for i in range(num_edge_type + 1):
        e_ids = torch.nonzero(etype == i).squeeze(-1)
        sg = dgl.graph((edges[0][e_ids], edges[1][e_ids]), num_nodes=g.num_nodes())
        # sg.edata['w'] = edata[e_ids]
        sg.edata["w"] = torch.ones(sg.num_edges(), device=ctx)
        graph_list.append(sg)
    if identity == True:
        x = torch.arange(0, g.num_nodes(), device=ctx)
        sg = dgl.graph((x, x))
        # sg.edata['w'] = edata[g.num_edges():]
        sg.edata["w"] = torch.ones(g.num_nodes(), device=ctx)
        graph_list.append(sg)
    return graph_list, g.ndata["h"], category_idx


class FastGTN(nn.Module):
    r"""
    Adapted from https://github.com/BUPT-GAMMA/OpenHGNN/tree/main

    FastGTN from paper `Graph Transformer Networks: Learning Meta-path Graphs to Improve GNNs
    <https://arxiv.org/abs/2106.06218>`__.
    It is the extension paper  of GTN.
    `Code from author <https://github.com/seongjunyun/Graph_Transformer_Networks>`__.

    Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\mathcal{R}`.Then we extract
    the single relation adjacency matrix list. In that, we can generate combination adjacency matrix by conv
    the single relation adjacency matrix list. We can generate :math:'l-length' meta-path adjacency matrix
    by multiplying combination adjacency matrix. Then we can generate node representation using a GCN layer.

    Parameters
    ----------
    num_edge_type : int
        Number of relations.
    num_channels : int
        Number of conv channels.
    in_dim : int
        The dimension of input feature.
    hidden_dim : int
        The dimension of hidden layer.
    num_class : int
        Number of classification type.
    num_layers : int
        Length of hybrid metapatorch.
    category : string
        Type of predicted nodes.
    norm : bool
        If True, the adjacency matrix will be normalized.
    identity : bool
        If True, the identity matrix will be added to relation matrix set.

    """

    def __init__(
        self,
        num_edge_type,
        num_channels,
        in_dim,
        hidden_dim,
        num_class,
        num_layers,
        category,
        norm,
        identity,
    ):
        super(FastGTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.identity = identity

        layers = []
        for i in range(num_layers):
            layers.append(GTConv(num_edge_type, num_channels))
        self.params = nn.ParameterList()
        for i in range(num_channels):
            self.params.append(nn.Parameter(torch.Tensor(in_dim, hidden_dim)))
        self.layers = nn.ModuleList(layers)
        self.gcn = GCNConv()
        self.norm = EdgeWeightNorm(norm="right")
        self.linear1 = nn.Linear(self.hidden_dim * self.num_channels, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_class)
        self.category = category
        self.category_idx = None
        self.A = None
        self.h = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.params is not None:
            for para in self.params:
                nn.init.xavier_uniform_(para)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            g = dgl.remove_self_loop(g)
            g.edata["w_sum"] = self.norm(g, g.edata["w_sum"])
            norm_H.append(g)
        return norm_H

    def forward(self, hg, h_dict):
        with hg.local_scope():
            max_feat_size = max(h.shape[1] for h in h_dict.values())

            # Pad node features to the same size
            for ntype in h_dict:
                if h_dict[ntype].shape[1] < max_feat_size:
                    pad_size = max_feat_size - h_dict[ntype].shape[1]
                    h_dict[ntype] = torch.cat(
                        [
                            h_dict[ntype],
                            torch.zeros(
                                h_dict[ntype].shape[0],
                                pad_size,
                                device=h_dict[ntype].device,
                            ),
                        ],
                        dim=1,
                    )

            hg.ndata["h"] = h_dict
            # * =============== Extract edges in original graph ================
            if self.category_idx is None:
                self.A, h_dict, self.category_idx = transform_relation_graph_list(
                    hg, category=self.category, identity=self.identity
                )
            else:
                g = dgl.to_homogeneous(hg, ndata="h")
                h_dict = g.ndata["h"]
            # X_ = self.gcn(g, self.h)
            A = self.A
            # * =============== Get new graph structure ================
            H = []
            for n_c in range(self.num_channels):
                H.append(torch.matmul(h_dict, self.params[n_c]))
            for i in range(self.num_layers):
                hat_A = self.layers[i](A)
                for n_c in range(self.num_channels):
                    edge_weight = self.norm(hat_A[n_c], hat_A[n_c].edata["w_sum"])
                    H[n_c] = self.gcn(hat_A[n_c], H[n_c], edge_weight=edge_weight)
            X_ = self.linear1(torch.cat(H, dim=1))
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return {self.category: y[self.category_idx]}


class GCNConv(nn.Module):
    def __init__(
        self,
    ):
        super(GCNConv, self).__init__()

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

                graph.srcdata["h"] = feat
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
        return rst


class GTConv(nn.Module):
    r"""
    We conv each sub adjacency matrix :math:`A_{R_{i}}` to a combination adjacency matrix :math:`A_{1}`:

    .. math::
        A_{1} = conv\left(A ; W_{c}\right)=\sum_{R_{i} \in R} w_{R_{i}} A_{R_{i}}

    where :math:`R_i \subseteq \mathcal{R}` and :math:`W_{c}` is the weight of each relation matrix
    """

    def __init__(self, in_channels, out_channels, softmax_flag=True):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.softmax_flag = softmax_flag
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, A):
        if self.softmax_flag:
            Filter = F.softmax(self.weight, dim=1)
        else:
            Filter = self.weight
        num_channels = Filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, g in enumerate(A):
                A[j].edata["w_sum"] = g.edata["w"] * Filter[i][j]
            sum_g = dgl.adj_sum_graph(A, "w_sum")
            results.append(sum_g)
        return results
