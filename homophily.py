import torch
import torch_scatter
from torch_sparse import SparseTensor


# Adapted from https://github.com/emalgorithm/directed-graph-neural-network/blob/main/src/homophily.py
class HomophilyCalculator:
    @classmethod
    def get_edge_homophily(cls, y, edge_index, edge_weight=None):
        """
        Return the weighted edge homophily, according to the weights in the provided adjacency matrix.
        """
        src, dst, edge_weight = cls.get_weighted_edges(edge_index, edge_weight)

        return (
            (y[src] == y[dst]).float().squeeze() * edge_weight
        ).sum() / edge_weight.sum()

    @classmethod
    def get_node_homophily(cls, y, edge_index, edge_weight=None):
        """
        Return the weighted node homophily, according to the weights in the provided adjacency matrix.
        """
        src, dst, edge_weight = cls.get_weighted_edges(edge_index, edge_weight)

        index = src
        mask = (y[src] == y[dst]).float().squeeze() * edge_weight
        per_node_masked_sum = torch_scatter.scatter_sum(mask, index)
        per_node_total_sum = torch_scatter.scatter_sum(edge_weight, index)

        non_zero_mask = per_node_total_sum != 0
        return (
            per_node_masked_sum[non_zero_mask] / per_node_total_sum[non_zero_mask]
        ).mean()

    @classmethod
    def get_compatibility_matrix(cls, y, edge_index, edge_weight=None):
        """
        Return the weighted compatibility matrix, according to the weights in the provided adjacency matrix.
        """
        src, dst, edge_weight = cls.get_weighted_edges(edge_index, edge_weight)

        num_classes = torch.unique(y).shape[0]
        H = torch.zeros((num_classes, num_classes))
        for i in range(src.shape[0]):
            y_src = y[src[i]]
            y_dst = y[dst[i]]
            H[y_src, y_dst] += edge_weight[i]

        return torch.nn.functional.normalize(H, p=1)

    @classmethod
    def get_weighted_edges(cls, edge_index, edge_weight=None):
        """
        Return (src, dst, edge_weight) tuple.
        """
        if isinstance(edge_index, SparseTensor):
            src, dst, edge_weight = edge_index.coo()
        else:
            src, dst = edge_index
            edge_weight = (
                edge_weight
                if edge_weight is not None
                else torch.ones((edge_index.size(1),), device=edge_index.device)
            )

        return src, dst, edge_weight
