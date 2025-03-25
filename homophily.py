import torch
import torch_scatter
from torch_sparse import SparseTensor

from torch_geometric.utils import degree


class HomophilyCalculator:
    @classmethod
    def get_class_adjusted_homophily(cls, label, edge_index):
        num_nodes = label.shape[0]
        num_edges = edge_index.shape[1]
        num_classes = int(label.max()) + 1

        row, col = edge_index[0], edge_index[1]
        out = torch.zeros(row.size(0), dtype=torch.float64)  # size = num_edges
        out[label[row] == label[col]] = (
            1.0  # Set 1 for edges that connect nodes of the same class
        )

        nomin = torch.zeros(num_classes, dtype=torch.float64)
        denomin = torch.zeros(num_classes, dtype=torch.float64)

        nomin.scatter_add_(
            0, label[col], out
        )  # For each class, adds up how many connections go to nodes of the same class.
        denomin.scatter_add_(
            0, label[col], out.new_ones(row.size(0))
        )  # For each class, counts the total number of connections.

        # Count nodes per class
        counts = label.bincount(minlength=num_classes)
        counts = counts.view(1, num_classes)
        proportions = counts / num_nodes

        h_adjs = []
        deg = degree(
            edge_index[1], num_nodes=num_nodes
        )  # Computes the in-degree of each node (how many edges point to it); edge_index[1] contains the destination node indices

        if float(denomin.sum()) > 0:
            # Edge homophily: proportion of edges connecting same-class nodes
            h_edge = float(nomin.sum() / denomin.sum())

            # Class-insensitive homophily: measures how much each class deviates from random mixing
            h_insensitive = torch.nan_to_num(nomin / denomin)
            h_insensitive = float(
                (h_insensitive - proportions).clamp_(min=0).sum(dim=-1)
            )
            h_insensitive /= num_classes - 1

            # Adjusted homophily: accounts for class size imbalance
            degree_sums = torch.zeros(num_classes)
            degree_sums.index_add_(
                dim=0, index=label, source=deg
            )  #  Aggregates the degrees of nodes in each class

            adjust = (degree_sums**2).sum() / float(
                num_edges**2
            )  # Calculates the expected homophily in a random graph with the same class distribution
            h_adj = (h_edge - adjust) / (1 - adjust)  # Normalize the score

            h_adjs.append(h_adj)

        return sum(h_adjs) / len(h_adjs)

    @classmethod
    def get_weighted_edge_homophily(cls, y, edge_index, edge_weight=None):
        """
        Return the weighted edge homophily, according to the weights in the provided adjacency matrix.

        """
        src, dst, edge_weight = cls.get_weighted_edges(edge_index, edge_weight)

        return (
            (y[src] == y[dst]).float().squeeze() * edge_weight
        ).sum() / edge_weight.sum()

    @classmethod
    def get_weighted_node_homophily(cls, y, edge_index, edge_weight=None):
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
