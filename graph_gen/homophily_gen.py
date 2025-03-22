from typing import List, Tuple

import numpy as np
import torch

from sklearn.preprocessing import normalize

import dgl
from dgl import DGLGraph

from utils import Util


class HomophilyGen:
    @classmethod
    def __fill(
        cls, hg: DGLGraph, n_features: int, train_split: float, val_split: float
    ):
        for ntype in hg.ntypes:
            num_nodes = hg.num_nodes(ntype)
            hg.nodes[ntype].data["feat"] = torch.randn(num_nodes, n_features)

            train_mask = torch.bernoulli(torch.ones(num_nodes) * train_split)
            inv_train_mask = 1 - train_mask

            val_mask = torch.bernoulli(inv_train_mask * val_split)
            test_mask = inv_train_mask - val_mask

            hg.nodes[ntype].data["train_mask"] = train_mask.to(torch.bool)
            hg.nodes[ntype].data["val_mask"] = train_mask.to(torch.bool)
            hg.nodes[ntype].data["test_mask"] = test_mask.to(torch.bool)

        return hg

    @classmethod
    def generate(
        cls,
        num_target_nodes: int,
        num_classes: int,
        metapaths: List[List[Tuple[str, str, str]]],
        target_node_type: str,
        homophily: float,
        max_neighbours_per_edge_type: int = 3,
        num_features=5,
        train_split=0.6,
        val_split=0.5,
    ):
        # Get compatibility matrix with given homophily
        H = np.random.rand(num_classes, num_classes)
        np.fill_diagonal(H, 0)
        H = (1 - homophily) * normalize(H, axis=1, norm="l1")
        np.fill_diagonal(H, homophily)
        np.testing.assert_allclose(
            H.sum(axis=1), np.ones(num_classes), rtol=1e-5, atol=0
        )

        edges = []
        for metapath in metapaths:
            edges.extend(metapath)

        # print(f"edges: {edges}")
        hg: DGLGraph = dgl.heterograph({edge: ([], []) for edge in edges})

        labels = []

        # Add nodes incrementally and create edges using preferential attachment
        for u in range(num_target_nodes):

            labels.append(np.random.choice(range(num_classes)))
            hg.add_nodes(1, ntype=target_node_type)

            if u == 0:
                continue

            for metapath in metapaths:
                source_type, edge_type_1, intermediate_type = metapath[0]
                _, edge_type_2, _ = metapath[1]

                valid_neighbors = list(range(u))  # All nodes added before u

                if not valid_neighbors:
                    continue

                # Calculate scores based on in-degree and homophily
                scores = []
                for v in valid_neighbors:
                    # Get in-degree of node v for the relevant edge type
                    in_degree = hg.in_degrees(v, etype=edge_type_2) + 0.01

                    # Calculate homophily score
                    homophily_score = H[labels[u], labels[v]]

                    # Final score is product of degree and homophily
                    scores.append(in_degree * homophily_score)

                scores = np.array(scores)

                # Normalize scores
                if scores.sum() > 0:
                    scores /= scores.sum()
                else:
                    scores = np.ones(len(valid_neighbors)) / len(valid_neighbors)

                # Determine number of connections to make
                num_edges = min(max_neighbours_per_edge_type, len(valid_neighbors))

                # Sample neighbors based on scores
                selected_neighbors = np.random.choice(
                    valid_neighbors,
                    size=num_edges,
                    replace=False,
                    p=scores,
                )

                # Create connections for each selected neighbor
                for v in selected_neighbors:
                    if u == v:
                        continue

                    # Create an intermediate node for this metapath
                    hg.add_nodes(1, ntype=intermediate_type)
                    intermediate_id = hg.num_nodes(ntype=intermediate_type) - 1

                    # Create the metapath edges
                    # First hop: u -> intermediate
                    hg.add_edges(u, intermediate_id, etype=edge_type_1)

                    # Second hop: intermediate -> v
                    hg.add_edges(intermediate_id, v, etype=edge_type_2)

                    # Create reverse edges if needed
                    if len(metapath) > 1:
                        reverse_edge_1 = metapath[1][
                            1
                        ]  # The edge type from intermediate back to u
                        reverse_edge_2 = metapath[0][
                            1
                        ]  # The edge type from v back to intermediate

                        # Add reverse edges
                        hg.add_edges(intermediate_id, u, etype=reverse_edge_1)
                        hg.add_edges(v, intermediate_id, etype=reverse_edge_2)



        print(f"labels: {labels}")
        hg.nodes[target_node_type].data["label"] = torch.tensor(
            np.array(labels), dtype=torch.long
        )
        hg = cls.__fill(hg, num_features, train_split, val_split)
        return hg
