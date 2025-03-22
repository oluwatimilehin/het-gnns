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

        for metapath in metapaths:
            source_type, etype_1, dest_type = metapath[0]
            etype_2 = metapath[1][1]

            for u in range(num_target_nodes):
                if u >= len(labels):
                    labels.append(np.random.choice(range(num_classes)))
                    hg.add_nodes(1, ntype=source_type)

                # Get probabilities for neighbors, proportional to in_degree and compatibility
                valid_neighbours = [v for v in hg.nodes(source_type) if v.item() <= u]
                scores = np.array(
                    [
                        (hg.in_degrees(v.item(), etype=etype_2) + 0.01)
                        * H[labels[u], labels[v.item()]]
                        for v in valid_neighbours
                    ]
                )
                # print(f"scores length: {len(scores)}; scores: {scores}")
                scores /= scores.sum()

                num_edges = (
                    max_neighbours_per_edge_type
                    if max_neighbours_per_edge_type <= len(valid_neighbours)
                    else len(valid_neighbours)
                )

                vs = np.random.choice(
                    valid_neighbours,
                    size=num_edges,
                    replace=False,
                    p=scores,
                )

                # print(f"vs: {vs} for {u} when processing {metapath}")
                for v in vs:
                    if u == v:
                        continue

                    # print(f"Adding edge between {u} and {v}")

                    hg.add_nodes(1, ntype=dest_type)

                    dest_node_id = hg.num_nodes(ntype=dest_type) - 1

                    # Create the metapath (and reverse edges)
                    hg.add_edges(u, dest_node_id, etype=etype_1)
                    hg.add_edges(dest_node_id, u, etype=etype_2)

                    # print(f"edges for {etype_1}: {hg.edges(etype=etype_1)}")
                    # print(f"edges for {etype_2}: {hg.edges(etype=etype_2)}")

                    hg.add_edges(dest_node_id, v, etype=etype_2)
                    hg.add_edges(v, dest_node_id, etype=etype_1)

                    # print(f"edges for {etype_1}: {hg.edges(etype=etype_1)}")
                    # print(f"edges for {etype_2}: {hg.edges(etype=etype_2)}")

        print(f"labels: {labels}")
        hg.nodes[target_node_type].data["label"] = torch.tensor(
            np.array(labels), dtype=torch.long
        )
        hg = cls.__fill(hg, num_features, train_split, val_split)
        return hg
