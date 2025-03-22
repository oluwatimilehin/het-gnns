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
        edge_types: List[
            Tuple[str, str, str]
        ],  # Assumes the source is always the target_node_type; we add the reverse edges in this function
        target_node_type: str,
        max_neighbours_per_edge_type: int = 5,
        homophily: float = 0.1,
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

        all_edge_types = list(edge_types)
        for src_type, etype, dest_type in edge_types:
            all_edge_types.append((dest_type, f"rev-{etype}", src_type))

        hg: DGLGraph = dgl.heterograph(
            {edge_type: ([], []) for edge_type in all_edge_types}
        )
        labels = {}

        for source_type, etype, dest_type in edge_types:
            rev_etype = f"rev-{etype}"
            print(f"computing: {etype}")
            for u in range(num_target_nodes):
                if not u in labels:
                    labels[u] = np.random.choice(range(num_classes))
                    hg.add_nodes(1, ntype=source_type)

                # Get probabilities for neighbors, proportional to in_degree and compatibility
                scores = np.array(
                    [
                        (hg.in_degrees(v.item(), etype=rev_etype) + 0.01)
                        * H[labels[u], labels[v.item()]]
                        for v in hg.nodes(source_type)
                    ]
                )
                scores /= scores.sum()

                num_edges = (
                    max_neighbours_per_edge_type
                    if max_neighbours_per_edge_type <= hg.num_nodes(source_type)
                    else hg.num_nodes(source_type)
                )

                vs = np.random.choice(
                    hg.nodes(source_type), size=num_edges, replace=False, p=scores
                )

                for v in enumerate(vs):
                    hg.add_nodes(1, ntype=dest_type)

                    if dest_type == source_type:
                        labels[hg.num_nodes(dest_type) - 1] = np.random.choice(
                            range(num_classes)
                        )

                    dest_node_id = hg.num_nodes() - 1

                    # Create the metapath (and reverse edges)
                    hg.add_edges(u, torch.tensor([dest_node_id]), etype=etype)
                    hg.add_edges(
                        torch.tensor([dest_node_id]), torch.tensor([u]), etype=rev_etype
                    )

                    hg.add_edges(torch.tensor([dest_node_id]), v, etype=rev_etype)
                    hg.add_edges(v, torch.tensor([dest_node_id]), etype=etype)

        hg.nodes[target_node_type].data["label"] = torch.tensor(
            np.array(list(labels.values())), dtype=torch.long
        )
        hg = cls.__fill(hg, num_features, train_split, val_split)
        return hg
