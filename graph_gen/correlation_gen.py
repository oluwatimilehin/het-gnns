from collections import Counter
from typing import List, Dict, Tuple

import dgl
from dgl import DGLGraph

import torch
from dgl import transforms as T

from utils import Util


class CorrelationGen:
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
        num_nodes_dict: Dict[str, int],
        num_edges_dict: Dict[Tuple, int],
        target_node_type: str,
        num_features=5,
        train_split=0.6,
        val_split=0.5,
    ) -> DGLGraph:
        data_dict = {}

        edge_types_to_target = []

        # Say we have num_edges_dict = {("author", "author-paper", "paper"): 10000}
        # This creates {num} edges from 'author' to 'paper'
        for etype, num in num_edges_dict.items():
            source_type = etype[0]
            dest_type = etype[2]

            src_nodes = torch.randint(
                low=0, high=num_nodes_dict[source_type], size=(num,)
            )
            dest_nodes = torch.randint(
                low=0, high=num_nodes_dict[dest_type], size=(num,)
            )
            data_dict[etype] = (src_nodes, dest_nodes)

            if dest_type == target_node_type:
                edge_types_to_target.append(etype)

        hg = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)

        transform = T.Compose([T.ToSimple(), T.AddReverse()])
        hg = transform(hg)

        hg = cls.__fill(hg, num_features, train_split, val_split)
        return hg

    @classmethod
    def label(
        cls,
        hg: DGLGraph,
        target_node_type: str,
        num_classes: int,
        correlation: float,
    ) -> DGLGraph:
        num_target_nodes = hg.num_nodes(target_node_type)
        hg.nodes[target_node_type].data["label"] = torch.zeros(
            num_target_nodes, dtype=torch.long
        )

        return hg
