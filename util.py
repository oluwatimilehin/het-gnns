from collections import Counter
import random

import dgl
from dgl import transforms as T
from dgl.heterograph import DGLGraph

import numpy as np

import torch


class Util:
    @staticmethod
    def generate_graph(
        num_nodes_dict,
        num_edges_dict,
        target_node_type,
        num_classes,
        num_features=5,
        homogeneity=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42,
    ) -> DGLGraph:
        data_dict = {}

        # np.random.seed(random_seed)
        # torch.manual_seed(random_seed * 32)

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

        num_target_nodes = num_nodes_dict[target_node_type]
        hg.nodes[target_node_type].data["feat"] = torch.randn(
            num_target_nodes, num_features
        )

        for node_type, count in num_nodes_dict.items():
            if node_type != target_node_type:
                hg.nodes[node_type].data["feat"] = torch.eye(count, num_features)

        # Assign labels
        initial_labels = torch.zeros(num_target_nodes, dtype=torch.long)

        # Shuffle all node indices first
        node_indices = torch.randperm(num_target_nodes)

        # Assign the dominant label (0) to the first `num_dominant` nodes
        num_dominant = int(homogeneity * num_target_nodes)
        initial_labels[node_indices[:num_dominant]] = (
            0  # Assign dominant label to first 'num_dominant' nodes
        )

        # Assign random labels to the remaining nodes
        remaining_indices = node_indices[num_dominant:]
        remaining_labels = torch.randint(1, num_classes, size=(len(remaining_indices),))
        initial_labels[remaining_indices] = remaining_labels

        # Update the graph with the labels
        hg.nodes[target_node_type].data["label"] = initial_labels

        # Create train, val and test masks
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Split ratios must sum to 1"

        indices = torch.randperm(num_target_nodes)

        num_train = int(train_ratio * num_target_nodes)
        num_val = int(val_ratio * num_target_nodes)

        train_mask = torch.zeros(num_target_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_target_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_target_nodes, dtype=torch.bool)

        train_mask[indices[:num_train]] = True
        val_mask[indices[num_train : num_train + num_val]] = True
        test_mask[indices[num_train + num_val :]] = True

        # Add masks to the graph
        hg.nodes[target_node_type].data["train_mask"] = train_mask
        hg.nodes[target_node_type].data["val_mask"] = val_mask
        hg.nodes[target_node_type].data["test_mask"] = test_mask

        return hg

    @staticmethod
    def compute_homogeneity(g: DGLGraph, target_node_type: str) -> float:
        all_labels = g.nodes[target_node_type].data["label"]

        total_max_freq = 0
        total_count = 0

        for src_type, edge, dest_type in g.canonical_etypes:
            if dest_type != target_node_type:
                continue

            src_max_freq = 0
            src_total_count = 0

            src_nodes = g.nodes(src_type).tolist()

            for node in src_nodes:
                target_nodes = g.successors(node, (edge)).tolist()
                if len(target_nodes) < 2:
                    continue

                labels = [all_labels[item].item() for item in target_nodes]
                max_freq = Counter(labels).most_common(1)[0][1]  # Get mode frequency

                src_max_freq += max_freq
                src_total_count += len(target_nodes)

            print(f"{src_type} homogeneity: {src_max_freq / src_total_count}")
            total_max_freq += src_max_freq
            total_count += src_total_count

        return total_max_freq / total_count if total_count > 0 else 0.0

    @staticmethod
    def to_homogeneous(g: DGLGraph, target_node_type):
        node_types = g.ntypes
        type_to_id = {ntype: i for i, ntype in enumerate(node_types)}

        homogeneous_g = dgl.to_homogeneous(g)

        node_type_ids = homogeneous_g.ndata[dgl.NTYPE]
        target_type_id = type_to_id[target_node_type]
        target_mask = node_type_ids == target_type_id

        target_indices = torch.where(target_mask)[0]

        for attr in ["feat", "train_mask", "val_mask", "test_mask", "label"]:
            if attr in g.nodes[target_node_type].data:
                target_attr = g.nodes[target_node_type].data[attr]
                shape = (
                    (homogeneous_g.num_nodes(),)
                    if target_attr.dim() == 1
                    else (homogeneous_g.num_nodes(), target_attr.shape[1])
                )
                all_attr = torch.zeros(
                    shape, dtype=target_attr.dtype, device=target_attr.device
                )
                all_attr[target_indices] = target_attr
                homogeneous_g.ndata[attr] = all_attr

        homogeneous_g.ndata["node_type"] = node_type_ids

        return homogeneous_g
