from collections import Counter, namedtuple
import random


import numpy as np
import torch

import dgl
from dgl import transforms as T
from dgl.heterograph import DGLGraph

from sklearn.metrics import f1_score


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), "es_checkpoint.pt")


Metric = namedtuple("Metric", ["micro_f1", "macro_f1", "accuracy"])


class Util:
    @classmethod
    def accuracy(cls, logits, labels):
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return round(correct.item() * 1.0 / len(labels), 5)

    @classmethod
    def micro_macro_f1_score(cls, logits, labels):
        prediction = torch.argmax(logits, dim=1).long().numpy()
        labels = labels.numpy()
        micro_f1 = f1_score(labels, prediction, average="micro")
        macro_f1 = round(f1_score(labels, prediction, average="macro"), 4)
        return (micro_f1, macro_f1)

    @classmethod
    def evaluate(cls, g, model, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(g, features)
            logits = logits[mask]
            labels = labels[mask]

            f1_score = cls.micro_macro_f1_score(logits, labels)
            return Metric(f1_score[0], f1_score[1], cls.accuracy(logits, labels))

    @classmethod
    def evaluate_dict(cls, g, model, features_dict, category, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(g, features_dict)[category]
            logits = logits[mask]
            labels = labels[mask]
            f1_score = cls.micro_macro_f1_score(logits, labels)
            return Metric(
                f1_score[0],
                f1_score[1],
                cls.accuracy(logits, labels),
            )

    @classmethod
    def generate_graph(
        cls,
        num_nodes_dict,
        num_edges_dict,
        target_node_type,
        num_classes,
        num_features=5,
        correlation=1,
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

        node_indices = torch.randperm(num_target_nodes)

        num_dominant = int(correlation * num_target_nodes)
        initial_labels[node_indices[:num_dominant]] = 0

        remaining_indices = node_indices[num_dominant:]
        remaining_labels = torch.randint(1, num_classes, size=(len(remaining_indices),))
        initial_labels[remaining_indices] = remaining_labels

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

        hg.nodes[target_node_type].data["train_mask"] = train_mask
        hg.nodes[target_node_type].data["val_mask"] = val_mask
        hg.nodes[target_node_type].data["test_mask"] = test_mask

        return hg

    @classmethod
    def compute_correlation(cls, g: DGLGraph, target_node_type: str) -> float:
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

    @classmethod
    def to_homogeneous(cls, g: DGLGraph, target_node_type):
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
