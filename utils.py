from collections import Counter, namedtuple

import torch

import dgl
from dgl.heterograph import DGLGraph

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


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
    def evaluate(cls, g, model, features, labels, mask) -> Metric:
        model.eval()
        with torch.no_grad():
            logits = model(g, features)
            logits = logits[mask]
            labels = labels[mask]

            f1_score = cls.micro_macro_f1_score(logits, labels)
            return Metric(f1_score[0], f1_score[1], cls.accuracy(logits, labels))

    @classmethod
    def evaluate_dict(cls, g, model, features_dict, category, labels, mask) -> Metric:
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

    @classmethod
    def split_idx(cls, samples, train_size, val_size, random_state=None):
        """Split samples into training, validation, and test sets, satisfying the following conditions (expressed as floating-point numbers):

        * 0 < train_size < 1
        * 0 < val_size < 1
        * train_size + val_size < 1

        :param samples: list/ndarray/tensor of samples
        :param train_size: int or float If it is an integer, it represents the absolute number of training samples; otherwise, it represents the proportion of training samples in the entire dataset
        :param val_size: int or float If it is an integer, it represents the absolute number of validation samples; otherwise, it represents the proportion of validation samples in the entire dataset
        :param random_state: int, optional Random seed
        :return: (train, val, test) with the same type as samples
        """
        train, val = train_test_split(
            samples, train_size=train_size, random_state=random_state
        )
        if isinstance(val_size, float):
            val_size *= len(samples) / len(val)
        val, test = train_test_split(
            val, train_size=val_size, random_state=random_state
        )
        return train, val, test
