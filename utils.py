import datetime

from collections import Counter, namedtuple
from typing import List, Tuple

import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import degree, from_dgl

import dgl
from dgl.heterograph import DGLGraph

from sklearn.metrics import f1_score

from homophily import HomophilyCalculator


class EarlyStopping:
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            dt.date(), dt.hour, dt.minute, dt.second
        )
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
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename, weights_only=False))


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
    def compute_homophily(
        cls,
        hg: DGLGraph,
        category: str,
        metapaths: List[List[Tuple[str, str, str]]] = [],
    ):
        """
        Determines homophily for a heterogeneous graph
        Code adapted from https://github.com/junhongmit/H2GB/blob/main/H2GB/calcHomophily.py
        and https://github.com/emalgorithm/directed-graph-neural-network/blob/main/src/homophily.py
        """
        data = from_dgl(hg)

        results = []

        def extract_metapath(edge_types, cur, metapath, hop, task_entity=None):
            if hop < 1:
                if task_entity is None:
                    results.append(metapath)
                elif cur == task_entity:
                    results.append(metapath)
                return
            for edge_type in edge_types:
                src, _, dst = edge_type
                # if src != dst and src == cur:
                if src == cur:
                    extract_metapath(
                        edge_types, dst, metapath + [edge_type], hop - 1, task_entity
                    )
            return results

        # Sample metapaths: [[('author', 'ap', 'paper'), ('paper', 'pa', 'author')]]
        if not metapaths:
            metapaths = extract_metapath(data.edge_types, category, [], 2, category)
        # print(f"metapaths: {metapaths}")

        label = data[category]["label"]
        device = label.device

        weighted_node_homs = []
        weighted_edge_homs = []
        class_adjusted_edge_homs = []

        for metapath in metapaths:
            src, rel, dst = metapath[0]
            m = data.num_nodes_dict[src]
            n = data.num_nodes_dict[dst]
            k = data.num_nodes_dict[src]

            edge_index_1 = data[metapath[0]].edge_index
            edge_index_2 = data[metapath[1]].edge_index

            adj_1 = SparseTensor(
                row=edge_index_1[0],
                col=edge_index_1[1],
                value=None,
                sparse_sizes=(m, n),
            ).to(device)

            adj_2 = SparseTensor(
                row=edge_index_2[0],
                col=edge_index_2[1],
                value=None,
                sparse_sizes=(n, k),
            ).to(device)

            result = adj_1 @ adj_2  # represents two-hop connections
            row, col, _ = result.coo()
            edge_index = torch.stack(
                [row, col], dim=0
            )  # Creates an edge index tensor where each column represents an edge (source, destination).

            weighted_node_homs.append(
                HomophilyCalculator.get_weighted_node_homophily(label, edge_index)
            )
            weighted_edge_homs.append(
                HomophilyCalculator.get_weighted_edge_homophily(label, edge_index)
            )

            class_adjusted_edge_homs.append(
                HomophilyCalculator.get_class_adjusted_homophily(label, edge_index)
            )

        return {
            "weighted_node_homs": sum(weighted_node_homs) / len(weighted_node_homs),
            "weighted_edge_homs": sum(weighted_edge_homs) / len(weighted_edge_homs),
            "class_adjusted_edge_homs": sum(class_adjusted_edge_homs)
            / len(class_adjusted_edge_homs),
        }
