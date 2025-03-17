from dgl.heterograph import DGLGraph
from collections import Counter
from typing import List, Tuple


class Util:
    @staticmethod
    def compute_homogeneity(
        g: DGLGraph, target_node_type: str, edge_types: List[Tuple[str, str]]
    ):
        all_labels = g.nodes[target_node_type].data["label"]

        total_max_freq = 0
        total_count = 0

        for src_type, edge in edge_types:
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
