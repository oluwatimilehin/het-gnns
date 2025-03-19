from graph_gen.simple_gen import SimpleGen

from trainers.FastGTNTrainer import FastGTNTrainer
from trainers.GATV2Trainer import GATV2Trainer
from trainers.HANTrainer import HANTrainer
from trainers.HGTTrainer import HGTTrainer
from trainers.SimpleHGNTrainer import SimpleHGNTrainer

from utils import Util, Metric
from typing import Dict


def run(
    graph, num_features, n_classes, category, meta_paths, num_epochs=100
) -> Dict[str, Metric]:
    results = {}

    results["FastGTN"] = FastGTNTrainer(
        graph,
        input_dim=num_features,
        output_dim=n_classes,
        category=category,
    ).run(num_epochs)

    results["GAT"] = GATV2Trainer(
        Util.to_homogeneous(graph, category),
        input_dim=num_features,
        output_dim=n_classes,
    ).run(num_epochs)

    results["HAN"] = HANTrainer(
        graph,
        input_dim=num_features,
        output_dim=n_classes,
        meta_paths=meta_paths,
        category=category,
    ).run(num_epochs)

    results["HGT"] = HGTTrainer(
        graph,
        input_dim=num_features,
        output_dim=n_classes,
        category=category,
    ).run(num_epochs)

    results["SimpleHGN"] = SimpleHGNTrainer(
        graph,
        input_dim=num_features,
        output_dim=n_classes,
        category=category,
    ).run(num_epochs)

    return results


if __name__ == "__main__":
    num_features = 20
    hg = SimpleGen.generate(
        n_node_types=3,
        n_het_edge_types=4,
        n_nodes_per_type=1000,
        n_edges_per_type=500,
        n_edges_across_types=200,
        n_features=num_features,
    )

    num_classes = 5
    category = "node_type0"

    meta_paths = SimpleGen.get_metapaths(hg, category)
    print(f"metapaths: {meta_paths}")

    # Fixed node importance, varying edge importance test
    fixed_node_importance_res = {}
    for i in range(0, 12, 2):
        importance = i / 10.0
        print(f"Running for homogeneous edge importance factor: {importance}")
        labelled_graph = SimpleGen.label(
            hg,
            n_classes=num_classes,
            node_feat_importance=1,
            hom_edge_importance_factor=importance,
        )

        fixed_node_importance_res[importance] = run(
            labelled_graph,
            num_features=num_features,
            n_classes=num_classes,
            category=category,
            meta_paths=meta_paths,
        )

        print(
            f"Current results for homogeneous edge importance {importance}: {fixed_node_importance_res}"
        )

    # Varying node importance, edges are equally important
    varying_node_importance_res = {}
    for i in range(0, 12, 2):
        print(f"Running for node importance: {i}")
        labelled_graph = SimpleGen.label(
            hg,
            n_classes=num_classes,
            node_feat_importance=i,
            hom_edge_importance_factor=1,
        )

        varying_node_importance_res[i] = run(
            labelled_graph,
            num_features=num_features,
            n_classes=num_classes,
            category=category,
            meta_paths=meta_paths,
        )
        print(f"Current results for node importance {i}: {varying_node_importance_res}")

    print(f"Results from fixed node importance experiment: {fixed_node_importance_res}")
    print(
        f"Results from varying node importance experiment: {varying_node_importance_res}"
    )
