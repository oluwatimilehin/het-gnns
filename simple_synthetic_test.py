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

    fixed_node_importance_res = {
        0.0: {
            "FastGTN": Metric(
                micro_f1=0.6888888888888889, macro_f1=0.3679, accuracy=0.68889
            ),
            "GAT": Metric(micro_f1=0.45, macro_f1=0.2905, accuracy=0.45),
            "HAN": Metric(
                micro_f1=0.6277777777777778, macro_f1=0.1543, accuracy=0.62778
            ),
            "HGT": Metric(
                micro_f1=0.8277777777777777, macro_f1=0.6722, accuracy=0.82778
            ),
            "SimpleHGN": Metric(micro_f1=0.75, macro_f1=0.4953, accuracy=0.75),
        },
        0.2: {
            "FastGTN": Metric(
                micro_f1=0.6777777777777778, macro_f1=0.258, accuracy=0.67778
            ),
            "GAT": Metric(
                micro_f1=0.5277777777777778, macro_f1=0.2978, accuracy=0.52778
            ),
            "HAN": Metric(
                micro_f1=0.6944444444444444, macro_f1=0.1639, accuracy=0.69444
            ),
            "HGT": Metric(
                micro_f1=0.8222222222222222, macro_f1=0.612, accuracy=0.82222
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7555555555555555, macro_f1=0.455, accuracy=0.75556
            ),
        },
        0.4: {
            "FastGTN": Metric(
                micro_f1=0.6888888888888889, macro_f1=0.2893, accuracy=0.68889
            ),
            "GAT": Metric(micro_f1=0.55, macro_f1=0.3504, accuracy=0.55),
            "HAN": Metric(micro_f1=0.6666666666666666, macro_f1=0.16, accuracy=0.66667),
            "HGT": Metric(
                micro_f1=0.8611111111111112, macro_f1=0.673, accuracy=0.86111
            ),
            "SimpleHGN": Metric(
                micro_f1=0.8111111111111111, macro_f1=0.5151, accuracy=0.81111
            ),
        },
        0.6: {
            "FastGTN": Metric(
                micro_f1=0.7111111111111111, macro_f1=0.3363, accuracy=0.71111
            ),
            "GAT": Metric(
                micro_f1=0.4722222222222222, macro_f1=0.2937, accuracy=0.47222
            ),
            "HAN": Metric(
                micro_f1=0.6833333333333333, macro_f1=0.1624, accuracy=0.68333
            ),
            "HGT": Metric(
                micro_f1=0.8388888888888889, macro_f1=0.6799, accuracy=0.83889
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7722222222222223, macro_f1=0.5511, accuracy=0.77222
            ),
        },
        0.8: {
            "FastGTN": Metric(
                micro_f1=0.6888888888888889, macro_f1=0.2418, accuracy=0.68889
            ),
            "GAT": Metric(micro_f1=0.55, macro_f1=0.319, accuracy=0.55),
            "HAN": Metric(
                micro_f1=0.7166666666666667, macro_f1=0.167, accuracy=0.71667
            ),
            "HGT": Metric(
                micro_f1=0.8111111111111111, macro_f1=0.5852, accuracy=0.81111
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7666666666666667, macro_f1=0.4425, accuracy=0.76667
            ),
        },
        1.0: {
            "FastGTN": Metric(
                micro_f1=0.7166666666666667, macro_f1=0.3933, accuracy=0.71667
            ),
            "GAT": Metric(
                micro_f1=0.5555555555555556, macro_f1=0.3097, accuracy=0.55556
            ),
            "HAN": Metric(
                micro_f1=0.6555555555555556, macro_f1=0.1584, accuracy=0.65556
            ),
            "HGT": Metric(
                micro_f1=0.8166666666666667, macro_f1=0.5995, accuracy=0.81667
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7666666666666667, macro_f1=0.5053, accuracy=0.76667
            ),
        },
    }
    varying_node_importance_res = {
        0: {
            "FastGTN": Metric(
                micro_f1=0.6666666666666666, macro_f1=0.3422, accuracy=0.66667
            ),
            "GAT": Metric(
                micro_f1=0.5333333333333333, macro_f1=0.3523, accuracy=0.53333
            ),
            "HAN": Metric(
                micro_f1=0.6111111111111112, macro_f1=0.1517, accuracy=0.61111
            ),
            "HGT": Metric(
                micro_f1=0.8388888888888889, macro_f1=0.699, accuracy=0.83889
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7111111111111111, macro_f1=0.4317, accuracy=0.71111
            ),
        },
        2: {
            "FastGTN": Metric(
                micro_f1=0.6722222222222223, macro_f1=0.3051, accuracy=0.67222
            ),
            "GAT": Metric(
                micro_f1=0.5055555555555555, macro_f1=0.296, accuracy=0.50556
            ),
            "HAN": Metric(
                micro_f1=0.6222222222222222, macro_f1=0.1534, accuracy=0.62222
            ),
            "HGT": Metric(
                micro_f1=0.8333333333333334, macro_f1=0.613, accuracy=0.83333
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7555555555555555, macro_f1=0.4684, accuracy=0.75556
            ),
        },
        4: {
            "FastGTN": Metric(
                micro_f1=0.6944444444444444, macro_f1=0.3848, accuracy=0.69444
            ),
            "GAT": Metric(
                micro_f1=0.48333333333333334, macro_f1=0.2979, accuracy=0.48333
            ),
            "HAN": Metric(
                micro_f1=0.6222222222222222, macro_f1=0.1534, accuracy=0.62222
            ),
            "HGT": Metric(
                micro_f1=0.8055555555555556, macro_f1=0.5965, accuracy=0.80556
            ),
            "SimpleHGN": Metric(micro_f1=0.75, macro_f1=0.4814, accuracy=0.75),
        },
        6: {
            "FastGTN": Metric(micro_f1=0.7, macro_f1=0.3859, accuracy=0.7),
            "GAT": Metric(
                micro_f1=0.5666666666666667, macro_f1=0.3305, accuracy=0.56667
            ),
            "HAN": Metric(
                micro_f1=0.6833333333333333, macro_f1=0.1624, accuracy=0.68333
            ),
            "HGT": Metric(
                micro_f1=0.8222222222222222, macro_f1=0.6497, accuracy=0.82222
            ),
            "SimpleHGN": Metric(micro_f1=0.8, macro_f1=0.5998, accuracy=0.8),
        },
        8: {
            "FastGTN": Metric(micro_f1=0.7, macro_f1=0.3956, accuracy=0.7),
            "GAT": Metric(
                micro_f1=0.4388888888888889, macro_f1=0.3128, accuracy=0.43889
            ),
            "HAN": Metric(
                micro_f1=0.6388888888888888, macro_f1=0.1559, accuracy=0.63889
            ),
            "HGT": Metric(
                micro_f1=0.8222222222222222, macro_f1=0.679, accuracy=0.82222
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7777777777777778, macro_f1=0.5109, accuracy=0.77778
            ),
        },
        10: {
            "FastGTN": Metric(
                micro_f1=0.6777777777777778, macro_f1=0.3094, accuracy=0.67778
            ),
            "GAT": Metric(micro_f1=0.5, macro_f1=0.2766, accuracy=0.5),
            "HAN": Metric(
                micro_f1=0.6555555555555556, macro_f1=0.1584, accuracy=0.65556
            ),
            "HGT": Metric(
                micro_f1=0.7944444444444444, macro_f1=0.5474, accuracy=0.79444
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7833333333333333, macro_f1=0.5175, accuracy=0.78333
            ),
        },
    }
