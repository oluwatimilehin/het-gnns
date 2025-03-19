from utils import Util, Metric
from typing import Dict, List
from dgl.data import DGLDataset

from graph_gen.simple_gen import SimpleGen
from graph_gen.correlation_gen import CorrelationGen

from data.acm import ACMDataset
from data.imdb import IMDbDataset
from data.dataset import (
    DBLPHeCoDataset,
    AMinerHeCoDataset,
    FreebaseHeCoDataset,
)

from trainers.FastGTNTrainer import FastGTNTrainer
from trainers.GATV2Trainer import GATV2Trainer
from trainers.HANTrainer import HANTrainer
from trainers.HGTTrainer import HGTTrainer
from trainers.SimpleHGNTrainer import SimpleHGNTrainer


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


def test_simple_gen():
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
    for i in range(0, 22, 4):
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
                micro_f1=0.6086956521739131, macro_f1=0.2147, accuracy=0.6087
            ),
            "GAT": Metric(
                micro_f1=0.5265700483091788, macro_f1=0.3479, accuracy=0.52657
            ),
            "HAN": Metric(micro_f1=0.6570048309178744, macro_f1=0.1586, accuracy=0.657),
            "HGT": Metric(
                micro_f1=0.7874396135265701, macro_f1=0.5251, accuracy=0.78744
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7101449275362319, macro_f1=0.3731, accuracy=0.71014
            ),
        },
        0.4: {
            "FastGTN": Metric(
                micro_f1=0.6570048309178744, macro_f1=0.3731, accuracy=0.657
            ),
            "GAT": Metric(
                micro_f1=0.5652173913043478, macro_f1=0.3867, accuracy=0.56522
            ),
            "HAN": Metric(
                micro_f1=0.5893719806763285, macro_f1=0.1483, accuracy=0.58937
            ),
            "HGT": Metric(
                micro_f1=0.8115942028985508, macro_f1=0.665, accuracy=0.81159
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7004830917874396, macro_f1=0.4777, accuracy=0.70048
            ),
        },
        0.8: {
            "FastGTN": Metric(
                micro_f1=0.6183574879227053, macro_f1=0.2779, accuracy=0.61836
            ),
            "GAT": Metric(
                micro_f1=0.5362318840579711, macro_f1=0.3153, accuracy=0.53623
            ),
            "HAN": Metric(
                micro_f1=0.6328502415458938, macro_f1=0.155, accuracy=0.63285
            ),
            "HGT": Metric(
                micro_f1=0.821256038647343, macro_f1=0.6085, accuracy=0.82126
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7536231884057971, macro_f1=0.4771, accuracy=0.75362
            ),
        },
        1.2: {
            "FastGTN": Metric(
                micro_f1=0.6280193236714976, macro_f1=0.308, accuracy=0.62802
            ),
            "GAT": Metric(
                micro_f1=0.4782608695652174, macro_f1=0.3308, accuracy=0.47826
            ),
            "HAN": Metric(
                micro_f1=0.6376811594202898, macro_f1=0.1558, accuracy=0.63768
            ),
            "HGT": Metric(
                micro_f1=0.7777777777777778, macro_f1=0.6219, accuracy=0.77778
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7053140096618358, macro_f1=0.4609, accuracy=0.70531
            ),
        },
        1.6: {
            "FastGTN": Metric(
                micro_f1=0.5942028985507246, macro_f1=0.2623, accuracy=0.5942
            ),
            "GAT": Metric(
                micro_f1=0.48792270531400966, macro_f1=0.3119, accuracy=0.48792
            ),
            "HAN": Metric(
                micro_f1=0.5797101449275363, macro_f1=0.1468, accuracy=0.57971
            ),
            "HGT": Metric(
                micro_f1=0.7391304347826086, macro_f1=0.557, accuracy=0.73913
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7342995169082126, macro_f1=0.5748, accuracy=0.7343
            ),
        },
        2.0: {
            "FastGTN": Metric(
                micro_f1=0.6328502415458938, macro_f1=0.3302, accuracy=0.63285
            ),
            "GAT": Metric(
                micro_f1=0.5314009661835749, macro_f1=0.3356, accuracy=0.5314
            ),
            "HAN": Metric(
                micro_f1=0.5797101449275363, macro_f1=0.1468, accuracy=0.57971
            ),
            "HGT": Metric(
                micro_f1=0.8405797101449275, macro_f1=0.6986, accuracy=0.84058
            ),
            "SimpleHGN": Metric(
                micro_f1=0.7536231884057971, macro_f1=0.5547, accuracy=0.75362
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


def test_correlator():
    num_user_classes = 3
    num_features = 5
    num_edges_dict = {
        ("user", "follow", "user"): 3000,
        ("user", "click", "item"): 5000,
        ("user", "dislike", "item"): 500,
    }
    num_nodes_dict = {"user": 1000, "item": 500}

    category = "user"
    meta_paths = [["dislike", "rev_dislike"], ["click", "rev_click"]]

    hg = CorrelationGen.generate(
        num_nodes_dict,
        num_edges_dict,
        category,
        num_user_classes,
        num_features=num_features,
        correlation=0.5,
    )

    correlation = 0.5
    labelled_hg = CorrelationGen.label(hg, category, num_user_classes, correlation)

    print(f"Correlation score: {Util.compute_correlation(labelled_hg, category)}")
    results = run(labelled_hg, num_features, num_user_classes, category, meta_paths)

    print(f"results: {results}")


def test_standard_datasets():
    datasets: List[DGLDataset] = [
        DBLPHeCoDataset,
        ACMDataset,
        FreebaseHeCoDataset,
        AMinerHeCoDataset,
        IMDbDataset,
    ]

    for dataset in datasets:
        data: DGLDataset = dataset()
        hg = data[0]
        category = data.predict_ntype

        print(f"Running {data.name} with correlation score: {data.correlation_score()}")
        num_features = hg.nodes[category].data["feat"].shape[1]
        results = run(
            hg, num_features, data.num_classes, category, data.metapaths, num_epochs=3
        )

        print(f"Results: {results}")


if __name__ == "__main__":

    test_standard_datasets()
    # test_simple_gen()
