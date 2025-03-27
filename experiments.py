from collections import defaultdict
from typing import Dict, List

import numpy as np

from dgl.data import DGLDataset

from graph_gen.simple_gen import SimpleGen
from graph_gen.correlation_gen import CorrelationGen
from graph_gen.homophily_gen import HomophilyGen

from utils import Util, Metric

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

from torch import tensor
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def run(
    graph, num_features, n_classes, category, meta_paths, num_epochs=100
) -> Dict[str, Metric]:
    results = {}

    results["FastGTN"] = FastGTNTrainer(
        graph,
        input_dim=num_features,
        output_dim=n_classes,
        category=category,
        gpu=0,
        hidden_dim=192
    ).run(num_epochs)

    results["GAT"] = GATV2Trainer(
        Util.to_homogeneous(graph, category),
        input_dim=num_features,
        output_dim=n_classes,
        gpu=0,
        hidden_dim=192
    ).run(num_epochs)

    results["HAN"] = HANTrainer(
        graph,
        input_dim=num_features,
        output_dim=n_classes,
        meta_paths=meta_paths,
        category=category,
        gpu=0,
        hidden_dim=192
    ).run(num_epochs)

    results["HGT"] = HGTTrainer(
        graph,
        input_dim=num_features,
        output_dim=n_classes,
        category=category,
        gpu=0,
        hidden_dim=192
    ).run(num_epochs)

    results["SimpleHGN"] = SimpleHGNTrainer(
        graph,
        input_dim=num_features,
        output_dim=n_classes,
        category=category,
        gpu=0,
        hidden_dim=192
    ).run(num_epochs)

    return results


def plot_micro_macro_simple(data, x_label):
    micro_f1 = {}
    macro_f1 = {}

    for x_val, results in results.items():
        micro_f1_per_model = defaultdict(list)
        macro_f1_per_model = defaultdict(list)

        # for res in results:
            # for model, metric in res.items():
        for model, res in results.items():
            for metric in res:
                micro_f1_per_model[model].append(metric.micro_f1)
                macro_f1_per_model[model].append(metric.macro_f1)

        mean_micro_f1_per_model = {
            model: round(np.mean(values), 3)
            for model, values in micro_f1_per_model.items()
        }

        mean_macro_f1_per_model = {
            model: round(np.mean(values), 3)
            for model, values in macro_f1_per_model.items()
        }

        micro_f1[x_val] = mean_micro_f1_per_model
        macro_f1[x_val] = mean_macro_f1_per_model

    plot_simple(micro_f1, x_label=x_label, y_label="Micro-F1 Score")
    plot_simple(macro_f1, x_label=x_label, y_label="Macro-F1 Score")

def get_labeled_graph(num_features, n_node_types, n_het_edge_types, n_nodes_per_type, n_edges_per_type, n_edges_across_types, n_classes, i_node, i_hom, i_het):
    hg = SimpleGen.generate(
        n_node_types=n_node_types,
        n_het_edge_types=n_het_edge_types,
        n_nodes_per_type=n_nodes_per_type,
        n_edges_per_type=n_edges_per_type,
        n_edges_across_types=n_edges_across_types,
        n_features=num_features,
    )
    # print(f"metapaths: {meta_paths}")

    labelled_node = SimpleGen.label(
        hg,
        n_classes=n_classes,
        node_feat_importance=i_node,
        hom_edge_importance=i_hom,
        het_edge_importance=i_het,
    )

    return labelled_node

def run_simple(num_features, n_node_types, n_het_edge_types, n_nodes_per_type, n_edges_per_type, n_edges_across_types, n_classes, i_node, i_hom, i_het):
    labelled_graph = get_labeled_graph(num_features=num_features, n_classes=n_classes, n_node_types=n_node_types, n_het_edge_types=n_het_edge_types, n_nodes_per_type=n_nodes_per_type, n_edges_per_type=n_edges_per_type, n_edges_across_types=n_edges_across_types,
            i_node = i_node,
            i_hom = i_hom,
            i_het = i_het)

    category = "node_type0"
    meta_paths = SimpleGen.get_metapaths(labelled_graph, category)
    
    return run(
            labelled_graph,
            num_features=num_features,
            n_classes=n_classes,
            category=category,
            meta_paths=meta_paths,
        )


def test_simple_gen():
    def run_vary_importances(i_node, i_hom, i_het):
        n_runs = 5

        num_features = 20
        num_classes = 5
        n_node_types = 5
        n_het_edge_types = 15
        n_nodes_per_type = 1000
        n_edges_per_type = 4000
        n_edges_across_types = 4000
        
        results = []

        for _ in range(n_runs):
            results.append(run_simple(num_features=num_features, n_classes=num_classes, n_node_types=n_node_types, n_het_edge_types=n_het_edge_types, n_nodes_per_type=n_nodes_per_type, n_edges_per_type=n_edges_per_type, n_edges_across_types=n_edges_across_types,
            i_node = i_node,
            i_hom = i_hom,
            i_het = i_het))
        
        return results

    # Fixed node importance, varying edge importance test
    varying_node_importance_res = {}
    varying_hom_edge_importance_res = {}
    varying_het_edge_importance_res = {}
    # for i in range(0, 100, 2):
    for i in range(0, 22, 4):
        importance = i / 10.0
        print(f"Running for homogeneous edge importance factor: {importance}")
        
        # node_labelled_graph = get_labeled_graph(num_features=num_features, n_node_types=n_node_types, n_het_edge_types=n_het_edge_types, n_nodes_per_type=n_nodes_per_type, n_edges_per_type=n_edges_per_type, n_edges_across_types=n_edges_across_types,
        #     i_node = importance,
        #     i_hom = 1,
        #     i_het = 1)
        
        # hom_labelled_graph = get_labeled_graph(num_features=num_features, n_node_types=n_node_types, n_het_edge_types=n_het_edge_types, n_nodes_per_type=n_nodes_per_type, n_edges_per_type=n_edges_per_type, n_edges_across_types=n_edges_across_types,
        #     i_node = 1,
        #     i_hom = importance,
        #     i_het = 1)

        # het_labelled_graph = get_labeled_graph(num_features=num_features, n_node_types=n_node_types, n_het_edge_types=n_het_edge_types, n_nodes_per_type=n_nodes_per_type, n_edges_per_type=n_edges_per_type, n_edges_across_types=n_edges_across_types,
        #     i_node = 1,
        #     i_hom = 1,
        #     i_het = importance)

        # varying_node_importance_res[importance] = run(
        #     node_labelled_graph,
        #     num_features=num_features,
        #     n_classes=num_classes,
        #     category=category,
        #     meta_paths=meta_paths,
        # )

        # varying_node_importance_res[importance] = run(
        #     node_labelled_graph,
        #     num_features=num_features,
        #     n_classes=num_classes,
        #     category=category,
        #     meta_paths=meta_paths,
        # )

        varying_node_importance_res[importance] = run_vary_importances(importance, 1, 1)
        varying_hom_edge_importance_res[importance] = run_vary_importances(1, importance, 1)
        varying_het_edge_importance_res[importance] = run_vary_importances(1, importance, 1)

        # print(
        #     f"Current results for homogeneous edge importance {importance}: {fixed_node_importance_res}"
        # )
    
    print(
        f"Results for varying node importance: {varying_node_importance_res}"
    )
    print(
        f"Results for varying hom edge importance: {varying_hom_edge_importance_res}"
    )
    print(
        f"Results for varying het edge importance: {varying_het_edge_importance_res}"
    )
    
    plot_micro_macro_simple(varying_node_importance_res, "i_node")
    plot_micro_macro_simple(varying_hom_edge_importance_res, "i_hom")
    plot_micro_macro_simple(varying_het_edge_importance_res, "i_het")


def test_homophily():
    num_target_nodes = 1000
    num_classes = 5
    num_features = 5

    target_node_type = "user"

    metapaths = [
        [("user", "click", "item"), ("item", "clicked_by", "user")],
        [("user", "dislike", "food"), ("food", "disliked_by", "user")],
    ]

    num_epochs = 5
    results = {}
    for i in range(0, 12, 2):
        homophily = i / 10
        print(f"intended homophily: {homophily}")
        hg = HomophilyGen.generate(
            num_target_nodes,
            num_classes,
            metapaths,
            target_node_type,
            homophily,
            num_features=num_features,
        )

        actual_homophily = Util.compute_homophily(hg, target_node_type, metapaths)[
            "weighted_node_homs"
        ]
        print(
            f"Weighted node homophily: {Util.compute_homophily(hg, target_node_type, metapaths)}"
        )

        for j in range(num_epochs):
            print(f"Run {j + 1} for intended_homophily: {homophily}")
            hom_results = results.get(actual_homophily, [])
            hom_results.append(
                run(
                    hg,
                    num_features,
                    num_classes,
                    target_node_type,
                    metapaths,
                    num_epochs=200,
                )
            )

            results[actual_homophily] = hom_results

        print(f"Results for {actual_homophily}: {results[actual_homophily]}")

    print(f"results: {results}")

    results = {
        (tensor(0.1930), tensor(0.1460)): [
            {
                "FastGTN": Metric(
                    micro_f1=0.1683673469387755, macro_f1=0.1682, accuracy=0.16837
                ),
                "GAT": Metric(
                    micro_f1=0.1683673469387755, macro_f1=0.1471, accuracy=0.16837
                ),
                "HAN": Metric(
                    micro_f1=0.12244897959183673, macro_f1=0.0498, accuracy=0.12245
                ),
                "HGT": Metric(
                    micro_f1=0.15816326530612246, macro_f1=0.155, accuracy=0.15816
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.20408163265306123, macro_f1=0.199, accuracy=0.20408
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.15306122448979592, macro_f1=0.1554, accuracy=0.15306
                ),
                "GAT": Metric(
                    micro_f1=0.14795918367346939, macro_f1=0.1346, accuracy=0.14796
                ),
                "HAN": Metric(
                    micro_f1=0.11734693877551021, macro_f1=0.0436, accuracy=0.11735
                ),
                "HGT": Metric(
                    micro_f1=0.20408163265306123, macro_f1=0.201, accuracy=0.20408
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.18877551020408162, macro_f1=0.1716, accuracy=0.18878
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.20918367346938777, macro_f1=0.1999, accuracy=0.20918
                ),
                "GAT": Metric(
                    micro_f1=0.1377551020408163, macro_f1=0.1085, accuracy=0.13776
                ),
                "HAN": Metric(
                    micro_f1=0.1377551020408163, macro_f1=0.0664, accuracy=0.13776
                ),
                "HGT": Metric(
                    micro_f1=0.19387755102040816, macro_f1=0.1818, accuracy=0.19388
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.14285714285714285, macro_f1=0.1393, accuracy=0.14286
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.17857142857142858, macro_f1=0.1807, accuracy=0.17857
                ),
                "GAT": Metric(
                    micro_f1=0.15816326530612246, macro_f1=0.1439, accuracy=0.15816
                ),
                "HAN": Metric(
                    micro_f1=0.1326530612244898, macro_f1=0.0473, accuracy=0.13265
                ),
                "HGT": Metric(
                    micro_f1=0.20918367346938777, macro_f1=0.1919, accuracy=0.20918
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.22448979591836735, macro_f1=0.2051, accuracy=0.22449
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.1836734693877551, macro_f1=0.1858, accuracy=0.18367
                ),
                "GAT": Metric(
                    micro_f1=0.12244897959183673, macro_f1=0.1202, accuracy=0.12245
                ),
                "HAN": Metric(
                    micro_f1=0.11734693877551021, macro_f1=0.0442, accuracy=0.11735
                ),
                "HGT": Metric(
                    micro_f1=0.19387755102040816, macro_f1=0.1887, accuracy=0.19388
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.1989795918367347, macro_f1=0.1955, accuracy=0.19898
                ),
            },
        ],
        (tensor(0.3566), tensor(0.3188)): [
            {
                "FastGTN": Metric(
                    micro_f1=0.23529411764705882, macro_f1=0.2282, accuracy=0.23529
                ),
                "GAT": Metric(
                    micro_f1=0.17647058823529413, macro_f1=0.1653, accuracy=0.17647
                ),
                "HAN": Metric(
                    micro_f1=0.23039215686274508, macro_f1=0.1726, accuracy=0.23039
                ),
                "HGT": Metric(
                    micro_f1=0.2549019607843137, macro_f1=0.2522, accuracy=0.2549
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.23529411764705882, macro_f1=0.2239, accuracy=0.23529
                ),
            },
            {
                "FastGTN": Metric(micro_f1=0.25, macro_f1=0.2437, accuracy=0.25),
                "GAT": Metric(
                    micro_f1=0.17647058823529413, macro_f1=0.169, accuracy=0.17647
                ),
                "HAN": Metric(
                    micro_f1=0.24509803921568626, macro_f1=0.1725, accuracy=0.2451
                ),
                "HGT": Metric(
                    micro_f1=0.2696078431372549, macro_f1=0.2719, accuracy=0.26961
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.28431372549019607, macro_f1=0.2465, accuracy=0.28431
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.2549019607843137, macro_f1=0.2455, accuracy=0.2549
                ),
                "GAT": Metric(
                    micro_f1=0.18627450980392157, macro_f1=0.1729, accuracy=0.18627
                ),
                "HAN": Metric(
                    micro_f1=0.23039215686274508, macro_f1=0.1518, accuracy=0.23039
                ),
                "HGT": Metric(
                    micro_f1=0.27941176470588236, macro_f1=0.2732, accuracy=0.27941
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.2647058823529412, macro_f1=0.2194, accuracy=0.26471
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.22058823529411764, macro_f1=0.2194, accuracy=0.22059
                ),
                "GAT": Metric(
                    micro_f1=0.23529411764705882, macro_f1=0.2114, accuracy=0.23529
                ),
                "HAN": Metric(
                    micro_f1=0.2549019607843137, macro_f1=0.1735, accuracy=0.2549
                ),
                "HGT": Metric(
                    micro_f1=0.22549019607843138, macro_f1=0.2244, accuracy=0.22549
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.24509803921568626, macro_f1=0.2239, accuracy=0.2451
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.23039215686274508, macro_f1=0.2245, accuracy=0.23039
                ),
                "GAT": Metric(
                    micro_f1=0.21568627450980393, macro_f1=0.1869, accuracy=0.21569
                ),
                "HAN": Metric(
                    micro_f1=0.2549019607843137, macro_f1=0.1724, accuracy=0.2549
                ),
                "HGT": Metric(
                    micro_f1=0.2107843137254902, macro_f1=0.2099, accuracy=0.21078
                ),
                "SimpleHGN": Metric(micro_f1=0.25, macro_f1=0.2005, accuracy=0.25),
            },
        ],
        (tensor(0.5058), tensor(0.4731)): [
            {
                "FastGTN": Metric(
                    micro_f1=0.22277227722772278, macro_f1=0.2179, accuracy=0.22277
                ),
                "GAT": Metric(
                    micro_f1=0.22772277227722773, macro_f1=0.1857, accuracy=0.22772
                ),
                "HAN": Metric(
                    micro_f1=0.22277227722772278, macro_f1=0.0827, accuracy=0.22277
                ),
                "HGT": Metric(
                    micro_f1=0.23267326732673269, macro_f1=0.2274, accuracy=0.23267
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.26732673267326734, macro_f1=0.2442, accuracy=0.26733
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.2079207920792079, macro_f1=0.1994, accuracy=0.20792
                ),
                "GAT": Metric(
                    micro_f1=0.2376237623762376, macro_f1=0.1961, accuracy=0.23762
                ),
                "HAN": Metric(
                    micro_f1=0.23267326732673269, macro_f1=0.1, accuracy=0.23267
                ),
                "HGT": Metric(
                    micro_f1=0.25742574257425743, macro_f1=0.2504, accuracy=0.25743
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.3118811881188119, macro_f1=0.2807, accuracy=0.31188
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.19801980198019803, macro_f1=0.1924, accuracy=0.19802
                ),
                "GAT": Metric(
                    micro_f1=0.26732673267326734, macro_f1=0.2617, accuracy=0.26733
                ),
                "HAN": Metric(
                    micro_f1=0.2376237623762376, macro_f1=0.1063, accuracy=0.23762
                ),
                "HGT": Metric(
                    micro_f1=0.23267326732673269, macro_f1=0.226, accuracy=0.23267
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.2079207920792079, macro_f1=0.1922, accuracy=0.20792
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.19801980198019803, macro_f1=0.1879, accuracy=0.19802
                ),
                "GAT": Metric(
                    micro_f1=0.22277227722772278, macro_f1=0.2116, accuracy=0.22277
                ),
                "HAN": Metric(
                    micro_f1=0.22772277227722773, macro_f1=0.0922, accuracy=0.22772
                ),
                "HGT": Metric(
                    micro_f1=0.24752475247524752, macro_f1=0.2422, accuracy=0.24752
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.3316831683168317, macro_f1=0.3101, accuracy=0.33168
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.2079207920792079, macro_f1=0.1994, accuracy=0.20792
                ),
                "GAT": Metric(
                    micro_f1=0.30198019801980197, macro_f1=0.2571, accuracy=0.30198
                ),
                "HAN": Metric(
                    micro_f1=0.22772277227722773, macro_f1=0.0852, accuracy=0.22772
                ),
                "HGT": Metric(
                    micro_f1=0.2722772277227723, macro_f1=0.2748, accuracy=0.27228
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.30198019801980197, macro_f1=0.2419, accuracy=0.30198
                ),
            },
        ],
        (tensor(0.6718), tensor(0.6481)): [
            {
                "FastGTN": Metric(
                    micro_f1=0.2111111111111111, macro_f1=0.2073, accuracy=0.21111
                ),
                "GAT": Metric(
                    micro_f1=0.20555555555555555, macro_f1=0.1922, accuracy=0.20556
                ),
                "HAN": Metric(
                    micro_f1=0.25555555555555554, macro_f1=0.1637, accuracy=0.25556
                ),
                "HGT": Metric(
                    micro_f1=0.28888888888888886, macro_f1=0.2863, accuracy=0.28889
                ),
                "SimpleHGN": Metric(micro_f1=0.3, macro_f1=0.28, accuracy=0.3),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.21666666666666667, macro_f1=0.2142, accuracy=0.21667
                ),
                "GAT": Metric(
                    micro_f1=0.2111111111111111, macro_f1=0.2047, accuracy=0.21111
                ),
                "HAN": Metric(
                    micro_f1=0.2611111111111111, macro_f1=0.1752, accuracy=0.26111
                ),
                "HGT": Metric(
                    micro_f1=0.25555555555555554, macro_f1=0.2546, accuracy=0.25556
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.32222222222222224, macro_f1=0.3012, accuracy=0.32222
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.2222222222222222, macro_f1=0.2176, accuracy=0.22222
                ),
                "GAT": Metric(
                    micro_f1=0.21666666666666667, macro_f1=0.1949, accuracy=0.21667
                ),
                "HAN": Metric(
                    micro_f1=0.2111111111111111, macro_f1=0.1272, accuracy=0.21111
                ),
                "HGT": Metric(
                    micro_f1=0.28888888888888886, macro_f1=0.29, accuracy=0.28889
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.29444444444444445, macro_f1=0.2904, accuracy=0.29444
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.22777777777777777, macro_f1=0.2277, accuracy=0.22778
                ),
                "GAT": Metric(micro_f1=0.25, macro_f1=0.2167, accuracy=0.25),
                "HAN": Metric(
                    micro_f1=0.2111111111111111, macro_f1=0.1249, accuracy=0.21111
                ),
                "HGT": Metric(
                    micro_f1=0.22777777777777777, macro_f1=0.2248, accuracy=0.22778
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.3277777777777778, macro_f1=0.332, accuracy=0.32778
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.2222222222222222, macro_f1=0.2231, accuracy=0.22222
                ),
                "GAT": Metric(
                    micro_f1=0.21666666666666667, macro_f1=0.1946, accuracy=0.21667
                ),
                "HAN": Metric(
                    micro_f1=0.25555555555555554, macro_f1=0.1681, accuracy=0.25556
                ),
                "HGT": Metric(
                    micro_f1=0.2722222222222222, macro_f1=0.2733, accuracy=0.27222
                ),
                "SimpleHGN": Metric(micro_f1=0.3, macro_f1=0.2693, accuracy=0.3),
            },
        ],
        (tensor(0.8281), tensor(0.8131)): [
            {
                "FastGTN": Metric(
                    micro_f1=0.2513089005235602, macro_f1=0.2476, accuracy=0.25131
                ),
                "GAT": Metric(
                    micro_f1=0.2617801047120419, macro_f1=0.2414, accuracy=0.26178
                ),
                "HAN": Metric(
                    micro_f1=0.3036649214659686, macro_f1=0.2325, accuracy=0.30366
                ),
                "HGT": Metric(
                    micro_f1=0.3403141361256545, macro_f1=0.334, accuracy=0.34031
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.42408376963350786, macro_f1=0.3824, accuracy=0.42408
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.2670157068062827, macro_f1=0.2635, accuracy=0.26702
                ),
                "GAT": Metric(
                    micro_f1=0.21465968586387435, macro_f1=0.1905, accuracy=0.21466
                ),
                "HAN": Metric(
                    micro_f1=0.27225130890052357, macro_f1=0.2113, accuracy=0.27225
                ),
                "HGT": Metric(
                    micro_f1=0.3193717277486911, macro_f1=0.3023, accuracy=0.31937
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.4397905759162304, macro_f1=0.4109, accuracy=0.43979
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.2198952879581152, macro_f1=0.2144, accuracy=0.2199
                ),
                "GAT": Metric(
                    micro_f1=0.225130890052356, macro_f1=0.2133, accuracy=0.22513
                ),
                "HAN": Metric(
                    micro_f1=0.3298429319371728, macro_f1=0.2562, accuracy=0.32984
                ),
                "HGT": Metric(
                    micro_f1=0.39267015706806285, macro_f1=0.3896, accuracy=0.39267
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.42408376963350786, macro_f1=0.4006, accuracy=0.42408
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.27225130890052357, macro_f1=0.2683, accuracy=0.27225
                ),
                "GAT": Metric(
                    micro_f1=0.24083769633507854, macro_f1=0.2227, accuracy=0.24084
                ),
                "HAN": Metric(
                    micro_f1=0.3089005235602094, macro_f1=0.2352, accuracy=0.3089
                ),
                "HGT": Metric(
                    micro_f1=0.3403141361256545, macro_f1=0.3341, accuracy=0.34031
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.4973821989528796, macro_f1=0.4701, accuracy=0.49738
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.225130890052356, macro_f1=0.2253, accuracy=0.22513
                ),
                "GAT": Metric(
                    micro_f1=0.2198952879581152, macro_f1=0.1907, accuracy=0.2199
                ),
                "HAN": Metric(
                    micro_f1=0.2670157068062827, macro_f1=0.2008, accuracy=0.26702
                ),
                "HGT": Metric(
                    micro_f1=0.3036649214659686, macro_f1=0.2898, accuracy=0.30366
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.4031413612565445, macro_f1=0.3895, accuracy=0.40314
                ),
            },
        ],
        (tensor(0.9997), tensor(0.9976)): [
            {
                "FastGTN": Metric(
                    micro_f1=0.24120603015075376, macro_f1=0.2359, accuracy=0.24121
                ),
                "GAT": Metric(
                    micro_f1=0.2814070351758794, macro_f1=0.2516, accuracy=0.28141
                ),
                "HAN": Metric(
                    micro_f1=0.3869346733668342, macro_f1=0.3465, accuracy=0.38693
                ),
                "HGT": Metric(
                    micro_f1=0.3768844221105528, macro_f1=0.3701, accuracy=0.37688
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.36683417085427134, macro_f1=0.3532, accuracy=0.36683
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.20100502512562815, macro_f1=0.201, accuracy=0.20101
                ),
                "GAT": Metric(
                    micro_f1=0.23115577889447236, macro_f1=0.223, accuracy=0.23116
                ),
                "HAN": Metric(
                    micro_f1=0.36180904522613067, macro_f1=0.3169, accuracy=0.36181
                ),
                "HGT": Metric(
                    micro_f1=0.4020100502512563, macro_f1=0.3969, accuracy=0.40201
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.4221105527638191, macro_f1=0.4191, accuracy=0.42211
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.18592964824120603, macro_f1=0.1819, accuracy=0.18593
                ),
                "GAT": Metric(
                    micro_f1=0.2562814070351759, macro_f1=0.2369, accuracy=0.25628
                ),
                "HAN": Metric(
                    micro_f1=0.3316582914572864, macro_f1=0.2915, accuracy=0.33166
                ),
                "HGT": Metric(
                    micro_f1=0.3969849246231156, macro_f1=0.3918, accuracy=0.39698
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.5326633165829145, macro_f1=0.5312, accuracy=0.53266
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.24623115577889448, macro_f1=0.2423, accuracy=0.24623
                ),
                "GAT": Metric(
                    micro_f1=0.2613065326633166, macro_f1=0.2483, accuracy=0.26131
                ),
                "HAN": Metric(
                    micro_f1=0.31155778894472363, macro_f1=0.2808, accuracy=0.31156
                ),
                "HGT": Metric(
                    micro_f1=0.4371859296482412, macro_f1=0.4336, accuracy=0.43719
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.44221105527638194, macro_f1=0.4233, accuracy=0.44221
                ),
            },
            {
                "FastGTN": Metric(
                    micro_f1=0.19095477386934673, macro_f1=0.1898, accuracy=0.19095
                ),
                "GAT": Metric(
                    micro_f1=0.2562814070351759, macro_f1=0.2424, accuracy=0.25628
                ),
                "HAN": Metric(
                    micro_f1=0.3417085427135678, macro_f1=0.3007, accuracy=0.34171
                ),
                "HGT": Metric(
                    micro_f1=0.5025125628140703, macro_f1=0.5013, accuracy=0.50251
                ),
                "SimpleHGN": Metric(
                    micro_f1=0.5175879396984925, macro_f1=0.5243, accuracy=0.51759
                ),
            },
        ],
    }

    node_homophily_micro_f1 = {}
    node_homophily_macro_f1 = {}

    edge_homophily_micro_f1 = {}
    edge_homophily_macro_f1 = {}
    for (node_homophily, edge_homophily), homophily_results in results.items():
        micro_f1_per_model = defaultdict(list)
        macro_f1_per_model = defaultdict(list)

        for res in homophily_results:
            for model, metric in res.items():
                micro_f1_per_model[model].append(metric.micro_f1)
                macro_f1_per_model[model].append(metric.macro_f1)

        mean_micro_f1_per_model = {
            model: round(np.mean(values), 3)
            for model, values in micro_f1_per_model.items()
        }

        mean_macro_f1_per_model = {
            model: round(np.mean(values), 3)
            for model, values in macro_f1_per_model.items()
        }

        rounded_node_hom = round(node_homophily.item(), 2)
        rounded_edge_hom = round(edge_homophily.item(), 2)

        node_homophily_micro_f1[rounded_node_hom] = mean_micro_f1_per_model
        node_homophily_macro_f1[rounded_node_hom] = mean_macro_f1_per_model

        edge_homophily_micro_f1[rounded_edge_hom] = mean_micro_f1_per_model
        edge_homophily_macro_f1[rounded_edge_hom] = mean_macro_f1_per_model

    print(f"Node_homophily_micro_f1: {node_homophily_micro_f1}")
    print(f"Node_homophily_macro_f1: {node_homophily_macro_f1}")

    print(f"Edge homophily micro f1: {edge_homophily_micro_f1}")
    print(f"Edge homophily macro f1: {edge_homophily_macro_f1}")
    plot(node_homophily_micro_f1, x_label="Node Homophily", y_label="Micro-F1 Score")
    plot(node_homophily_macro_f1, x_label="Node Homophily", y_label="Macro-F1 Score")

    plot(edge_homophily_micro_f1, x_label="Edge Homophily", y_label="Micro-F1 Score")
    plot(edge_homophily_macro_f1, x_label="Edge Homophily", y_label="Macro-F1 Score")


def plot(data: Dict[float, Dict[str, float]], x_label: str, y_label: str):
    homophily_values = sorted(data.keys())
    models = data[homophily_values[0]].keys()

    plt.figure(figsize=(10, 6))

    for model in models:
        scores = [data[h][model] for h in homophily_values]
        # Create smooth lines using spline interpolation
        x_new = np.linspace(min(homophily_values), max(homophily_values), 300)
        spline = make_interp_spline(homophily_values, scores, k=3)
        y_smooth = spline(x_new)
        plt.plot(x_new, y_smooth, label=model)
        # plt.plot(homophily_values, scores, marker="o", label=model)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(f"{x_label} vs {y_label} Score for Different Models")
    plt.legend(title="Models", loc="best")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(
        f'{x_label.lower().replace(" ", "_")}_{y_label.lower().replace(" ", "_")}.png'
    )
    # plt.show()


def test_standard_datasets():
    datasets: List[DGLDataset] = [
        ACMDataset,
        FreebaseHeCoDataset,
        AMinerHeCoDataset,
        IMDbDataset,
        DBLPHeCoDataset,
    ]

    results: Dict[str, List[Metric]] = {}
    num_runs = 5
    for dataset in datasets:
        data: DGLDataset = dataset()
        hg = data[0]
        category = data.predict_ntype

        dataset_name = data.name
        # print(f"metapaths: {data.metapaths}")
        print(f"{dataset_name}; hg: {hg}")
        print(
            f"Running {dataset_name} with correlation score: {data.correlation_score()} and homophily: {Util.compute_homophily(hg, category)}"
        )

        num_features = hg.nodes[category].data["feat"].shape[1]

        for i in range(num_runs):
            print(f"Run {i + 1} for {dataset_name}")
            dataset_results = results.get(dataset_name, [])
            dataset_results.append(
                run(
                    hg,
                    num_features,
                    data.num_classes,
                    category,
                    data.metapaths,
                    num_epochs=200,
                )
            )

            results[dataset_name] = dataset_results

    print(f"Results: {results}")


if __name__ == "__main__":
    # test_homophily()
    # test_standard_datasets()
    test_simple_gen()
