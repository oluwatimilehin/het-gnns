import torch

from data.acm import ACMDataset

from trainers.FastGTNTrainer import FastGTNTrainer
from trainers.GATV2Trainer import GATV2Trainer
from trainers.HANTrainer import HANTrainer
from trainers.HGTTrainer import HGTTrainer
from trainers.SimpleHGNTrainer import SimpleHGNTrainer

from util import Util


def populate_graph_node_data(
    hetero_graph, node_type, num_nodes, num_classes, num_features
):
    hetero_graph.nodes[node_type].data["feat"] = torch.randn(num_nodes, num_features)
    hetero_graph.nodes[node_type].data["label"] = torch.randint(
        0, num_classes, (num_nodes,)
    )
    hetero_graph.nodes[node_type].data["train_mask"] = torch.zeros(
        num_nodes, dtype=torch.bool
    ).bernoulli(0.6)
    hetero_graph.nodes[node_type].data["val_mask"] = torch.zeros(
        num_nodes, dtype=torch.bool
    ).bernoulli(0.2)
    hetero_graph.nodes[node_type].data["test_mask"] = torch.zeros(
        num_nodes, dtype=torch.bool
    ).bernoulli(0.2)

    return hetero_graph


def test_acm():
    # Function to verify the heterogenous models by evaluating them on the ACM dataset
    data = ACMDataset()
    g = data[0]

    num_epochs = 100
    ntype = data.predict_ntype
    input_dim = g.nodes[ntype].data["feat"].shape[1]

    print(f"acm het: {data.score()}")

    # gat_trainer = GATV2Trainer(
    #     Util.to_homogeneous(g, ntype),
    #     input_dim=input_dim,
    #     output_dim=data.num_classes,
    # )
    # gat_trainer.run()

    # hgt_trainer = HGTTrainer(
    #     g, input_dim=input_dim, output_dim=data.num_classes, category=ntype
    # )
    # hgt_trainer.run(num_epochs=num_epochs)

    fastgtn_trainer = FastGTNTrainer(
        g, input_dim=input_dim, output_dim=data.num_classes, category=ntype
    )
    fastgtn_trainer.run(num_epochs=num_epochs)

    simplehgt_trainer = SimpleHGNTrainer(
        g, input_dim=input_dim, output_dim=data.num_classes, category=ntype
    )
    simplehgt_trainer.run(num_epochs=num_epochs)

    han_trainer = HANTrainer(
        g,
        input_dim=input_dim,
        output_dim=data.num_classes,
        meta_paths=data.metapaths,
        category=ntype,
    )
    han_trainer.run(num_epochs=num_epochs)


if __name__ == "__main__":

    n_user_classes = 5
    num_edges_dict = {("author", "author-paper", "paper"): 10000}
    num_nodes_dict = {
        "paper": 1000,
        "author": 100,
    }

    hg = Util.generate_graph(
        num_nodes_dict, num_edges_dict, "paper", n_user_classes, homogeneity=0.2
    )
    print(f'homogeneity: {Util.compute_homogeneity(hg, "paper")}')

    n_user_classes = 5
    n_hetero_features = 5
    num_edges_dict = {
        ("user", "follow", "user"): 3000,
        ("user", "click", "item"): 5000,
        ("user", "dislike", "item"): 500,
    }
    num_nodes_dict = {"user": 1000, "item": 500}

    category = "user"
    hetero_graph = Util.generate_graph(
        num_nodes_dict,
        num_edges_dict,
        category,
        n_user_classes,
        num_features=n_hetero_features,
        homogeneity=0.6,
    )
    print(f"homogeneity: {Util.compute_homogeneity(hetero_graph, category)}")

    # test_acm()

    num_epochs = 100

    fastgtn_trainer = FastGTNTrainer(
        hetero_graph,
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
        category=category,
    )
    fastgtn_trainer.run(num_epochs)

    # SimpleHGN
    simple_hgn_trainer = SimpleHGNTrainer(
        hetero_graph,
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
        category=category,
    )
    simple_hgn_trainer.run(num_epochs)

    # # HAN
    han_trainer = HANTrainer(
        hetero_graph,
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
        meta_paths=[["dislike", "rev_dislike"], ["click", "rev_click"]],
        category=category,
    )
    han_trainer.run(num_epochs)

    gat_trainer = GATV2Trainer(
        Util.to_homogeneous(hetero_graph, category),
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
    )
    gat_trainer.run(num_epochs)

    # HGT
    hgt_trainer = HGTTrainer(
        hetero_graph,
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
        category=category,
    )
    hgt_trainer.run(num_epochs)
