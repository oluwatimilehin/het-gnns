import numpy as np
import dgl
import torch

from data.acm import ACMDataset

from trainers.FastGTNTrainer import FastGTNTrainer
from trainers.GATV2Trainer import GATV2Trainer
from trainers.HANTrainer import HANTrainer
from trainers.HGTTrainer import HGTTrainer
from trainers.SimpleHGNTrainer import SimpleHGNTrainer


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
    ).bernoulli(0.6)
    hetero_graph.nodes[node_type].data["test_mask"] = torch.zeros(
        num_nodes, dtype=torch.bool
    ).bernoulli(0.6)

    return hetero_graph


def test_acm():
    # Function to verify the heterogenous models by evaluating them on the ACM dataset
    data = ACMDataset()
    g = data[0]

    num_epochs = 100
    ntype = data.predict_ntype
    input_dim = g.nodes[ntype].data["feat"].shape[1]

    hgt_trainer = HGTTrainer(
        g, input_dim=input_dim, output_dim=data.num_classes, category=ntype
    )
    hgt_trainer.run(num_epochs=num_epochs)

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

    n_users = 1000
    n_items = 500
    n_follows = 3000
    n_clicks = 5000
    n_dislikes = 500
    n_hetero_features = 10
    n_user_classes = 5
    n_max_clicks = 10

    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)

    hetero_graph = dgl.heterograph(
        {
            ("user", "follow", "user"): (follow_src, follow_dst),
            ("user", "followed-by", "user"): (follow_dst, follow_src),
            ("user", "click", "item"): (click_src, click_dst),
            ("item", "clicked-by", "user"): (click_dst, click_src),
            ("user", "dislike", "item"): (dislike_src, dislike_dst),
            ("item", "disliked-by", "user"): (dislike_dst, dislike_src),
        }
    )

    hetero_graph = populate_graph_node_data(
        hetero_graph=hetero_graph,
        node_type="user",
        num_nodes=n_users,
        num_classes=n_user_classes,
        num_features=n_hetero_features,
    )

    hetero_graph = populate_graph_node_data(
        hetero_graph=hetero_graph,
        node_type="item",
        num_nodes=n_items,
        num_classes=5,
        num_features=n_hetero_features,
    )

    hetero_graph.edges["click"].data["label"] = torch.randint(
        1, n_max_clicks, (n_clicks,)
    ).float()

    hetero_graph.edges["click"].data["train_mask"] = torch.zeros(
        n_clicks, dtype=torch.bool
    ).bernoulli(0.6)

    test_acm()

    category = "user"

    # HGT
    hgt_trainer = HGTTrainer(
        hetero_graph,
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
        category=category,
    )
    hgt_trainer.run()

    # SimpleHGN
    simple_hgn_trainer = SimpleHGNTrainer(
        hetero_graph,
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
        category=category,
    )
    simple_hgn_trainer.run()

    # HAN
    han_trainer = HANTrainer(
        hetero_graph,
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
        meta_paths=[["dislike", "disliked-by"], ["click", "clicked-by"]],
        category=category,
    )
    han_trainer.run()

    # GATV2
    # TODO: I think I need to zero-pad any node types, instead of ensuring that all the types have the same features
    gat_trainer = GATV2Trainer(
        dgl.to_homogeneous(
            hetero_graph,
            ndata=["feat", "label", "test_mask", "train_mask", "val_mask"],
        ),
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
    )
    gat_trainer.run()
