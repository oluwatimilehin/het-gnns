import torch

from data.acm import ACMDataset
from data.imdb import IMDbDataset
from data.dataset import (
    ACMHeCoDataset,
    DBLPHeCoDataset,
    AMinerHeCoDataset,
    FreebaseHeCoDataset,
)

from trainers.FastGTNTrainer import FastGTNTrainer
from trainers.GATV2Trainer import GATV2Trainer
from trainers.HANTrainer import HANTrainer
from trainers.HGTTrainer import HGTTrainer
from trainers.SimpleHGNTrainer import SimpleHGNTrainer

from utils import Util


def test_acm():
    # Function to verify the heterogenous models by evaluating them on the ACM dataset
    data = ACMDataset()
    g = data[0]

    num_epochs = 100
    ntype = data.predict_ntype
    input_dim = g.nodes[ntype].data["feat"].shape[1]

    print(f"acm het: {data.correlation_score()}")

    gat_trainer = GATV2Trainer(
        Util.to_homogeneous(g, ntype),
        input_dim=input_dim,
        output_dim=data.num_classes,
    )
    gat_trainer.run(num_epochs=num_epochs)

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
    print(f"DBLPHeco Correlation: {DBLPHeCoDataset().correlation_score()}")
    print(f"ACMHeco Correlation : {ACMHeCoDataset().correlation_score()}")
    print(f"FreebaseHeco Correlation : {FreebaseHeCoDataset().correlation_score()}")
    print(f"AMiner Correlation: {AMinerHeCoDataset().correlation_score()}")
    print(f"IMDb Correlation: {IMDbDataset().correlation_score()}")

    # raise ValueError("Stop")

    n_user_classes = 5
    num_edges_dict = {("author", "author-paper", "paper"): 10000}
    num_nodes_dict = {
        "paper": 1000,
        "author": 100,
    }

    hg = Util.generate_graph(
        num_nodes_dict, num_edges_dict, "paper", n_user_classes, correlation=0.2
    )
    print(f'correlation: {Util.compute_correlation(hg, "paper")}')

    n_user_classes = 3
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
        correlation=0.5,
    )
    print(f"homogeneity: {Util.compute_correlation(hetero_graph, category)}")

    test_acm()
