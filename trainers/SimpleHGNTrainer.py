import dgl
import torch
import torch.nn.functional as F

from models.SimpleHGN import SimpleHGN
from utils import Util


class SimpleHGNTrainer:
    def __init__(
        self,
        g,
        input_dim,
        output_dim,
        edge_dim=64,
        gpu=-1,
        hidden_dim=256,
        num_layers=1,
        num_heads=8,
        feat_dropout=0.5,
        negative_slope=0.05,
        residual=True,
        beta=0.05,
    ):
        if gpu < 0:
            self.cuda = False
        else:
            self.cuda = True
            g = g.int().to(gpu)

        self.g = g
        self.model = SimpleHGN(
            edge_dim=edge_dim,
            num_etypes=len(g.etypes),
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=output_dim,
            num_layers=num_layers,
            heads=[num_heads] * num_layers + [1],
            feat_drop=feat_dropout,
            negative_slope=negative_slope,
            residual=residual,
            beta=beta,
            ntypes=g.ntypes,
        )

        if self.cuda:
            self.model.cuda()

    def run(self, predicted_node_type, num_epochs=200, lr=1e-3, weight_decay=5e-4):
        print(f"Running SimpleHGNTrainer")

        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        features = self.g.nodes[predicted_node_type].data["feat"]
        labels = self.g.nodes[predicted_node_type].data["label"]
        test_mask = self.g.nodes[predicted_node_type].data["test_mask"]
        train_mask = self.g.nodes[predicted_node_type].data["train_mask"]
        val_mask = self.g.nodes[predicted_node_type].data["val_mask"]

        h_dict = {ntype: self.g.nodes[ntype].data["feat"] for ntype in self.g.ntypes}

        for epoch in range(num_epochs):
            self.model.train()

            logits = self.model(self.g, h_dict)[predicted_node_type]
            loss = F.cross_entropy(
                logits[train_mask],
                labels[train_mask],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                train_acc = Util.accuracy(
                    logits[train_mask],
                    labels[train_mask],
                )

                val_acc = Util.evaluate_dict(
                    self.g,
                    self.model,
                    h_dict,
                    predicted_node_type,
                    labels,
                    val_mask,
                )

                print(
                    f"Epoch {epoch:05d}  | Loss {loss.item():.4f} | "
                    f"TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f}"
                )

        acc = Util.evaluate_dict(
            self.gs, self.model, h_dict, predicted_node_type, labels, test_mask
        )
        print(f"Test Accuracy {acc:.4f}")
