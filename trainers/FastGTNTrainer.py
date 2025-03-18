import torch
import torch.nn.functional as F

from models.FastGTN import FastGTN
from utils import Util


class FastGTNTrainer:
    def __init__(
        self,
        g,
        input_dim,
        output_dim,
        category,
        num_channels=2,
        gpu=-1,
        hidden_dim=128,
        num_layers=2,
        norm=True,
        identity=False,
    ):
        if gpu < 0:
            self.cuda = False
        else:
            self.cuda = True
            g = g.int().to(gpu)

        self.g = g
        self.category = category

        self.model = FastGTN(
            num_edge_type=len(g.etypes),
            num_channels=num_channels,
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            num_class=output_dim,
            num_layers=num_layers,
            norm=norm,
            identity=identity,
            category=category,
        )

        if self.cuda:
            self.model.cuda()

    def run(self, num_epochs=200, lr=0.005, weight_decay=0.001):
        print(f"Running FastGTNTrainer")

        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        labels = self.g.nodes[self.category].data["label"]
        test_mask = self.g.nodes[self.category].data["test_mask"]
        train_mask = self.g.nodes[self.category].data["train_mask"]
        val_mask = self.g.nodes[self.category].data["val_mask"]

        h_dict = {ntype: self.g.nodes[ntype].data["feat"] for ntype in self.g.ntypes}

        for epoch in range(num_epochs):
            self.model.train()

            logits = self.model(self.g, h_dict)[self.category]
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

                val_res = Util.evaluate_dict(
                    self.g,
                    self.model,
                    h_dict,
                    self.category,
                    labels,
                    val_mask,
                )

                print(
                    f"Epoch {epoch:05d}  | Loss {loss.item():.4f} | "
                    f"TrainAcc {train_acc:.4f} | ValRes: {val_res}"
                )

        res = Util.evaluate_dict(
            self.g, self.model, h_dict, self.category, labels, test_mask
        )
        print(f"Test results: {res}")
        return res
