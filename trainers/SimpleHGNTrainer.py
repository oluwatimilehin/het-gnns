import torch
import torch.nn.functional as F

from models.SimpleHGN import SimpleHGN
from utils import Util, EarlyStopping


class SimpleHGNTrainer:
    def __init__(
        self,
        g,
        input_dim,
        output_dim,
        category,
        edge_dim=64,
        gpu=-1,
        hidden_dim=256,
        num_layers=3,
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

        self.category = category
        if self.cuda:
            self.model.cuda()

    def run(self, num_epochs=200, lr=1e-3, weight_decay=5e-4):
        print(f"Running SimpleHGNTrainer")

        stopper = EarlyStopping(patience=100)

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

                val_res = Util.accuracy(logits[val_mask], labels[val_mask])

                early_stop = stopper.step(val_res, self.model)
                if early_stop:
                    break

                print(
                    f"Epoch {epoch:05d}  | Loss {loss.item():.4f} | "
                    f"TrainAcc {train_acc:.4f} | ValRes: {val_res}"
                )

        stopper.load_checkpoint(self.model)
        res = Util.evaluate_dict(
            self.g, self.model, h_dict, self.category, labels, test_mask
        )
        print(f"Test results: {res}")
        return res
