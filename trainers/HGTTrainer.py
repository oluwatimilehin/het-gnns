import torch
import torch.nn.functional as F

from models.HGT import HGT
from utils import Util


class HGTTrainer:
    def __init__(
        self,
        g,
        input_dim,
        output_dim,
        category,
        hidden_dim=256,
        gpu=-1,
        num_heads=4,
        num_layers=2,
        use_norm=True,
    ):

        if gpu < 0:
            self.cuda = False
        else:
            self.cuda = True
            g = g.int().to(gpu)

        self.g = g

        self.model = HGT(
            in_dims={ntype: g.nodes[ntype].data["feat"].shape[1] for ntype in g.ntypes},
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            num_heads=num_heads,
            ntypes=g.ntypes,
            etypes=g.canonical_etypes,
            category=category,
            num_layers=num_layers,
            use_norm=use_norm,
        )

        self.category = category

        if self.cuda:
            self.model.cuda()

    def run(self, num_epochs=200, max_lr=0.005, clip=1):
        print(f"Running HGTTrainer")
        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=num_epochs, max_lr=max_lr
        )

        features = {ntype: self.g.nodes[ntype].data["feat"] for ntype in self.g.ntypes}
        labels = self.g.nodes[self.category].data["label"]
        test_mask = self.g.nodes[self.category].data["test_mask"]
        train_mask = self.g.nodes[self.category].data["train_mask"]
        val_mask = self.g.nodes[self.category].data["val_mask"]

        for epoch in range(num_epochs):
            self.model.train()

            logits = self.model(self.g, features)

            loss = F.cross_entropy(
                logits[train_mask],
                labels[train_mask],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            optimizer.step()
            scheduler.step()

            if epoch >= 3:
                train_acc = Util.accuracy(
                    logits[train_mask],
                    labels[train_mask],
                )

                val_acc = Util.evaluate(
                    self.g,
                    self.model,
                    features,
                    labels,
                    val_mask,
                )

                print(
                    f"Epoch {epoch:05d}  | Loss {loss.item():.4f} | "
                    f"TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f}"
                )

        acc = Util.evaluate(
            self.g,
            self.model,
            features,
            labels,
            test_mask,
        )
        print(f"Test Accuracy {acc:.4f}")
