import dgl
import torch
import torch.nn.functional as F

from models.HGT import HGT
from utils import EarlyStopping, Util


class HGTTrainer:
    def __init__(
        self,
        g,
        gpu=-1,
        num_heads=4,
        num_layers=2,
        num_hidden=256,
        input_dim=256,
        use_norm=True,
    ):

        if gpu < 0:
            self.cuda = False
        else:
            self.cuda = True
            g = g.int().to(gpu)

        self.g = g
        self.features = g.ndata["feat"]
        num_feats = self.features.shape[1]

        self.labels = g.ndata["label"]
        self.train_mask = g.ndata["train_mask"]
        self.val_mask = g.ndata["val_mask"]
        self.test_mask = g.ndata["test_mask"]

        # TODO: how to get node_dict and edge_dict from G

        self.model = HGT(
            g,
            node_dict={},
            edge_dict={},
            n_inp=input_dim,
            n_hid=num_hidden,
            n_out=self.labels.max().item() + 1,
            n_layers=num_layers,
            n_heads=num_heads,
            use_norm=use_norm,
        )

    def train(self, num_epochs=200, max_lr=1e-3, clip=1):
        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=num_epochs, max_lr=max_lr
        )

        for epoch in range(num_epochs):
            self.model.train()

            logits = self.model(self.g, self.features)

            loss = F.cross_entropy(
                logits[self.train_mask], self.labels[self.train_mask]
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            optimizer.step()
            scheduler.step()

            if epoch >= 3:
                train_acc = Util.accuracy(
                    logits[self.train_mask], self.labels[self.train_mask]
                )

                val_acc = Util.evaluate(
                    self.g, self.model, self.features, self.labels, self.val_mask
                )

                print(
                    f"Epoch {epoch:05d}  | Loss {loss.item():.4f} | "
                    f"TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f}"
                )

        acc = Util.evaluate(
            self.g, self.model, self.features, self.labels, self.test_mask
        )
        print(f"Test Accuracy {acc:.4f}")
