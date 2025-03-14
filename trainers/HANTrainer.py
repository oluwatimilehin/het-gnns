import dgl
import torch
import torch.nn.functional as F

from models.HAN import HAN
from utils import Util


class HANTrainer:
    def __init__(
        self,
        g,
        input_dim,
        output_dim,
        meta_paths,
        category,
        hidden_dim=256,
        gpu=-1,
        num_heads=4,
        dropout=0.7,
    ):

        if gpu < 0:
            self.cuda = False
        else:
            self.cuda = True
            g = g.int().to(gpu)

        gs = [dgl.metapath_reachable_graph(g, meta_path) for meta_path in meta_paths]
        for i in range(len(gs)):
            gs[i] = dgl.add_self_loop(dgl.remove_self_loop(gs[i]))

        self.g = g
        self.gs = gs
        self.model = HAN(
            num_meta_paths=len(meta_paths),
            in_size=input_dim,
            hidden_size=hidden_dim,
            out_size=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.category = category
        if self.cuda:
            self.model.cuda()

    def run(self, num_epochs=200, lr=1e-3, weight_decay=5e-4):
        print(f"Running HANTrainer")
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        features = self.g.nodes[self.category].data["feat"]
        labels = self.g.nodes[self.category].data["label"]
        test_mask = self.g.nodes[self.category].data["test_mask"]
        train_mask = self.g.nodes[self.category].data["train_mask"]
        val_mask = self.g.nodes[self.category].data["val_mask"]

        for epoch in range(num_epochs):
            self.model.train()

            logits = self.model(self.gs, features)
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

                val_acc = Util.evaluate(
                    self.gs,
                    self.model,
                    features,
                    labels,
                    val_mask,
                )

                print(
                    f"Epoch {epoch:05d}  | Loss {loss.item():.4f} | "
                    f"TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f}"
                )

        acc = Util.evaluate(self.gs, self.model, features, labels, test_mask)
        print(f"Test Accuracy {acc:.4f}")
