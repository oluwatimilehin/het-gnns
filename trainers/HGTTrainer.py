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
        gpu=-1,
        num_heads=4,
        num_layers=2,
        num_hidden=256,
        use_norm=True,
    ):

        if gpu < 0:
            self.cuda = False
        else:
            self.cuda = True
            g = g.int().to(gpu)

        self.g = g

        node_dict = {}
        edge_dict = {}
        for ntype in self.g.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in self.g.etypes:
            edge_dict[etype] = len(edge_dict)
            self.g.edges[etype].data["id"] = (
                torch.ones(self.g.num_edges(etype), dtype=torch.long) * edge_dict[etype]
            )

        self.model = HGT(
            g,
            node_dict=node_dict,
            edge_dict=edge_dict,
            n_inp=input_dim,
            n_hid=num_hidden,
            n_out=output_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            use_norm=use_norm,
        )

    def run(self, predict_node_type, num_epochs=200, max_lr=1e-3, clip=1):
        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=num_epochs, max_lr=max_lr
        )

        labels = self.g.nodes[predict_node_type].data["label"]
        test_mask = self.g.nodes[predict_node_type].data["test_mask"]
        train_mask = self.g.nodes[predict_node_type].data["train_mask"]
        val_mask = self.g.nodes[predict_node_type].data["val_mask"]

        for epoch in range(num_epochs):
            self.model.train()

            logits = self.model(self.g, predict_node_type)

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
                    predict_node_type,
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
            predict_node_type,
            labels,
            test_mask,
        )
        print(f"Test Accuracy {acc:.4f}")
