import dgl
import torch
import torch.nn.functional as F

from models.GATV2 import GATv2
from utils import EarlyStopping, Util


class GATV2Trainer:
    def __init__(
        self,
        g,
        input_dim,
        output_dim,
        hidden_dim=256,
        gpu=-1,
        num_heads=4,
        num_out_heads=1,
        num_layers=3,
        residual=False,
        input_feature_dropout=0.7,
        attention_dropout=0.7,
        negative_slope=0.2,
    ):

        if gpu < 0:
            self.cuda = False
        else:
            self.cuda = True
            g = g.int().to(gpu)

        g = dgl.remove_self_loop(g)
        self.g = dgl.add_self_loop(g)

        heads = ([num_heads] * num_layers) + [num_out_heads]
        self.model = GATv2(
            num_layers,
            input_dim,
            hidden_dim,
            output_dim,
            heads,
            F.elu,
            input_feature_dropout,
            attention_dropout,
            negative_slope,
            residual,
        )

        if self.cuda:
            self.model.cuda()

    def run(
        self,
        num_epochs=200,
        lr=0.005,
        weight_decay=5e-4,
    ):
        print(f"Running GATTrainer")
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        stopper = EarlyStopping(patience=100)

        features = self.g.ndata["feat"]

        labels = self.g.ndata["label"]
        train_mask = self.g.ndata["train_mask"]
        val_mask = self.g.ndata["val_mask"]
        test_mask = self.g.ndata["test_mask"]

        for epoch in range(num_epochs):
            self.model.train()

            # forward
            logits = self.model(self.g, features)

            loss_fcn = torch.nn.CrossEntropyLoss()
            loss = loss_fcn(logits[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                train_acc = Util.accuracy(logits[train_mask], labels[train_mask])
                val_res = Util.accuracy(logits[val_mask], labels[val_mask])

                early_stop = stopper.step(val_res, self.model)
                if early_stop:
                    break

                print(
                    f"Epoch {epoch:05d} | Loss {loss.item():.4f} | "
                    f"TrainAcc {train_acc:.4f} | ValRes: {val_res} "
                )

        stopper.load_checkpoint(self.model)
        res = Util.evaluate(self.g, self.model, features, labels, test_mask)
        print(f"Test results: {res}")

        return res
