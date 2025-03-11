import time

import dgl
import torch
import torch.nn.functional as F

from models.GATV2 import GATv2
from utils import EarlyStopping, Util


class GATV2Trainer:
    def __init__(
        self,
        g,
        num_classes,
        gpu=-1,
        num_heads=8,
        num_out_heads=1,
        num_layers=1,
        num_hidden=8,
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

        self.features = g.ndata["feat"]
        num_feats = self.features.shape[1]

        self.labels = g.ndata["label"]
        self.train_mask = g.ndata["train_mask"]
        self.val_mask = g.ndata["val_mask"]
        self.test_mask = g.ndata["test_mask"]

        g = dgl.remove_self_loop(g)
        self.g = dgl.add_self_loop(g)

        heads = ([num_heads] * num_layers) + [num_out_heads]
        self.model = GATv2(
            num_layers,
            num_feats,
            num_hidden,
            num_classes,
            heads,
            F.elu,
            input_feature_dropout,
            attention_dropout,
            negative_slope,
            residual,
        )

        if self.cuda:
            self.model.cuda()

    def train(
        self,
        num_epochs=200,
        lr=0.005,
        weight_decay=5e-4,
        early_stop=False,
        fast_mode=False,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay)

        if early_stop:
            stopper = EarlyStopping(patience=100)

        mean = 0
        for epoch in range(num_epochs):
            self.model.train()

            if epoch >= 3:
                t0 = time.time()

            # forward
            logits = self.model(self.g, self.features)

            loss_fcn = torch.nn.CrossEntropyLoss()
            loss = loss_fcn(logits[self.train_mask], self.labels[self.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= 3:
                train_acc = Util.accuracy(
                    logits[self.train_mask], self.labels[self.train_mask]
                )

                if fast_mode:
                    val_acc = Util.accuracy(
                        logits[self.val_mask], self.labels[self.val_mask]
                    )
                else:
                    val_acc = Util.evaluate(
                        self.g, self.model, self.features, self.labels, self.val_mask
                    )
                    if early_stop:
                        if stopper.step(val_acc, self.model):
                            break

                print(
                    f"Epoch {epoch:05d} | Loss {loss.item():.4f} | "
                    f"TrainAcc {train_acc:.4f} | ValAcc {val_acc:.4f} "
                )

        if early_stop:
            self.model.load_state_dict(
                torch.load("es_checkpoint.pt", weights_only=False)
            )

        acc = Util.evaluate(
            self.g, self.model, self.features, self.labels, self.test_mask
        )
        print(f"Test Accuracy {acc:.4f}")
