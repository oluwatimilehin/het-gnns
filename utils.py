import torch
import torch.nn.functional as F


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), "es_checkpoint.pt")


class Util:
    @classmethod
    def accuracy(cls, logits, labels):
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

    @classmethod
    def evaluate(cls, g, model, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(g, features)
            logits = logits[mask]
            labels = labels[mask]
            return cls.accuracy(logits, labels)
