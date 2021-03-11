import pytorch_lightning as pl

import torch
from torch import nn
from torch.nn import Linear
from torch.nn import functional as F


def accuracy(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    """
    Computes accuracy for binary or multi-label classification
    :param y:
    :param y_hat:
    :return:
    """
    y_hat = torch.round(y_hat)
    acc = (y == y_hat).sum().item() / y.numel()
    return acc


class AttentionPooling(nn.Module):
    # Batch first !
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.d = Linear(in_features, 1)

    def forward(self, x):
        x1 = F.softmax(self.d(x), dim=1).expand(x.size())
        x2 = x1 * x
        v = x2.sum(1)
        return v


class RNN(pl.LightningModule):
    def __init__(self, p: float = 0.5, n_out: int = 1, vocab_limit: int = 101):
        super().__init__()
        self.save_hyperparameters()
        self.do = torch.nn.Dropout(p=p)
        self.embeddings = torch.nn.Embedding(vocab_limit, embedding_dim=64)
        self.lstm = torch.nn.LSTM(
            input_size=64, hidden_size=128, bidirectional=True, batch_first=True
        )

        self.attn_pool = AttentionPooling(256)
        self.out = Linear(256, n_out)

    def forward(self, x):
        x = self.do(self.embeddings(x))
        x, _ = self.lstm(x)

        x = self.do(x)

        x = self.attn_pool(x)

        y_hat = F.sigmoid(self.out(x))

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.binary_cross_entropy(y_hat, y)

        acc = accuracy(y, y_hat)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.binary_cross_entropy(y_hat, y)

        acc = accuracy(y, y_hat)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.binary_cross_entropy(y_hat, y)

        acc = accuracy(y, y_hat)

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
