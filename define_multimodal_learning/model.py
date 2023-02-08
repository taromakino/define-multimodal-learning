import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import Accuracy


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        module_list = []
        last_in_dim = input_dim
        for hidden_dim in hidden_dims:
            module_list.append(nn.Linear(last_in_dim, hidden_dim))
            module_list.append(nn.LeakyReLU())
            last_in_dim = hidden_dim
        module_list.append(nn.Linear(last_in_dim, output_dim))
        self.module_list = nn.Sequential(*module_list)

    def forward(self, *args):
        return self.module_list(torch.hstack(args))


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(*args):
    return [arg.to(device()) for arg in args]


class UnimodalEnsemble(pl.LightningModule):
    def __init__(self, seed, dpath, hidden_dims, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.lr = lr
        self.model0 = MLP(2, hidden_dims, 1)
        self.model1 = MLP(1, hidden_dims, 1)
        self.test_acc = Accuracy("binary")


    def forward(self, x):
        x0, xp = x[:, :2], x[:, 2]
        pred0 = torch.sigmoid(self.model0(x0))
        predp = torch.sigmoid(self.model1(xp[:, None]))
        return (pred0 + predp) / 2


    def loss(self, x, y):
        return F.binary_cross_entropy(self.forward(x), y)


    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(*batch)
        self.test_acc(self.forward(x), y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class Multimodal(pl.LightningModule):
    def __init__(self, seed, dpath, hidden_dims, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.dpath = dpath
        self.lr = lr
        self.model = MLP(3, hidden_dims, 1)
        self.test_acc = Accuracy("binary")


    def forward(self, x):
        return torch.sigmoid(self.model(x))


    def loss(self, x, y):
        return F.binary_cross_entropy(self.forward(x), y)


    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(*batch)
        self.test_acc(self.forward(x), y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)