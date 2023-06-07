import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score


class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        if self.num_classes == 2:
            self.train_acc = Accuracy(task="binary")
        else:
            self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        return x

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        if self.num_classes == 2:
            loss = F.binary_cross_entropy(logits, y)
        else:
            loss = F.cross_entropy(logits, y)

        accuracy = self.train_acc(logits, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", accuracy, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        if self.num_classes == 2:
            loss = F.binary_cross_entropy(logits, y)
        else:
            loss = F.cross_entropy(logits, y)

        acc = self.train_acc(logits, y)

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
