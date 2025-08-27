from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class MLPClassifier(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, lr: float, weight_decay: float, dropout: float = 0.0):
        super().__init__()
        self.save_hyperparameters()  # loga no checkpoint

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, num_classes)]
        self.net = nn.Sequential(*layers)

        self.acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = self.acc(preds, y)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc",  acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)