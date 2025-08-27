from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

@dataclass
class SyntheticConfig:
    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int
    n_repeated: int
    n_classes: int
    class_sep: float
    random_state: int
    test_size: float
    val_size: float
    batch_size: int
    num_workers: int

class NumpyClassificationDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

class SyntheticClassifDM(pl.LightningDataModule):
    def __init__(self, cfg: SyntheticConfig):
        super().__init__()
        self.cfg = cfg
        self._train = self._val = self._test = None

    def prepare_data(self) -> None:
        # nada para baixar
        pass

    def setup(self, stage: str | None = None) -> None:
        X, y = make_classification(
            n_samples=self.cfg.n_samples,
            n_features=self.cfg.n_features,
            n_informative=self.cfg.n_informative,
            n_redundant=self.cfg.n_redundant,
            n_repeated=self.cfg.n_repeated,
            n_classes=self.cfg.n_classes,
            class_sep=self.cfg.class_sep,
            random_state=self.cfg.random_state,
        )
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=self.cfg.test_size + self.cfg.val_size,
            random_state=self.cfg.random_state, stratify=y
        )
        rel_val = self.cfg.val_size / (self.cfg.test_size + self.cfg.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=1 - rel_val, random_state=self.cfg.random_state, stratify=y_tmp
        )

        self._train = NumpyClassificationDataset(X_train, y_train)
        self._val   = NumpyClassificationDataset(X_val,   y_val)
        self._test  = NumpyClassificationDataset(X_test,  y_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)