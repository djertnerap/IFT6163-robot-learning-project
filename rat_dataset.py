import os
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from walker import run_random_walk


class RatDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._train_dataset = None

    def prepare_data(self):
        data_dir = Path(self._data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if not any(data_dir.iterdir()):
            run_random_walk(600, 0, self._data_dir)

    def setup(self, stage: str):
        self._train_dataset = ImageFolder(root=str(Path(self._data_dir).parent), transform=ToTensor())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self._train_dataset, batch_size=self._batch_size)
