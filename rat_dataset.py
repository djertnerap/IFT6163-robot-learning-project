from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import generate_data
from agents.walker import run_random_walk


class RatDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 50, num_workers: int = 0, img_size: int = 64):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._img_size = img_size
        self._img_dataset = None

    def prepare_data(self):
        data_dir = Path(self._data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if not any(data_dir.iterdir()):
            run_random_walk(1200, 0, self._data_dir, img_size=self._img_size, save_traj=False)

    def setup(self, stage: str):
        self._img_dataset = ImageFolder(root=str(Path(self._data_dir).parent), transform=ToTensor())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self._img_dataset, batch_size=self._batch_size, num_workers=self._num_workers)


class SequenceDataset(Dataset):
    def __init__(self, root: str, seq_length):
        self._data_dir = root
        self.path = Path(root)
        self.n_trajs = 0
        self.seq_lenght = seq_length
        self.transform = ToTensor()

        arrs = []
        for child in self.path.iterdir():
            if self.n_trajs == 0:
                self._subfloder = str(child)[:-1]
            self.n_trajs += 1
        for i in range(self.n_trajs):
            arrs.append(np.load(self._subfloder + str(i) + "/traj.npy")[None])
        self.trajs = np.concatenate(arrs, axis=0)

        self.traj_chunks = self.trajs.shape[-1] // seq_length

    def __len__(self):
        return self.n_trajs * self.traj_chunks

    def __getitem__(self, index):
        traj = index // self.traj_chunks
        chunk = index % self.traj_chunks

        acts = self.trajs[traj, :, chunk * self.seq_lenght : (chunk + 1) * self.seq_lenght].T

        first_img_idx = chunk * self.seq_lenght
        img_path = Path(str(self._subfloder) + str(traj) + "/Images")
        first_img = self.transform(Image.open(img_path / (str(first_img_idx) + ".png")))
        imgs = torch.zeros((self.seq_lenght, *first_img.shape))
        imgs[0] = first_img
        for i in range(self.seq_lenght - 1):
            img = Image.open(img_path / (str(first_img_idx + i + 1) + ".png"))
            imgs[i + 1] = self.transform(img)

        # imgs is a Tensor of size (seq_length, 3, 64, 64) and acts is a Tensor of size (seq_length, 2)
        return imgs, self.transform(acts).squeeze()


class SequencedDataModule(RatDataModule):
    def __init__(self, data_dir: str, config: DictConfig, seq_length: int = 50, batch_size: int = 32):
        super().__init__(data_dir=data_dir, batch_size=batch_size)
        self.seq_length = seq_length
        self._config = config

    def prepare_data(self):
        if True:
            generate_data.generate_data(self._config)

    def setup(self, stage: str):
        self._img_dataset = SequenceDataset(root=self._data_dir, seq_length=self.seq_length)
