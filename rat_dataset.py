from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Sized

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import generate_data


class RatDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, config: DictConfig, batch_size: int = 50, num_workers: int = 0, img_size: int = 64
    ):
        super().__init__()
        self._data_dir = data_dir
        self._config = config
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._img_size = img_size
        self._img_dataset = None

    def prepare_data(self):
        original_cwd = hydra.utils.get_original_cwd()
        data_dir = os.path.abspath(original_cwd + self._config.hardware.smp_dataset_folder_path)
        if not Path(data_dir).exists():
            generate_data.generate_data(self._config)

    def setup(self, stage: str):
        self._img_dataset = ImageFolder(root=self._data_dir, transform=ToTensor())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self._img_dataset,
            shuffle=True,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=True if self._num_workers > 0 else False,
        )


class RandomTrajectoriesSampler(Sampler[int]):
    r"""Samples elements randomly from different trajectories, without replacement.

    Args:
        n_traj: number of trajectories
        chunks: number of chunks in every trajectory
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source: SequenceDataset, batch_size, generator=None) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.n_trajs = data_source.n_trajs
        self.chunks = data_source.traj_chunks
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        idxs = np.arange(self.n_trajs * self.chunks).reshape((self.n_trajs, self.chunks))
        for i in range(self.n_trajs):
            idxs[i] = np.random.permutation(idxs[i])
        for i in range(self.chunks):
            so_idxs = np.random.permutation(self.n_trajs)
            idxs[:, i:] = idxs[:, i:][so_idxs]
        idxs = idxs.reshape(self.batch_size, -1, order="A")
        rand_idxs = idxs.T.flatten()
        yield from iter(rand_idxs.tolist())

    def __len__(self) -> int:
        return len(self.data_source)


class SequenceDataset(Dataset):
    def __init__(self, root: str, seq_length: int):
        self._data_dir = root
        self.path = Path(root)
        self.n_trajs = 0
        self.seq_length = seq_length
        self.transform = ToTensor()

        arrs = []
        for child in self.path.iterdir():
            if self.n_trajs == 0:
                self._subfloder = str(child)[:-1]
                while self._subfloder[-1].isdigit():
                    self._subfloder = self._subfloder[:-1]
            self.n_trajs += 1
        for i in range(self.n_trajs):
            arrs.append(np.load(self._subfloder + str(i) + "/traj.npy")[None])
        self.trajs = np.concatenate(arrs, axis=0, dtype="float32")

        self.traj_chunks = self.trajs.shape[-1] // seq_length

    def __len__(self):
        return self.n_trajs * self.traj_chunks

    def __getitem__(self, index):
        traj = index // self.traj_chunks
        chunk = index % self.traj_chunks

        acts = self.trajs[traj, :, chunk * self.seq_length : (chunk + 1) * self.seq_length].T
        

        first_img_idx = chunk * self.seq_length
        img_path = Path(str(self._subfloder) + str(traj) + "/Images")
        first_img = self.transform(Image.open(img_path / (str(first_img_idx) + ".png")))
        imgs = torch.zeros((self.seq_length, *first_img.shape))
        imgs[0] = first_img
        for i in range(self.seq_length - 1):
            img = Image.open(img_path / (str(first_img_idx + i + 1) + ".png"))
            imgs[i + 1] = self.transform(img)

        # imgs is a Tensor of size (seq_length, 3, 64, 64) and acts is a Tensor of size (seq_length, 2)
        return imgs, self.transform(acts).squeeze()


class SequencedDataModule(RatDataModule):
    def __init__(
        self,
        data_dir: str,
        config: DictConfig,
        bptt_unroll_length: int = 50,
        batch_size: int = 32,
        num_workers: int = 0,
        img_size: int = 64,
    ):
        super().__init__(
            data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, img_size=img_size, config=config
        )
        self._sampler = None
        self._seq_dataset = None
        self._bptt_unroll_length = bptt_unroll_length
        self.seq_length = config.smp.bptt_unroll_length

    def setup(self, stage):
        self._seq_dataset = SequenceDataset(root=self._data_dir, seq_length=self.seq_length)
        self._sampler = RandomTrajectoriesSampler(self._seq_dataset, self._batch_size)

    def prepare_data(self):
        original_cwd = hydra.utils.get_original_cwd()
        data_dir = os.path.abspath(original_cwd + self._config.hardware.smp_dataset_folder_path)
        if not Path(data_dir).exists():
            generate_data.generate_data(self._config)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self._seq_dataset,
            sampler=self._sampler,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=True if self._num_workers > 0 else False,
        )
