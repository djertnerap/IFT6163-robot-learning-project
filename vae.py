# Visual Autoencoder - AE minimizing reconstruction error by distance squared |y_raw - y_hat_raw|^2

import os
import time
from typing import Tuple, List

import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image
import pytorch_lightning as pl

from ae_model import ConvolutionalAE

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, rotate=None):
        self.img_dir = img_dir
        self.transform = transform
        self.rotate = rotate

    def __len__(self):
        return len([name for name in os.listdir(self.img_dir) if os.path.isfile(self.img_dir + name)])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir)
        # image = read_image(img_path)
        conv = ToTensor()
        image = Image.open(img_path + str(idx) + '.png').convert("RGB")
        image = conv(image)
        return image


def get_dataloaders(config, transform=None):

    cwd = hydra.utils.get_original_cwd() + '/'
    path = cwd + config['data']['img_dir']

    dataset = CustomImageDataset(img_dir=path, transform=transform)
    train_dataloader = DataLoader(dataset,
                                  batch_size=config['alg']['train_batch_size'],
                                  shuffle=True,
                                  drop_last=True)
    # test_dataloader = DataLoader(dataset,
    #                               batch_size=config['alg']['train_batch_size'],
    #                               shuffle=True,
    #                               drop_last=True)

    return train_dataloader


def train(config, model, optimizer):

    # 200,000 minibatches * 50 images =
    # rained separate AE for each env (square, circular, trapezoid)
    #
    train_dataloader = get_dataloaders(config=config, transform=None)

    start = time.time()
    for epoch in range(config['alg']['n_epochs']):
        model.train()
        for batch in train_dataloader:

            optimizer.zero_grad()

            # input
            x = batch.to(model.device)

            # recon
            recon = model(x)

            # loss
            loss = model.loss_function(x, recon)

            loss.backward()
            optimizer.step()
            print('Time:', time.time() - start)

        print('loss', loss)


def create_network_config(config):
    out_channels: List[int, ...] = list(config['alg']['output_channels'])
    kernel_sizes: List[int, ...] = list(config['alg']['kernel_sizes'])
    strides: List[int, ...] = list(config['alg']['strides'])
    paddings: List = list(config['alg']['paddings'])
    return [out_channels, kernel_sizes, strides, paddings]


@hydra.main(config_path="config", config_name="config_vae")
def main(config: DictConfig):
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    net_config = create_network_config(config=config)
    
    model = ConvolutionalAE(
        in_channels=config['alg']['in_channels'],
        latent_dim=config['alg']['latent_dim'],
        net_config=net_config,
        device=device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['alg']['learning_rate'])

    model.to(device)

    if config['alg']['do_train']:
        train(config=config, model=model, optimizer=optimizer)


if __name__ == "__main__":
    main()
