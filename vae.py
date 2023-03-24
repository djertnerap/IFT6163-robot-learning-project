# Visual Autoencoder - AE minimizing reconstruction error by distance squared |y_raw - y_hat_raw|^2

import os
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
from torch.utils.data import DataLoader

from ae_model import ConvolutionalAE, NetworkConfiguration


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir)
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image


def get_dataloaders(config, transform=None):
    dataset = CustomImageDataset(img_dir=config['data']['img_dir'], transform=transform)
    train_dataloader = DataLoader(dataset["train"],
                                  batch_size=config['alg']['train_batch_size'],
                                  shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(dataset["test"],
                                  batch_size=config['alg']['train_batch_size'],
                                  shuffle=True,
                                  drop_last=True)

    return train_dataloader, test_dataloader

def train(config, model, optimizer):

    # 200,000 minibatches * 50 images =
    # rained separate AE for each env (square, circular, trapezoid)
    #
    train_dataloader, _ = get_dataloaders(config=config, transform=None)

    for epoch in range(config['alg']['epochs']):
        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                optimizer.zero_grad()

                # input
                x = batch["pixel_values"].to(model.device)

                # recon
                recon, input_img = model(x)[0:1]

                # loss
                loss = model.loss_function(recon, input_img)

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())


def create_config(config):
    out_channels: List[int, ...] = list(config['alg']['out_channels'])
    kernel_sizes: List[int, ...] = list(config['alg']['kernel_sizes'])
    strides: List[int, ...] = list(config['alg']['strides'])
    paddings: List = list(config['alg']['paddings'])
    return [out_channels, kernel_sizes, strides, paddings]


@hydra.main(config_path="config", config_name="config_vae")
def main(config: DictConfig):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_config = create_config(config=config)
    
    model = ConvolutionalAE(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim'],
        net_config=net_config,
        device=device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['alg']['learning_rate'])

    model.to(device)
    optimizer.to(device)

    train_dataloader, test_dataloader = get_dataloaders(config=config, transform=None)

    for epoch in range(1, config['alg']['n_epochs'] + 1):
        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            model.train()
            for batch in tepoch:
                tepoch.set_description(f"Epoch: {epoch}")

                optimizer.zero_grad()

                x = batch.to(device)

                recons = model(x)

                loss = model.loss(x, recons)
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())


if __name__ == "__main__":
    main()
