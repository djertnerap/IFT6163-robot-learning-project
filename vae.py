import os

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from rat_dataset import RatDataModule


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, learning_rate: float, net_config: tuple, in_channels: int, latent_dim: int):
        super().__init__()
        self._learning_rate = learning_rate
        activation = nn.ReLU()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        n_channels, kernel_sizes, strides, paddings, output_paddings = net_config
        in_channels = self.in_channels

        ###########################
        # 1. Build Encoder
        ###########################
        modules = []

        # CNN
        for i in range(len(n_channels)):
            modules.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=n_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            )
            modules.append(activation)
            in_channels = n_channels[i]

        # Flatten and Linear encoder
        modules.append(nn.Flatten())
        modules.append(nn.Linear(n_channels[-1] * 16 * 16, self.latent_dim))

        self.encoder = nn.Sequential(*modules)

        ###########################
        # 2. Build Decoder
        ###########################
        modules = []

        n_channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        paddings.reverse()
        n_channels.append(self.in_channels)

        # Flatten and Linear encoder
        modules.append(nn.Flatten())
        decoder_lin = nn.Sequential(
            nn.Linear(self.latent_dim, n_channels[0] * 16 * 16),
            nn.Unflatten(dim=1, unflattened_size=(n_channels[0], 16, 16)),
            activation,
        )
        modules.append(decoder_lin)

        # reverse CNN
        for i in range(len(n_channels) - 1):

            modules.append(
                nn.ConvTranspose2d(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i],
                ),
            )
            modules.append(activation)

        self.decoder = nn.Sequential(*modules)

        self.save_hyperparameters(ignore=["net_config"])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def generate(self, x):
        return self.forward(x)[0]

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


def run_vae_experiment(config: DictConfig):
    original_cwd = hydra.utils.get_original_cwd()
    rat_data_module = RatDataModule(
        data_dir=os.path.abspath(original_cwd + config["hardware"]["smp_dataset_folder_path"]),
        config=config,
        batch_size=config["vae"]["train_batch_size"],
        num_workers=config["hardware"]["num_data_loader_workers"],
        img_size=config["env"]["img_size"],
    )

    ae = LitAutoEncoder(
        learning_rate=config["vae"]["learning_rate"],
        net_config=config["vae"]["net_config"].values(),
        in_channels=config["vae"]["in_channels"],
        latent_dim=config["vae"]["latent_dim"],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        filename="rat_ae-{epoch:02d}-{train_loss:.6f}",
        save_top_k=3,
        mode="min",
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(
        max_steps=config["vae"]["max_steps"],
        max_epochs=config["vae"]["max_epochs"],
        callbacks=[checkpoint_callback],
        default_root_dir=original_cwd,
        logger=tb_logger,
        # profiler="simple",
    )
    trainer.fit(ae, datamodule=rat_data_module)
