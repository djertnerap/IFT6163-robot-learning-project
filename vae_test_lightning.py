import os
import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
import torch
from torch import nn
from torch.nn import functional as F

from rat_dataset import RatDataModule


# TODO: Spatial memory pipeline like this
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._learning_rate = self.config['vae']['learning_rate']
        self.activation = nn.ReLU()
        self.net_config = self.config['vae']['net_config'].values()
        self.in_channels = self.config['vae']['in_channels']
        self.latent_dim = self.config['vae']['latent_dim']

        n_channels, kernel_sizes, strides, paddings, output_paddings = self.net_config
        in_channels = self.in_channels

        ###########################
        # 1. Build Encoder
        ###########################
        modules = []

        # CNN
        for i in range(len(n_channels)):
            modules.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=n_channels[i],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          padding=paddings[i])
            )
            modules.append(self.activation)
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
        self.decoder_lin = nn.Sequential(
            nn.Linear(self.latent_dim, n_channels[0] * 16 * 16),
            nn.Unflatten(dim=1, unflattened_size=(n_channels[0], 16, 16)),
            self.activation
        )
        modules.append(self.decoder_lin)

        # reverse CNN
        for i in range(len(n_channels) - 1):

            modules.append(
                nn.ConvTranspose2d(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i]),
            )
            modules.append(self.activation)

        self.decoder = nn.Sequential(*modules)

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


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):

    # OLD STUFF
    # hydra.initialize(config_path="config", job_name="rat_random_walk_dataset_generator", version_base=None)
    # config = hydra.compose(config_name="config", return_hydra_config=True)
    # hydra.runtime.output_dir + # version_output = trainer.logger.version ?

    original_cwd = hydra.utils.get_original_cwd()
    rat_data_module = RatDataModule(original_cwd + config["hardware"]["dataset_folder_path"])

    ae = LitAutoEncoder(config=config)

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        filename='rat_autoencoder-{epoch:02d}-{train_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(max_epochs=4, callbacks=[checkpoint_callback], default_root_dir=original_cwd, logger=tb_logger)
    trainer.fit(ae, datamodule=rat_data_module)


if __name__ == "__main__":
    main()
