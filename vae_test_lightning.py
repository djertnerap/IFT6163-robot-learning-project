import hydra
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from rat_dataset import RatDataModule


# TODO: Spatial memory pipeline like this
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, learning_rate: float, config):
        super().__init__()
        self.config = config
        # self.encoder, self.decoder = encoder, decoder
        self._learning_rate = learning_rate
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

        ## Flatten and Linear encoder
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


def main():
    hydra.initialize(config_path="config", job_name="rat_random_walk_dataset_generator", version_base=None)
    config = hydra.compose(config_name="config")

    rat_data_module = RatDataModule(config["hardware"]["dataset_folder_path"])

    encoder = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=8192, out_features=64),
    )
    decoder = nn.Sequential(
        nn.Linear(in_features=64, out_features=8192),
        nn.Unflatten(dim=1, unflattened_size=(32, 16, 16)),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(),
    )
    ae = LitAutoEncoder(encoder=encoder, decoder=decoder, learning_rate=0.0001, config=config)

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(ae, datamodule=rat_data_module)

if __name__ == "__main__":
    main()
