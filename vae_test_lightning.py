import hydra
import pytorch_lightning as pl
import torch
from torch import nn

from rat_dataset import RatDataModule


# TODO: Spatial memory pipeline like this
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, learning_rate: float):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self._learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


def main():
    hydra.initialize(config_path="config", job_name="rat_random_walk_dataset_generator", version_base=None)
    config = hydra.compose(config_name="config_test")

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
    ae = LitAutoEncoder(encoder, decoder, 0.0001)

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(ae, datamodule=rat_data_module)




if __name__ == "__main__":
    main()
