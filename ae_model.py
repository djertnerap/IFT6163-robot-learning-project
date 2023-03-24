import torch
from abc import abstractmethod
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple, List, NamedTuple, Any

from functools import reduce
from operator import __add__


class BaseAE(nn.Module):

    def __init__(self) -> None:
        super(BaseAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


def calculate_same_padding(k, s, i):
    return round(((s - 1) * i - s + k) / 2)


class ConvolutionalAE(BaseAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 net_config: List[List],
                 device=torch.device,
                 **kwargs) -> None:
        super(ConvolutionalAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.net_config = net_config
        self.device = device

        self.activation = nn.ReLU()

        n_channels, kernel_sizes, strides, paddings = self.net_config

        # Build Encoder
        modules = []

        for i in range(len(n_channels)):
            if strides[i] > 1 and paddings[i] == 'same':
                paddings[i] = calculate_same_padding(kernel_sizes[i], strides[i], in_channels)
            modules.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=n_channels[i],
                          kernel_size=kernel_sizes[i],
                          stride=strides[i],
                          padding=paddings[i])
            )
            modules.append(self.activation)
            in_channels = n_channels[i]

        modules.append(nn.Flatten())

        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, n_channels[-1] * latent_dim * latent_dim * 32)

        modules.append(self.decoder_input)

        n_channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        paddings.reverse()

        for i in range(len(n_channels) - 1):
            if strides[i] > 1 and paddings[i] == 'same':
                paddings[i] = calculate_same_padding(kernel_sizes[i], strides[i], n_channels[i])
            modules.append(
                nn.ConvTranspose2d(
                    in_channels=n_channels[i],
                    out_channels=n_channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=1),
            )
            modules.append(self.activation)

        self.decoder = nn.Sequential(*modules)

    def encode(self, x: Tensor) -> List[Tensor]:
        return self.encoder(x)

    def decode(self, x_encoded: Tensor) -> Any:
        return self.decoder(x_encoded)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    @abstractmethod
    def loss_function(self, x, recons) -> Tensor:

        loss = F.mse_loss(recons, x)
        return loss

