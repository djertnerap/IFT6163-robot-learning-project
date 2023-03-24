import torch
from abc import abstractmethod
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Tuple, Callable, List, NamedTuple, Any


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    paddings: Tuple[int, ...] = (0, 0, 0)
    dense_hiddens: Tuple[int, ...] = (256, 256)


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


class ConvolutionalAE(BaseAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 net_config: NetworkConfiguration,
                 device=torch.device,
                 **kwargs) -> None:
        super(ConvolutionalAE, self).__init__()

        self.net_config = net_config
        self.latent_dim = latent_dim
        self.device = device

        n_channels, kernel_sizes, strides, paddings, dense_hiddens = self.net_config

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels=h_dim,
                          kernel_size=3,
                          stride=2,
                          padding='same'),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels=h_dim,
                          kernel_size=3,
                          stride=2,
                          padding='same'),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels=h_dim,
                          kernel_size=3,
                          stride=2,
                          padding='same'),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels=h_dim,
                          kernel_size=3,
                          stride=2,
                          padding='same'),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            )
        )

        # Build Decoder
        modules = []

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        return self.encoder(input)

    def decode(self, input: Tensor) -> Any:
        return self.decoder(input)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        return decoded

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:

        recons = inputs[0]
        input = inputs[1]

        loss = F.mse_loss(recons, input)
        return loss
