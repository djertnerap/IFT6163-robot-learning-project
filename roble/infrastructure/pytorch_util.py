#########################################################################################
# The code in this file has been taken & modified from the homeworks of the course IFT6163 at UdeM.
# Original authors: Glen Berseth, Lucas Maes, Aton Kamanda
# Date: 2023-04-06
# Title: ift6163_homeworks_2023
# Code version: 5a7e39e78a9260e078555305e669ebcb93ef6e6c
# Type: Source code
# URL: https://github.com/milarobotlearningcourse/ift6163_homeworks_2023
#########################################################################################

from typing import Union

from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = "tanh",
    output_activation: Activation = "identity",
) -> nn.Module:
    """
    Builds a feedforward neural network
    arguments:
        input_placeholder: placeholder variable for the state (batch_size, input_size)
        scope: variable scope of the network
        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer
        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer
    returns:
        output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)
