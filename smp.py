from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn


class SpatialMemoryPipeline(pl.LightningModule):
    def __init__(self, learning_rate: float, auto_encoder: pl.LightningModule, entropy_reactivation_target: float):
        super().__init__()
        self._learning_rate = learning_rate

        self._lstm_angular_velocity = nn.LSTM(batch_first=True)
        self._lstm_angular_velocity_correction = nn.LSTM(batch_first=True)
        self._pi_angular_velocity = torch.rand(size=[1])

        self._lstm_angular_velocity_and_speed = nn.LSTM(batch_first=True)
        self._lstm_angular_velocity_and_speed_correction = nn.LSTM(batch_first=True)
        self._pi_angular_velocity_and_speed = torch.rand(size=[1])

        self._lstm_no_self_motion = nn.LSTM(batch_first=True)
        self._lstm_no_self_motion_correction = nn.LSTM(batch_first=True)
        self._pi_no_self_motion = torch.rand(size=[1])

        self._auto_encoder = auto_encoder
        self._auto_encoder.requires_grad_(False)

        self._entropy_reactivation_target = entropy_reactivation_target
        self._beta = 1  # TODO: find initial beta

        # TODO: find how to init the memory slots
        self._memories = torch.rand(size=(4, 512, 64))
        # 1 (visual inputs) + 3 (nb of RNNs) X nb of slots X encoding dimension

    @property
    def _visual_memories(self) -> torch.Tensor:
        return torch.squeeze(self._memories[0])

    @property
    def _angular_velocity_memories(self) -> torch.Tensor:
        return torch.squeeze(self._memories[1])

    @property
    def _angular_velocity_and_speed_memories(self) -> torch.Tensor:
        return torch.squeeze(self._memories[2])

    @property
    def _no_self_motion_memories(self) -> torch.Tensor:
        return torch.squeeze(self._memories[3])

    @staticmethod
    def _batch_vector_dot(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(torch.matmul(torch.unsqueeze(v1, dim=-1), torch.unsqueeze(v2, dim=-2)))

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # velocities: Batch X Sequence length X 3 (sine angular velocity, cosine angular velocity, linear speed)
        visual_input, velocities = batch

        y_enc = self._auto_encoder.forward(visual_input)  # Batch X Sequence length X encoding dimension

        # Batch X Sequence length X nb of slots
        p_react = self._calculate_activation(self._beta, y_enc, self._visual_memories)

        x_1, h_1 = self._lstm_angular_velocity(velocities[:, :, :2])  # x: Batch X Sequence length X encoding dimension
        x_2, h_2 = self._lstm_angular_velocity_and_speed(velocities)
        x_3, h_3 = self._lstm_no_self_motion(torch.empty(size=velocities.shape, device=self.device))
        # TODO: these do not account for correction step. Might need to do a for loop.

        p_pred = (
            self._calculate_activation(self._pi_angular_velocity, x_1, self._angular_velocity_memories)
            * self._calculate_activation(
                self._pi_angular_velocity_and_speed, x_2, self._angular_velocity_and_speed_memories
            )
            * self._calculate_activation(self._pi_no_self_motion, x_3, self._no_self_motion_memories)
        )  # Batch X Sequence length X nb of slots

        loss = torch.sum(p_react * torch.log_(p_pred))
        return loss

    @classmethod
    def _calculate_activation(
        cls, entropy_coeff: Union[float, torch.Tensor], activation_vector: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp_(entropy_coeff * cls._batch_vector_dot(activation_vector, memory))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    def update_beta(self, p_react: float):
        """
        Gets the newly regulated parameter beta used to calculate the target memory reactivation.
        **This should be called after every trajectory**
        """
        beta_logit = np.log(self._beta)
        # Perhaps we could apply a certain rounding that defines when they are close enough for us not to change it?
        if p_react < self._entropy_reactivation_target:
            beta_logit += 0.001
        elif p_react > self._entropy_reactivation_target:
            beta_logit -= 0.001
        self._beta = np.exp(beta_logit)
