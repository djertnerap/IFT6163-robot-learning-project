import os
from typing import Tuple, Union

import hydra
import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch import nn

from rat_dataset import SequencedDataModule
from vae import LitAutoEncoder


class SpatialMemoryPipeline(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        memory_slot_learning_rate: float,
        auto_encoder: LitAutoEncoder,
        entropy_reactivation_target: float,
        memory_slot_size: int,
        nb_memory_slots: int,
        probability_correction: float,
    ):
        super().__init__()
        self._learning_rate = learning_rate
        self._memory_slot_learning_rate = memory_slot_learning_rate
        self._probability_correction = probability_correction

        self._lstm_angular_velocity = nn.LSTM(input_size=2, hidden_size=memory_slot_size, batch_first=True)
        self._lstm_angular_velocity_correction = nn.LSTM(
            input_size=2 * memory_slot_size, hidden_size=memory_slot_size, batch_first=True
        )
        self._pi_angular_velocity = nn.Parameter(torch.rand(size=[1]))

        self._lstm_angular_velocity_and_speed = nn.LSTM(input_size=3, hidden_size=memory_slot_size, batch_first=True)
        self._lstm_angular_velocity_and_speed_correction = nn.LSTM(
            input_size=2 * memory_slot_size, hidden_size=memory_slot_size, batch_first=True
        )
        self._pi_angular_velocity_and_speed = nn.Parameter(torch.rand(size=[1]))

        self._lstm_no_self_motion = nn.LSTM(input_size=1, hidden_size=memory_slot_size, batch_first=True)
        self._lstm_no_self_motion_correction = nn.LSTM(
            input_size=2 * memory_slot_size, hidden_size=memory_slot_size, batch_first=True
        )
        self._pi_no_self_motion = nn.Parameter(torch.rand(size=[1]))

        self._auto_encoder = auto_encoder
        self._auto_encoder.requires_grad_(False)

        self._entropy_reactivation_target = entropy_reactivation_target
        self._beta = 1  # TODO: find initial beta

        # TODO: find how to init the memory slots
        self._memories = nn.Parameter(torch.rand(size=(4, nb_memory_slots, memory_slot_size)))
        # 1 (visual inputs) + 3 (nb of RNNs) X nb of slots X encoding dimension

        self._gamma = nn.Parameter(torch.rand((1,)))

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
        return torch.squeeze(
            torch.matmul(torch.unsqueeze(torch.unsqueeze(v1, dim=-2), dim=-2), torch.unsqueeze(v2, dim=-1))
        )

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        # visual_input: Vatch X Sequence length X nb channels X img_size X img_size
        # velocities: Batch X Sequence length X 2 (linear speed, angular velocity)
        visual_input, velocities = batch

        y_enc = self._auto_encoder.encode(torch.flatten(visual_input, start_dim=0, end_dim=1)).unflatten(
            dim=0, sizes=visual_input.shape[:2]
        )  # Batch X Sequence length X encoding dimension

        # Batch X Sequence length X nb of slots
        p_react = self._calculate_activation(self._beta, y_enc, self._visual_memories)

        angular_velocity = velocities[:, :, 1]
        velocities = torch.concat(
            [
                10
                * (
                    torch.concat(
                        [
                            torch.unsqueeze(torch.cos_(angular_velocity), dim=-1),
                            torch.unsqueeze(torch.sin_(angular_velocity), dim=-1),
                        ],
                        dim=-1,
                    )
                ),
                torch.unsqueeze(velocities[:, :, 0], dim=-1),
            ],
            dim=-1,
        )
        x_1, h_1 = self._lstm_angular_velocity(velocities[:, :, :2])  # x: Batch X Sequence length X encoding dimension
        x_2, h_2 = self._lstm_angular_velocity_and_speed(velocities)
        x_3, h_3 = self._lstm_no_self_motion(torch.empty(size=list(velocities.shape[:2]) + [1], device=self.device))
        # TODO: these do not account for correction step. Might need to do a for loop.

        # Correction step
        # C Step 1 calculate weights
        unscaled_visual_activations = self._gamma * self._batch_vector_dot(y_enc, self._visual_memories)
        weights = torch.unsqueeze(nn.functional.softmax(unscaled_visual_activations, dim=-1), dim=-1)

        # C Step 2 Calculate weighted memories
        angular_velocity_x_tilde = torch.sum(weights * self._angular_velocity_memories, dim=-2)
        angular_velocity_and_speed_x_tilde = torch.sum(weights * self._angular_velocity_and_speed_memories, dim=-2)
        no_self_motion_x_tilde = torch.sum(weights * self._no_self_motion_memories, dim=-2)

        # C Step 3 Calculate P_correction mask
        should_be_corrected = torch.unsqueeze(
            torch.rand(size=(p_react.shape[:-1]), device=self.device) < self._probability_correction, dim=-1
        )

        # C Step 4 Apply corrections
        corrected_x_1, _ = self._lstm_angular_velocity_correction(torch.concat([x_1, angular_velocity_x_tilde], dim=-1))
        x_1 = should_be_corrected * corrected_x_1 + ~should_be_corrected * x_1
        corrected_x_2, _ = self._lstm_angular_velocity_and_speed_correction(
            torch.concat([x_2, angular_velocity_and_speed_x_tilde], dim=-1)
        )
        x_2 = should_be_corrected * corrected_x_2 + ~should_be_corrected * x_2
        corrected_x_3, _ = self._lstm_no_self_motion_correction(torch.concat([x_3, no_self_motion_x_tilde], dim=-1))
        x_3 = should_be_corrected * corrected_x_3 + ~should_be_corrected * x_3

        # p_pred
        p_pred = (
            self._calculate_activation(self._pi_angular_velocity, x_1, self._angular_velocity_memories)
            * self._calculate_activation(
                self._pi_angular_velocity_and_speed, x_2, self._angular_velocity_and_speed_memories
            )
            * self._calculate_activation(self._pi_no_self_motion, x_3, self._no_self_motion_memories)
        )  # Batch X Sequence length X nb of slots

        loss = torch.sum(p_react * torch.log(p_pred))
        self.log("train_loss", loss)
        self.log("batch", float(batch_idx))
        return loss

    @classmethod
    def _calculate_activation(
        cls, entropy_coeff: Union[float, torch.Tensor], activation_vector: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        return torch.exp_(entropy_coeff * cls._batch_vector_dot(activation_vector, memory))

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self._memories, "lr": self._memory_slot_learning_rate},
                {"params": self._pi_no_self_motion},
                {"params": self._pi_angular_velocity},
                {"params": self._pi_angular_velocity_and_speed},
                {"params": self._gamma},
                {"params": self._lstm_angular_velocity.parameters()},
                {"params": self._lstm_angular_velocity_and_speed.parameters()},
                {"params": self._lstm_no_self_motion.parameters()},
            ],
            lr=self._learning_rate,
        )

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


def run_smp_experiment(config: DictConfig):
    original_cwd = hydra.utils.get_original_cwd()
    data_dir = os.path.abspath(original_cwd + config["hardware"]["smp_dataset_folder_path"])
    rat_sequence_data_module = SequencedDataModule(
        data_dir=data_dir,
        config=config,
        bptt_unroll_length=config["smp"]["bptt_unroll_length"],
        batch_size=config["smp"]["batch_size"],
        num_workers=config["hardware"]["num_data_loader_workers"],
        img_size=config["env"]["img_size"],
    )

    checkpoint_path = os.path.abspath(original_cwd + config["smp"]["ae_checkpoint_path"])
    ae = LitAutoEncoder.load_from_checkpoint(
        checkpoint_path, in_channels=config["vae"]["in_channels"], net_config=config["vae"]["net_config"].values()
    )
    ae.eval()
    ae.freeze()

    smp = SpatialMemoryPipeline(
        learning_rate=config["smp"]["learning_rate"],
        memory_slot_learning_rate=config["smp"]["memory_slot_learning_rate"],
        auto_encoder=ae,
        entropy_reactivation_target=config["smp"]["entropy_reactivation_target"],
        memory_slot_size=config["vae"]["latent_dim"],
        nb_memory_slots=config["smp"]["nb_memory_slots"],
        probability_correction=config["smp"]["prob_correction"],
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(max_epochs=1, default_root_dir=original_cwd, logger=tb_logger, log_every_n_steps=1)
    trainer.fit(smp, datamodule=rat_sequence_data_module)
