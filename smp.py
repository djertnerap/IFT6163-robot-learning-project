import os
from typing import Tuple, Union

import hydra
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn.functional import normalize
from lightning import pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
from torch import nn
from torchvision.utils import save_image

from rat_dataset import SequencedDataModule
from vae import LitAutoEncoder


class SpatialMemoryPipeline(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        memory_slot_learning_rate: float,
        auto_encoder: LitAutoEncoder,
        beta: float,
        entropy_reactivation_target: float,
        memory_slot_size: int,
        hidden_size_RNN1: int,
        hidden_size_RNN2: int,
        hidden_size_RNN3: int,
        nb_memory_slots: int,
        probability_correction: float,
        probability_storage: float,
        sequence_length: int,
        update_beta_every: int = 10,
    ):
        super().__init__()
        self.batch_size = batch_size
        self._sequence_length = sequence_length
        self._learning_rate = learning_rate
        self._memory_slot_learning_rate = memory_slot_learning_rate
        self._nb_memory_slots = nb_memory_slots
        self._probability_correction = probability_correction
        self._probability_storage = probability_storage
        self._update_beta_every = update_beta_every
        self._memory_slot_size = memory_slot_size
        self._hidden_size_RNN1 = hidden_size_RNN1
        self._hidden_size_RNN2 = hidden_size_RNN2
        self._hidden_size_RNN3 = hidden_size_RNN3

        self._lstm_angular_velocity = nn.LSTMCell(input_size=2, hidden_size=hidden_size_RNN1)
        self._lstm_angular_velocity_correction = nn.LSTMCell(input_size=hidden_size_RNN1, hidden_size=hidden_size_RNN1)
        self._pi_angular_velocity = nn.Parameter(torch.rand(size=[1]))
        self._angular_velocity_memories = nn.Parameter(torch.rand(size=(hidden_size_RNN1, nb_memory_slots)))

        self._lstm_angular_velocity_and_speed = nn.LSTMCell(input_size=3, hidden_size=hidden_size_RNN2)
        self._lstm_angular_velocity_and_speed_correction = nn.LSTMCell(
            input_size=hidden_size_RNN2, hidden_size=hidden_size_RNN2
        )
        self._pi_angular_velocity_and_speed = nn.Parameter(torch.rand(size=[1]))
        self._angular_velocity_and_speed_memories = nn.Parameter(torch.rand(size=(hidden_size_RNN2, nb_memory_slots)))

        self._lstm_no_self_motion = nn.LSTMCell(input_size=1, hidden_size=hidden_size_RNN3)
        self._lstm_no_self_motion_correction = nn.LSTMCell(input_size=hidden_size_RNN3, hidden_size=hidden_size_RNN3)
        self._pi_no_self_motion = nn.Parameter(torch.rand(size=[1]))
        self._no_self_motion_memories = nn.Parameter(torch.rand(size=(hidden_size_RNN3, nb_memory_slots)))

        self._auto_encoder = auto_encoder
        self._auto_encoder.requires_grad_(False)

        self._entropy_reactivation_target = entropy_reactivation_target
        self._beta = beta  # TODO: find initial beta (nooooo =()

        # TODO: use Sigmoid-LSTM
        self._visual_memories = nn.Parameter(torch.rand(size=(memory_slot_size, nb_memory_slots)), requires_grad=False)
        self._gamma = nn.Parameter(torch.rand((1,)))

        # self._last_y_enc = None
        self.slots_to_store = []

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        # visual_input: Batch X Sequence length X nb channels X img_size X img_size
        # velocities: Batch X Sequence length X 2 (linear speed, angular velocity)

        visual_input, velocities = batch

        # A: Calculate the target memories
        # A1: encode observations
        y_enc = self._auto_encoder.encode(torch.flatten(visual_input, start_dim=0, end_dim=1)).unflatten(
            dim=0, sizes=visual_input.shape[:2]
        )  # Batch X Sequence length X encoding dimension
        # self._last_y_enc = y_enc.detach()

        # A2: initialize P distributions
        # Batch X Sequence length X nb of slots
        p_react = torch.zeros(size=[self.batch_size, self._sequence_length, self._nb_memory_slots], device=self.device)
        p_pred = torch.zeros(size=[self.batch_size, self._sequence_length, self._nb_memory_slots], device=self.device)

        # A3: Prepare data
        angular_velocity = velocities[:, :, 1]
        velocities = torch.concat(
            [
                10
                * torch.concat(
                    [
                        torch.unsqueeze(torch.cos(angular_velocity), dim=-1),
                        torch.unsqueeze(torch.sin(angular_velocity), dim=-1),
                    ],
                    dim=-1,
                ),
                torch.unsqueeze(velocities[:, :, 0], dim=-1),
            ],
            dim=-1,
        )

        # B: For loop
        x_1, h_1 = torch.zeros((self.batch_size, self._hidden_size_RNN1), device=self.device), torch.zeros(
            (self.batch_size, self._hidden_size_RNN1), device=self.device
        )
        x_2, h_2 = torch.zeros((self.batch_size, self._hidden_size_RNN2), device=self.device), torch.zeros(
            (self.batch_size, self._hidden_size_RNN2), device=self.device
        )
        x_3, h_3 = torch.zeros((self.batch_size, self._hidden_size_RNN3), device=self.device), torch.zeros(
            (self.batch_size, self._hidden_size_RNN3), device=self.device
        )

        xs_1 = torch.zeros(size=list(velocities.shape[:2]) + [self._hidden_size_RNN1], device=self.device)
        xs_2 = torch.zeros(size=list(velocities.shape[:2]) + [self._hidden_size_RNN2], device=self.device)
        xs_3 = torch.zeros(size=list(velocities.shape[:2]) + [self._hidden_size_RNN3], device=self.device)
        y_raw_activation = torch.zeros(size=list(velocities.shape[:2]) + [self._memory_slot_size], device=self.device)

        self.mask_1 = [torch.ones(size=[self._hidden_size_RNN1, self._nb_memory_slots], device=self.device)]
        self.mask_2 = [torch.ones(size=[self._hidden_size_RNN2, self._nb_memory_slots], device=self.device)]
        self.mask_3 = [torch.ones(size=[self._hidden_size_RNN3, self._nb_memory_slots], device=self.device)]

        seq_len = velocities.shape[1]
        correction_samples = np.random.random(size=(seq_len,))
        storage_samples = np.random.random(size=(seq_len,))
        storage_samples[0] = 1
        start_t = [0]

        for t in range(seq_len):

            # P_storage
            if storage_samples[t] < self._probability_storage:
                
                p_react[:, start_t[-1]:t, :] = nn.functional.softmax(self._calculate_activation(self._beta,
                                                                                                y_enc[:, start_t[-1]:t, :],
                                                                                                self._visual_memories),
                                                                    dim=-1)
                start_t.append(t)

                self.vis_range = torch.max(torch.abs(torch.Tensor([self._visual_memories.max(),
                                                          self._visual_memories.min()]))).detach()
                self.av_range = torch.max(torch.abs(torch.Tensor([self._angular_velocity_memories.max(),
                                                            self._angular_velocity_memories.min()]))).detach()
                self.avs_range = torch.max(torch.abs(torch.Tensor([self._angular_velocity_and_speed_memories.max(),
                                                            self._angular_velocity_and_speed_memories.min()]))).detach()
                self.nsm_range = torch.max(torch.abs(torch.Tensor([self._no_self_motion_memories.max(),
                                                            self._no_self_motion_memories.min()]))).detach()

                self.slots_to_store.append(torch.randperm(self._nb_memory_slots)[0])

                batch_idx = torch.randperm(self.batch_size)[0]

                self._visual_memories[:, self.slots_to_store[-1]] = normalize(y_enc[batch_idx, t][None]) * self.vis_range
                self.mask_1.append(self.mask_1[-1].clone())
                self.mask_2.append(self.mask_2[-1].clone())
                self.mask_3.append(self.mask_3[-1].clone())
                self.mask_1[-1][:, self.slots_to_store[-1]] = (normalize(x_1[batch_idx].detach()[None])
                                                            * self.av_range
                                                            / self._angular_velocity_memories[:, self.slots_to_store[-1]])
                self.mask_2[-1][:, self.slots_to_store[-1]] = (normalize(x_2[batch_idx].detach()[None])
                                                            * self.avs_range
                                                            / self._angular_velocity_and_speed_memories[:, self.slots_to_store[-1]])
                self.mask_3[-1][:, self.slots_to_store[-1]] = (normalize(x_3[batch_idx].detach()[None])
                                                            * self.nsm_range
                                                            / self._no_self_motion_memories[:, self.slots_to_store[-1]])

            y_raw_activation = y_enc[:,t,:] @ self._visual_memories

            x_1, h_1 = self._lstm_angular_velocity(
                velocities[:, t, :2].squeeze(), (x_1, h_1)
            )  # x: Batch X 1 X encoding dimension
            x_2, h_2 = self._lstm_angular_velocity_and_speed(velocities[:, t, :].squeeze(), (x_2, h_2))
            x_3, h_3 = self._lstm_no_self_motion(torch.ones(size=(self.batch_size, 1), device=self.device), (x_3, h_3))

            out_1 = nn.functional.dropout(x_1, p=0.5)
            out_2 = nn.functional.dropout(x_2, p=0.5)
            out_3 = nn.functional.dropout(x_3, p=0.5)

            # Correction step
            # C1: Decide if the correction is happening
            if correction_samples[t] < self._probability_correction:
                # C2: calculate weights
                unscaled_visual_activations = self._gamma * y_raw_activation.squeeze()
                weights = torch.unsqueeze(nn.functional.softmax(unscaled_visual_activations, dim=-1), dim=-1)

                # C3: Calculate weighted memories
                angular_velocity_x_tilde = torch.sum(weights * (self._angular_velocity_memories*self.mask_1[-1]).T, dim=-2)
                angular_velocity_and_speed_x_tilde = torch.sum(
                    weights * (self._angular_velocity_and_speed_memories*self.mask_2[-1]).T, dim=-2
                )
                no_self_motion_x_tilde = torch.sum(weights * (self._no_self_motion_memories*self.mask_3[-1]).T, dim=-2)

                # C4: Apply corrections
                x_1, h_1 = self._lstm_angular_velocity_correction(angular_velocity_x_tilde, (x_1, h_1))
                out_1 = nn.functional.dropout(x_1, p=0.5)

                x_2, h_2 = self._lstm_angular_velocity_and_speed_correction(
                    angular_velocity_and_speed_x_tilde, (x_2, h_2)
                )
                out_2 = nn.functional.dropout(x_2, p=0.5)

                x_3, h_3 = self._lstm_no_self_motion_correction(no_self_motion_x_tilde, (x_3, h_3))
                out_3 = nn.functional.dropout(x_3, p=0.5)

            xs_1[:, t, :] = out_1
            xs_2[:, t, :] = out_2
            xs_3[:, t, :] = out_3

        # D: Calculate the predictions of the RNNs
        # D1: Calculate the rest of p_react
        p_react[:, start_t[-1]:, :] = nn.functional.softmax(self._calculate_activation(self._beta,
                                                                                       y_enc[:, start_t[-1]:, :],
                                                                                       self._visual_memories),
                                                            dim=-1)
        p_react_entropy = -torch.sum(p_react * torch.log(p_react + 1e-43), dim=-1).mean()
        self.update_beta(p_react_entropy)
        # D2: Calculate the predictions
        start_t.append(50)
        for i in range(len(start_t)-1):
            p_pred[:, start_t[i]:start_t[i+1], :] = nn.functional.softmax(
                self._calculate_activation(self._pi_angular_velocity,
                                        xs_1[:, start_t[i]:start_t[i+1], :],
                                        self._angular_velocity_memories*self.mask_1[i])
                + self._calculate_activation(self._pi_angular_velocity_and_speed,
                                            xs_2[:, start_t[i]:start_t[i+1], :],
                                            self._angular_velocity_and_speed_memories*self.mask_2[i])
                + self._calculate_activation(self._pi_no_self_motion,
                                            xs_3[:, start_t[i]:start_t[i+1], :],
                                            self._no_self_motion_memories*self.mask_3[i]),
                dim=-1
            )  # Batch X Sequence length X nb of slots

        # E: Calculate the loss
        loss = -torch.sum(torch.flatten(p_react, end_dim=1) * torch.log(torch.flatten(p_pred, end_dim=1)+1e-43), dim=-1).mean()

        if torch.isnan(loss):
            print("loss is NaN")
        self.log("train_loss", loss)
        self.log("batch", float(batch_idx))
        self.log('stored_memory_slots', len(self.slots_to_store))
        self.log('beta', self._beta)
        return loss

    @classmethod
    def _calculate_activation(
        cls, entropy_coeff: Union[float, torch.Tensor], activation_vector: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        return entropy_coeff * (activation_vector @ memory)

    def on_after_backward(self):
        with torch.no_grad():
            self._angular_velocity_memories.data = self._angular_velocity_memories*self.mask_1[-1]
            self._angular_velocity_and_speed_memories.data = self._angular_velocity_and_speed_memories*self.mask_2[-1]
            self._no_self_motion_memories.data = self._no_self_motion_memories*self.mask_3[-1]

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self._visual_memories, "lr": self._memory_slot_learning_rate},
                {"params": self._angular_velocity_memories, "lr": self._memory_slot_learning_rate},
                {"params": self._angular_velocity_and_speed_memories, "lr": self._memory_slot_learning_rate},
                {"params": self._no_self_motion_memories, "lr": self._memory_slot_learning_rate},
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

    def update_beta(self, p_react_entropy: torch.Tensor):
        """
        Gets the newly regulated parameter beta used to calculate the target memory reactivation.
        **This should be called after every trajectory**
        """
        beta_logit = np.log(self._beta)
        # Perhaps we could apply a certain rounding that defines when they are close enough for us not to change it?
        if p_react_entropy < self._entropy_reactivation_target:
            beta_logit -= 0.001
        elif p_react_entropy > self._entropy_reactivation_target:
            beta_logit += 0.001
        self._beta = float(np.exp(beta_logit))

    def fill_slot(self):
        """
        Chooses a memory slot at random and fills it with RNNs/VAE outputs
        """
        slot = np.random.choice(self._nb_memory_slots)


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

    if config["smp"]["saved_model_path"]:
        checkpoint_path = os.path.abspath(original_cwd + config["smp"]["saved_model_path"])
        smp = SpatialMemoryPipeline.load_from_checkpoint(
            checkpoint_path,
            batch_size=config["smp"]["batch_size"],
            learning_rate=config["smp"]["learning_rate"],
            memory_slot_learning_rate=config["smp"]["memory_slot_learning_rate"],
            auto_encoder=ae,
            beta=config["smp"]["beta"],
            entropy_reactivation_target=config["smp"]["entropy_reactivation_target"],
            memory_slot_size=config["vae"]["latent_dim"],
            nb_memory_slots=config["smp"]["nb_memory_slots"],
            probability_correction=config["smp"]["prob_correction"],
            probability_storage=config["smp"]["prob_storage"],
            hidden_size_RNN1=config["smp"]["hidden_size_RNN1"],
            hidden_size_RNN2=config["smp"]["hidden_size_RNN2"],
            hidden_size_RNN3=config["smp"]["hidden_size_RNN3"],
            sequence_length=config["smp"]["bptt_unroll_length"],
        )
    else:
        smp = SpatialMemoryPipeline(
            batch_size=config["smp"]["batch_size"],
            learning_rate=config["smp"]["learning_rate"],
            memory_slot_learning_rate=config["smp"]["memory_slot_learning_rate"],
            auto_encoder=ae,
            beta=config["smp"]["beta"],
            entropy_reactivation_target=config["smp"]["entropy_reactivation_target"],
            memory_slot_size=config["vae"]["latent_dim"],
            nb_memory_slots=config["smp"]["nb_memory_slots"],
            probability_correction=config["smp"]["prob_correction"],
            probability_storage=config["smp"]["prob_storage"],
            hidden_size_RNN1=config["smp"]["hidden_size_RNN1"],
            hidden_size_RNN2=config["smp"]["hidden_size_RNN2"],
            hidden_size_RNN3=config["smp"]["hidden_size_RNN3"],
            sequence_length=config["smp"]["bptt_unroll_length"],
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        filename="smp-{epoch:02d}-{train_loss:.3f}",
        save_top_k=3,
        save_last=True,
        mode="min",
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(
        accelerator=config["hardware"]["accelerator"],
        max_epochs=config["smp"]["max_epochs"],
        max_steps=config["smp"]["max_steps"],
        default_root_dir=original_cwd,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        profiler="simple",
        # detect_anomaly=True
    )
    trainer.fit(smp, datamodule=rat_sequence_data_module)
