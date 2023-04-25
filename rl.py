import os
from typing import Callable, Iterator, Tuple, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from torch.nn.functional import normalize
from lightning import pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from torchvision.utils import save_image

from environ import OpenField
from roble.infrastructure.memory_optimized_replay_buffer import MemoryOptimizedReplayBuffer
from roble.policies.MLP_policy import ConcatMLP, MLPPolicyStochastic
from utils.trajectory import generate_traj
from vae import LitAutoEncoder

# TODO MENU
# Done: Test the for loop with ratinabox env
# Done: Iterate over both images and velocities
# Done: Do the multi model multi optimizer trick
# Done: Carry over SAC or other RL into module & train with dummy linear layer SMP
# Done: Carry over SMP into this module
# Done: Edit Replay buffer to have observations at t+1 for velocities
# Done: Calculate the SMP output for the current and future batch as the input to the RL train batch
# Done: ensure that we are using the policy's actions to generate data
# TODO: Test the training loop
# TODO: Ensure that reward signal is good in the gym env for random target chosen at start of training
# TODO: Enable video generation from validation trajectories
# TODO: Reference homework code


class RLDataset(IterableDataset):
    def __init__(self, replay_buffer: MemoryOptimizedReplayBuffer, sample_size: int = 2):
        self._replay_buffer = replay_buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        (
            (visual_inputs, velocities),
            actions,
            rewards,
            (new_visual_inputs, new_velocities),
            dones,
        ) = self._replay_buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield visual_inputs[i].astype(np.float32), velocities[i], actions[i], rewards[i], new_visual_inputs[
                i
            ].astype(np.float32), new_velocities[i], dones[i]


class SACWithSpatialMemoryPipeline(pl.LightningModule):
    def __init__(
        self,
        # Shared param
        img_size: int,
        batch_size: int,
        learning_rate: float,
        # SAC params
        bptt_unroll_len: int,
        sac_entropy_coeff: float,
        policy_n_layers: int,
        policy_net_size: int,
        sac_gradient_clipping: float,
        target_update_freq: int,
        polyak_avg: float,
        sac_gamma: float,
        # SMP params
        hidden_size_RNN1: int,
        hidden_size_RNN2: int,
        hidden_size_RNN3: int,
        memory_slot_learning_rate: float,
        auto_encoder: LitAutoEncoder,
        beta: float,
        entropy_reactivation_target: float,
        memory_slot_size: int,
        nb_memory_slots: int,
        probability_correction: float,
        probability_storage: float,
        replay_buffer_size: int,
        max_steps: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["auto_encoder"])
        self.automatic_optimization = False

        self.env: OpenField = gym.make(
            "MiniWorld-OpenField-Goal-v1",
            view="agent",
            render_mode="rgb_array",
            obs_width=img_size,
            obs_height=img_size,
            window_width=img_size,
            window_height=img_size,
        )
        action_dim = 2

        self.replay_buffer = MemoryOptimizedReplayBuffer(
            size=replay_buffer_size, frame_history_len=bptt_unroll_len, continuous_actions=True, ac_dim=action_dim
        )

        self.warm_start_replay_buffer(batch_size * bptt_unroll_len + batch_size)

        # Live loop parameters
        self.last_obs, _ = self.env.reset(seed=1)
        self.last_obs = self.last_obs / 255
        self.last_velocities = np.zeros(2)

        self.x_1, self.h_1 = torch.zeros((hidden_size_RNN1,), device="cuda"), torch.zeros(
            (hidden_size_RNN1,), device="cuda"
        )
        self.x_2, self.h_2 = torch.zeros((hidden_size_RNN2,), device="cuda"), torch.zeros(
            (hidden_size_RNN2,), device="cuda"
        )
        self.x_3, self.h_3 = torch.zeros((hidden_size_RNN3,), device="cuda"), torch.zeros(
            (hidden_size_RNN3,), device="cuda"
        )

        # SAC models
        # previous t-step reward, previous action, SMP output, SMP output when last reached goal
        # observation_dim = 1 + 2 + 2 * nb_memory_slots

        # For now, don't include the previous t-step reward and action
        observation_dim = 2 * (hidden_size_RNN1+hidden_size_RNN2+hidden_size_RNN3)
        self.actor = MLPPolicyStochastic(
            entropy_coeff=sac_entropy_coeff,
            ac_dim=action_dim,
            ob_dim=observation_dim,
            n_layers=policy_n_layers,
            size=policy_net_size,
        )
        self.q_net = ConcatMLP(
            ac_dim=1,
            ob_dim=observation_dim + action_dim,
            n_layers=policy_n_layers,
            size=policy_net_size,
            deterministic=True,
        )
        self.q_net2 = ConcatMLP(
            ac_dim=1,
            ob_dim=observation_dim + action_dim,
            n_layers=policy_n_layers,
            size=policy_net_size,
            deterministic=True,
        )
        self.q_net_target = ConcatMLP(
            ac_dim=1,
            ob_dim=observation_dim + action_dim,
            n_layers=policy_n_layers,
            size=policy_net_size,
            deterministic=True,
        )
        self.q_net_target2 = ConcatMLP(
            ac_dim=1,
            ob_dim=observation_dim + action_dim,
            n_layers=policy_n_layers,
            size=policy_net_size,
            deterministic=True,
        )
        self.q_net_loss = nn.SmoothL1Loss()
        self.rewards=[]

        self._last_success_smp_output = torch.zeros(size=(hidden_size_RNN1+hidden_size_RNN2+hidden_size_RNN3,), device="cuda")

        # SMP models
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

        self._visual_memories = nn.Parameter(torch.rand(size=(memory_slot_size, nb_memory_slots)), requires_grad=False)
        self._gamma = nn.Parameter(torch.rand((1,)))
        self._beta = beta

        self._last_y_enc = None
        self.slots_to_store = []

    def warm_start_replay_buffer(self, steps: int = 15):
        last_obs, _ = self.env.reset()
        last_velocity = np.zeros(2)

        pos = self.env.agent.pos
        direction = self.env.agent.dir

        _, traj = generate_traj((np.array([pos[0], pos[2]]) - 0.5) / 10, -direction, 1, steps * 50e-3)
        for t in tqdm(range(traj.shape[1]), desc=f"Warm start replay buffer"):
            # Slow movement speed to minimize resets
            action = traj[:, t]
            replay_buffer_idx = self.replay_buffer.store_frame(last_obs / 255, last_velocity)

            last_velocity = action
            last_obs, reward, termination, truncation, info = self.env.step(action)

            self.replay_buffer.store_effect(replay_buffer_idx, action, reward, termination)

            if termination or truncation:
                last_obs, _ = self.env.reset()

    @classmethod
    def _calculate_activation(
        cls, entropy_coeff: Union[float, torch.Tensor], activation_vector: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        return entropy_coeff * (activation_vector @ memory)

    def update_beta(self, p_react_entropy: torch.Tensor):
        """
        Gets the newly regulated parameter beta used to calculate the target memory reactivation.
        **This should be called after every trajectory**
        """
        beta_logit = np.log(self._beta)
        # Perhaps we could apply a certain rounding that defines when they are close enough for us not to change it?
        if p_react_entropy < self.hparams.entropy_reactivation_target:
            beta_logit -= 0.001
        elif p_react_entropy > self.hparams.entropy_reactivation_target:
            beta_logit += 0.001
        self._beta = float(np.exp(beta_logit))

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        visual_obs, velocities, actions, reward, next_visual_obs, next_velocities, done = batch

        # Generate data with current policy
        self._take_env_step()

        # Train SMP
        self._perform_smp_training_step(batch, batch_idx)

        # Train SAC agent
        last_success_smp_output = torch.unsqueeze(self._last_success_smp_output, dim=0).expand(
            self.hparams.batch_size, -1
        )
        smp_predictions = self._smp_prediction(visual_obs, velocities)  # Take most up-to-date prediction
        obs = torch.concat([smp_predictions, last_success_smp_output], dim=1)
        next_smp_predictions = self._smp_prediction(next_visual_obs, next_velocities)
        next_obs = torch.concat([next_smp_predictions, last_success_smp_output], dim=1)
        self._sac_training_step(obs, next_obs, actions, reward, done, batch_idx)

    def _take_env_step(self):
        replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs, self.last_velocities)

        # SMP prediction
        last_obs = torch.from_numpy(self.last_obs).float().to(self.device)
        visual_input = torch.unsqueeze(torch.permute(last_obs, dims=(2, 0, 1)), dim=0)
        y_enc = self._auto_encoder.encode(visual_input)
        y_raw_activation = y_enc @ self._visual_memories

        last_velocities = torch.from_numpy(self.last_velocities).float().to(self.device)
        velocities = torch.squeeze(
            self._prepare_velocities(torch.unsqueeze(torch.unsqueeze(last_velocities, dim=0), dim=0))
        )

        self.x_1, self.h_1 = self._lstm_angular_velocity(
            velocities[:2], (self.x_1, self.h_1)
        )  # x: Batch X 1 X encoding dimension
        self.x_2, self.h_2 = self._lstm_angular_velocity_and_speed(velocities, (self.x_2, self.h_2))
        self.x_3, self.h_3 = self._lstm_no_self_motion(torch.ones(size=(1,), device=self.device), (self.x_3, self.h_3))

        # Correction step
        # C1: Decide if the correction is happening
        if np.random.random() < self.hparams.probability_correction:
            # C2: calculate weights
            unscaled_visual_activations = self._gamma * y_raw_activation.squeeze()
            weights = torch.unsqueeze(nn.functional.softmax(unscaled_visual_activations, dim=-1), dim=-1)

            # C3: Calculate weighted memories
            angular_velocity_x_tilde = torch.sum(weights * self._angular_velocity_memories.T, dim=-2)
            angular_velocity_and_speed_x_tilde = torch.sum(weights * self._angular_velocity_and_speed_memories.T, dim=-2)
            no_self_motion_x_tilde = torch.sum(weights * self._no_self_motion_memories.T, dim=-2)

            # C4: Apply corrections
            self.x_1, self.h_1 = self._lstm_angular_velocity_correction(angular_velocity_x_tilde, (self.x_1, self.h_1))
            self.x_2, self.h_2 = self._lstm_angular_velocity_and_speed_correction(
                angular_velocity_and_speed_x_tilde, (self.x_2, self.h_2)
            )
            self.x_3, self.h_3 = self._lstm_no_self_motion_correction(no_self_motion_x_tilde, (self.x_3, self.h_3))

        output = torch.concat([self.x_1, self.x_2, self.x_3])
        policy_input = torch.concat([output, self._last_success_smp_output])

        action = torch.squeeze(self.actor.get_action(torch.unsqueeze(policy_input, dim=0)))
        self.last_velocities = action.cpu().detach().numpy()
        self.last_obs, reward, done, truncation, info = self.env.step(self.last_velocities)
        self.last_obs = self.last_obs / 255

        if reward:
            self._last_success_smp_output = output

        self.replay_buffer.store_effect(idx=replay_buffer_idx, action=self.last_velocities, reward=reward, done=done)
        self.log("Last_reward", float(reward))
        self.rewards.append(float(reward))
        self.log('Average_reward', np.mean(self.rewards[-300:]))
        self.log('Sum_reward', np.sum(self.rewards))

    def _perform_smp_training_step(self, batch, batch_idx: int):
        visual_input, velocities, _, _, _, _, _ = batch
        visual_input = torch.permute(visual_input, dims=(0, 1, 4, 2, 3))

        # A: Calculate the target memories
        # A1: encode observations
        y_enc = self._auto_encoder.encode(torch.flatten(visual_input, start_dim=0, end_dim=1)).unflatten(
            dim=0, sizes=visual_input.shape[:2]
        )  # Batch X Sequence length X encoding dimension

        # A3: Prepare data
        velocities = self._prepare_velocities(velocities)

        # B: For loop
        p_react, p_pred = self._calculate_p(velocities, y_enc, smp=True)[:2]

        # E: Calculate the loss
        loss = -torch.sum(torch.flatten(p_react, end_dim=1) * torch.log(torch.flatten(p_pred, end_dim=1)+1e-43), dim=-1).mean()
        self.log("SMP_loss", loss)
        self.log('stored_memory_slots', len(self.slots_to_store))
        self.log('beta', self._beta)

        smp_optimizer = self.optimizers()[0]
        smp_optimizer.zero_grad()
        self.manual_backward(loss)
        smp_optimizer.step()

        # P_storage
        with torch.no_grad():
            self._angular_velocity_memories.data = self._angular_velocity_memories*self.mask_1[-1]
            self._angular_velocity_and_speed_memories.data = self._angular_velocity_and_speed_memories*self.mask_2[-1]
            self._no_self_motion_memories.data = self._no_self_motion_memories*self.mask_3[-1]

    def _smp_prediction(self, visual_input: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        visual_input = torch.permute(visual_input, dims=(0, 1, 4, 2, 3))
        y_enc = self._auto_encoder.encode(torch.flatten(visual_input, start_dim=0, end_dim=1)).unflatten(
            dim=0, sizes=visual_input.shape[:2]
        )
        velocities = self._prepare_velocities(velocities)
        self.eval()
        out = self._calculate_p(velocities, y_enc, smp=False)[2:]
        self.train()
        return torch.concat(out, dim=1)

    def _prepare_velocities(self, velocities: torch.Tensor) -> torch.Tensor:
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
        return velocities

    def _calculate_p(self, velocities: torch.Tensor, y_enc: torch.Tensor, smp) -> torch.Tensor:

        # Initialize P distributions
        # Batch X Sequence length X nb of slots
        p_react = torch.zeros(size=[self.hparams.batch_size, self.hparams.bptt_unroll_len, self.hparams.nb_memory_slots], device=self.device)
        p_pred = torch.zeros(size=[self.hparams.batch_size, self.hparams.bptt_unroll_len, self.hparams.nb_memory_slots], device=self.device)

        x_1, h_1 = torch.zeros(
            (self.hparams.batch_size, self.hparams.hidden_size_RNN1), device=self.device
        ), torch.zeros((self.hparams.batch_size, self.hparams.hidden_size_RNN1), device=self.device)
        x_2, h_2 = torch.zeros(
            (self.hparams.batch_size, self.hparams.hidden_size_RNN2), device=self.device
        ), torch.zeros((self.hparams.batch_size, self.hparams.hidden_size_RNN2), device=self.device)
        x_3, h_3 = torch.zeros(
            (self.hparams.batch_size, self.hparams.hidden_size_RNN3), device=self.device
        ), torch.zeros((self.hparams.batch_size, self.hparams.hidden_size_RNN3), device=self.device)

        xs_1 = torch.zeros(size=list(velocities.shape[:2]) + [self.hparams.hidden_size_RNN1], device=self.device)
        xs_2 = torch.zeros(size=list(velocities.shape[:2]) + [self.hparams.hidden_size_RNN2], device=self.device)
        xs_3 = torch.zeros(size=list(velocities.shape[:2]) + [self.hparams.hidden_size_RNN3], device=self.device)
        y_raw_activation = torch.zeros(size=list(velocities.shape[:2]) + [self.hparams.memory_slot_size], device=self.device)

        self.mask_1 = [torch.ones(size=[self.hparams.hidden_size_RNN1, self.hparams.nb_memory_slots], device=self.device)]
        self.mask_2 = [torch.ones(size=[self.hparams.hidden_size_RNN2, self.hparams.nb_memory_slots], device=self.device)]
        self.mask_3 = [torch.ones(size=[self.hparams.hidden_size_RNN3, self.hparams.nb_memory_slots], device=self.device)]

        seq_len = velocities.shape[1]
        correction_samples = np.random.random(size=(seq_len,))
        storage_samples = np.random.random(size=(seq_len,))
        storage_samples[0] = 1
        start_t = [0]

        for t in range(velocities.shape[1]):
            # P_storage
            if (storage_samples[t] < self.hparams.probability_storage) and smp:

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

                self.slots_to_store.append(torch.randperm(self.hparams.nb_memory_slots)[0])

                batch_idx = torch.randperm(self.hparams.batch_size)[0]

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
            x_3, h_3 = self._lstm_no_self_motion(
                torch.ones(size=(self.hparams.batch_size, 1), device=self.device), (x_3, h_3)
            )
            
            out_1 = nn.functional.dropout(x_1, p=0.5)
            out_2 = nn.functional.dropout(x_2, p=0.5)
            out_3 = nn.functional.dropout(x_3, p=0.5)

            # Correction step
            # C1: Decide if the correction is happening
            if correction_samples[t] < self.hparams.probability_correction:
                # C2: calculate weights
                unscaled_visual_activations = self._gamma * y_raw_activation.squeeze()
                weights = torch.unsqueeze(nn.functional.softmax(unscaled_visual_activations, dim=-1), dim=-1)

                # C3: Calculate weighted memories
                angular_velocity_x_tilde = torch.sum(weights * (self._angular_velocity_memorie*self.mask_1[-1]).T, dim=-2)
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
        if smp:
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
            return p_react, p_pred
        
        return xs_1[:, -1, :], xs_2[:, -1, :], xs_3[:, -1, :]

    def _sac_training_step(self, observation, next_observation, actions, rewards, terminal, batch_idx):
        """
        observation: Batch X Nb memory slots
        """

        # Critic Update
        self._train_critic(observation, next_observation, actions, rewards, terminal)
        # Actor Update
        self._train_actor(observation)

        if batch_idx % self.hparams.target_update_freq == 0:
            self._update_target_network()

    def _train_critic(self, observation, next_observation, actions, rewards, terminal):
        _, _, q_net_optimizer, q_net2_optimizer = self.optimizers()

        qa_t_values = self.q_net(observation, actions)
        q_t_values = torch.squeeze(qa_t_values)

        qa_t_values2 = self.q_net2(observation, actions)
        q_t_values2 = torch.squeeze(qa_t_values2)

        action_distribution = self.actor(next_observation)
        target_actions = action_distribution.rsample()
        qa_tp1_values1 = self.q_net_target(next_observation, target_actions)
        qa_tp1_values2 = self.q_net_target2(next_observation, target_actions)
        qa_tp1_values = torch.squeeze(torch.minimum(qa_tp1_values1, qa_tp1_values2))

        qa_tp1_values_reg = qa_tp1_values - (
            self.hparams.sac_entropy_coeff * action_distribution.log_prob(target_actions)
        )

        target = rewards + self.hparams.sac_gamma * qa_tp1_values_reg * (1 - terminal)
        target = target.detach()

        loss = self.q_net_loss(q_t_values, target)
        self.log("Q_net1_loss", loss)
        q_net_optimizer.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        self.clip_gradients(
            q_net_optimizer, gradient_clip_val=self.hparams.sac_gradient_clipping, gradient_clip_algorithm="norm"
        )
        q_net_optimizer.step()

        loss2 = self.q_net_loss(q_t_values2, target)
        self.log("Q_net2_loss", loss)
        q_net2_optimizer.zero_grad()
        self.manual_backward(loss2, retain_graph=True)
        self.clip_gradients(
            q_net2_optimizer, gradient_clip_val=self.hparams.sac_gradient_clipping, gradient_clip_algorithm="norm"
        )
        q_net2_optimizer.step()

    def _train_actor(self, observation):
        _, actor_optimizer, _, _ = self.optimizers()

        action_distribution = self.actor.forward(observation)
        actions = action_distribution.rsample()
        q_values1 = self.q_net(observation, actions)
        q_values2 = self.q_net2(observation, actions)
        loss = torch.mean(
            -(
                torch.squeeze(torch.minimum(q_values1, q_values2))
                - self.hparams.sac_entropy_coeff * action_distribution.log_prob(actions)
            )
        )
        self.log("Actor_loss", loss)
        actor_optimizer.zero_grad()
        self.manual_backward(loss)
        actor_optimizer.step()

    def _update_target_network(self):
        self._apply_polyak_avg(self.q_net_target, self.q_net)
        self._apply_polyak_avg(self.q_net_target2, self.q_net2)

    def _apply_polyak_avg(self, target_net_params: nn.Module, net_params: nn.Module):
        for target_param, param in zip(target_net_params.parameters(), net_params.parameters()):
            ## Perform Polyak averaging
            target_param.data.copy_(self.hparams.polyak_avg * target_param + (1 - self.hparams.polyak_avg) * param)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=RLDataset(
                self.replay_buffer, sample_size=self.hparams.batch_size * self.hparams.bptt_unroll_len
            ),  # This does not have to be batch_size to allow bigger epochs
            batch_size=self.hparams.batch_size,
        )

    def configure_optimizers(self):
        return [
            torch.optim.Adam(
                [
                    {"params": self._visual_memories, "lr": self.hparams.memory_slot_learning_rate},
                    {"params": self._angular_velocity_memories, "lr": self.hparams.memory_slot_learning_rate},
                    {"params": self._angular_velocity_and_speed_memories, "lr": self.hparams.memory_slot_learning_rate},
                    {"params": self._no_self_motion_memories, "lr": self.hparams.memory_slot_learning_rate},
                    {"params": self._pi_no_self_motion},
                    {"params": self._pi_angular_velocity},
                    {"params": self._pi_angular_velocity_and_speed},
                    {"params": self._gamma},
                    {"params": self._lstm_angular_velocity.parameters()},
                    {"params": self._lstm_angular_velocity_and_speed.parameters()},
                    {"params": self._lstm_no_self_motion.parameters()},
                ],
                lr=self.hparams.learning_rate,
            ),
            torch.optim.Adam(self.actor.parameters(), lr=self.hparams.learning_rate),
            torch.optim.Adam(self.q_net.parameters(), lr=self.hparams.learning_rate),
            torch.optim.Adam(self.q_net2.parameters(), lr=self.hparams.learning_rate),
        ]


def run_rl_experiment(config: DictConfig) -> None:
    """Run RL experiment."""
    original_cwd = hydra.utils.get_original_cwd()

    checkpoint_path = os.path.abspath(original_cwd + config["rlsmp"]["ae_checkpoint_path"])
    ae = LitAutoEncoder.load_from_checkpoint(
        checkpoint_path, in_channels=config["vae"]["in_channels"], net_config=config["vae"]["net_config"].values()
    )
    ae.eval()
    ae.freeze()

    model = SACWithSpatialMemoryPipeline(
        img_size=config["env"]["img_size"],
        batch_size=config["rlsmp"]["batch_size"],
        learning_rate=config["rlsmp"]["learning_rate"],
        bptt_unroll_len=config["rlsmp"]["bptt_unroll_length"],
        sac_entropy_coeff=config["rlsmp"]["sac_entropy_coeff"],
        hidden_size_RNN1=config["rlsmp"]["hidden_size_RNN1"],
        hidden_size_RNN2=config["rlsmp"]["hidden_size_RNN2"],
        hidden_size_RNN3=config["rlsmp"]["hidden_size_RNN3"],
        policy_n_layers=config["rlsmp"]["policy_n_layers"],
        policy_net_size=config["rlsmp"]["policy_net_size"],
        sac_gradient_clipping=config["rlsmp"]["sac_grad_norm_clipping"],
        target_update_freq=config["rlsmp"]["target_update_freq"],
        polyak_avg=config["rlsmp"]["polyak_avg"],
        sac_gamma=config["rlsmp"]["sac_gamma"],
        memory_slot_learning_rate=config["rlsmp"]["memory_slot_learning_rate"],
        auto_encoder=ae,
        beta=config["rlsmp"]["beta"],
        entropy_reactivation_target=config["rlsmp"]["entropy_reactivation_target"],
        memory_slot_size=config["vae"]["latent_dim"],
        nb_memory_slots=config["rlsmp"]["nb_memory_slots"],
        probability_correction=config["rlsmp"]["prob_correction"],
        probability_storage=config["rlsmp"]["prob_storage"],
        replay_buffer_size=config["rlsmp"]["replay_buffer_size"],
        max_steps=config["rlsmp"]["max_steps"],
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(
        # max_epochs=5,
        max_steps=config["rlsmp"]["max_steps"],
        default_root_dir=original_cwd,
        logger=tb_logger,
        log_every_n_steps=1,
        profiler="simple",
    )

    trainer.fit(model)
