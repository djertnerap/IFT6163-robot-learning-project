import os
from typing import Iterator, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from environ import OpenField
from roble.infrastructure.memory_optimized_replay_buffer import MemoryOptimizedReplayBuffer
from roble.policies.MLP_policy import ConcatMLP, MLPPolicyStochastic
from utils.trajectory import generate_traj

# TODO MENU
# Done: Test the for loop with ratinabox env
# Done: Iterate over both images and velocities
# Done: Do the multi model multi optimizer trick
# Done: Carry over SAC or other RL into module & train with dummy linear layer SMP
# TODO: Carry over SMP into this module
# TODO: Ensure that reward signal is good in the gym env for random target chosen at start of training
# TODO: Enable video generation from validation trajectories
# TODO: Reference homework code


class DummyAgent:
    def __init__(self, env: gym.Env, replay_buffer: MemoryOptimizedReplayBuffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self.last_obs, _ = env.reset(seed=1)
        self.last_velocities = np.zeros(2)

    def step_env(self):
        replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs, self.last_velocities)

        action = self.env.action_space.sample()
        self.last_velocities = action
        self.last_obs, reward, done, truncation, info = self.env.step(action)

        self.replay_buffer.store_effect(idx=replay_buffer_idx, action=action, reward=reward, done=done)


class RLDataset(IterableDataset):
    def __init__(self, replay_buffer: MemoryOptimizedReplayBuffer, sample_size: int = 2):
        self._replay_buffer = replay_buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        (visual_inputs, velocities), actions, rewards, new_states, dones = self._replay_buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield visual_inputs[i], velocities[i], actions[i], rewards[i], new_states[i], dones[i]


class SACWithSpatialMemoryPipeline(pl.LightningModule):
    def __init__(
        self,
        img_size: int,
        batch_size: int,
        learning_rate: float,
        bptt_unroll_len: int,
        sac_entropy_coeff: float,
        hidden_size_RNN1: int,
        hidden_size_RNN2: int,
        hidden_size_RNN3: int,
        policy_n_layers: int,
        policy_net_size: int,
        sac_gradient_clipping: float,
        target_update_freq: int,
        polyak_avg: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.env: OpenField = gym.make(
            "MiniWorld-OpenField-v1",
            view="agent",
            render_mode="rgb_array",
            obs_width=img_size,
            obs_height=img_size,
            window_width=img_size,
            window_height=img_size,
        )
        action_dim = 2

        self.replay_buffer = MemoryOptimizedReplayBuffer(
            size=1000, frame_history_len=bptt_unroll_len, continuous_actions=True, ac_dim=action_dim
        )
        self.dummy_agent = DummyAgent(env=self.env, replay_buffer=self.replay_buffer)
        self.dummy_critic = torch.nn.Linear(2, 1)
        self.dummy_smp = torch.nn.Linear(2, 1)
        self.warm_start_replay_buffer()

        # previous t-step reward, previous action, SMP output, SMP output when last reached goal
        observation_dim = 1 + 2 + 2 * (hidden_size_RNN1 + hidden_size_RNN2 + hidden_size_RNN3)
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
        )
        self.q_net2 = ConcatMLP(
            ac_dim=1,
            ob_dim=observation_dim + action_dim,
            n_layers=policy_n_layers,
            size=policy_net_size,
        )
        self.q_net_target = ConcatMLP(
            ac_dim=1,
            ob_dim=observation_dim + action_dim,
            n_layers=policy_n_layers,
            size=policy_net_size,
        )
        self.q_net_target2 = ConcatMLP(
            ac_dim=1,
            ob_dim=observation_dim + action_dim,
            n_layers=policy_n_layers,
            size=policy_net_size,
        )

    def warm_start_replay_buffer(self, steps: int = 15):
        last_obs, _ = self.env.reset()
        last_velocity = np.zeros(2)

        pos = self.env.agent.pos
        direction = self.env.agent.dir

        _, traj = generate_traj((np.array([pos[0], pos[2]]) - 0.5) / 10, -direction, 1, steps * 50e-3)
        for t in tqdm(range(traj.shape[1]), desc=f"Warm start replay buffer"):
            # Slow movement speed to minimize resets
            action = traj[:, t]
            replay_buffer_idx = self.replay_buffer.store_frame(last_obs, last_velocity)

            last_velocity = action
            last_obs, reward, termination, truncation, info = self.env.step(action)

            self.replay_buffer.store_effect(replay_buffer_idx, action, reward, termination)

            if termination or truncation:
                last_obs, _ = self.env.reset()

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        visual_obs, velocities, actions, reward, next_obs, done = batch

        self.dummy_agent.step_env()

        smp_predictions, next_smp_predictions = self._perform_smp_training_step(batch, batch_idx)
        self._sac_training_step(smp_predictions, next_smp_predictions, actions, reward, done, batch_idx)

    def _perform_smp_training_step(self, batch, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        smp_optimizer, _ = self.optimizers()
        visual_obs, velocities, actions, reward, next_obs, done = batch

        smp_loss = torch.sum(1 - self.dummy_smp(velocities))

        smp_optimizer.zero_grad()
        self.manual_backward(smp_loss)
        smp_optimizer.step()

    def _sac_training_step(self, observation, next_observation, actions, rewards, terminal, batch_idx):
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

        qa_tp1_values_reg = qa_tp1_values - (self.sac_entropy_coeff * action_distribution.log_prob(target_actions))

        target = rewards + self.gamma * qa_tp1_values_reg * (1 - terminal)
        target = target.detach()

        loss = self.loss(q_t_values, target)
        self.log("Q_net1_loss", loss)
        q_net_optimizer.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(
            q_net_optimizer, gradient_clip_val=self.hparams.sac_gradient_clipping, gradient_clip_algorithm="norm"
        )
        q_net_optimizer.step()

        loss2 = self.loss(q_t_values2, target)
        self.log("Q_net2_loss", loss)
        q_net2_optimizer.zero_grad()
        self.manual_backward(loss2)
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
                - self.entropy_coeff * action_distribution.log_prob(actions)
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
                self.replay_buffer, sample_size=self.hparams.batch_size
            ),  # This does not have to be batch_size to allow bigger epochs
            batch_size=self.hparams.batch_size,
        )

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.dummy_smp.parameters(), lr=self.hparams.learning_rate),  # SMP
            torch.optim.Adam(self.actor.parameters(), lr=self.hparams.learning_rate),
            torch.optim.Adam(self.q_net.parameters(), lr=self.hparams.learning_rate),
            torch.optim.Adam(self.q_net2.parameters(), lr=self.hparams.learning_rate),
        ]


def run_rl_experiment(config: DictConfig) -> None:
    """Run RL experiment."""
    original_cwd = hydra.utils.get_original_cwd()

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
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    trainer = pl.Trainer(
        max_epochs=5,
        max_steps=10,
        default_root_dir=original_cwd,
        logger=tb_logger,
        log_every_n_steps=1,
        profiler="simple",
    )

    trainer.fit(model)

    # data_dir = os.path.abspath(original_cwd + config["hardware"]["smp_dataset_folder_path"])
    # rat_sequence_data_module = SequencedDataModule(
    #     data_dir=data_dir,
    #     config=config,
    #     bptt_unroll_length=config["smp"]["bptt_unroll_length"],
    #     batch_size=config["smp"]["batch_size"],
    #     num_workers=config["hardware"]["num_data_loader_workers"],
    #     img_size=config["env"]["img_size"],
    # )
    #
    # checkpoint_path = os.path.abspath(original_cwd + config["smp"]["ae_checkpoint_path"])
    # ae = LitAutoEncoder.load_from_checkpoint(
    #     checkpoint_path, in_channels=config["vae"]["in_channels"], net_config=config["vae"]["net_config"].values()
    # )
    # ae.eval()
    # ae.freeze()
    #
    # # Need to handle the speed of the perpendicular direction of heading for RNN2
    # smp = SpatialMemoryPipeline(
    #     batch_size=config["rlsmp"]["batch_size"],
    #     learning_rate=config["rlsmp"]["learning_rate"],
    #     memory_slot_learning_rate=config["rlsmp"]["memory_slot_learning_rate"],
    #     auto_encoder=ae,
    #     entropy_reactivation_target=config["rlsmp"]["entropy_reactivation_target"],
    #     memory_slot_size=config["vae"]["latent_dim"],
    #     nb_memory_slots=config["rlsmp"]["nb_memory_slots"],
    #     probability_correction=config["rlsmp"]["prob_correction"],
    #     probability_storage=config["rlsmp"]["prob_storage"],
    #     hidden_size_RNN1=config["rlsmp"]["hidden_size_RNN1"],
    #     hidden_size_RNN2=config["rlsmp"]["hidden_size_RNN2"],
    #     hidden_size_RNN3=config["rlsmp"]["hidden_size_RNN3"],
    # )
    #
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.getcwd())
    # trainer = pl.Trainer(
    #     max_epochs=4, default_root_dir=original_cwd, logger=tb_logger, log_every_n_steps=1, profiler="simple"
    # )
    # trainer.fit(smp, datamodule=rat_sequence_data_module)

    # https://lightning-bolts.readthedocs.io/en/0.5.0/deprecated/models/reinforce_learn.html#soft-actor-critic-sac

    # # set seed
    # seed = config["hardware"]["seed"]
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    #
    # # set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # set up environment
    # env = gym.make(config["environment"]["name"])
    # env.seed(seed)
    # env = env.unwrapped
    # env = Monitor(env, directory=None, allow_early_resets=True)
    #
    # # set up agent
    # agent = Agent(
    #     state_size=env.observation_space.shape[0],
    #     action_size=env.action_space.n,
    #     seed=seed,
    #     config=config,
    # )
    #
    # # set up training
    # scores = train(agent, env, config)
    #
    # # save model
    # torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
    #
    # # plot scores
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(scores)), scores)
    # plt.ylabel("Score")
    # plt.xlabel("Episode #")
    # plt.show()
    #
    # # close environment
    # env.close()
