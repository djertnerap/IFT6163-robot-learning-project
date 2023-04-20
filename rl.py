import os
from typing import Iterator, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from environ import OpenField
from memory_optimized_replay_buffer import MemoryOptimizedReplayBuffer
from rat_dataset import SequencedDataModule
from utils.trajectory import generate_traj
from vae import LitAutoEncoder

# TODO MENU
# Done: Test the for loop with ratinabox env
# TODO: Do the multi model multi optimizer trick
# TODO: Carry over SAC or other RL into module & train with dummy linear layer SMP
# TODO: Carry over SMP into this module
# TODO: Ensure that reward signal is good in the gym env for random target chosen at start of training
# TODO: Enable video generation from validation trajectories
# TODO: Reference homework code


class DummyAgent:
    def __init__(self, env: gym.Env, replay_buffer: MemoryOptimizedReplayBuffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self.last_obs, _ = env.reset(seed=1)

    def step_env(self):
        replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        action = self.env.action_space.sample()
        last_obs, reward, done, truncation, info = self.env.step(action)

        self.replay_buffer.store_effect(idx=replay_buffer_idx, action=action, reward=reward, done=done)


class RLDataset(IterableDataset):
    def __init__(self, replay_buffer: MemoryOptimizedReplayBuffer, sample_size: int = 2):
        self._replay_buffer = replay_buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, new_states, dones = self._replay_buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class SACWithSpatialMemoryPipeline(pl.LightningModule):
    def __init__(self, img_size: int, batch_size: int, learning_rate: float, bptt_unroll_len: int):
        super().__init__()
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._bptt_unroll_len = bptt_unroll_len

        self.save_hyperparameters()

        self.env: OpenField = gym.make(
            "MiniWorld-OpenField-v1",
            view="agent",
            render_mode="rgb_array",
            obs_width=img_size,
            obs_height=img_size,
            window_width=img_size,
            window_height=img_size,
        )

        self.replay_buffer = MemoryOptimizedReplayBuffer(
            size=1000, frame_history_len=bptt_unroll_len, continuous_actions=True, ac_dim=2
        )
        self.dummy_agent = DummyAgent(env=self.env, replay_buffer=self.replay_buffer)
        self.dummy_critic = torch.nn.Linear(2, 3)
        self.warm_start_replay_buffer()

    def warm_start_replay_buffer(self, steps: int = 15):
        last_obs, _ = self.env.reset()

        pos = self.env.agent.pos
        direction = self.env.agent.dir

        _, traj = generate_traj((np.array([pos[0], pos[2]]) - 0.5) / 10, -direction, 1, steps * 50e-3)
        for t in tqdm(range(traj.shape[1]), desc=f"Warm start replay buffer"):
            # Slow movement speed to minimize resets
            action = traj[:, t]
            replay_buffer_idx = self.replay_buffer.store_frame(last_obs)

            last_obs, reward, termination, truncation, info = self.env.step(action)

            self.replay_buffer.store_effect(replay_buffer_idx, action, reward, termination)

            if termination or truncation:
                last_obs, _ = self.env.reset()

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        obs, actions, reward, next_obs, done = batch
        self.dummy_agent.step_env()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=RLDataset(
                self.replay_buffer, sample_size=self._batch_size
            ),  # This does not have to be batch_size to allow bigger epochs
            batch_size=self._batch_size,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)


def run_rl_experiment(config: DictConfig) -> None:
    """Run RL experiment."""
    original_cwd = hydra.utils.get_original_cwd()

    model = SACWithSpatialMemoryPipeline(
        img_size=config["env"]["img_size"],
        batch_size=config["rlsmp"]["batch_size"],
        learning_rate=config["rlsmp"]["learning_rate"],
        bptt_unroll_len=config["rlsmp"]["bptt_unroll_length"],
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
