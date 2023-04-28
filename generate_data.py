import faulthandler
import os.path
from pathlib import Path

import gymnasium as gym
import hydra
from omegaconf import DictConfig



def generate_data(cfg: DictConfig):
    # Generates n=logging.n_traj trajectories and saves them to the "data" folder

    faulthandler.enable()

    data_path = os.path.abspath(cfg.hardware.smp_dataset_folder_path)

    import environ
    env = gym.make(
        "MiniWorld-OpenField-v1",
        view="agent",
        render_mode="rgb_array",
        obs_width=cfg.env.img_size,
        obs_height=cfg.env.img_size,
        window_width=cfg.env.img_size,
        window_height=cfg.env.img_size,
    )
    for i in range(cfg.logging.n_traj):
        exp_name = cfg.logging.exp + "_" + str(i)
        logdir = os.path.join(data_path, exp_name)

        Path(logdir).mkdir(parents=True, exist_ok=True)

        seed = cfg.logging.seed + i

        from agents.walker import run_random_walk
        run_random_walk(
            time=cfg.smp.episode_len,
            seed=seed,
            dataset_folder_path=logdir,
            traj_nb=i,
            img_size=cfg.env.img_size,
            env=env,
            timescale=cfg.env.traj_timescale
        )
