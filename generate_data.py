import os.path
import time

import hydra
import numpy as np
import gymnasium as gym
from PIL import Image
from tqdm import tqdm
from omegaconf import DictConfig

from agents.walker import run_random_walk

import environ

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):

    # Generates n=logging.n_traj trajectories and saves them to the "data" folder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    for i in range(cfg.logging.n_traj):
        exp_name = cfg.logging.exp + '_' + str(i)
        logdir = os.path.join(data_path, exp_name)
        img_size = cfg.env.img_size
        
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        traj_time = cfg.smp.episode_len
        seed = cfg.logging.seed + i

        run_random_walk(traj_time, seed, logdir, img_size)



if __name__ == "__main__":
    main()