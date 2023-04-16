import os.path
from pathlib import Path

import hydra
from omegaconf import DictConfig

import environ
from agents.walker import run_random_walk

import faulthandler


def generate_data(cfg: DictConfig):
    # Generates n=logging.n_traj trajectories and saves them to the "data" folder

    faulthandler.enable()

    data_path = os.path.abspath(hydra.utils.get_original_cwd() + cfg.hardware.smp_dataset_folder_path)
    for i in range(cfg.logging.n_traj):
        exp_name = cfg.logging.exp + "_" + str(i)
        logdir = os.path.join(data_path, exp_name)

        Path(logdir).mkdir(parents=True, exist_ok=True)

        seed = cfg.logging.seed + i

        run_random_walk(time=cfg.smp.episode_len, seed=seed, dataset_folder_path=logdir, traj_nb=i, img_size=cfg.env.img_size)


# @hydra.main(config_path="config", config_name="config")
# def main(cfg: DictConfig):
#     generate_data(cfg)
#
#
# if __name__ == "__main__":
#     main()
