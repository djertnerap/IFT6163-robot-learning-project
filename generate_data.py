import os.path

import hydra
from omegaconf import DictConfig

import environ
from agents.walker import run_random_walk


def generate_data(cfg: DictConfig):
    # Generates n=logging.n_traj trajectories and saves them to the "data" folder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), cfg.hardware.smp_dataset_folder_path)
    for i in range(cfg.logging.n_traj):
        exp_name = cfg.logging.exp + "_" + str(i)
        logdir = os.path.join(data_path, exp_name)

        if not os.path.exists(logdir):
            os.mkdir(logdir)

        seed = cfg.logging.seed + i

        run_random_walk(time=cfg.smp.episode_len, seed=seed, dataset_folder_path=logdir, img_size=cfg.env.img_size)


# @hydra.main(config_path="config", config_name="config")
# def main(cfg: DictConfig):
#     generate_data(cfg)
#
#
# if __name__ == "__main__":
#     main()
