import os.path

import hydra
import numpy as np
import gymnasium as gym
from PIL import Image
from tqdm import tqdm

from utils.trajectory import generate_traj

import environ


def run_random_walk(time, seed):
    hydra.initialize(config_path="config", job_name="rat_random_walk_dataset_generator", version_base=None)
    config = hydra.compose(config_name="config_test")
    dataset_folder_path = config["hardware"]["dataset_folder_path"]

    if not os.path.exists(dataset_folder_path):
        os.mkdir(dataset_folder_path)

    env = gym.make('MiniWorld-OpenField-v1',
                   view="agent",
                   render_mode="rgb_array",
                   obs_width=64,
                   obs_height=64,
                   window_width=64,
                   window_height=64)
    env.reset(seed=seed)

    pos = env.agent.pos
    direction = env.agent.dir

    ag, traj = generate_traj((np.array([pos[0], pos[2]]) - 0.5) / 10, -direction, time)

    for t in tqdm(range(traj.shape[1] - 1)):
        # insert tqdm for image saving here
        Image.fromarray(env.render()).save(os.path.join(dataset_folder_path, f"{t}.png"))

        # Slow movement speed to minimize resets
        action = traj[:, t + 1]
        obs, reward, termination, truncation, info = env.step(action)

        if termination or truncation:
            env.reset()

    env.close()


if __name__ == "__main__":
    run_random_walk(600, 0)
