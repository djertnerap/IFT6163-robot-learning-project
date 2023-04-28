import functools
import os

import gymnasium as gym
import hydra
import numpy as np
from PIL import Image
from tqdm import tqdm

import environ
from utils.trajectory import generate_traj


def run_random_walk(time: int, seed: int, dataset_folder_path: str, traj_nb, img_size=64, save_traj=True, env=None, timescale=50):
    if env == None:
        env = gym.make(
            "MiniWorld-OpenField-v1",
            view="agent",
            render_mode="rgb_array",
            obs_width=img_size,
            obs_height=img_size,
            window_width=img_size,
            window_height=img_size,
        )
    env.reset(seed=seed)

    imgdir = os.path.join(dataset_folder_path, "Images")
    os.mkdir(imgdir)

    pos = env.agent.pos
    direction = env.agent.dir

    ag, traj = generate_traj((np.array([pos[0], pos[2]]) - 0.5) / 10, -direction, traj_nb, timescale)

    for t in tqdm(range(traj.shape[1]), desc=f"Render Images for trajectory #{traj_nb}"):
        # insert tqdm for image saving here
        Image.fromarray(env.render()).save(os.path.join(imgdir, f"{t}.png"))

        # Slow movement speed to minimize resets
        action = traj[:, t]
        obs, reward, termination, truncation, info = env.step(action)

        if termination or truncation:
            env.reset()

    if save_traj:
        np.save(os.path.join(dataset_folder_path, "traj.npy"), traj)

    env.close()
