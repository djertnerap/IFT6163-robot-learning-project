import os

import hydra
import numpy as np
import gymnasium as gym
from PIL import Image
from tqdm import tqdm

from utils.trajectory import generate_traj

import environ

def run_random_walk(time, seed, dataset_folder_path, img_size=64, save_traj=True):

    env = gym.make('MiniWorld-OpenField-v1',
                   view="agent",
                   render_mode="rgb_array",
                   obs_width=img_size,
                   obs_height=img_size,
                   window_width=img_size,
                   window_height=img_size)
    env.reset(seed=seed)

    imgdir = os.path.join(dataset_folder_path, 'Images')
    os.mkdir(imgdir)

    pos = env.agent.pos
    direction = env.agent.dir

    ag, traj = generate_traj((np.array([pos[0], pos[2]]) - 0.5) / 10, -direction, time)

    for t in tqdm(range(traj.shape[1])):
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