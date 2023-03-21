import numpy as np
import gymnasium as gym

from utils.trajectory import generate_traj

import environ


def run_random_walk(time, seed):
    env = gym.make('MiniWorld-OpenField-v1', view="agent", render_mode="human")
    env.reset(seed=seed)

    pos = env.agent.pos
    direction = env.agent.dir

    ag, traj = generate_traj((np.array([pos[0], pos[2]]) - 0.5) / 10, -direction, time)

    for t in range(traj.shape[1] - 1):
        env.render()

        # Slow movement speed to minimize resets
        action = traj[:, t + 1]
        obs, reward, termination, truncation, info = env.step(action)

        if termination or truncation:
            env.reset()

    env.close()


if __name__ == "__main__":
    run_random_walk(600, 0)
