import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ContinuousActions(gym.Wrapper):

    def __init__(self, env):
        self.action_space = spaces.Box(
            low=np.array([0,1]), high=np.array([-1,1]), shape=(2,)
        )

    def move_agent(self, speed):
        """
        Move the agent forward
        """

        next_pos = (
            self.agent.pos
            + self.agent.dir_vec * speed
        )

        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False

        self.agent.pos = next_pos

        return True

    def step(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        self.turn_agent(action[1])
        self.move_agent(action[0])

        # Generate the current camera image
        obs = self.render_obs()

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            termination = False
            truncation = True
            reward = 0
            return obs, reward, termination, truncation, {}

        reward = 0
        termination = False
        truncation = False

        return obs, reward, termination, truncation, {}
