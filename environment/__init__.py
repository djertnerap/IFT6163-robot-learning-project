import gymnasium as gym
from .env import OpenField


# Register the environment with OpenAI Gym
gym_id = "MiniWorld-OpenField-v0"
entry_point = "environment:OpenField"

gym.envs.registration.register(
    id=gym_id,
    entry_point=entry_point,
)
