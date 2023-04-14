import gymnasium as gym

from environ.env import OpenField

# Register the environment with OpenAI Gym
gym_id = "MiniWorld-OpenField-v0"
entry_point = "environ:OpenField"

gym.envs.registration.register(
    id=gym_id,
    entry_point=entry_point,
)

gym_id = "MiniWorld-OpenField-v1"
entry_point = "environ:OpenField"

gym.envs.registration.register(id=gym_id, entry_point=entry_point, kwargs={"continuous": True})
