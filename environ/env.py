import math

from abc import ABC
from typing import Optional

import numpy as np
from gymnasium import spaces
from miniworld.entity import Agent, MeshEnt, Entity
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS

class Goal(Entity):
    def __init__(self, radius=0.1):
        super().__init__()
        self.radius = radius
    def render(self):
        pass

class Rat(Agent):
    def __init__(self):
        super().__init__()
        self.cam_height = 0.75
        self.radius = 0.2
        self.height = 0.9
        self.cam_fwd_disp = 0

    def randomize(self, *args):
        pass


class RatWorldEnv(MiniWorldEnv, ABC):
    def __init__(
        self,
        max_episode_steps: int = 12000,
        obs_width: int = 80,
        obs_height: int = 60,
        window_width: int = 800,
        window_height: int = 600,
        params=DEFAULT_PARAMS,
        domain_rand: bool = False,
        render_mode: Optional[str] = None,
        view: str = "agent",
    ):
        super().__init__(
            max_episode_steps=max_episode_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            window_width=window_width,
            window_height=window_height,
            params=params,
            domain_rand=domain_rand,
            render_mode=render_mode,
            view=view,
        )

    def reset(self, *, seed=None, options=None):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """
        super().reset(seed=seed)

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Rat()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []

        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.np_random if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"])

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max("forward_step")

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs, {}


class OpenField(RatWorldEnv):
    DEFAULT_DRIFT = 0
    """
    ## Description

    Room

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move_back                   |
    | 4   | pickup                      |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:



    ## Arguments

    ```python
    PickupObjects(size=22, continuous=False)
    ```

    `size`: size of world
    'continuous': True if action space should be continuous

    """

    def __init__(self, size=23, continuous=False, target=False, **kwargs):
        assert size >= 2
        self.size = size
        self.continuous = continuous
        self.target = target
        super().__init__(**kwargs)

        if continuous:
            self.action_space = spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), shape=(2,))

        else:
            # Reduce the action space
            self.action_space = spaces.Discrete(self.actions.pickup + 1)

    def _gen_world(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        # colorlist = list(COLOR_NAMES)

        self.place_entity(
            MeshEnt(mesh_name="building", height=20),
            pos=np.array([40, 0, 35]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="barrel", height=25),
            pos=np.array([-40, 0, 20]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="cone", height=25),
            pos=np.array([-30, 0, -20]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="duckie", height=25),
            pos=np.array([0, 0, 35]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="tree", height=25),
            pos=np.array([0, 0, -35]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="potion", height=25),
            pos=np.array([40, 0, -35]),
            dir=-math.pi,
        )

        self.place_entity(
            MeshEnt(mesh_name="office_chair", height=25),
            pos=np.array([40, 0, 12]),
            dir=-math.pi,
        )

        if self.target:
            self.goal = self.place_entity(Goal(), pos=np.array([5, 0, 5]))

        self.place_agent()

    def turn_agent_cont(self, turn_angle):
        """
        Turn the agent left or right
        """

        orig_dir = self.agent.dir

        self.agent.dir += turn_angle

        return True

    def move_agent_cont(self, speed):
        """
        Move the agent forward
        """

        next_pos = self.agent.pos + self.agent.dir_vec * speed

        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False

        self.agent.pos = next_pos

        return True

    def step(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        if self.continuous:
            self.turn_agent_cont(action[1])
            self.move_agent_cont(action[0])
        else:
            rand = self.np_random if self.domain_rand else None
            fwd_step = self.params.sample(rand, "forward_step")
            fwd_drift = self.params.sample(rand, "forward_drift")
            turn_step = self.params.sample(rand, "turn_step")

            if action == self.actions.move_forward:
                self.move_agent(fwd_step, fwd_drift)

            elif action == self.actions.move_back:
                self.move_agent(-fwd_step, fwd_drift)

            elif action == self.actions.turn_left:
                self.turn_agent(turn_step)

            elif action == self.actions.turn_right:
                self.turn_agent(-turn_step)

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            termination = False
            truncation = True
            reward = 0
            # return obs, reward, termination, truncation, {}
        # If the goal is reached
        elif self.target and self.near(self.goal):
            self.step_count = 0
            self.place_agent()
            reward = self._reward()
            termination = False
            truncation = False

        else:
            reward = 0
            termination = False
            truncation = False

        # Generate the current camera image
        obs = self.render_obs()

        return obs, reward, termination, truncation, {}
