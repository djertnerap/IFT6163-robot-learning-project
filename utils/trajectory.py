import numpy as np
import ratinabox
from ratinabox.Agent import Agent
from ratinabox.Environment import Environment
from ratinabox.utils import get_angle
from tqdm import tqdm


class Agent_New(Agent):
    def __init__(self, Environment, params={"wall_repel_distance": 0.2}):
        super().__init__(Environment, params)
        self.history["t"] = [0]
        self.history["pos"] = [self.pos]
        self.history["vel"] = [self.velocity]
        self.history["rot_vel"] = [self.rotational_velocity]
        self.history["speed"] = [np.linalg.norm(self.velocity)]
        self.history["rotation"] = [0]
        self.history["angle"] = [get_angle(self.velocity)]
        # self.history["rotation"] = []
        # self.history["speed"] = []

    def update(self, dt=None, drift_velocity=None, drift_to_random_strength_ratio=1):
        super().update(dt, drift_velocity, drift_to_random_strength_ratio)
        self.history["speed"].append(
            np.linalg.norm(np.array(self.history["pos"][-1]) - np.array(self.history["pos"][-2]))
        )

        angle_now = get_angle(np.array(self.history["pos"][-1]) - np.array(self.history["pos"][-2]))
        angle_before = self.history["angle"][-1]
        if abs(angle_now - angle_before) > np.pi:
            if angle_now > angle_before:
                angle_now -= 2 * np.pi
            elif angle_now < angle_before:
                angle_before -= 2 * np.pi
        self.history["rotation"].append(angle_now - angle_before)
        self.history["angle"].append(angle_now)
        return


def generate_traj(pos, direction, traj_nb, T=600):
    Env = Environment(params={"aspect": 1, "scale": 2.2})

    Ag = Agent_New(Env)
    Ag.pos = pos
    Ag.velocity = Ag.speed_std * np.array([np.cos(direction), np.sin(direction)])
    Ag.history["pos"] = [Ag.pos]
    Ag.history["vel"] = [Ag.velocity]
    Ag.history["speed"] = [np.linalg.norm(Ag.velocity)]
    Ag.history["angle"] = [get_angle(Ag.velocity)]
    Ag.speed_mean = 0.2

    dt = 50e-3

    for i in tqdm(range(int(T / dt)), desc=f"Render Updates for trajectory #{traj_nb}"):
        Ag.update(dt=dt)

    traj = np.vstack((np.array(Ag.history["speed"]) * 10, -np.array(Ag.history["rotation"])))

    return Ag, traj[:, 1:]
