#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse
import math

import gymnasium as gym
import miniworld
import pyglet
from pyglet.window import key

import environ

# import sys


parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="MiniWorld-OpenField-v0")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--no-time-limit", action="store_true", help="ignore time step limits")
parser.add_argument(
    "--top_view",
    action="store_true",
    help="show the top view instead of the agent view",
)
args = parser.parse_args()
view_mode = "top" if args.top_view else "agent"

environment = gym.make(args.env_name, view=view_mode, render_mode="human")

if args.no_time_limit:
    environment.max_episode_steps = math.inf
if args.domain_rand:
    environment.domain_rand = True

print("============")
print("Instructions")
print("============")
print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
print("============")

environment.reset()

# Create the display window
environment.render()


def step(action):
    print(
        "step {}/{}: {}".format(
            environment.step_count + 1, environment.max_episode_steps, environment.actions(action).name
        )
    )

    obs, reward, termination, truncation, info = environment.step(action)

    if reward > 0:
        print(f"reward={reward:.2f}")

    if termination or truncation:
        print("done!")
        environment.reset()

    environment.render()


@environment.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        environment.reset()
        environment.render()
        return

    if symbol == key.ESCAPE:
        environment.close()
        # sys.exit(0)

    if symbol == key.UP:
        step(environment.actions.move_forward)
    elif symbol == key.DOWN:
        step(environment.actions.move_back)

    elif symbol == key.LEFT:
        step(environment.actions.turn_left)
    elif symbol == key.RIGHT:
        step(environment.actions.turn_right)

    elif symbol == key.PAGEUP or symbol == key.P:
        step(environment.actions.pickup)
    elif symbol == key.PAGEDOWN or symbol == key.D:
        step(environment.actions.drop)

    elif symbol == key.ENTER:
        step(environment.actions.done)


@environment.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass


@environment.unwrapped.window.event
def on_draw():
    environment.render()


@environment.unwrapped.window.event
def on_close():
    pyglet.app.exit()


# Enter main event loop
pyglet.app.run()

environment.close()
