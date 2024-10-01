# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Orbit environments."""

"""Launch Isaac Sim Simulator first."""
from trainer import NN

import argparse
import numpy as np

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Orbit environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=1, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch

from omni.isaac.orbit.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.manipulation.lift import mdp
from omni.isaac.orbit_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.orbit_tasks.utils.parse_cfg import parse_env_cfg


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    assert (
        args_cli.task == "Isaac-Lift-Cube-Franka-IK-Rel-v0"
    ), "Only 'Isaac-Lift-Cube-Franka-IK-Rel-v0' is supported currently."
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)

    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False #这里是将观测值返回为字典而不是张量

    # add termination condition for reaching the goal otherwise the environment won't reset
    env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # TODO: load nn model
    model = NN() # load nn model
    model.load_state_dict(torch.load('model.pth')) # load model from file
    model.to("cuda")
    model.eval() 

    # reset environment
    obs_dict, _ = env.reset()
    # TODO: get action from nn model using obs_dict
    obs = obs_dict["policy"]
    act_pre = model.predict(self, obs) # action from nn model

    
    # TODO: 找到具体的action与act_pre的对应关系
    act_clow =   #从输出的act_pre转化为具体的action

    goal=obs_dict #obs_dict中goal的部分
    his_act = []
    his_obs = []


    while simulation_app.is_running():
        with torch.inference_mode():
            # perform action on environment
            obs_dict = env.step(actions)
            obs = obs_dict["policy"]

            obs = (8 - len(his_obs)) * [0] + his_obs[-8:]
            act = (8 - len(his_act)) * [0] + his_act[-8:]
            

            # TODO: get action from nn model using obs_dict
            act_pre = model.predict(self, goal, obs, act) # action from nn model
            act = act_pre.cpu().numpy()
            history_act = torch.cat([history_act[:, 1:], act_pre.view(1, 1, -1)], dim=1)
            
            
            his_obs.append(obs)
            his_act.append(actions)

            
            

            if env.unwrapped.sim.is_stopped():
                break

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
