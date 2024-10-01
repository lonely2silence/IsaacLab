# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/05_controllers/ik_control.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.") #franka_panda, ur10
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")#此处直接修改机器人个数
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.controllers import DifferentialIKController, DifferentialIKControllerCfg #逆运动学控制器
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.orbit_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG, COBOTTA_CFG# isort:skip #这里是机器人的配置文件，新建“cobotta”

from get_data import controller_test

@configclass
class TableTopSceneCfg(InteractiveSceneCfg): #这里是场景的配置文件
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg( #这里是地面的配置文件
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg( #这里是灯光的配置文件
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg( #这里機器人底座下的桌子的配置文件
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)   
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") #替换机器人的配置文件
    elif args_cli.robot == "cobotta":
        robot = COBOTTA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers#作用尚不清楚，感觉不太重要 
    frame_marker_cfg = FRAME_MARKER_CFG.copy() 
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current")) #这里是显示的末端位置
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")) #这里是显示的目标位置

    # Define goals for the arm  #这里使用的定义是位置（3）加四元数（4）（姿态）#将手柄获得的这些数加到这里面
    ee_goals = [
        [0.4, 0, 0.3, 0.7071, 0, 0.7071, 0]
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # print(ee_goals) #后面多了个device=cuda
    # Track the given command
    current_goal_idx = 0 #当存在多个目标位置时，用于循环
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx] #将目标位置赋值给ik_commands

    # Specify robot-specific parameters  #
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"]) #设置机器人的关节的末端
    elif args_cli.robot == "cobotta":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["onrobot_rg6_base_link"]) #设置机器人的关节的末端
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene) 
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians. 
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # # Simulation loop
    #  # reset joint state
    # joint_pos = robot.data.default_joint_pos.clone() #设为原始预备状态
    # joint_vel = robot.data.default_joint_vel.clone()
    # robot.write_joint_state_to_sim(joint_pos, joint_vel) #将关节状态写入仿真
    # robot.reset()
    
    # diff = controller_test.main()#insert   
    # print("diff:",diff)
    # time.sleep(1)
    
    # diff = torch.tensor(diff, dtype=torch.float32)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # diff = diff.to(device) 

    while simulation_app.is_running():
        # reset
        if count % 1000 == 0: #当开始本次循环时，设定初始位置和目标位置
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()#设置回了原始状态00000
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]#循环至下一个目标位置

            # diff = controller_test.main()#insert  
            # print("diff:",diff)
            # time.sleep(0.5) #等待一段时间
            # diff = torch.tensor(diff, dtype=torch.float32)
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # diff = diff.to(device)

            # ik_commands = ik_commands+diff
            # print(ik_commands)
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset() #重置控制器
            diff_ik_controller.set_command(ik_commands)
            # change goal
            #current_goal_idx = (current_goal_idx + 1) % len(ee_goals) 
        else:
            # obtain quantities from simulation 
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            #看看jacobi矩阵
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids] 
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos) #通过末端位置和四元数，雅可比矩阵，当前关节位置计算关节位置

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids) #应用需要的关节位置
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7] #获取当前的末端位置
        # update marker positions #
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7]) 


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
