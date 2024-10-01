# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.assets import RigidObjectCfg
from omni.isaac.orbit.sensors import FrameTransformerCfg
from omni.isaac.orbit.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.orbit.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit_tasks.manipulation.lift import mdp
from omni.isaac.orbit_tasks.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.orbit_assets.franka import FRANKA_PANDA_CFG  # isort: skip
from omni.isaac.orbit_assets.cobotta import COBOTTA_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = COBOTTA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_.*"], scale=0.5, use_default_offset=True #可以由articulation的joint_pos来设置
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg( #设置夹爪的动作
            asset_name="robot",
            joint_names=["finger_joint",
            "right_outer_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint"],
            close_command_expr={
                "finger_joint": 0.5, 
                "right_outer_knuckle_joint": -0.5, 
                "left_inner_finger_joint": -0.5, 
                "right_inner_finger_joint": 0.5,},
            open_command_expr={
                "finger_joint": 0.0, 
                "right_outer_knuckle_joint": -0.0, 
                "left_inner_finger_joint": -0.0, 
                "right_inner_finger_joint": 0.0,},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "onrobot_rg6_base_link"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False, #修改：不受重力影响
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/onrobot_rg6_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.25], #修改：末端执行器的位置,但是没有在isaac中反应出来
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
