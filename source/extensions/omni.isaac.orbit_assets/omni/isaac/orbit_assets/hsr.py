import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR


HSR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/wang/Documents/isaac_hsr-master/usd/hsrb4s/hsrb4s.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_roll_joint": 0.0,
            "wrist_flex_joint": 0.0,
            "arm_flex_joint": 0.0,
            "wrist_roll_joint": 0.0,
            "arm_lift_joint": 0.0,
            "wrist_ft_sensor_frame_joint": 0.0,
            "base_b_bumper_joint": 0.0,
            "base_f_bumper_joint": 0.0,
            "base_roll_joint": 0.0,
            "torso_lift_joint": 0.0,
            "hand_l_proximal_joint": 0.0,
            "hand_motor_joint": 0.0,
            "hand_r_proximal_joint": 0.0,
            "base_l_drive_wheel_joint": 0.0,
            "base_l_passive_wheel_x_frame_joint": 0.0,
            "base_r_drive_wheel_joint": 0.0,
            "base_r_passive_wheel_x_frame_joint": 0.0,
            "head_pan_joint": 0.0,
            "hand_l_spring_proximal_joint": 0.0,
            "hand_r_spring_proximal_joint": 0.0,
            "base_l_passive_wheel_y_frame_joint": 0.0,
            "base_r_passive_wheel_y_frame_joint": 0.0,
            "head_tilt_joint":  0.0,
            "hand_l_mimic_distal_joint": 0.0,
            "hand_r_mimic_distal_joint": 0.0,
            "base_l_passive_wheel_z_joint": 0.0,
            "base_r_passive_wheel_z_joint": 0.0,
            "hand_l_distal_joint": 0.0,
            "hand_r_distal_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
   
)