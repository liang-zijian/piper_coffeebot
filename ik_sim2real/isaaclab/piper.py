import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/ubuntu/workspace/piper_ws/ik_sim2real/piper_description/piper_description.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        # 关节初始角度（弧度）
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
            "joint8": 0.0,
        },
    ),

    # 执行器（隐式 PD 模型） --------------------------------------------------
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[r"joint[1-8]"],
            velocity_limit=100.0,        # rad / s
            effort_limit=100.0,          # N·m
            stiffness=10000.0,            # N·m / rad
            damping=100.0,               # N·m / (rad/s)
        ),
    },
)
"""Configuration of Piper 8-DOF arm using implicit PD actuators."""