import isaaclab.sim as sim_utils
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
from isaaclab.assets import (
    RigidObjectCfg,
    AssetBaseCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.schemas import MassPropertiesCfg

TABLE_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/table",
    spawn=UsdFileCfg(
        usd_path=f"/home/ubuntu/workspace/piper_ws/simulations/assets/table.usd",
        scale=(0.01, 0.01, 0.01),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
            angular_damping=20.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.14382, -0.79955, -0.1012),
        rot=(0.707, 0.0 , 0.0, -0.707),
    ),
)