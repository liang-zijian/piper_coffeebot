# Copyright (c) 2022-2025, Zijian Liang.
# All rights reserved.

"""
1. deactivate the virtual environment if needed
2. launch the script:
        ~/IsaacLab-2.0.2/isaaclab.sh -p piper_ik_s2r.py 

"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="piper", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--real_robot", type=bool, default=False, help="Whether to use the real robot.")
# append AppLauncher cli argsisaa
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from rich.console import Console
from rich.table import Table

console = Console()

def print_joints_table(joint_positions):
    """Print the joints table."""
    joints_table = Table(title="Joints Targets", show_header=True)
    joints_table.add_column("Joint Name", justify="left")
    joints_table.add_column("Target Position", justify="right")

    for i, pos in enumerate(joint_positions):
        joints_table.add_row(f"joint_{i + 1}", f"{pos:.2f}")

    console.print(joints_table)

from piper import PIPER_CFG  
from table import TABLE_CFG
from piper_controller import PiperController

@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation for piper
    robot = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # table
    table = TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/table")

def run_sim2real(sim: sim_utils.SimulationContext, scene: InteractiveScene, piper_controller: PiperController):
    """Runs the sim2real loop."""
    # enable piper controller
    if args_cli.real_robot:
        piper_controller.enable_control()
        piper_controller.reset_to_initial_positions()
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.2, 0.002, 0.20, 0.707, 0, 0.707, 0],
        [0.2, 0.1, 0.20, 0.707, 0, 0.707, 0],
        [0.2, -0.1, 0.20, 0.707, 0, 0.707, 0],
        [0.2, 0.0, 0.3, 0.707, 0, 0.707, 0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)

    # Specify robot-specific parameters
    if args_cli.robot == "piper":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint[1-8]"], body_names=["link6"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: piper")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Get default joint positions for reset
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()

    # Wait for simulation to be ready
    while sim.is_stopped():
        sim.step()

    # -- Go through all the goals
    for i, goal in enumerate(ee_goals):
        if not simulation_app.is_running():
            break
        
        console.clear()
        console.print(f"Moving to goal {i}...")
        ik_commands = goal.unsqueeze(0)
        diff_ik_controller.set_command(ik_commands)
        diff_ik_controller.reset()

        # target joint positions
        target_joint_pos = None

        # run for some steps to reach the goal
        for _ in range(100):
            if not simulation_app.is_running():
                break
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            target_joint_pos = joint_pos_des.clone()
            # apply actions
            robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
            scene.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            scene.update(sim_dt)
            # update marker positions
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + torch.tensor(TABLE_CFG.init_state.pos, device=sim.device), ik_commands[:, 3:7])

        if not simulation_app.is_running():
            break

        # pause for 2 seconds
        if args_cli.real_robot:
            piper_controller.move_joint_real(target_joint_pos[0].cpu().numpy())
        print_joints_table(target_joint_pos[0].cpu().numpy())
        pause_start_time = time.time()
        while time.time() - pause_start_time < 2.0:
            if not simulation_app.is_running():
                break
            sim.step()
            scene.update(sim_dt)

    # -- Reset robot
    if simulation_app.is_running():
        console.print("[yellow]Resetting robot...[/yellow]")
        #robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        # robot.reset()
        if args_cli.real_robot:
            piper_controller.reset_to_initial_positions()
        # Need to step sim to see the reset
        for _ in range(10):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    # keep stepping simulation
    # while simulation_app.is_running():
    #     sim.step()
    #     scene.update(sim_dt)
    
    simulation_app.close()
    return

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = BaseSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Create Piper controller
    if args_cli.real_robot:
        piper_controller = PiperController()
    else:
        piper_controller = None
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_sim2real(sim, scene, piper_controller)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
