import time
import os
from pathlib import Path
from isaaclab.app import AppLauncher
import numpy as np
import pygame
import torch
import gymnasium as gym                   
import sys
import copy
import cv2, json
import dataclasses
import tyro
from typing import Optional, List

from utils.helper import print_dict_shapes
from utils.data_recorder import DataRecorder, RecorderConfig

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg


@dataclasses.dataclass
class CoffeeGamepadConfig:
    """Franka机器人咖啡控制参数配置"""
    
    # CPU相关设置
    use_cpu: bool = dataclasses.field(
        default=False,
        metadata={"help": "使用CPU进行模拟计算（默认使用GPU）"}
    )
    
    # Fabric相关设置
    disable_fabric: bool = dataclasses.field(
        default=False,
        metadata={"help": "禁用Fabric，使用USD I/O操作"}
    )
    
    # 任务相关设置
    task: str = dataclasses.field(
        default="Isaac-Franka-Coffee-Human-Control-Direct-v0",
        metadata={"help": "任务名称"}
    )
    
    # 环境相关设置
    num_envs: int = dataclasses.field(
        default=1,
        metadata={"help": "要模拟的环境数量"}
    )
    
    # 数据录制相关配置
    recorder: RecorderConfig = dataclasses.field(
        default_factory=RecorderConfig,
        metadata={"help": "数据录制相关配置"}
    )
    
    # AppLauncher相关设置
    headless: bool = dataclasses.field(
        default=False,
        metadata={"help": "是否使用无头模式（不显示界面）"}
    )
    
    enable_cameras: bool = dataclasses.field(
        default=True,
        metadata={"help": "是否启用摄像头"}
    )
    
    viewport_width: int = dataclasses.field(
        default=1280,
        metadata={"help": "视口宽度"}
    )
    
    viewport_height: int = dataclasses.field(
        default=720,
        metadata={"help": "视口高度"}
    )


class PygameTeleop:
    """
    左摇杆:  xy面平移
    右摇杆:  偏航(z轴)，俯仰(y轴)
    LT/RT:  z轴平移
    LB/RB:  横滚(x轴)
    A:      夹爪open/close
    """
    def __init__(self, pos_sensitivity: float = 1.0, rot_sensitivity: float = 1.6, debug_mode: bool = False):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("pygame未检测到手柄。")
        else:
            print(f"pygame检测到手柄: {pygame.joystick.Joystick(0).get_name()}")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        self.pos_scale = 0.015 * pos_sensitivity  # 每个周期满偏转的[米]
        self.rot_scale = 0.05 * rot_sensitivity   # 每个周期满偏转的[弧度]
        self._prev_grip_button_state = False
        self._grip_toggle_latch = False
        self.debug_mode = debug_mode  # 新增：调试模式开关

    # ---------------------------------------------------------------------
    # 脚本期望的公共API
    # ---------------------------------------------------------------------
    def advance(self):
        """轮询手柄并返回*(6,) delta_pose, grip_toggle*。

        *delta_pose*是*(dx, dy, dz, droll, dpitch, dyaw)*，单位为*米/弧度*。
        *grip_toggle*仅在*A/十字*按钮的*上升沿*为*True*
        这样夹爪状态每次按下只会切换一次。
        """
        pygame.event.pump()  # 非阻塞
        delta_pose = self._read_joystick()
        grip_toggle = self._read_grip_toggle()
        return delta_pose, grip_toggle

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------
    def _print_all_inputs(self):
        """调试函数：打印所有手柄输入"""
        print("\n--- 手柄输入调试信息 ---")
        print(f"轴数量: {self.joy.get_numaxes()}")
        for i in range(self.joy.get_numaxes()):
            print(f"轴 {i}: {self.joy.get_axis(i):.3f}")
        
        print(f"按钮数量: {self.joy.get_numbuttons()}")
        for i in range(self.joy.get_numbuttons()):
            print(f"按钮 {i}: {self.joy.get_button(i)}")
        
        print(f"Hat数量: {self.joy.get_numhats()}")
        for i in range(self.joy.get_numhats()):
            print(f"Hat {i}: {self.joy.get_hat(i)}")
        print("----------------------")

    def _read_joystick(self):
        """读取手柄输入并转换为机器人控制信号"""
        if self.debug_mode:
            self._print_all_inputs()
        lx = -self.joy.get_axis(1)   
        ly = -self.joy.get_axis(0)  
        rx = -self.joy.get_axis(3)   
        ry = self.joy.get_axis(4)  
        
        lt = (self.joy.get_axis(2) + 1) / 2  
        rt = (self.joy.get_axis(5) + 1) / 2  

        dx = lx * self.pos_scale
        dy = ly * self.pos_scale
        dz = (rt - lt) * self.pos_scale

        roll_left = self.joy.get_button(4)  
        roll_right = self.joy.get_button(5) 
        droll = (roll_right - roll_left) * self.rot_scale
        dpitch = ry * self.rot_scale
        dyaw = rx * self.rot_scale

        if self.debug_mode:
            print(f"控制信号: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, droll={droll:.4f}, dpitch={dpitch:.4f}, dyaw={dyaw:.4f}")

        return np.array([dx, dy, dz, droll, dpitch, dyaw], dtype=np.float32)

    def _read_grip_toggle(self):
        """当A/十字按钮被按下时返回*True*一次"""
        grip_pressed = bool(self.joy.get_button(0))  # 按钮0 = XInput上的A
        toggle = grip_pressed and not self._prev_grip_button_state
        self._prev_grip_button_state = grip_pressed
        return toggle


def main(cfg: CoffeeGamepadConfig):
    """主函数，处理机器人控制与数据录制"""
    
    # launch omniverse app
    app_launcher = AppLauncher(
        headless=cfg.headless, 
        enable_cameras=cfg.enable_cameras,
        viewport_width=cfg.viewport_width,
        viewport_height=cfg.viewport_height
    )
    simulation_app = app_launcher.app

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    SIM_WARM_UP_STEP = 30

    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  
    
    # parse configuration
    env_cfg = parse_env_cfg(
        cfg.task,
        device="cuda:0" if not cfg.use_cpu else "cpu",
        num_envs=cfg.num_envs,
        use_fabric=not cfg.disable_fabric,
    )
    
    # create environment
    env = gym.make(cfg.task, cfg=env_cfg)
    
    # 初始化数据录制器
    recorder = DataRecorder(cfg.recorder, env, env.device)

    # reset environment at start
    obs, _ = env.reset()
    print("初始化环境...")

    count = 0
    episode_idx = recorder.get_current_episode()

    # 手柄控制器
    teleop = PygameTeleop(pos_sensitivity=5, rot_sensitivity=7)

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if count == 0:
                print(f"\n进入模拟循环...{episode_idx}")

            count += 1

            # give some time to warm up
            if count > SIM_WARM_UP_STEP:
                print(f"开始控制! Episode {episode_idx}/{recorder.get_target_episodes()}, step: {count}")

                # --- 读取手柄 ---
                delta_pose_np, grip_toggle = teleop.advance()
                # 形状 => (num_envs, 6)
                delta_pose = torch.tensor(delta_pose_np,
                                        dtype=torch.float,
                                        device=env.device).repeat(env.unwrapped.num_envs, 1)

                # --- 维护夹爪状态 ---
                if 'gripper_state' not in locals():
                    gripper_state = torch.ones((env.unwrapped.num_envs, 1), device=env.device)
                if grip_toggle:  # 按A时翻转
                    gripper_state *= -1.0

                # --- 任务空间 -> 关节空间  ---
                ee_pos, ee_quat = env.unwrapped._compute_frame_pose()
                dpos         = delta_pose[:, :3]
                drot_euler   = delta_pose[:, 3:]
                target_pos   = ee_pos + dpos
                dquat = quat_from_euler_xyz(
                            drot_euler[:, 0],
                            drot_euler[:, 1],
                            drot_euler[:, 2])
                target_quat  = quat_mul(dquat, ee_quat)
                target_pose  = torch.cat([target_pos, target_quat], dim=-1)

                jac    = env.unwrapped._compute_ee_jacobian()
                q_curr = env.unwrapped._robot.data.joint_pos[:, env.unwrapped.arm_joint_ids]

                diff_ik = diff_ik if 'diff_ik' in locals() else DifferentialIKController(
                    DifferentialIKControllerCfg(command_type="pose",
                                                use_relative_mode=False,
                                                ik_method="dls"),
                    num_envs=env.unwrapped.num_envs,
                    device=env.device)
                diff_ik.set_command(target_pose, ee_pos, ee_quat)
                q_target = diff_ik.compute(ee_pos, ee_quat, jac, q_curr)

                actions = torch.cat([q_target, gripper_state], dim=-1)
                print(f"actions: {actions.shape}")
                import pdb; pdb.set_trace()

                obs, reward, terminated, truncated, info = env.step(actions)

                dones = terminated | truncated 

                # 记录数据，并检查是否完成episode
                episode_finished = recorder.record_frame(obs, actions, dones)
                
                # 如果不记录数据集，只需要在任务完成时增加episode计数
                if episode_finished or (dones.any() and not cfg.recorder.record):
                    episode_idx = recorder.get_current_episode()
                
                # 检查是否完成所有录制目标
                if recorder.is_complete():
                    break

            if recorder.is_complete():
                break

    # close the environment
    env.close()
    # close sim app
    simulation_app.close()


if __name__ == "__main__":
    # 通过tyro解析命令行参数
    config = tyro.cli(CoffeeGamepadConfig)
    main(config)