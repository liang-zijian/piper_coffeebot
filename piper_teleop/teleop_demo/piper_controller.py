import pygame
import numpy as np
import threading
from piper_sdk import *
import time
from scipy.spatial.transform import Rotation as R
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from datetime import datetime
import genesis as gs

console = Console()


class PiperController:
    def __init__(self, piper, scene, cams=[], lock=None):
        self.piper = piper
        self.cams = cams
        self.lock = lock
        self.scene = scene
        self.running = True
        self.piper_real = C_PiperInterface_V2("can0")
        self.init_pos = np.array([0.06, 0.002, 0.22])
        self.target_pos = None
        self.target_quat = None

        # 控制参数
        self.pos_scale = 0.01
        self.rot_scale = 0.01
        
        # 夹爪状态
        self.is_grasp = False
        self._prev_grip_button_state = False
        
        # 关节名称映射
        self.JOINT_NAMES = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8'
        ]
        self.dofs_idx = [piper.get_joint(name).dof_idx_local for name in self.JOINT_NAMES]
        
        # 状态变量
        self.last_update_time = time.time()
        self.update_interval = 0.5  # 更新间隔
        
    def start(self):
        """启动手柄控制线程"""
        self.enable_realtime()
        self.move_joint_real(self.piper.get_qpos()[:6].cpu().numpy())

        threading.Thread(target=self._run_target_loop, daemon=True).start()
        self.running = True

    def _run_target_loop(self):
        while self.running:
            self.handle_target_input()
            pygame.time.wait(50)

    def handle_target_input(self):

        with self.lock:
            self.scene.clear_debug_objects()
            if True:                
                # 获取末端执行器当前位置
                end_effector = self.piper.get_link("link6")
                # --- 可视化link6的位姿  ---
                try:
                    link_pos_tensor = end_effector.get_pos()
                    link_quat_tensor = end_effector.get_quat() # w, x, y, z
                    if link_pos_tensor is not None and link_quat_tensor is not None:
                        link_pos = link_pos_tensor.cpu().numpy()
                        link_quat = link_quat_tensor.cpu().numpy()

                        # Genesis 使用 w-x-y-z, scipy 使用 x-y-z-w
                        rot = R.from_quat([link_quat[1], link_quat[2], link_quat[3], link_quat[0]])
                        
                        axis_length = 0.1
                        x_axis = rot.apply([1, 0, 0])
                        y_axis = rot.apply([0, 1, 0])
                        z_axis = rot.apply([0, 0, 1])
                        
                        self.scene.draw_debug_arrow(pos=link_pos, vec=x_axis * axis_length, color=(1, 0, 0, 1)) # red
                        self.scene.draw_debug_arrow(pos=link_pos, vec=y_axis * axis_length, color=(0, 1, 0, 1)) # green
                        self.scene.draw_debug_arrow(pos=link_pos, vec=z_axis * axis_length, color=(0, 0, 1, 1)) # blue

                except Exception as e:
                    console.print(f"[yellow]无法可视化link6位姿: {e}[/yellow]")

                current_pos = end_effector.get_pos().cpu().numpy().copy()
                
                # 切换夹爪状态
                if self.is_grasp:
                    self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
                else:
                    self.piper_real.GripperCtrl(5000 * 1000, 1000, 0x01, 0)
                
                # 逆运动学计算
                try:
                    new_qpos = self.piper.inverse_kinematics(
                        link=end_effector, 
                        pos=self.target_pos if self.target_pos is not None else current_pos,
                        quat=self.target_quat if self.target_quat is not None else np.array([0.707, 0, 0.707, 0]),
                        max_solver_iters = 60
                    )
                    
                    target_qpos = new_qpos.clone()
                    
                    self.update_dofs_position(target_qpos, self.dofs_idx)
                    
                except Exception as e:
                    console.print(f"[red]IK solved fail: {e}[/red]")
                
                self.run_sim()
                #self.move_joint_real(self.piper.get_qpos()[:6].cpu().numpy())

    def update_dofs_position(self, target_qpos, dofs_idx):
        """更新仿真关节位置"""
        if self.is_grasp:
            target_qpos[-2:] = 0.0
        else:
            target_qpos[-2] = 0.04
            target_qpos[-1] = -0.04
        self.target_qpos = target_qpos.clone()
        self.piper.control_dofs_position(target_qpos[:-2], dofs_idx[:-2])

    def control_grasp(self, target_qpos, dofs_idx):
        """控制夹爪"""
        if self.is_grasp:
            target_qpos[-2:] = 0.0
        else:
            target_qpos[-2] = 0.04
            target_qpos[-1] = -0.04
        self.piper.control_dofs_position(target_qpos[-2:], dofs_idx[-2:])

    def run_sim(self, step=1):
        """运行仿真步骤"""
        for i in range(step):
            self.scene.step()

    def enable_realtime(self):
        """使能实时控制"""
        self.piper_real.ConnectPort()
        self.piper_real.GripperCtrl(0, 1000, 0x00, 0)
        time.sleep(1.5)
        enable_fun(self.piper_real)

    def reset_to_initial_positions(self):
        """ Reset the robot to its initial positions."""
        self.move_joint_real([0] * 6)
        self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
        time.sleep(1)
        console.print("[green]Reset to initial positions complete[/green]")

    def move_joint_real(self, target_joints):
        """移动指定关节到目标位置"""
        factor = 57295.7795
        joints = [0] * 6
        for i in range(0, 6):
            joints[i] = round(target_joints[i] * factor)
        self.piper_real.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        self.piper_real.JointCtrl(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])
