import pygame
import numpy as np
from piper_sdk import *
import time
from scipy.spatial.transform import Rotation as R
from rich.console import Console
from rich.table import Table
from rich.text import Text


console = Console()

def enable_fun(piper: C_PiperInterface_V2):
    """使能机械臂并检测使能状态"""
    enable_flag = False
    timeout = 5
    start_time = time.time()
    
    while not enable_flag:
        elapsed_time = time.time() - start_time
        
        # 获取使能状态
        arm_info = piper.GetArmLowSpdInfoMsgs()
        motor_status = [
            arm_info.motor_1.foc_status.driver_enable_status,
            arm_info.motor_2.foc_status.driver_enable_status,
            arm_info.motor_3.foc_status.driver_enable_status,
            arm_info.motor_4.foc_status.driver_enable_status,
            arm_info.motor_5.foc_status.driver_enable_status,
            arm_info.motor_6.foc_status.driver_enable_status
        ]
        
        enable_flag = all(motor_status)
        
        # 清屏并显示状态
        console.clear()
        table = Table(title=f"使能状态 ({elapsed_time:.1f}s)", show_header=True)
        table.add_column("电机", style="cyan")
        table.add_column("状态")
        
        for i, status in enumerate(motor_status):
            status_text = Text("✓", style="green") if status else Text("✗", style="red")
            table.add_row(f"Motor {i+1}", status_text)
        
        console.print(table)
        
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        
        if elapsed_time > timeout:
            console.print("[red]使能超时，程序退出[/red]")
            time.sleep(2)
            exit(0)
        
        time.sleep(0.5)


class PiperController:
    def __init__(self):
        self.piper_real = C_PiperInterface_V2("can0")

    def enable_control(self):
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

    def move_joint_real(self, target_joints, grasp=False):
        """Move joints to target positions in real-time."""
        if grasp:
            self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
        else:
            self.piper_real.GripperCtrl(5000 * 1000, 1000, 0x01, 0)

        factor = 57295.7795
        joints = [0] * 6
        for i in range(0, 6):
            joints[i] = round(target_joints[i] * factor)
        self.piper_real.MotionCtrl_2(0x01, 0x01, 60, 0x00)
        try:
            self.piper_real.JointCtrl(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])
        except Exception as e:
            console.print(f"[red]Error moving joints: {e}[/red]")
            return