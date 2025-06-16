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
    def __init__(self, piper, scene, cams=[], lock=None):
        self.piper = piper
        self.target_qpos = piper.get_qpos().clone()
        self.control_qlimit = piper.get_dofs_limit()
        self.joint_step_size = (self.control_qlimit[1] - self.control_qlimit[0]) / 50
        self.cams = cams
        self.lock = lock
        self.scene = scene
        self.running = True
        self.piper_real = C_PiperInterface_V2("can0")
        
        # 初始化 Pygame 手柄
        pygame.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("未检测到游戏手柄！")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
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
        
        # 死区设置
        self.dead_zone = 0.15
        
        # 关节5和6的当前角度
        self.joint5_angle = 0.0
        self.joint6_angle = 0.0
        
        # 状态变量
        self.last_update_time = time.time()
        self.update_interval = 0.5  # 更新间隔
        
    def print_status(self, delta_pos, joint5_delta, joint6_delta):
        """打印状态信息（限制频率）"""
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            console.clear()
            
            # 创建状态表格
            table = Table(title=f"Piper 控制状态 - {datetime.now().strftime('%H:%M:%S')}")
            table.add_column("参数", style="cyan")
            table.add_column("值")
            
            # 位置增量
            table.add_row("位置增量 X", f"{delta_pos[0]:7.4f}")
            table.add_row("位置增量 Y", f"{delta_pos[1]:7.4f}")
            table.add_row("位置增量 Z", f"{delta_pos[2]:7.4f}")
            table.add_row("", "")  # 空行
            
            # 关节角度
            table.add_row("关节5角度", f"{self.joint5_angle:7.3f} rad")
            table.add_row("关节5增量", f"{joint5_delta:7.4f}")
            table.add_row("关节6角度", f"{self.joint6_angle:7.3f} rad") 
            table.add_row("关节6增量", f"{joint6_delta:7.4f}")
            table.add_row("", "")  # 空行

            # 当前关节位置
            joints_position = self.piper.get_dofs_position().cpu().numpy()
            table.add_row("joint1", f"{joints_position[0]:7.4f}")
            table.add_row("joint2", f"{joints_position[1]:7.4f}")
            table.add_row("joint3", f"{joints_position[2]:7.4f}")
            table.add_row("joint4", f"{joints_position[3]:7.4f}")
            table.add_row("joint5", f"{joints_position[4]:7.4f}")
            table.add_row("joint6", f"{joints_position[5]:7.4f}")
            
            # 夹爪状态
            gripper_color = "green" if self.is_grasp else "yellow"
            table.add_row("夹爪状态", f"[{gripper_color}]{('闭合' if self.is_grasp else '张开')}[/]")
            
            console.print(table)
            self.last_update_time = current_time

    def start(self):
        """启动手柄控制线程"""
        self.enable_realtime()
        self.move_joint_real(self.piper.get_qpos()[:6].cpu().numpy())
        
        # 初始化关节5和6的角度
        current_qpos = self.piper.get_qpos().cpu().numpy()
        self.joint5_angle = current_qpos[4]
        self.joint6_angle = current_qpos[5]
        
        threading.Thread(target=self._run_joystick_loop, daemon=True).start()
        self.running = True

    def _run_joystick_loop(self):
        while self.running:
            self.handle_joystick_input()
            pygame.time.wait(50)

    def is_moving(self, val):
        """判断摇杆是否在死区外"""
        return abs(val) > self.dead_zone

    def read_joystick_inputs(self):
        """读取手柄输入并转换为控制信号"""
        # 左摇杆: XY平移
        lx = -self.joystick.get_axis(1)
        ly = -self.joystick.get_axis(0)
        
        # 右摇杆
        rx = -self.joystick.get_axis(3)
        ry = self.joystick.get_axis(4)
        
        # LT/RT: Z轴平移
        lt = (self.joystick.get_axis(2) + 1) / 2
        rt = (self.joystick.get_axis(5) + 1) / 2
        
        # LB/RB: 控制关节6
        lb = self.joystick.get_button(4)
        rb = self.joystick.get_button(5)
        
        # A按钮: 夹爪开合
        grip_button = self.joystick.get_button(0)
        
        # 计算位置增量
        dx = lx * self.pos_scale if self.is_moving(lx) else 0.0
        dy = ly * self.pos_scale if self.is_moving(ly) else 0.0
        
        trigger_diff = rt - lt
        dz = trigger_diff * self.pos_scale if abs(trigger_diff) > self.dead_zone else 0.0
        
        # 计算关节角度增量
        joint5_delta = 0.0
        if self.is_moving(ry):
            joint5_delta += ry * self.rot_scale
        
        joint6_delta = (rb - lb) * self.rot_scale
        
        # 检测夹爪按钮
        grip_toggle = grip_button and not self._prev_grip_button_state
        self._prev_grip_button_state = grip_button
        
        delta_pos = np.array([dx, dy, dz], dtype=np.float32)
        
        return delta_pos, joint5_delta, joint6_delta, grip_toggle

    def handle_joystick_input(self):
        """处理手柄输入事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

        with self.lock:
            # 获取手柄输入
            delta_pos, joint5_delta, joint6_delta, grip_toggle = self.read_joystick_inputs()
            
            # 检查是否有有效的输入变化
            has_pos_change = np.any(np.abs(delta_pos) > 0.001)
            has_joint5_change = abs(joint5_delta) > 0.001
            has_joint6_change = abs(joint6_delta) > 0.001
            
            if has_pos_change or has_joint5_change or has_joint6_change or grip_toggle:
                # 打印状态
                self.print_status(delta_pos, joint5_delta, joint6_delta)
                
                # 获取末端执行器当前位置
                end_effector = self.piper.get_link("link6")
                current_pos = end_effector.get_pos().cpu().numpy().copy()
                
                if has_pos_change:
                    new_pos = current_pos + delta_pos
                else:
                    new_pos = current_pos
                
                # 更新关节角度
                self.joint5_angle += joint5_delta
                self.joint6_angle += joint6_delta
                
                # 限制关节角度范围
                if hasattr(self.control_qlimit, 'cpu'):
                    qlimit = self.control_qlimit.cpu().numpy()
                else:
                    lower_limits, upper_limits = self.control_qlimit
                    if hasattr(lower_limits, 'cpu'):
                        lower_limits = lower_limits.cpu().numpy()
                        upper_limits = upper_limits.cpu().numpy()
                    qlimit = np.array([lower_limits, upper_limits])
                
                self.joint5_angle = np.clip(self.joint5_angle, qlimit[0, 4], qlimit[1, 4])
                self.joint6_angle = np.clip(self.joint6_angle, qlimit[0, 5], qlimit[1, 5])
                
                # 切换夹爪状态
                if grip_toggle:
                    self.is_grasp = not self.is_grasp
                    if self.is_grasp:
                        self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
                    else:
                        self.piper_real.GripperCtrl(5000 * 1000, 1000, 0x01, 0)
                
                # 使用逆运动学计算
                try:
                    new_qpos = self.piper.inverse_kinematics(
                        link=end_effector, 
                        pos=new_pos
                    )
                    
                    target_qpos = new_qpos.clone()
                    target_qpos[4] = self.joint5_angle
                    target_qpos[5] = self.joint6_angle
                    
                    self.update_dofs_position(target_qpos, self.dofs_idx)
                    
                except Exception as e:
                    console.print(f"[red]IK求解失败: {e}[/red]")
                
                self.run_sim()
                self.move_joint_real(self.piper.get_qpos()[:6].cpu().numpy())

    def update_dofs_position(self, target_qpos, dofs_idx):
        """更新关节位置"""
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
        """将关节恢复到初始位置"""
        console.print("[yellow]恢复到初始位置...[/yellow]")
        self.move_joint_real([0] * 6)
        self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
        console.print("[green]已恢复到初始位置。[/green]")

    def move_joint_real(self, target_joints):
        """移动指定关节到目标位置"""
        factor = 57295.7795
        joints = [0] * 6
        for i in range(0, 6):
            joints[i] = round(target_joints[i] * factor)
        self.piper_real.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper_real.JointCtrl(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])