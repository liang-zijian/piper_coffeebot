import pygame
import numpy as np
import threading
from piper_sdk import *
import time
from scipy.spatial.transform import Rotation as R


def enable_fun(piper: C_PiperInterface_V2):
    """
    使能机械臂并检测使能状态，尝试5秒，如果使能超时则退出程序。
    """
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False

    while not enable_flag:
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:", enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        print("--------------------")

        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)

    if elapsed_time_flag:
        print("程序自动使能超时，退出程序")
        exit(0)


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
        print(f"检测到手柄: {self.joystick.get_name()}")

        # 控制参数
        self.pos_scale = 0.01  # 位置增量缩放因子 (米/周期)
        self.rot_scale = 0.01  # 旋转增量缩放因子 (弧度/周期)
        
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
            pygame.time.wait(50)  # 控制刷新频率

    def is_moving(self, val):
        """判断摇杆是否在死区外"""
        return abs(val) > self.dead_zone

    def read_joystick_inputs(self):
        """读取手柄输入并转换为控制信号
        返回: (delta_pos, joint5_delta, joint6_delta, grip_toggle)
        delta_pos: [dx, dy, dz]
        joint5_delta: 关节5的角度增量
        joint6_delta: 关节6的角度增量
        grip_toggle: 是否切换夹爪状态
        """
        # 左摇杆: XY平移
        lx = -self.joystick.get_axis(1)  # 前后
        ly = -self.joystick.get_axis(0)  # 左右
        
        # 右摇杆: 两个轴都控制关节5
        rx = -self.joystick.get_axis(3)  # 右摇杆X轴
        ry = self.joystick.get_axis(4)   # 右摇杆Y轴
        
        # LT/RT: Z轴平移
        lt = (self.joystick.get_axis(2) + 1) / 2  # 左扳机，归一化到[0,1]
        rt = (self.joystick.get_axis(5) + 1) / 2  # 右扳机，归一化到[0,1]
        
        # LB/RB: 控制关节6
        lb = self.joystick.get_button(4)  # 左肩键
        rb = self.joystick.get_button(5)  # 右肩键
        
        # A按钮: 夹爪开合
        grip_button = self.joystick.get_button(0)  # A按钮
        
        # 计算位置增量
        dx = lx * self.pos_scale if self.is_moving(lx) else 0.0
        dy = ly * self.pos_scale if self.is_moving(ly) else 0.0
        dz = (rt - lt) * self.pos_scale
        
        # 计算关节5的角度增量（使用右摇杆的合成）
        # 选择使用Y轴
        joint5_delta = 0.0
        # if self.is_moving(rx):
        #     joint5_delta += rx * self.rot_scale
        if self.is_moving(ry):
            joint5_delta += ry * self.rot_scale
        
        # 计算关节6的角度增量（使用LB/RB）
        joint6_delta = (rb - lb) * self.rot_scale
        
        # 检测夹爪按钮的上升沿
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
            
            # 获取末端执行器当前位置
            end_effector = self.piper.get_link("link6")
            current_pos = end_effector.get_pos().cpu().numpy().copy()
            
            # 更新位置
            new_pos = current_pos + delta_pos
            
            # 更新关节5和6的角度
            self.joint5_angle += joint5_delta
            self.joint6_angle += joint6_delta
            
            # 限制关节角度范围（根据实际机械臂的限制调整）
            # 获取关节限制
            if hasattr(self.control_qlimit, 'cpu'):
                # 如果是 tensor 对象
                qlimit = self.control_qlimit.cpu().numpy()
            else:
                # 如果是 tuple 对象，假设是 (lower, upper) 的格式
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
                print(f"夹爪状态切换: {'闭合' if self.is_grasp else '张开'}")
                if self.is_grasp:
                    self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
                else:
                    self.piper_real.GripperCtrl(5000 * 1000, 1000, 0x01, 0)
            
            # 使用逆运动学计算前4个关节角度（只考虑位置）
            try:
                # 只传入位置参数进行IK求解
                new_qpos = self.piper.inverse_kinematics(
                    link=end_effector, 
                    pos=new_pos
                )
                
                # 复制当前的关节角度
                target_qpos = new_qpos.clone()
                
                # 保持关节5和6为直接控制的值
                target_qpos[4] = self.joint5_angle
                target_qpos[5] = self.joint6_angle
                
                self.update_dofs_position(target_qpos, self.dofs_idx)
                
                # 打印调试信息
                if any(abs(delta_pos) > 0.001) or abs(joint5_delta) > 0.001 or abs(joint6_delta) > 0.001:
                    print(f"Delta pos: [{delta_pos[0]:.3f}, {delta_pos[1]:.3f}, {delta_pos[2]:.3f}]")
                    print(f"Joint 5: {self.joint5_angle:.3f} (delta: {joint5_delta:.3f})")
                    print(f"Joint 6: {self.joint6_angle:.3f} (delta: {joint6_delta:.3f})")
                    
            except Exception as e:
                print(f"IK求解失败: {e}")
            
            self.run_sim()
            self.move_joint_real(self.piper.get_qpos()[:6].cpu().numpy())

    def update_dofs_position(self, target_qpos, dofs_idx):
        """更新关节位置"""
        # 夹爪开合逻辑
        if self.is_grasp:
            target_qpos[-2:] = 0.0
        else:
            target_qpos[-2] = 0.04
            target_qpos[-1] = -0.04
        self.target_qpos = target_qpos.clone()
        self.piper.control_dofs_position(target_qpos[:-2], dofs_idx[:-2])
        print("target_qpos:", [f"{x:.3f}" for x in self.target_qpos])

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
        # 初始化连接
        self.piper_real.ConnectPort()
        # 夹爪设置0点
        self.piper_real.GripperCtrl(0, 1000, 0x00, 0)
        time.sleep(1.5)
        enable_fun(self.piper_real)

    def reset_to_initial_positions(self):
        """将关节恢复到初始位置"""
        print("恢复到初始位置...")
        self.move_joint_real([0] * 6)
        self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
        print("已恢复到初始位置。")

    def move_joint_real(self, target_joints):
        """移动指定关节到目标位置"""
        factor = 57295.7795  # 1000*180/3.1415926 弧度转角度
        joints = [0] * 6
        for i in range(0, 6):
            joints[i] = round(target_joints[i] * factor)
        self.piper_real.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper_real.JointCtrl(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])