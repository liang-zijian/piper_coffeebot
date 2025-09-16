#!/usr/bin/env python3
"""
手柄控制模块
集成手柄控制和录制功能，支持X键停止录制
"""

import pygame
import numpy as np
import threading
import time
from typing import Tuple, Callable, Optional

# 导入全局日志管理器
from global_logger import log_message, log_info, log_warning, log_error, log_success

class GamepadController:
    """手柄控制器，集成录制功能"""
    
    def __init__(self, 
                 pos_scale: float = 0.01,
                 rot_scale: float = 0.01,
                 dead_zone: float = 0.15):
        """
        初始化手柄控制器
        
        Args:
            pos_scale: 位置控制比例
            rot_scale: 旋转控制比例
            dead_zone: 摇杆死区
        """
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.dead_zone = dead_zone
        
        # 手柄状态
        self.joystick = None
        self.running = False
        self.is_recording = False
        
        # 按钮状态记录
        self._prev_grip_button_state = False
        self._prev_record_button_state = False
        self._prev_stop_record_button_state = False
        
        # 控制状态
        self.delta_pos = np.zeros(3, dtype=np.float32)
        self.joint5_delta = 0.0
        self.joint6_delta = 0.0
        self.grip_toggle = False
        self.record_toggle = False
        self.stop_record_toggle = False
        
        # 回调函数
        self.on_movement_callback = None
        self.on_grip_callback = None
        self.on_record_start_callback = None
        self.on_record_stop_callback = None
        
        self._init_gamepad()
    
    def _init_gamepad(self):
        """初始化手柄"""
        try:
            pygame.init()
            
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("未检测到游戏手柄！")
            
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            
            log_message(f"✅ 手柄初始化成功: {self.joystick.get_name()}", "success", "Gamepad")
            
        except Exception as e:
            log_message(f"❌ 手柄初始化失败: {e}", "error", "Gamepad")
            raise
    
    def set_movement_callback(self, callback: Callable[[np.ndarray, float, float], None]):
        """设置移动回调函数"""
        self.on_movement_callback = callback
    
    def set_grip_callback(self, callback: Callable[[bool], None]):
        """设置夹爪回调函数"""
        self.on_grip_callback = callback
    
    def set_record_start_callback(self, callback: Callable[[], None]):
        """设置开始录制回调函数"""
        self.on_record_start_callback = callback
    
    def set_record_stop_callback(self, callback: Callable[[], None]):
        """设置停止录制回调函数"""
        self.on_record_stop_callback = callback
    
    def is_moving(self, val: float) -> bool:
        """判断摇杆是否在死区外"""
        return abs(val) > self.dead_zone
    
    def read_gamepad_inputs(self) -> Tuple[np.ndarray, float, float, bool, bool, bool]:
        """
        读取手柄输入并转换为控制信号
        
        Returns:
            (delta_pos, joint5_delta, joint6_delta, grip_toggle, record_toggle, stop_record_toggle)
        """
        if self.joystick is None:
            return np.zeros(3), 0.0, 0.0, False, False, False
        
        # 左摇杆: XY平移
        lx = -self.joystick.get_axis(1)  # 前后
        ly = -self.joystick.get_axis(0)  # 左右
        
        # 右摇杆: 关节5控制
        ry = self.joystick.get_axis(3)
        
        # LT/RT: Z轴平移
        lt = (self.joystick.get_axis(5) + 1) / 2
        rt = (self.joystick.get_axis(4) + 1) / 2
        
        # LB/RB: 控制关节6
        lb = self.joystick.get_button(6)
        rb = self.joystick.get_button(7)
        
        # A按钮: 夹爪开合
        grip_button = self.joystick.get_button(0)
        
        # Y按钮: 开始录制
        record_button = self.joystick.get_button(3)
        
        # X按钮: 停止录制 (对应 self.joystick.get_button(2))
        stop_record_button = self.joystick.get_button(2)
        
        # 计算位置增量
        dx = lx * self.pos_scale if self.is_moving(lx) else 0.0
        dy = ly * self.pos_scale if self.is_moving(ly) else 0.0
        
        trigger_diff = rt - lt
        dz = trigger_diff * self.pos_scale if abs(trigger_diff) > self.dead_zone else 0.0
        
        # 计算关节角度增量
        joint5_delta = ry * self.rot_scale if self.is_moving(ry) else 0.0
        joint6_delta = (rb - lb) * self.rot_scale
        
        # 检测按钮切换
        grip_toggle = grip_button and not self._prev_grip_button_state
        record_toggle = record_button and not self._prev_record_button_state
        stop_record_toggle = stop_record_button and not self._prev_stop_record_button_state
        
        # 更新按钮状态
        self._prev_grip_button_state = grip_button
        self._prev_record_button_state = record_button
        self._prev_stop_record_button_state = stop_record_button
        
        delta_pos = np.array([dx, dy, dz], dtype=np.float32)
        
        return delta_pos, joint5_delta, joint6_delta, grip_toggle, record_toggle, stop_record_toggle
    
    def update(self):
        """更新手柄状态，应该在主循环中调用"""
        if not self.running:
            return
        
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
        
        # 读取手柄输入
        (self.delta_pos, self.joint5_delta, self.joint6_delta, 
         self.grip_toggle, self.record_toggle, self.stop_record_toggle) = self.read_gamepad_inputs()
        
        # 调用回调函数
        if self.on_movement_callback and (np.any(np.abs(self.delta_pos) > 0.001) or 
                                         abs(self.joint5_delta) > 0.001 or 
                                         abs(self.joint6_delta) > 0.001):
            self.on_movement_callback(self.delta_pos, self.joint5_delta, self.joint6_delta)
        
        if self.on_grip_callback and self.grip_toggle:
            self.on_grip_callback(self.grip_toggle)
        
        if self.on_record_start_callback and self.record_toggle:
            log_message("🎮 检测到Y键按下，开始录制", "info", "Gamepad")
            self.is_recording = True
            self.on_record_start_callback()
        
        if self.on_record_stop_callback and self.stop_record_toggle:
            log_message("🎮 检测到X键按下，停止录制", "info", "Gamepad")
            self.is_recording = False
            self.on_record_stop_callback()
    
    def start(self):
        """启动手柄控制"""
        if self.joystick is None:
            log_message("手柄未初始化", "error", "Gamepad")
            return False
        
        self.running = True
        log_message("🎮 手柄控制已启动", "info", "Gamepad")
        log_message("🎮 控制说明:", "info", "Gamepad")
        log_message("   左摇杆: XY平移", "info", "Gamepad")
        log_message("   LT/RT: Z轴平移", "info", "Gamepad")
        log_message("   右摇杆Y: 关节5控制", "info", "Gamepad")
        log_message("   LB/RB: 关节6控制", "info", "Gamepad")
        log_message("   A键: 夹爪开合", "info", "Gamepad")
        log_message("   Y键: 开始录制", "info", "Gamepad")
        log_message("   X键: 停止录制", "info", "Gamepad")
        
        return True
    
    def stop(self):
        """停止手柄控制"""
        self.running = False
        if self.joystick:
            self.joystick.quit()
        pygame.quit()
        log_message("🎮 手柄控制已停止", "info", "Gamepad")
    
    def get_current_input(self) -> dict:
        """获取当前输入状态"""
        return {
            "delta_pos": self.delta_pos.copy(),
            "joint5_delta": self.joint5_delta,
            "joint6_delta": self.joint6_delta,
            "grip_toggle": self.grip_toggle,
            "record_toggle": self.record_toggle,
            "stop_record_toggle": self.stop_record_toggle,
            "is_recording": self.is_recording
        }


class ThreadedGamepadController(GamepadController):
    """多线程版本的手柄控制器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.control_thread = None
        self.thread_running = False
        self.lock = threading.Lock()
    
    def _control_loop(self):
        """控制循环线程"""
        while self.thread_running and self.running:
            with self.lock:
                self.update()
            time.sleep(0.02)  # 50Hz更新频率
    
    def start_threaded(self):
        """启动多线程控制"""
        if not self.start():
            return False
        
        self.thread_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        log_message("🎮 多线程手柄控制已启动", "info", "Gamepad")
        return True
    
    def stop_threaded(self):
        """停止多线程控制"""
        self.thread_running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join()
        self.stop()
        log_message("🎮 多线程手柄控制已停止", "info", "Gamepad")
    
    def get_current_input_safe(self) -> dict:
        """线程安全地获取当前输入状态"""
        with self.lock:
            return self.get_current_input()


def test_gamepad_controller():
    """测试手柄控制器"""
    log_message("开始测试手柄控制器", "info", "Gamepad")
    
    def on_movement(delta_pos, joint5_delta, joint6_delta):
        log_message(f"Movement: pos={delta_pos}, j5={joint5_delta:.3f}, j6={joint6_delta:.3f}", "info", "Gamepad")
    
    def on_grip(toggle):
        log_message(f"Gripper toggle: {toggle}", "info", "Gamepad")
    
    def on_record_start():
        log_message("📹 Start recording", "info", "Gamepad")
    
    def on_record_stop():
        log_message("⏹️ Stop recording", "info", "Gamepad")
    
    try:
        # 创建手柄控制器
        controller = GamepadController()
        
        # 设置回调函数
        controller.set_movement_callback(on_movement)
        controller.set_grip_callback(on_grip)
        controller.set_record_start_callback(on_record_start)
        controller.set_record_stop_callback(on_record_stop)
        
        # 启动控制
        if not controller.start():
            log_message("启动手柄控制失败", "error", "Gamepad")
            return
        
        log_message("手柄控制测试运行中，按Ctrl+C退出...", "info", "Gamepad")
        
        # 主循环
        try:
            while controller.running:
                controller.update()
                time.sleep(0.02)
        except KeyboardInterrupt:
            log_message("收到中断信号", "info", "Gamepad")
        
    except Exception as e:
        log_message(f"测试过程中出错: {e}", "error", "Gamepad")
    
    finally:
        if 'controller' in locals():
            controller.stop()
        log_message("测试结束", "info", "Gamepad")


if __name__ == "__main__":
    test_gamepad_controller()