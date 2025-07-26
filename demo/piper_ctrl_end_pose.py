#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import time
from pynput import keyboard # 替换 keyboard 库
from piper_sdk import *

# --- 初始化 Piper 机械臂 ---
# ... (与原代码相同) ...
try:
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
except Exception as e:
    print(f"Error connecting to Piper: {e}")
    exit()
piper.EnableArm(7, 0x02)

# --- 参数和状态变量 ---
RX_DEG, RY_DEG, RZ_DEG = 0.0, 85.0, 0.0
current_X, current_Y, current_Z = 157.0, 0.0, 120.0
INCREMENT_MM = 10.0

# 使用一个字典来跟踪哪些键被按下
keys_pressed = {
    'w': False, 's': False,
    'a': False, 'd': False,
    'up': False, 'down': False
}

# --- 辅助函数 ---
# ... (mm, deg, reset_arm 函数与原代码相同) ...
def mm(val): return int(round(val * 1000))
def deg(val): return int(round(val * 1000))

def reset_arm(piper_instance):
    print("Resetting arm...")
    piper_instance.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    piper_instance.JointCtrl(0, 0, 0, 0, 0, 0)
    time.sleep(2)

def move_to_target(piper_instance, x, y, z, rx, ry, rz):
    piper_instance.EndPoseCtrl(mm(x), mm(y), mm(z), deg(rx), deg(ry), deg(rz))
    print(f"Target Pose -> X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}", end='\r')

# --- pynput 键盘事件处理 ---
def on_press(key):
    """按键按下时的回调函数"""
    global keys_pressed
    key_name = get_key_name(key)
    if key_name in keys_pressed:
        keys_pressed[key_name] = True

def on_release(key):
    """按键释放时的回调函数"""
    global keys_pressed
    key_name = get_key_name(key)
    if key_name in keys_pressed:
        keys_pressed[key_name] = False
    if key == keyboard.KeyCode.from_char('q'):
        # 按下 'q' 键停止监听器
        return False

def get_key_name(key):
    """获取按键的名称，处理特殊键"""
    if isinstance(key, keyboard.KeyCode):
        return key.char
    elif isinstance(key, keyboard.Key):
        return key.name
    return None

# --- 主程序 ---
if __name__ == "__main__":
    try:
        reset_arm(piper)
        time.sleep(2.0)
        piper.MotionCtrl_2(0x01, 0x00, 80, 0x00)

        print(f"Moving to starting position...")
        move_to_target(piper, current_X, current_Y, current_Z, RX_DEG, RY_DEG, RZ_DEG)
        time.sleep(2)

        print("\n" + "="*50)
        print("pynput Control Active")
        print("  - W/S: Control X-axis (+/-)")
        print("  - A/D: Control Y-axis (-/+)")
        print("  - Up/Down Arrow: Control Z-axis (+/-)")
        print("  - Q: Quit")
        print("="*50 + "\n")

        # 启动非阻塞的键盘监听器
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        while listener.running:
            # 根据按下的键更新坐标
            # 这样做可以实现按住不放持续移动
            if keys_pressed['w']: current_X += INCREMENT_MM
            if keys_pressed['s']: current_X -= INCREMENT_MM
            if keys_pressed['d']: current_Y += INCREMENT_MM
            if keys_pressed['a']: current_Y -= INCREMENT_MM
            if keys_pressed['up']: current_Z += INCREMENT_MM
            if keys_pressed['down']: current_Z -= INCREMENT_MM
            
            if any(keys_pressed.values()):
                move_to_target(piper, current_X, current_Y, current_Z, RX_DEG, RY_DEG, RZ_DEG)
            
            time.sleep(0.05) # 循环延时，控制指令发送频率

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nProgram finished. Resetting arm one last time.")
        reset_arm(piper)
        piper.DisconnectPort()
        print("Disconnected.")