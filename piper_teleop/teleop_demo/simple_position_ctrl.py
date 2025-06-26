#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_position_ctrl.py
Ubuntu-22.04 + piper_sdk-0.2.19  示例：
依次把 PiPER 六轴机械臂移动到 3 个预设关节位姿
Author : Derek     Date : 2025-06-14
"""

import time
from math import radians
from piper_sdk import *

# ---------------- 0. 用户可配置参数 ----------------
CAN_PORT      = "can0"     # 如果你的 CAN 口名不同，请改这里
SETTLE_TIME   = 4.0        # 粗略等待到位的秒数

# 三个目标关节位姿（弧度）
WAYPOINTS = [
    [0, 0, 0, 0, 0, 0],                                   # Home
    [0, 0, 0, radians( 30), radians(20), radians(-15)],   # Point-A 
    # [radians(-40), radians(10), radians(-35), 0, 0, 0],   # Point-B 
]

def enable_fun(piper: C_PiperInterface_V2):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
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
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("使能超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
    
    if elapsed_time_flag:
        print("程序自动使能超时,退出程序")
        return False
    
    print("机械臂使能成功!")
    return True

def disable_fun(piper: C_PiperInterface_V2):
    '''
    失能机械臂
    '''
    enable_flag = True
    loop_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    
    while not loop_flag:
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_list = []
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
        enable_list.append(piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
        
        enable_flag = any(enable_list)
        piper.DisableArm(7)
        piper.GripperCtrl(0, 1000, 0x02, 0)
        
        print(f"使能状态: {enable_flag}")
        print("--------------------")
        
        if not enable_flag:
            loop_flag = True
        else: 
            loop_flag = False
            
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("失能超时....")
            elapsed_time_flag = True
            enable_flag = False
            loop_flag = True
            break
        time.sleep(0.5)
    
    resp = not enable_flag
    print(f"失能结果: {resp}")
    return resp

# ---------------- 1. 建立接口并连接 CAN ----------------
piper = C_PiperInterface_V2(CAN_PORT)
piper.ConnectPort()

# ---------------- 2. 使能机械臂 ----------------
if not enable_fun(piper):
    print("使能失败，退出程序")
    exit(1)

# 弧度转换为0.001度的系数
factor = 57295.7795  # 1000*180/3.1415926

# ---------------- 3. 逐点运动 ----------------
for idx, q in enumerate(WAYPOINTS):
    print(f"▶ 正在去往 Waypoint {idx}: {[round(a*180/3.1416,1) for a in q]}°")
    
    # 将弧度转换为0.001度单位
    joint_0 = round(q[0] * factor)
    joint_1 = round(q[1] * factor)
    joint_2 = round(q[2] * factor)
    joint_3 = round(q[3] * factor)
    joint_4 = round(q[4] * factor)
    joint_5 = round(q[5] * factor)
    
    # 设置运动控制模式
    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    
    # 发送关节角度指令
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    
    # 等待运动完成
    time.sleep(SETTLE_TIME)
    
    # 打印当前状态
    print(f"运动状态: {piper.GetArmStatus()}")

print("✅ 已完成全部点位！开始关机操作…")

# ---------------- 4. 失能机械臂 ----------------
if disable_fun(piper):
    print("失能成功!")
else:
    print("失能失败!")

# 断开连接
piper.DisconnectPort()
print("程序运行完成，连接已断开")
