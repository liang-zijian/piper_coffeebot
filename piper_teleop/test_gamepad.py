#!/usr/bin/env python3
import pygame
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("未检测到游戏手柄")
    exit(1)

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"已连接游戏手柄: {joystick.get_name()}")
print("请移动摇杆或按按钮，将显示输入值（Ctrl+C退出）...")

try:
    while True:
        pygame.event.pump()
        
        # 检查轴输入
        axes_data = []
        for i in range(joystick.get_numaxes()):
            value = joystick.get_axis(i)
            if abs(value) > 0.1:  # 死区过滤
                axes_data.append(f"轴{i}: {value:.3f}")
        
        # 检查按钮输入
        buttons_data = []
        for i in range(joystick.get_numbuttons()):
            if joystick.get_button(i):
                buttons_data.append(f"按钮{i}")
        
        # 输出非零值
        if axes_data or buttons_data:
            output = []
            if axes_data:
                output.append("轴: " + ", ".join(axes_data))
            if buttons_data:
                output.append("按钮: " + ", ".join(buttons_data))
            print(" | ".join(output))
        
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n测试结束")
finally:
    pygame.quit() 