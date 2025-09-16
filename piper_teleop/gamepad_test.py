import pygame
import sys

# 初始化pygame和手柄
pygame.init()
pygame.joystick.init()

# 检查手柄连接
if pygame.joystick.get_count() == 0:
    print("未检测到手柄")
    sys.exit()

# 获取第一个手柄
joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"手柄已连接: {joystick.get_name()}")
print(f"轴数量: {joystick.get_numaxes()}")
print(f"按钮数量: {joystick.get_numbuttons()}")
print("按Ctrl+C退出\n")

# 主循环
clock = pygame.time.Clock()
running = True

try:
    while running:
        pygame.event.pump()
        
        # 清屏（移动光标到起始位置）
        print("\033[H\033[J", end="")
        
        # 打印所有轴的值
        print("轴 (Axes):")
        for i in range(joystick.get_numaxes()):
            value = joystick.get_axis(i)
            print(f"  轴{i}: {value:7.4f}")
        
        # 打印所有按钮的值
        print("\n按钮 (Buttons):")
        for i in range(joystick.get_numbuttons()):
            value = joystick.get_button(i)
            print(f"  按钮{i:2d}: {value}", end="  ")
            if (i + 1) % 5 == 0:  # 每5个按钮换行
                print()
        
        clock.tick(60)
        
except KeyboardInterrupt:
    print("\n程序退出")
finally:
    pygame.quit()