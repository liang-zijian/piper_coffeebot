#!/bin/bash

# Piper机械臂数据录制启动脚本

echo "=================================="
echo "  Piper机械臂数据录制系统"
echo "=================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    exit 1
fi

# 检查必要的Python包
echo "检查依赖包..."
python3 -c "
import sys
missing_packages = []

try:
    import rich
except ImportError:
    missing_packages.append('rich')

try:
    import pygame
except ImportError:
    missing_packages.append('pygame')

try:
    import pyrealsense2
except ImportError:
    missing_packages.append('pyrealsense2')

try:
    import numpy
except ImportError:
    missing_packages.append('numpy')

try:
    import cv2
except ImportError:
    missing_packages.append('opencv-python')

try:
    import genesis
except ImportError:
    missing_packages.append('genesis-world')

if missing_packages:
    print('缺少以下依赖包:', ', '.join(missing_packages))
    print('请使用以下命令安装:')
    print('pip install', ' '.join(missing_packages))
    sys.exit(1)
else:
    print('✅ 所有依赖包已安装')
"

if [ $? -ne 0 ]; then
    exit 1
fi

# 检查硬件连接
echo ""
echo "检查硬件连接..."

# 检查CAN接口
if ip link show can0 &> /dev/null; then
    echo "✅ CAN接口 can0 已配置"
else
    echo "⚠️  CAN接口 can0 未找到，请检查机械臂连接"
fi

# 检查RealSense相机
python3 -c "
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) > 0:
    print(f'✅ 检测到 {len(devices)} 个RealSense设备')
    for i, device in enumerate(devices):
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        print(f'   设备{i+1}: {name} (序列号: {serial})')
else:
    print('⚠️  未检测到RealSense设备')
"

# 检查游戏手柄
python3 -c "
import pygame
pygame.init()
joystick_count = pygame.joystick.get_count()
if joystick_count > 0:
    print(f'✅ 检测到 {joystick_count} 个游戏手柄')
    for i in range(joystick_count):
        joy = pygame.joystick.Joystick(i)
        joy.init()
        print(f'   手柄{i+1}: {joy.get_name()}')
        joy.quit()
else:
    print('⚠️  未检测到游戏手柄')
pygame.quit()
"

echo ""
echo "硬件检查完成"
echo ""

# 设置默认参数
DATASET_DIR="piper_real_dataset"
RESUME=""
FPS="30.0"
TASK="Real robot manipulation with Piper arm"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --dataset_dir DIR    数据集保存目录 (默认: piper_real_dataset)"
            echo "  --resume            从现有数据集继续录制"
            echo "  --fps FPS           录制帧率 (默认: 30.0)"
            echo "  --task DESCRIPTION  任务描述"
            echo "  --help             显示此帮助信息"
            echo ""
            echo "手柄控制说明:"
            echo "  左摇杆: XY平移"
            echo "  LT/RT: Z轴平移"
            echo "  右摇杆Y: 关节5控制"
            echo "  LB/RB: 关节6控制"
            echo "  A键: 夹爪开合"
            echo "  Y键: 开始录制"
            echo "  X键: 停止录制"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "启动参数:"
echo "  数据集目录: $DATASET_DIR"
echo "  录制帧率: $FPS"
echo "  任务描述: $TASK"
if [ -n "$RESUME" ]; then
    echo "  模式: 增量录制"
else
    echo "  模式: 新建数据集"
fi

echo ""
echo "⚠️  安全提醒："
echo "   1. 确保机械臂运动空间安全"
echo "   2. 随时准备紧急停止"
echo "   3. 录制前请检查所有连接"
echo ""

read -p "按Enter键开始启动录制系统，或Ctrl+C取消..."

echo ""
echo "🚀 启动数据录制系统..."
echo ""

# 启动录制系统
python3 piper_real_data_recorder.py \
    --dataset_dir "$DATASET_DIR" \
    --fps "$FPS" \
    --task_description "$TASK" \
    $RESUME \
    --vis

echo ""
echo "📝 录制系统已退出"