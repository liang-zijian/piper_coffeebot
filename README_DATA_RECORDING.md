# Piper机械臂真实环境数据录制系统

本系统实现了Piper机械臂在真实环境中的数据录制功能，支持三个RealSense相机同时采集数据，并将数据保存为LeRobot格式的数据集。

## 功能特性

- ✅ **三相机同步采集**: 支持腕部相机、两个侧视相机同时采集RGB图像
- ✅ **机械臂状态记录**: 实时记录关节位置、速度和控制动作
- ✅ **LeRobot格式输出**: 兼容LeRobot训练框架的数据格式
- ✅ **增量录制**: 支持从现有数据集继续录制
- ✅ **手柄控制**: 使用游戏手柄控制机械臂和录制
- ✅ **实时状态显示**: 使用Rich库显示系统运行状态
- ✅ **模块化设计**: 各功能模块独立，便于维护和扩展

## 系统架构

```
piper_real_data_recorder.py          # 主录制脚本
├── multi_realsense_cameras.py       # 多相机管理模块
├── robot_state_recorder.py          # 机械臂状态记录模块
├── lerobot_dataset_manager.py       # LeRobot数据集管理模块
├── gamepad_controller.py            # 手柄控制模块
├── live_status_display.py           # 实时状态显示模块
└── piper_xiaoji_real.py             # Piper控制器（现有）
```

## 安装依赖

```bash
# 安装Python依赖
pip install rich pygame pyrealsense2 numpy opencv-python scipy

# 安装LeRobot (如果尚未安装)
pip install lerobot

# 安装Genesis仿真引擎 (如果尚未安装)
pip install genesis-world
```

## 硬件要求

1. **Piper机械臂**: 配置了CAN总线通信
2. **RealSense相机**: 最多支持3个D435i相机
   - 腕部相机 (ee_cam)
   - 侧视相机1 (rgb_rs_0)
   - 侧视相机2 (rgb_rs_1)
3. **游戏手柄**: 支持Xbox控制器或兼容手柄
4. **CAN接口**: 用于机械臂通信

## 使用方法

### 1. 基本使用

```bash
# 启动数据录制系统
python piper_real_data_recorder.py

# 指定数据集目录
python piper_real_data_recorder.py --dataset_dir my_dataset

# 从现有数据集继续录制
python piper_real_data_recorder.py --resume --dataset_dir existing_dataset
```

### 2. 命令行参数

```bash
python piper_real_data_recorder.py --help

可选参数:
  --dataset_dir DATASET_DIR     数据集保存目录 (默认: piper_real_dataset)
  --repo_id REPO_ID            数据集仓库ID (默认: piper/real-manipulation)
  --fps FPS                    录制帧率 (默认: 30.0)
  --task_description TASK      任务描述 (默认: Real robot manipulation with Piper arm)
  --vis                        是否显示仿真场景 (默认: True)
  --resume                     是否从现有数据集继续录制 (默认: False)
```

### 3. 手柄控制说明

| 按键/摇杆 | 功能 |
|-----------|------|
| 左摇杆 | XY平移控制 |
| LT/RT | Z轴平移控制 |
| 右摇杆Y轴 | 关节5控制 |
| LB/RB | 关节6控制 |
| A键 | 夹爪开合 |
| Y键 | 开始录制 |
| **X键** | **停止录制** |

### 4. 操作流程

1. **启动系统**: 运行主脚本，等待所有模块初始化完成
2. **检查状态**: 查看实时状态面板，确认所有设备正常连接
3. **开始录制**: 按Y键开始录制新的episode
4. **操作机械臂**: 使用手柄控制机械臂完成操作任务
5. **停止录制**: 按X键停止当前episode录制
6. **重复录制**: 可以重复步骤3-5录制多个episodes
7. **安全退出**: 按Ctrl+C安全退出程序

## 数据集格式

每个录制帧包含以下数据：

```python
frame = {
    # 摄像头图像 (CHW格式, RGB)
    "observation.images.ee_cam": (3, 480, 640),        # 腕部相机
    "observation.images.rgb_rs_0": (3, 480, 640), # 侧视相机1
    "observation.images.rgb_rs_1": (3, 480, 640), # 侧视相机2
    
    # 机械臂状态 (17维: 位置9维 + 速度8维)
    "observation.state": (17,),
    
    # 控制动作 (8维关节控制)
    "actions": (8,),
    
    # 任务描述
    "task": "move the coffee cup to the coffee machine"
}
```

## 状态显示界面

系统提供实时状态显示，包括：

- **系统状态**: FPS、录制状态、episode计数
- **机械臂状态**: 关节位置、速度、夹爪状态
- **相机状态**: 连接状态、FPS
- **手柄状态**: 连接状态、当前输入
- **数据集状态**: 目录、episode数量、大小
- **系统消息**: 最近的操作日志

## 模块说明

### 1. multi_realsense_cameras.py
负责管理多个RealSense相机，支持自动设备检测和图像同步获取。

### 2. robot_state_recorder.py
记录机械臂的关节状态和控制动作，计算位置增量。

### 3. lerobot_dataset_manager.py
管理LeRobot格式数据集的创建、保存和增量录制。

### 4. gamepad_controller.py
处理游戏手柄输入，提供录制控制接口。

### 5. live_status_display.py
使用Rich库实现实时状态显示界面。

## 故障排除

### 1. 相机相关问题

```bash
# 检查相机连接
python -c "import pyrealsense2 as rs; print(rs.context().query_devices())"

# 测试多相机功能
python multi_realsense_cameras.py
```

### 2. 机械臂连接问题

```bash
# 检查CAN总线
ip link show can0

# 测试机械臂连接
python piper_teleop/test_gamepad.py
```

### 3. 手柄问题

```bash
# 检测手柄
python -c "import pygame; pygame.init(); print(f'检测到 {pygame.joystick.get_count()} 个手柄')"

# 测试手柄功能
python gamepad_controller.py
```

### 4. 数据集问题

```bash
# 测试LeRobot数据集功能
python lerobot_dataset_manager.py

# 检查数据集内容
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('your_dataset_path'); print(len(ds))"
```

## 注意事项

1. **安全第一**: 使用前请确保机械臂运动空间安全，随时准备紧急停止
2. **数据备份**: 重要数据请及时备份，避免意外丢失
3. **资源监控**: 录制过程会占用较多磁盘空间，请监控可用空间
4. **网络要求**: 如需上传到HuggingFace Hub，请确保网络连接正常
5. **权限问题**: 可能需要sudo权限访问CAN设备和相机

## 扩展功能

系统采用模块化设计，可以轻松扩展：

- 添加新的相机类型
- 支持不同的机械臂型号
- 集成其他传感器数据
- 自定义数据集格式
- 添加实时数据分析功能

## 技术支持

如遇到问题，请检查：

1. 硬件连接是否正常
2. 依赖库是否正确安装
3. 权限设置是否正确
4. 日志输出中的错误信息

更多详细信息请参考各模块的代码注释和测试函数。