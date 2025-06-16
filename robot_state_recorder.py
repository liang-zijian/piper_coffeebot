#!/usr/bin/env python3
"""
机械臂状态记录模块
用于获取机械臂关节位置和记录控制动作
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.logging import RichHandler
import logging
from collections import deque

# 配置rich日志
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("RobotStateRecorder")

class RobotStateRecorder:
    """机械臂状态记录器"""
    
    def __init__(self, piper_robot, buffer_size: int = 1000):
        """
        初始化机械臂状态记录器
        
        Args:
            piper_robot: Piper机械臂实例
            buffer_size: 历史数据缓冲区大小
        """
        self.piper = piper_robot
        self.buffer_size = buffer_size
        
        # 历史数据缓冲区
        self.position_history = deque(maxlen=buffer_size)
        self.velocity_history = deque(maxlen=buffer_size)
        self.action_history = deque(maxlen=buffer_size)
        self.timestamp_history = deque(maxlen=buffer_size)
        
        # 当前状态
        self.current_position = None
        self.current_velocity = None
        self.current_action = None
        self.last_position = None
        
        # 关节名称
        self.JOINT_NAMES = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8'
        ]
        
        # 获取关节索引
        self.dofs_idx = [self.piper.get_joint(name).dof_idx_local for name in self.JOINT_NAMES]
        
        logger.info("✅ 机械臂状态记录器初始化完成")
    
    def update_robot_state(self) -> Dict[str, np.ndarray]:
        """
        更新机械臂状态
        
        Returns:
            包含当前关节位置和速度的字典
        """
        try:
            # 获取关节位置和速度
            current_position = self.piper.get_dofs_position().cpu().numpy()
            current_velocity = self.piper.get_dofs_velocity().cpu().numpy()
            
            # 更新当前状态
            self.current_position = current_position.copy()
            self.current_velocity = current_velocity.copy()
            
            # 添加到历史记录
            timestamp = time.time()
            self.position_history.append(current_position.copy())
            self.velocity_history.append(current_velocity.copy())
            self.timestamp_history.append(timestamp)
            
            return {
                "joint_position": current_position,
                "joint_velocity": current_velocity,
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"更新机械臂状态失败: {e}")
            return {}
    
    def record_action(self, action: np.ndarray) -> None:
        """
        记录控制动作
        
        Args:
            action: 控制动作数组
        """
        if action is not None:
            self.current_action = action.copy()
            self.action_history.append(action.copy())
    
    def calculate_position_delta(self) -> Optional[np.ndarray]:
        """
        计算位置增量（相对于上一帧的变化）
        
        Returns:
            位置增量数组，如果无法计算则返回None
        """
        if self.current_position is None:
            return None
        
        if self.last_position is None:
            # 第一次调用，返回零增量
            self.last_position = self.current_position.copy()
            return np.zeros_like(self.current_position)
        
        # 计算增量
        delta = self.current_position - self.last_position
        self.last_position = self.current_position.copy()
        
        return delta
    
    def get_state_vector_for_lerobot(self) -> Optional[np.ndarray]:
        """
        获取适用于lerobot数据集的状态向量
        状态向量格式：[joint_positions(9), joint_velocities(8)] = 17维
        
        Returns:
            17维状态向量，如果数据不可用则返回None
        """
        if self.current_position is None or self.current_velocity is None:
            return None
        
        try:
            # 确保关节位置为9维（如果是8维则复制夹爪值）
            if len(self.current_position) == 8:
                # 复制夹爪值以使其成为9维
                joint_pos = np.concatenate([
                    self.current_position, 
                    [self.current_position[-1]]  # 复制最后一个夹爪值
                ])
            else:
                joint_pos = self.current_position[:9]  # 取前9维
            
            # 关节速度取前8维
            joint_vel = self.current_velocity[:8]
            
            # 拼接状态向量：位置(9) + 速度(8) = 17维
            state_vec = np.concatenate([joint_pos, joint_vel]).astype(np.float32)
            
            return state_vec
            
        except Exception as e:
            logger.error(f"构建状态向量失败: {e}")
            return None
    
    def get_action_for_lerobot(self, action_type: str = "position_delta") -> Optional[np.ndarray]:
        """
        获取适用于lerobot数据集的动作向量
        
        Args:
            action_type: 动作类型 ("position_delta", "direct_action")
            
        Returns:
            8维动作向量，如果数据不可用则返回None
        """
        if action_type == "position_delta":
            delta = self.calculate_position_delta()
            if delta is not None:
                # 确保为8维
                if len(delta) >= 8:
                    return delta[:8].astype(np.float32)
                else:
                    # 如果不足8维，用零填充
                    padded_delta = np.zeros(8, dtype=np.float32)
                    padded_delta[:len(delta)] = delta
                    return padded_delta
        
        elif action_type == "direct_action":
            if self.current_action is not None:
                # 确保为8维
                if len(self.current_action) >= 8:
                    return self.current_action[:8].astype(np.float32)
                else:
                    # 如果不足8维，用零填充
                    padded_action = np.zeros(8, dtype=np.float32)
                    padded_action[:len(self.current_action)] = self.current_action
                    return padded_action
        
        return None
    
    def get_frame_data_for_lerobot(self, action_type: str = "position_delta") -> Dict[str, np.ndarray]:
        """
        获取适用于lerobot数据集的帧数据
        
        Args:
            action_type: 动作类型
            
        Returns:
            包含observation.state和actions的字典
        """
        # 更新机械臂状态
        self.update_robot_state()
        
        # 获取状态向量
        state_vec = self.get_state_vector_for_lerobot()
        if state_vec is None:
            return {}
        
        # 获取动作向量
        action_vec = self.get_action_for_lerobot(action_type)
        if action_vec is None:
            # 如果没有动作，使用零向量
            action_vec = np.zeros(8, dtype=np.float32)
        
        return {
            "observation.state": state_vec,
            "actions": action_vec
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """获取统计信息"""
        if len(self.position_history) == 0:
            return {"message": "没有历史数据"}
        
        positions = np.array(list(self.position_history))
        velocities = np.array(list(self.velocity_history))
        
        stats = {
            "buffer_size": len(self.position_history),
            "position_stats": {
                "mean": np.mean(positions, axis=0),
                "std": np.std(positions, axis=0),
                "min": np.min(positions, axis=0),
                "max": np.max(positions, axis=0)
            },
            "velocity_stats": {
                "mean": np.mean(velocities, axis=0),
                "std": np.std(velocities, axis=0),
                "min": np.min(velocities, axis=0),
                "max": np.max(velocities, axis=0)
            }
        }
        
        return stats
    
    def clear_history(self):
        """清空历史数据"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.action_history.clear()
        self.timestamp_history.clear()
        logger.info("历史数据已清空")
    
    def get_current_joint_info(self) -> Dict[str, float]:
        """获取当前关节信息的字典格式"""
        if self.current_position is None:
            return {}
        
        joint_info = {}
        for i, name in enumerate(self.JOINT_NAMES):
            if i < len(self.current_position):
                joint_info[name] = float(self.current_position[i])
        
        return joint_info


def test_robot_state_recorder():
    """测试机械臂状态记录器（需要真实的机械臂实例）"""
    # 这个测试函数需要真实的机械臂实例
    # 在实际使用中，piper_robot应该是一个有效的Piper机械臂实例
    logger.info("机械臂状态记录器测试函数")
    logger.info("需要真实的机械臂实例才能运行测试")
    
    # 模拟测试
    class MockPiperRobot:
        def __init__(self):
            self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
            self.position = np.random.rand(8) * 2 - 1  # 随机位置
            self.velocity = np.random.rand(8) * 0.1 - 0.05  # 随机速度
        
        def get_joint(self, name):
            class MockJoint:
                def __init__(self, idx):
                    self.dof_idx_local = idx
            return MockJoint(self.joint_names.index(name))
        
        def get_dofs_position(self):
            class MockTensor:
                def __init__(self, data):
                    self.data = data
                def cpu(self):
                    return self
                def numpy(self):
                    return self.data
            return MockTensor(self.position)
        
        def get_dofs_velocity(self):
            class MockTensor:
                def __init__(self, data):
                    self.data = data
                def cpu(self):
                    return self
                def numpy(self):
                    return self.data
            return MockTensor(self.velocity)
    
    # 创建模拟机械臂
    mock_robot = MockPiperRobot()
    
    # 创建状态记录器
    recorder = RobotStateRecorder(mock_robot)
    
    # 测试几次更新
    for i in range(5):
        # 更新机械臂状态
        state_data = recorder.update_robot_state()
        logger.info(f"更新 {i+1}: 关节位置均值 = {np.mean(state_data['joint_position']):.4f}")
        
        # 模拟动作
        mock_action = np.random.rand(8) * 0.01
        recorder.record_action(mock_action)
        
        # 获取lerobot格式数据
        frame_data = recorder.get_frame_data_for_lerobot()
        logger.info(f"状态向量维度: {frame_data['observation.state'].shape}")
        logger.info(f"动作向量维度: {frame_data['actions'].shape}")
        
        time.sleep(0.1)
    
    # 获取统计信息
    stats = recorder.get_statistics()
    logger.info(f"缓冲区大小: {stats['buffer_size']}")
    
    logger.info("测试完成")


if __name__ == "__main__":
    test_robot_state_recorder()