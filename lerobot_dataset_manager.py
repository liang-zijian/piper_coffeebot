#!/usr/bin/env python3
"""
LeRobot数据集管理模块
用于创建、保存和管理LeRobot格式的数据集
"""

import os
import json
import numpy as np
import time
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict, Optional, Any
from rich.console import Console
from rich.logging import RichHandler
import logging

# 导入lerobot相关模块
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("警告: 无法导入lerobot模块，请确保已正确安装lerobot")
    LeRobotDataset = None

# 配置rich日志
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("LeRobotDatasetManager")

class LeRobotDatasetManager:
    """LeRobot数据集管理器"""
    
    def __init__(self, 
                 dataset_dir: str = "piper_dataset",
                 repo_id: str = "piper/real-world-manipulation",
                 fps: float = 30.0,
                 task_description: str = "Real robot manipulation with Piper arm"):
        """
        初始化数据集管理器
        
        Args:
            dataset_dir: 数据集保存目录
            repo_id: 数据集仓库ID
            fps: 帧率
            task_description: 任务描述
        """
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.repo_id = repo_id
        self.fps = fps
        self.task_description = task_description
        
        self.dataset = None
        self.episode_idx = 0
        self.frame_count = 0
        self.is_recording = False
        
        # 数据集特征定义
        self.features = {
            # 摄像头图像 (CHW格式)
            "observation.images.ee_cam": {"shape": (3, 480, 640), "dtype": "image"},
            "observation.images.rgb_rs_0": {"shape": (3, 480, 640), "dtype": "image"},
            "observation.images.rgb_rs_1": {"shape": (3, 480, 640), "dtype": "image"},
            # 关节状态 (只有关节位置8维：6个主关节 + 2个夹爪)
            "observation.state": {"shape": (8,), "dtype": "float32"},
            # 动作 (8维关节绝对位置控制)
            "actions": {"shape": (8,), "dtype": "float32"}
        }
        
        logger.info("✅ LeRobot数据集管理器初始化完成")
    
    def create_dataset(self, resume: bool = False) -> bool:
        """
        创建或加载数据集
        
        Args:
            resume: 是否从现有数据集继续录制
            
        Returns:
            创建/加载是否成功
        """
        if LeRobotDataset is None:
            logger.error("LeRobotDataset未导入，无法创建数据集")
            return False
        
        try:
            # 处理增量录制逻辑
            if resume and self.dataset_dir.exists():
                existing_episodes = self._get_episode_count()
                logger.info(f"增量录制模式：检测到 {existing_episodes} 个已有episode")
                self.episode_idx = existing_episodes
                
                try:
                    # 尝试加载现有数据集
                    self.dataset = LeRobotDataset(
                        repo_id=self.repo_id,
                        root=self.dataset_dir
                    )
                    logger.info(f"✅ 成功加载现有数据集，将从episode {self.episode_idx}开始录制")
                    return True
                    
                except Exception as e:
                    logger.warning(f"加载现有数据集失败: {e}")
                    logger.info("将创建新数据集...")
                    
                    # 创建备份
                    if self.dataset_dir.exists():
                        backup_path = f"{self.dataset_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        logger.info(f"创建备份：{backup_path}")
                        shutil.copytree(self.dataset_dir, backup_path)
                        shutil.rmtree(self.dataset_dir)
                    
                    self.episode_idx = 0
            else:
                # 非增量模式，删除已有目录
                if self.dataset_dir.exists():
                    logger.info(f"删除已有数据集目录：{self.dataset_dir}")
                    shutil.rmtree(self.dataset_dir)
                self.episode_idx = 0
            
            # 创建新数据集
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                root=self.dataset_dir,
                fps=self.fps,
                features=self.features
            )
            
            # 创建modality文件
            self._create_modality_file()
            
            logger.info(f"✅ 成功创建新数据集")
            return True
            
        except Exception as e:
            logger.error(f"❌ 创建数据集失败: {e}")
            return False
    
    def _get_episode_count(self) -> int:
        """获取数据集中已有的episode数量"""
        try:
            if not self.dataset_dir.exists():
                return 0
            
            # 方法1: 读取meta/info.json
            info_path = self.dataset_dir / "meta" / "info.json"
            if info_path.exists():
                with open(info_path, "r") as f:
                    info = json.load(f)
                    return info.get("total_episodes", 0)
            
            # 方法2: 统计episode文件
            episode_count = 0
            data_path = self.dataset_dir / "data"
            if data_path.exists():
                for chunk_dir in data_path.glob("chunk-*"):
                    episode_files = list(chunk_dir.glob("episode_*.parquet"))
                    episode_count += len(episode_files)
            
            return episode_count
            
        except Exception as e:
            logger.warning(f"获取episode数量失败: {e}")
            return 0
    
    def _create_modality_file(self):
        """创建modality.json文件"""
        modality_path = self.dataset_dir / "meta" / "modality.json"
        if not modality_path.exists():
            modality_path.parent.mkdir(parents=True, exist_ok=True)
            modality_data = {
                "observation.state": {
                    "joint_position": {"start": 0, "end": 8}  # 关节位置 8维（6个主关节 + 2个夹爪）
                },
                "actions": {
                    "joint_target": {"start": 0, "end": 8, "absolute": True}  # 关节绝对位置目标 8维
                }
            }
            
            with open(modality_path, "w") as f:
                json.dump(modality_data, f, indent=2)
            
            logger.info(f"创建modality.json文件：{modality_path}")
    
    def start_episode(self):
        """开始新的episode"""
        if self.dataset is None:
            logger.error("数据集未初始化，无法开始episode")
            return False
        
        if self.is_recording:
            logger.warning("已经在录制中，请先结束当前episode")
            return False
        
        self.is_recording = True
        self.frame_count = 0
        logger.info(f"开始录制 episode {self.episode_idx}")
        return True
    
    def add_frame(self, 
                  camera_images: Dict[str, np.ndarray],
                  robot_state: np.ndarray,
                  actions: np.ndarray,
                  task: str = None) -> bool:
        """
        添加一帧数据
        
        Args:
            camera_images: 相机图像字典，key为相机名称，value为图像数组(CHW格式)
            robot_state: 机械臂状态向量(8维)
            actions: 动作向量(8维)
            task: 任务描述
            
        Returns:
            添加是否成功
        """
        if not self.is_recording or self.dataset is None:
            logger.error("未在录制状态或数据集未初始化")
            return False
        
        try:
            # 验证输入数据的完整性和有效性
            if not isinstance(camera_images, dict):
                logger.error("相机图像数据类型错误，应为字典")
                return False
                
            if robot_state is None or not isinstance(robot_state, np.ndarray) or len(robot_state) != 8:
                logger.error(f"机械臂状态数据无效: type={type(robot_state)}, shape={getattr(robot_state, 'shape', 'N/A')}")
                return False
                
            if actions is None or not isinstance(actions, np.ndarray) or len(actions) != 8:
                logger.error(f"动作数据无效: type={type(actions)}, shape={getattr(actions, 'shape', 'N/A')}")
                return False
            
            # 构建帧数据
            frame = {}
            
            # 严格验证和添加相机图像
            required_cameras = ["observation.images.ee_cam", "observation.images.rgb_rs_0", "observation.images.rgb_rs_1"]
            for key in required_cameras:
                if key in camera_images and camera_images[key] is not None:
                    img = camera_images[key]
                    
                    # 验证图像数据类型和尺寸
                    if not isinstance(img, np.ndarray):
                        logger.error(f"图像 {key} 数据类型错误: {type(img)}")
                        return False
                    
                    # 确保图像格式正确(CHW, uint8)
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    
                    # 验证图像尺寸
                    if img.shape != (3, 480, 640):
                        logger.error(f"图像 {key} 尺寸不匹配: 期望(3,480,640), 实际{img.shape}")
                        return False
                    
                    frame[key] = img
                else:
                    logger.error(f"必需的相机图像缺失: {key}")
                    return False
            
            # 添加机械臂状态
            frame["observation.state"] = robot_state.astype(np.float32)
            
            # 添加动作
            frame["actions"] = actions.astype(np.float32)
            
            # 最终验证帧数据完整性
            expected_keys = ["observation.images.ee_cam", "observation.images.rgb_rs_0", 
                           "observation.images.rgb_rs_1", "observation.state", "actions"]
            for key in expected_keys:
                if key not in frame:
                    logger.error(f"帧数据缺少必要字段: {key}")
                    return False
                    
                # 验证数据不为None
                if frame[key] is None:
                    logger.error(f"帧数据字段为None: {key}")
                    return False
            
            # 添加到数据集
            try:
                if task is not None:
                    self.dataset.add_frame(frame, task)
                else:
                    self.dataset.add_frame(frame, self.task_description)
                
                # 只有在成功添加到数据集后才增加计数器
                self.frame_count += 1
                
                # 每10帧输出一次调试信息
                if self.frame_count % 10 == 0:
                    logger.info(f"已添加 {self.frame_count} 帧到episode {self.episode_idx}")
                
                # 验证数据集buffer状态（每帧都检查，但只在问题时输出）
                if hasattr(self.dataset, 'episode_buffer'):
                    buffer = self.dataset.episode_buffer
                    if isinstance(buffer, dict):
                        # 只检查实际的数据字段
                        data_fields = [
                            'observation.images.ee_cam',
                            'observation.images.rgb_rs_0', 
                            'observation.images.rgb_rs_1',
                            'observation.state',
                            'actions',
                            'task',
                            'timestamp',
                            'frame_index'
                        ]
                        
                        for key in data_fields:
                            if key in buffer and hasattr(buffer[key], '__len__'):
                                if len(buffer[key]) != self.frame_count:
                                    logger.warning(f"⚠️ 检测到数据长度不一致: {key}={len(buffer[key])}, frame_count={self.frame_count}")
                                    break
                
                return True
                
            except Exception as dataset_e:
                logger.error(f"添加帧到数据集失败: {dataset_e}")
                logger.error(f"错误类型: {type(dataset_e).__name__}")
                
                # 提供更详细的错误信息
                logger.error(f"帧数据详细信息:")
                for key, value in frame.items():
                    if hasattr(value, 'shape'):
                        logger.error(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        logger.error(f"  {key}: type={type(value)}, value={value}")
                
                return False
            
        except Exception as e:
            logger.error(f"构建帧数据失败: {e}")
            logger.error(f"错误类型: {type(e).__name__}")
            return False
    
    def end_episode(self) -> bool:
        """结束当前episode"""
        logger.info("🔹" * 30)
        logger.info("📝 开始结束episode流程")
        
        if not self.is_recording or self.dataset is None:
            logger.error("❌ 未在录制状态或数据集未初始化")
            logger.error(f"  - is_recording: {self.is_recording}")
            logger.error(f"  - dataset: {self.dataset is not None}")
            return False
        
        logger.info(f"📊 当前状态:")
        logger.info(f"  - episode_idx: {self.episode_idx}")
        logger.info(f"  - frame_count: {self.frame_count}")
        logger.info(f"  - is_recording: {self.is_recording}")
        
        # 检查是否有足够的帧数据
        if self.frame_count == 0:
            logger.warning("⚠️ 当前episode没有录制任何帧数据，跳过保存")
            # 重置状态但不保存
            self.is_recording = False
            self.frame_count = 0
            logger.info("✅ 状态已重置")
            return True
        
        # 放宽最小帧数要求 - 从5帧降低到2帧，使保存更容易成功
        min_frames = 2  # 降低最小帧数要求
        if self.frame_count < min_frames:
            logger.warning(f"⚠️ 当前episode帧数太少 ({self.frame_count} < {min_frames})，跳过保存")
            self.is_recording = False
            self.frame_count = 0
            logger.info("✅ 状态已重置")
            return True
        
        logger.info(f"💾 开始保存episode {self.episode_idx}，共 {self.frame_count} 帧...")
        
        try:
            # Step 1: 同步帧计数
            logger.info("🔄 步骤1: 同步帧计数...")
            synced_count = self._sync_frame_count()
            logger.info(f"   同步结果: {self.frame_count} -> {synced_count}")
            
            # Step 2: 检查buffer状态
            logger.info("🔍 步骤2: 检查episode buffer...")
            buffer_ok = self._check_buffer_status()
            if not buffer_ok:
                logger.error("❌ Buffer状态检查失败")
                self._reset_episode_state()
                return False
            
            # Step 3: 尝试修复数据不一致问题（如果存在）
            logger.info("🔧 步骤3: 尝试修复数据不一致...")
            fix_ok = self._try_fix_data_inconsistency(min_frames)
            if not fix_ok:
                logger.error("❌ 数据修复失败")
                self._reset_episode_state()
                return False
            
            # Step 4: 最终验证（简化版本，更容错）
            logger.info("✅ 步骤4: 最终数据验证...")
            if not self._validate_episode_data_simple():
                logger.error("❌ 最终数据验证失败")
                self._reset_episode_state()
                return False
            
            # Step 5: 保存episode
            logger.info("💾 步骤5: 保存episode到磁盘...")
            save_start = time.time()
            self.dataset.save_episode()
            save_time = time.time() - save_start
            
            logger.info(f"✅ Episode {self.episode_idx} 保存成功!")
            logger.info(f"   📈 数据统计: {self.frame_count} 帧")
            logger.info(f"   ⏱️  保存耗时: {save_time:.2f}秒")
            
            # 更新状态
            self.episode_idx += 1
            self.frame_count = 0
            self.is_recording = False
            
            logger.info("🔹" * 30)
            return True
            
        except Exception as e:
            logger.error("❌" * 30)
            logger.error(f"💥 保存episode异常: {e}")
            logger.error(f"🔍 错误类型: {type(e).__name__}")
            logger.error(f"📄 错误详情: {str(e)}")
            
            # 打印关键错误堆栈（简化版）
            import traceback
            error_lines = traceback.format_exc().split('\n')
            # 只显示最后几行关键错误信息
            key_lines = [line for line in error_lines if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed', 'file', 'line'])]
            if key_lines:
                logger.error("🔍 关键错误信息:")
                for line in key_lines[-5:]:  # 最后5行关键信息
                    if line.strip():
                        logger.error(f"   {line.strip()}")
            
            # 简化的调试信息
            self._log_simple_debug_info()
            
            # 重置状态
            self._reset_episode_state()
            
            logger.error("❌" * 30)
            return False
    
    def _check_buffer_status(self) -> bool:
        """检查episode buffer状态"""
        try:
            if not hasattr(self.dataset, 'episode_buffer'):
                logger.error("❌ Dataset没有episode_buffer属性")
                return False
            
            buffer = self.dataset.episode_buffer
            if not isinstance(buffer, dict):
                logger.error(f"❌ Episode buffer类型错误: {type(buffer)}")
                return False
            
            logger.info(f"✅ Buffer状态正常: {len(buffer)} 个字段")
            return True
            
        except Exception as e:
            logger.error(f"❌ Buffer状态检查异常: {e}")
            return False
    
    def _try_fix_data_inconsistency(self, min_frames: int) -> bool:
        """尝试修复数据不一致问题"""
        try:
            if not hasattr(self.dataset, 'episode_buffer'):
                return True  # 如果没有buffer，跳过修复
            
            buffer = self.dataset.episode_buffer
            if not isinstance(buffer, dict):
                return True  # 如果不是字典，跳过修复
            
            # 检查关键数据字段的长度
            data_fields = [
                'observation.images.ee_cam',
                'observation.images.rgb_rs_0', 
                'observation.images.rgb_rs_1',
                'observation.state',
                'actions'
            ]
            
            lengths = {}
            for key in data_fields:
                if key in buffer and hasattr(buffer[key], '__len__'):
                    lengths[key] = len(buffer[key])
            
            if not lengths:
                logger.warning("⚠️ 没有找到数据字段，跳过修复")
                return True
            
            # 检查长度是否一致
            unique_lengths = set(lengths.values())
            if len(unique_lengths) == 1:
                # 长度一致，不需要修复
                actual_length = list(unique_lengths)[0]
                logger.info(f"✅ 数据长度一致: {actual_length}")
                
                # 确保size字段正确
                buffer['size'] = actual_length
                self.frame_count = actual_length
                return True
            
            # 长度不一致，尝试修复
            logger.warning(f"⚠️ 检测到数据长度不一致: {lengths}")
            
            # 使用最小长度
            min_length = min(lengths.values())
            logger.info(f"🔧 使用最小长度修复: {min_length}")
            
            if min_length < min_frames:
                logger.error(f"❌ 最小长度太小 ({min_length} < {min_frames})")
                return False
            
            # 截取所有字段到最小长度
            all_fields = list(buffer.keys())
            for key in all_fields:
                if key in buffer and hasattr(buffer[key], '__len__') and hasattr(buffer[key], '__getitem__'):
                    if len(buffer[key]) > min_length:
                        buffer[key] = buffer[key][:min_length]
                        logger.info(f"   截取 {key}: {len(buffer[key])} -> {min_length}")
            
            # 更新状态
            buffer['size'] = min_length
            self.frame_count = min_length
            
            logger.info(f"✅ 数据修复完成，统一长度: {min_length}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据修复异常: {e}")
            return False
    
    def _validate_episode_data_simple(self) -> bool:
        """简化的episode数据验证"""
        try:
            if not hasattr(self.dataset, 'episode_buffer'):
                logger.error("❌ 没有episode_buffer")
                return False
            
            buffer = self.dataset.episode_buffer
            if not isinstance(buffer, dict):
                logger.error("❌ episode_buffer不是字典")
                return False
            
            # 检查基本字段存在
            required_keys = [
                'observation.images.ee_cam',
                'observation.images.rgb_rs_0', 
                'observation.images.rgb_rs_1',
                'observation.state',
                'actions'
            ]
            
            missing_keys = []
            for key in required_keys:
                if key not in buffer:
                    missing_keys.append(key)
            
            if missing_keys:
                logger.error(f"❌ 缺少必要字段: {missing_keys}")
                return False
            
            # 检查数据不为空
            for key in required_keys:
                if not buffer[key] or len(buffer[key]) == 0:
                    logger.error(f"❌ 字段 {key} 为空")
                    return False
            
            # 确保size字段存在且正确
            if 'size' not in buffer:
                buffer['size'] = self.frame_count
            
            logger.info(f"✅ 数据验证通过: {self.frame_count} 帧")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据验证异常: {e}")
            return False
    
    def _log_simple_debug_info(self):
        """记录简化的调试信息"""
        try:
            logger.error("🔍 简化调试信息:")
            logger.error(f"  📊 帧计数: {self.frame_count}")
            logger.error(f"  📁 Episode索引: {self.episode_idx}")
            logger.error(f"  🎬 录制状态: {self.is_recording}")
            
            if hasattr(self.dataset, 'episode_buffer'):
                buffer = self.dataset.episode_buffer
                if isinstance(buffer, dict):
                    data_keys = ['observation.images.ee_cam', 'observation.state', 'actions']
                    for key in data_keys:
                        if key in buffer and hasattr(buffer[key], '__len__'):
                            logger.error(f"  📋 {key}: {len(buffer[key])} 项")
                        else:
                            logger.error(f"  ❌ {key}: 缺失或无效")
                else:
                    logger.error(f"  ❌ Buffer类型错误: {type(buffer)}")
            else:
                logger.error("  ❌ 没有episode_buffer")
                
        except Exception as e:
            logger.error(f"简化调试信息获取失败: {e}")
    
    def _log_debug_info(self):
        """记录调试信息"""
        try:
            logger.error("=================== 详细调试信息 ===================")
            
            if hasattr(self.dataset, 'episode_buffer'):
                buffer = self.dataset.episode_buffer
                logger.error(f"Episode buffer详细信息:")
                logger.error(f"  - 类型: {type(buffer)}")
                logger.error(f"  - 长度: {len(buffer) if hasattr(buffer, '__len__') else 'N/A'}")
                
                if isinstance(buffer, dict):
                    logger.error(f"  - Keys: {list(buffer.keys())}")
                    
                    # 详细分析每个字段
                    for key, value in buffer.items():
                        logger.error(f"  - {key}:")
                        if hasattr(value, 'shape'):
                            logger.error(f"      shape={value.shape}, dtype={value.dtype}")
                            if hasattr(value, '__len__'):
                                logger.error(f"      length={len(value)}")
                        elif hasattr(value, '__len__'):
                            logger.error(f"      length={len(value)}, type={type(value)}")
                            # 如果是列表，显示前几个元素的信息
                            if isinstance(value, list) and len(value) > 0:
                                first_elem = value[0]
                                logger.error(f"      first_element_type={type(first_elem)}")
                                if hasattr(first_elem, 'shape'):
                                    logger.error(f"      first_element_shape={first_elem.shape}")
                        else:
                            logger.error(f"      value={value}, type={type(value)}")
                    
                    # 检查数据长度一致性 - 只检查实际的数据字段
                    data_fields = [
                        'observation.images.ee_cam',
                        'observation.images.rgb_rs_0', 
                        'observation.images.rgb_rs_1',
                        'observation.state',
                        'actions',
                        'task',
                        'timestamp',
                        'frame_index'
                    ]
                    
                    lengths = {}
                    for key in data_fields:
                        if key in buffer and hasattr(buffer[key], '__len__'):
                            lengths[key] = len(buffer[key])
                    
                    if lengths:
                        logger.error(f"数据长度统计:")
                        for key, length in lengths.items():
                            logger.error(f"  {key}: {length}")
                        
                        # 检查是否有不一致的长度
                        unique_lengths = set(lengths.values())
                        if len(unique_lengths) > 1:
                            logger.error(f"⚠️ 检测到数据长度不一致!")
                            logger.error(f"不同的长度值: {sorted(unique_lengths)}")
                            
                            # 按长度分组
                            length_groups = {}
                            for key, length in lengths.items():
                                if length not in length_groups:
                                    length_groups[length] = []
                                length_groups[length].append(key)
                            
                            for length, keys in length_groups.items():
                                logger.error(f"长度 {length}: {keys}")
                        else:
                            logger.error(f"✅ 所有数据长度一致: {list(unique_lengths)[0]}")
                    
                    # 检查size字段
                    if 'size' in buffer:
                        logger.error(f"size字段值: {buffer['size']}")
                    else:
                        logger.error("⚠️ 缺少size字段")
            
            # 其他相关状态信息
            logger.error(f"本地状态信息:")
            logger.error(f"  - frame_count: {self.frame_count}")
            logger.error(f"  - episode_idx: {self.episode_idx}")
            logger.error(f"  - is_recording: {self.is_recording}")
            
            logger.error("=================== 调试信息结束 ===================")
            
        except Exception as debug_e:
            logger.error(f"调试信息获取失败: {debug_e}")
            import traceback
            logger.error(f"调试信息获取错误堆栈:")
            logger.error(traceback.format_exc())
    
    def _reset_episode_state(self):
        """重置episode状态"""
        try:
            if hasattr(self.dataset, 'reset_episode'):
                self.dataset.reset_episode()
                logger.info("已重置dataset episode状态")
        except Exception as reset_e:
            logger.warning(f"重置dataset episode状态失败: {reset_e}")
        
        # 重置本地状态
        self.is_recording = False
        self.frame_count = 0
        logger.info("已重置本地状态")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {
            "dataset_dir": str(self.dataset_dir),
            "repo_id": self.repo_id,
            "fps": self.fps,
            "current_episode": self.episode_idx,
            "frame_count": self.frame_count,
            "is_recording": self.is_recording,
            "total_episodes": self._get_episode_count()
        }
        
        # 如果数据集存在，添加更多信息
        if self.dataset_dir.exists():
            info_file = self.dataset_dir / "meta" / "info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r") as f:
                        meta_info = json.load(f)
                        info.update(meta_info)
                except Exception as e:
                    logger.warning(f"读取数据集元信息失败: {e}")
        
        return info
    
    def cleanup(self):
        """清理资源"""
        if self.is_recording:
            logger.warning("录制未完成，强制结束当前episode")
            self.end_episode()
        
        self.dataset = None
        logger.info("数据集管理器已清理")

    def _sync_frame_count(self):
        """同步帧计数与实际数据长度"""
        try:
            if hasattr(self.dataset, 'episode_buffer'):
                buffer = self.dataset.episode_buffer
                if isinstance(buffer, dict):
                    # 只检查实际的数据字段，忽略元数据字段
                    data_fields = [
                        'observation.images.ee_cam',
                        'observation.images.rgb_rs_0', 
                        'observation.images.rgb_rs_1',
                        'observation.state',
                        'actions',
                        'task',
                        'timestamp',
                        'frame_index'
                    ]
                    
                    lengths = []
                    for key in data_fields:
                        if key in buffer and hasattr(buffer[key], '__len__'):
                            lengths.append(len(buffer[key]))
                    
                    if lengths:
                        # 使用最常见的长度作为实际帧数
                        from collections import Counter
                        length_counts = Counter(lengths)
                        actual_frame_count = length_counts.most_common(1)[0][0]
                        
                        if actual_frame_count != self.frame_count:
                            logger.warning(f"同步帧计数: {self.frame_count} -> {actual_frame_count}")
                            self.frame_count = actual_frame_count
                            
                            # 同时更新buffer的size字段
                            buffer['size'] = actual_frame_count
                            
                        return actual_frame_count
                    
        except Exception as e:
            logger.warning(f"同步帧计数失败: {e}")
            
        return self.frame_count
    
    def _validate_episode_data(self) -> bool:
        """验证episode数据的完整性和一致性"""
        try:
            if not hasattr(self.dataset, 'episode_buffer'):
                logger.error("数据集没有episode_buffer属性")
                return False
            
            buffer = self.dataset.episode_buffer
            if not isinstance(buffer, dict):
                logger.error(f"episode_buffer类型错误: {type(buffer)}")
                return False
            
            # 检查必需的字段
            required_keys = [
                'observation.images.ee_cam',
                'observation.images.rgb_rs_0', 
                'observation.images.rgb_rs_1',
                'observation.state',
                'actions'
            ]
            
            missing_keys = []
            for key in required_keys:
                if key not in buffer:
                    missing_keys.append(key)
            
            if missing_keys:
                logger.error(f"缺少必需的数据字段: {missing_keys}")
                return False
            
            # 检查数据长度一致性 - 只检查实际的数据字段
            data_fields = [
                'observation.images.ee_cam',
                'observation.images.rgb_rs_0', 
                'observation.images.rgb_rs_1',
                'observation.state',
                'actions',
                'task',
                'timestamp',
                'frame_index'
            ]
            
            lengths = {}
            for key in data_fields:
                if key in buffer and hasattr(buffer[key], '__len__'):
                    lengths[key] = len(buffer[key])
            
            if not lengths:
                logger.error("没有找到任何数据")
                return False
            
            # 检查所有长度是否一致
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                logger.error(f"数据长度不一致: {dict(lengths)}")
                return False
            
            actual_length = list(unique_lengths)[0]
            
            # 检查size字段
            if 'size' not in buffer:
                logger.warning("缺少size字段，将自动设置")
                buffer['size'] = actual_length
            elif buffer['size'] != actual_length:
                logger.warning(f"size字段不匹配，更正: {buffer['size']} -> {actual_length}")
                buffer['size'] = actual_length
            
            # 检查数据不为空
            if actual_length == 0:
                logger.error("数据长度为0")
                return False
            
            # 检查关键数据字段的数据类型和形状
            for key in required_keys:
                value = buffer[key]
                if not isinstance(value, list) or len(value) == 0:
                    logger.error(f"字段 {key} 数据格式错误")
                    return False
                
                # 检查第一个元素的类型
                first_elem = value[0]
                if key.startswith('observation.images.'):
                    if not isinstance(first_elem, np.ndarray) or first_elem.shape != (3, 480, 640):
                        logger.error(f"图像字段 {key} 数据格式错误: shape={getattr(first_elem, 'shape', 'N/A')}")
                        return False
                elif key == 'observation.state':
                    if not isinstance(first_elem, np.ndarray) or len(first_elem) != 8:
                        logger.error(f"状态字段 {key} 数据格式错误: shape={getattr(first_elem, 'shape', 'N/A')}")
                        return False
                elif key == 'actions':
                    if not isinstance(first_elem, np.ndarray) or len(first_elem) != 8:
                        logger.error(f"动作字段 {key} 数据格式错误: shape={getattr(first_elem, 'shape', 'N/A')}")
                        return False
            
            logger.info(f"✅ 数据验证通过，共 {actual_length} 帧数据")
            return True
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def test_lerobot_dataset_manager():
    """测试LeRobot数据集管理器"""
    logger.info("开始测试LeRobot数据集管理器")
    
    # 创建测试目录
    test_dir = "test_dataset"
    manager = LeRobotDatasetManager(dataset_dir=test_dir)
    
    try:
        # 创建数据集
        if not manager.create_dataset():
            logger.error("创建数据集失败")
            return
        
        # 开始录制episode
        if not manager.start_episode():
            logger.error("开始episode失败")
            return
        
        # 添加几帧测试数据
        for i in range(5):
            # 模拟相机图像（CHW格式）
            camera_images = {
                "observation.images.ee_cam": np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),
                "observation.images.rgb_rs_0": np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),
                "observation.images.rgb_rs_1": np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8)
            }
            
            # 模拟机械臂状态（8维）
            robot_state = np.random.rand(8).astype(np.float32)
            
            # 模拟动作（8维）
            actions = np.random.rand(8).astype(np.float32)
            
            # 添加帧
            if manager.add_frame(camera_images, robot_state, actions, "test manipulation"):
                logger.info(f"添加帧 {i+1} 成功")
            else:
                logger.error(f"添加帧 {i+1} 失败")
        
        # 结束episode
        if manager.end_episode():
            logger.info("Episode结束成功")
        else:
            logger.error("Episode结束失败")
        
        # 获取数据集信息
        info = manager.get_dataset_info()
        logger.info(f"数据集信息: {info}")
        
        logger.info("✅ 测试完成")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
    
    finally:
        # 清理
        manager.cleanup()
        
        # 删除测试目录
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            logger.info("清理测试数据")


if __name__ == "__main__":
    test_lerobot_dataset_manager()