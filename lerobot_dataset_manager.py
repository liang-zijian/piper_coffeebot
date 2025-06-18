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
# 导入全局日志管理器
from global_logger import log_message, log_info, log_warning, log_error, log_success

# 导入lerobot相关模块
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    log_message("警告: 无法导入lerobot模块，请确保已正确安装lerobot", "warning", "LeRobot")
    LeRobotDataset = None

# 导入多相机管理器用于批量图像处理
from multi_realsense_cameras import MultiRealSenseManager

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
        
        # 临时存储原始BGR图像数据，用于批量处理
        self.episode_bgr_images = {}
        
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
        
        log_message("✅ LeRobot数据集管理器初始化完成", "success", "LeRobot")
    
    def create_dataset(self, resume: bool = False) -> bool:
        """
        创建或加载数据集
        
        Args:
            resume: 是否从现有数据集继续录制
            
        Returns:
            创建/加载是否成功
        """
        if LeRobotDataset is None:
            log_message("LeRobotDataset未导入，无法创建数据集", "error", "LeRobot")
            return False
        
        try:
            # 处理增量录制逻辑
            if resume and self.dataset_dir.exists():
                existing_episodes = self._get_episode_count()
                log_message(f"增量录制模式：检测到 {existing_episodes} 个已有episode", "info", "LeRobot")
                self.episode_idx = existing_episodes
                
                try:
                    # 尝试加载现有数据集
                    self.dataset = LeRobotDataset(
                        repo_id=self.repo_id,
                        root=self.dataset_dir
                    )
                    log_message(f"✅ 成功加载现有数据集，将从episode {self.episode_idx}开始录制", "success", "LeRobot")
                    return True
                    
                except Exception as e:
                    log_message(f"加载现有数据集失败: {e}", "warning", "LeRobot")
                    log_message("将创建新数据集...", "info", "LeRobot")
                    
                    # 创建备份
                    if self.dataset_dir.exists():
                        backup_path = f"{self.dataset_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        log_message(f"创建备份：{backup_path}", "info", "LeRobot")
                        shutil.copytree(self.dataset_dir, backup_path)
                        shutil.rmtree(self.dataset_dir)
                    
                    self.episode_idx = 0
            else:
                # 非增量模式，删除已有目录
                if self.dataset_dir.exists():
                    log_message(f"删除已有数据集目录：{self.dataset_dir}", "info", "LeRobot")
                    shutil.rmtree(self.dataset_dir)
                self.episode_idx = 0
            
            # 创建新数据集
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                root=self.dataset_dir,
                fps=self.fps,
                features=self.features,
                image_writer_processes=16,
                image_writer_threads=16
            )
            
            # 创建modality文件
            self._create_modality_file()
            
            log_message(f"✅ 成功创建新数据集", "success", "LeRobot")
            return True
            
        except Exception as e:
            log_message(f"❌ 创建数据集失败: {e}", "error", "LeRobot")
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
            log_message(f"获取episode数量失败: {e}", "info", "LeRobot")
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
            
            log_message(f"创建modality.json文件：{modality_path}", "info", "LeRobot")
    
    def start_episode(self):
        """开始新的episode"""
        if self.dataset is None:
            log_message("数据集未初始化，无法开始episode", "error", "LeRobot")
            return False
        
        if self.is_recording:
            log_message("已经在录制中，请先结束当前episode", "warning", "LeRobot")
            return False
        
        self.is_recording = True
        self.frame_count = 0
        # 初始化BGR图像临时存储
        self.episode_bgr_images = {
            "observation.images.ee_cam": [],
            "observation.images.rgb_rs_0": [],
            "observation.images.rgb_rs_1": []
        }
        # 初始化机械臂状态和动作临时存储
        self.episode_robot_states = []
        self.episode_actions = []
        self.episode_tasks = []
        log_message(f"Start episode {self.episode_idx}", "info", "LeRobot")
        return True
    
    def add_frame(self, 
                  camera_images: Dict[str, np.ndarray],
                  robot_state: np.ndarray,
                  actions: np.ndarray,
                  task: str = None) -> bool:
        """
        添加一帧数据
        录制时存储原始BGR图像，避免耗时转换
        
        Args:
            camera_images: 相机图像字典，key为相机名称，value为原始BGR图像数组(HWC格式)
            robot_state: 机械臂状态向量(8维)
            actions: 动作向量(8维)
            task: 任务描述
            
        Returns:
            添加是否成功
        """
        if not self.is_recording or self.dataset is None:
            log_message("未在录制状态或数据集未初始化", "info", "LeRobot")
            return False
        
        try:
            # 验证输入数据的完整性和有效性
            if not isinstance(camera_images, dict):
                log_message("相机图像数据类型错误，应为字典", "info", "LeRobot")
                return False
                
            if robot_state is None or not isinstance(robot_state, np.ndarray) or len(robot_state) != 8:
                log_message(f"机械臂状态数据无效: type={type(robot_state)}, shape={getattr(robot_state, 'shape', 'N/A')}", "info", "LeRobot")
                return False
                
            if actions is None or not isinstance(actions, np.ndarray) or len(actions) != 8:
                log_message(f"动作数据无效: type={type(actions)}, shape={getattr(actions, 'shape', 'N/A')}", "info", "LeRobot")
                return False
            
            # 构建帧数据
            frame = {}
            
            # 验证和临时存储原始BGR图像
            required_cameras = ["observation.images.ee_cam", "observation.images.rgb_rs_0", "observation.images.rgb_rs_1"]
            for key in required_cameras:
                if key in camera_images and camera_images[key] is not None:
                    img = camera_images[key]
                    
                    # 验证图像数据类型和尺寸
                    if not isinstance(img, np.ndarray):
                        log_message(f"图像 {key} 数据类型错误: {type(img)}", "info", "LeRobot")
                        return False
                    
                    # 确保图像格式正确(HWC, uint8) - BGR格式
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    
                    # 验证图像尺寸 - 录制时为HWC格式(480, 640, 3)
                    if img.shape != (480, 640, 3):
                        log_message(f"图像 {key} 尺寸不匹配: 期望(480,640,3), 实际{img.shape}", "info", "LeRobot")
                        return False
                    
                    # 存储到临时BGR图像缓存，暂不添加到帧数据
                    self.episode_bgr_images[key].append(img.copy())
                else:
                    log_message(f"必需的相机图像缺失: {key}", "info", "LeRobot")
                    return False
            
            # 临时存储机械臂状态和动作数据，用于后续批量处理
            self.episode_robot_states.append(robot_state.astype(np.float32))
            self.episode_actions.append(actions.astype(np.float32))
            self.episode_tasks.append(task if task is not None else self.task_description)
            
            # 录制时只增加帧计数，不进行实际数据集操作
            self.frame_count += 1
            
            # 每20帧输出一次调试信息
            if self.frame_count % 20 == 0:
                log_message(f"已缓存 {self.frame_count} 帧到episode {self.episode_idx} (BGR格式)", "info", "LeRobot")
            
            return True
            
        except Exception as e:
            log_message(f"构建帧数据失败: {e}", "info", "LeRobot")
            log_message(f"错误类型: {type(e).__name__}", "info", "LeRobot")
            return False
    
    def end_episode(self) -> bool:
        """
        结束当前episode并执行批量图像处理
        保存时批量处理BGR→RGB→CHW转换
        """
        log_message("🔹" * 30, "info", "LeRobot")
        log_message("📝 开始结束episode流程（批量处理模式）", "info", "LeRobot")
        
        if not self.is_recording or self.dataset is None:
            log_message("❌ 未在录制状态或数据集未初始化", "info", "LeRobot")
            return False
        
        log_message(f"📊 当前状态:", "info", "LeRobot")
        log_message(f"  - episode_idx: {self.episode_idx}", "info", "LeRobot")
        log_message(f"  - frame_count: {self.frame_count}", "info", "LeRobot")
        
        # 检查是否有足够的帧数据
        if self.frame_count == 0:
            log_message("⚠️ 当前episode没有录制任何帧数据，跳过保存", "warning", "LeRobot")
            self._reset_episode_state()
            return True
        
        min_frames = 2
        if self.frame_count < min_frames:
            log_message(f"⚠️ 当前episode帧数太少 ({self.frame_count} < {min_frames})，跳过保存", "info", "LeRobot")
            self._reset_episode_state()
            return True
        
        try:
            # Step 1: 批量处理BGR图像转换为RGB CHW格式
            batch_start_time = time.time()
            processed_images = MultiRealSenseManager.batch_convert_bgr_to_rgb_chw(self.episode_bgr_images)
            batch_duration = time.time() - batch_start_time
            log_message(f"   批量转换耗时: {batch_duration:.3f}秒 ({self.frame_count} 帧)", "info", "LeRobot")
            
            # Step 2: 验证处理后的图像数据
            for camera_key, images in processed_images.items():
                if len(images) != self.frame_count:
                    log_message(f"❌ 相机 {camera_key} 图像数量不匹配: {len(images)} != {self.frame_count}", "info", "LeRobot")
                    self._reset_episode_state()
                    return False
            
            # Step 3: 批量添加帧到数据集
            add_start_time = time.time()
            for frame_idx in range(self.frame_count):
                frame = {}
                
                # 添加处理后的图像
                for camera_key in ["observation.images.ee_cam", "observation.images.rgb_rs_0", "observation.images.rgb_rs_1"]:
                    if camera_key in processed_images and frame_idx < len(processed_images[camera_key]):
                        frame[camera_key] = processed_images[camera_key][frame_idx]
                    else:
                        self._reset_episode_state()
                        return False
                
                # 添加机械臂状态和动作
                frame["observation.state"] = self.episode_robot_states[frame_idx]
                frame["actions"] = self.episode_actions[frame_idx]
                
                # 添加到数据集
                try:
                    self.dataset.add_frame(frame, self.episode_tasks[frame_idx])
                except Exception as e:
                    log_message(f"❌ 添加帧 {frame_idx} 失败: {e}", "info", "LeRobot")
                    self._reset_episode_state()
                    return False
            add_duration = time.time() - add_start_time
            
            # Step 4: 保存episode到磁盘
            log_message("💾 步骤3: 保存episode到磁盘...", "info", "LeRobot")
            save_start = time.time()
            self.dataset.save_episode()
            save_duration = time.time() - save_start
            
            total_duration = time.time() - batch_start_time
            
            log_message(f"✅ Episode {self.episode_idx} 保存成功!", "info", "LeRobot")
            log_message(f"   📈 数据统计: {self.frame_count} 帧", "info", "LeRobot")
            log_message(f"   🎨 批量转换耗时: {batch_duration:.3f}秒", "info", "LeRobot")
            log_message(f"   💾 磁盘保存耗时: {save_duration:.3f}秒", "info", "LeRobot")
            log_message(f"   💾 添加帧耗时: {add_duration:.3f}秒", "info", "LeRobot")
            log_message(f"   ⏱️  总耗时: {total_duration:.3f}秒", "info", "LeRobot")
            
            # 更新状态
            self.episode_idx += 1
            self._reset_episode_state()
            
            log_message("🔹" * 30, "info", "LeRobot")
            return True
            
        except Exception as e:
            log_message(f"❌ 保存episode异常: {e}", "info", "LeRobot")
            log_message(f"错误类型: {type(e).__name__}", "info", "LeRobot")
            import traceback
            log_message("错误堆栈:", "info", "LeRobot")
            log_message(traceback.format_exc(), "info", "LeRobot")
            
            self._reset_episode_state()
            return False
    
    def _reset_episode_state(self):
        """重置episode状态"""
        self.is_recording = False
        self.frame_count = 0
        self.episode_bgr_images = {}
        if hasattr(self, 'episode_robot_states'):
            self.episode_robot_states = []
            self.episode_actions = []
            self.episode_tasks = []
    
    def _check_buffer_status(self) -> bool:
        """检查episode buffer状态"""
        try:
            if not hasattr(self.dataset, 'episode_buffer'):
                log_message("❌ Dataset没有episode_buffer属性", "info", "LeRobot")
                return False
            
            buffer = self.dataset.episode_buffer
            if not isinstance(buffer, dict):
                log_message(f"❌ Episode buffer类型错误: {type(buffer)}", "info", "LeRobot")
                return False
            
            log_message(f"✅ Buffer状态正常: {len(buffer)} 个字段", "info", "LeRobot")
            return True
            
        except Exception as e:
            log_message(f"❌ Buffer状态检查异常: {e}", "info", "LeRobot")
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
                log_message("⚠️ 没有找到数据字段，跳过修复", "warning", "LeRobot")
                return True
            
            # 检查长度是否一致
            unique_lengths = set(lengths.values())
            if len(unique_lengths) == 1:
                # 长度一致，不需要修复
                actual_length = list(unique_lengths)[0]
                log_message(f"✅ 数据长度一致: {actual_length}", "info", "LeRobot")
                
                # 确保size字段正确
                buffer['size'] = actual_length
                self.frame_count = actual_length
                return True
            
            # 长度不一致，尝试修复
            log_message(f"⚠️ 检测到数据长度不一致: {lengths}", "info", "LeRobot")
            
            # 使用最小长度
            min_length = min(lengths.values())
            log_message(f"🔧 使用最小长度修复: {min_length}", "info", "LeRobot")
            
            if min_length < min_frames:
                log_message(f"❌ 最小长度太小 ({min_length} < {min_frames})", "info", "LeRobot")
                return False
            
            # 截取所有字段到最小长度
            all_fields = list(buffer.keys())
            for key in all_fields:
                if key in buffer and hasattr(buffer[key], '__len__') and hasattr(buffer[key], '__getitem__'):
                    if len(buffer[key]) > min_length:
                        buffer[key] = buffer[key][:min_length]
                        log_message(f"   截取 {key}: {len(buffer[key])} -> {min_length}", "info", "LeRobot")
            
            # 更新状态
            buffer['size'] = min_length
            self.frame_count = min_length
            
            log_message(f"✅ 数据修复完成，统一长度: {min_length}", "info", "LeRobot")
            return True
            
        except Exception as e:
            log_message(f"❌ 数据修复异常: {e}", "info", "LeRobot")
            return False
    
    def _validate_episode_data_simple(self) -> bool:
        """简化的episode数据验证"""
        try:
            if not hasattr(self.dataset, 'episode_buffer'):
                log_message("❌ 没有episode_buffer", "info", "LeRobot")
                return False
            
            buffer = self.dataset.episode_buffer
            if not isinstance(buffer, dict):
                log_message("❌ episode_buffer不是字典", "info", "LeRobot")
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
                log_message(f"❌ 缺少必要字段: {missing_keys}", "info", "LeRobot")
                return False
            
            # 检查数据不为空
            for key in required_keys:
                if not buffer[key] or len(buffer[key]) == 0:
                    log_message(f"❌ 字段 {key} 为空", "info", "LeRobot")
                    return False
            
            # 确保size字段存在且正确
            if 'size' not in buffer:
                buffer['size'] = self.frame_count
            
            log_message(f"✅ 数据验证通过: {self.frame_count} 帧", "info", "LeRobot")
            return True
            
        except Exception as e:
            log_message(f"❌ 数据验证异常: {e}", "info", "LeRobot")
            return False
    
    def _log_simple_debug_info(self):
        """记录简化的调试信息"""
        try:
            log_message("🔍 简化调试信息:", "info", "LeRobot")
            log_message(f"  📊 帧计数: {self.frame_count}", "info", "LeRobot")
            log_message(f"  📁 Episode索引: {self.episode_idx}", "info", "LeRobot")
            log_message(f"  🎬 录制状态: {self.is_recording}", "info", "LeRobot")
            
            if hasattr(self.dataset, 'episode_buffer'):
                buffer = self.dataset.episode_buffer
                if isinstance(buffer, dict):
                    data_keys = ['observation.images.ee_cam', 'observation.state', 'actions']
                    for key in data_keys:
                        if key in buffer and hasattr(buffer[key], '__len__'):
                            log_message(f"  📋 {key}: {len(buffer[key])} 项", "info", "LeRobot")
                        else:
                            log_message(f"  ❌ {key}: 缺失或无效", "info", "LeRobot")
                else:
                    log_message(f"  ❌ Buffer类型错误: {type(buffer)}", "info", "LeRobot")
            else:
                log_message("  ❌ 没有episode_buffer", "info", "LeRobot")
                
        except Exception as e:
            log_message(f"简化调试信息获取失败: {e}", "info", "LeRobot")
    
    def _log_debug_info(self):
        """记录调试信息"""
        try:
            log_message("=================== 详细调试信息 ===================", "info", "LeRobot")
            
            if hasattr(self.dataset, 'episode_buffer'):
                buffer = self.dataset.episode_buffer
                log_message(f"Episode buffer详细信息:", "info", "LeRobot")
                log_message(f"  - 类型: {type(buffer)}", "info", "LeRobot")
                log_message(f"  - 长度: {len(buffer) if hasattr(buffer, '__len__') else 'N/A'}", "info", "LeRobot")
                
                if isinstance(buffer, dict):
                    log_message(f"  - Keys: {list(buffer.keys())}", "info", "LeRobot")
                    
                    # 详细分析每个字段
                    for key, value in buffer.items():
                        log_message(f"  - {key}:", "info", "LeRobot")
                        if hasattr(value, 'shape'):
                            log_message(f"      shape={value.shape}, dtype={value.dtype}", "info", "LeRobot")
                            if hasattr(value, '__len__'):
                                log_message(f"      length={len(value)}", "info", "LeRobot")
                        elif hasattr(value, '__len__'):
                            log_message(f"      length={len(value)}, type={type(value)}", "info", "LeRobot")
                            # 如果是列表，显示前几个元素的信息
                            if isinstance(value, list) and len(value) > 0:
                                first_elem = value[0]
                                log_message(f"      first_element_type={type(first_elem)}", "info", "LeRobot")
                                if hasattr(first_elem, 'shape'):
                                    log_message(f"      first_element_shape={first_elem.shape}", "info", "LeRobot")
                        else:
                            log_message(f"      value={value}, type={type(value)}", "info", "LeRobot")
                    
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
                        log_message(f"数据长度统计:", "info", "LeRobot")
                        for key, length in lengths.items():
                            log_message(f"  {key}: {length}", "info", "LeRobot")
                        
                        # 检查是否有不一致的长度
                        unique_lengths = set(lengths.values())
                        if len(unique_lengths) > 1:
                            log_message(f"⚠️ 检测到数据长度不一致!", "info", "LeRobot")
                            log_message(f"不同的长度值: {sorted(unique_lengths)}", "info", "LeRobot")
                            
                            # 按长度分组
                            length_groups = {}
                            for key, length in lengths.items():
                                if length not in length_groups:
                                    length_groups[length] = []
                                length_groups[length].append(key)
                            
                            for length, keys in length_groups.items():
                                log_message(f"长度 {length}: {keys}", "info", "LeRobot")
                        else:
                            log_message(f"✅ 所有数据长度一致: {list(unique_lengths)[0]}", "info", "LeRobot")
                    
                    # 检查size字段
                    if 'size' in buffer:
                        log_message(f"size字段值: {buffer['size']}", "info", "LeRobot")
                    else:
                        log_message("⚠️ 缺少size字段", "warning", "LeRobot")
            
            # 其他相关状态信息
            log_message(f"本地状态信息:", "info", "LeRobot")
            log_message(f"  - frame_count: {self.frame_count}", "info", "LeRobot")
            log_message(f"  - episode_idx: {self.episode_idx}", "info", "LeRobot")
            log_message(f"  - is_recording: {self.is_recording}", "info", "LeRobot")
            
            log_message("=================== 调试信息结束 ===================", "info", "LeRobot")
            
        except Exception as debug_e:
            log_message(f"调试信息获取失败: {debug_e}", "info", "LeRobot")
            import traceback
            log_message(f"调试信息获取错误堆栈:", "info", "LeRobot")
            log_message(traceback.format_exc(), "info", "LeRobot")
    
    def _reset_episode_state(self):
        """重置episode状态"""
        try:
            if hasattr(self.dataset, 'reset_episode'):
                self.dataset.reset_episode()
                log_message("已重置dataset episode状态", "info", "LeRobot")
        except Exception as reset_e:
            log_message(f"重置dataset episode状态失败: {reset_e}", "info", "LeRobot")
        
        # 重置本地状态
        self.is_recording = False
        self.frame_count = 0
        log_message("已重置本地状态", "info", "LeRobot")
    
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
                    log_message(f"读取数据集元信息失败: {e}", "info", "LeRobot")
        
        return info
    
    def cleanup(self):
        """清理资源"""
        if self.is_recording:
            log_message("录制未完成，强制结束当前episode", "info", "LeRobot")
            self.end_episode()
        
        self.dataset = None
        log_message("数据集管理器已清理", "info", "LeRobot")

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
                            log_message(f"同步帧计数: {self.frame_count} -> {actual_frame_count}", "info", "LeRobot")
                            self.frame_count = actual_frame_count
                            
                            # 同时更新buffer的size字段
                            buffer['size'] = actual_frame_count
                            
                        return actual_frame_count
                    
        except Exception as e:
            log_message(f"同步帧计数失败: {e}", "info", "LeRobot")
            
        return self.frame_count
    
    def _validate_episode_data(self) -> bool:
        """验证episode数据的完整性和一致性"""
        try:
            if not hasattr(self.dataset, 'episode_buffer'):
                log_message("数据集没有episode_buffer属性", "info", "LeRobot")
                return False
            
            buffer = self.dataset.episode_buffer
            if not isinstance(buffer, dict):
                log_message(f"episode_buffer类型错误: {type(buffer)}", "info", "LeRobot")
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
                log_message(f"缺少必需的数据字段: {missing_keys}", "info", "LeRobot")
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
                log_message("没有找到任何数据", "info", "LeRobot")
                return False
            
            # 检查所有长度是否一致
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                log_message(f"数据长度不一致: {dict(lengths)}", "info", "LeRobot")
                return False
            
            actual_length = list(unique_lengths)[0]
            
            # 检查size字段
            if 'size' not in buffer:
                log_message("缺少size字段，将自动设置", "info", "LeRobot")
                buffer['size'] = actual_length
            elif buffer['size'] != actual_length:
                log_message(f"size字段不匹配，更正: {buffer['size']} -> {actual_length}", "info", "LeRobot")
                buffer['size'] = actual_length
            
            # 检查数据不为空
            if actual_length == 0:
                log_message("数据长度为0", "info", "LeRobot")
                return False
            
            # 检查关键数据字段的数据类型和形状
            for key in required_keys:
                value = buffer[key]
                if not isinstance(value, list) or len(value) == 0:
                    log_message(f"字段 {key} 数据格式错误", "info", "LeRobot")
                    return False
                
                # 检查第一个元素的类型
                first_elem = value[0]
                if key.startswith('observation.images.'):
                    if not isinstance(first_elem, np.ndarray) or first_elem.shape != (3, 480, 640):
                        log_message(f"图像字段 {key} 数据格式错误: shape={getattr(first_elem, 'shape', 'N/A')}", "info", "LeRobot")
                        return False
                elif key == 'observation.state':
                    if not isinstance(first_elem, np.ndarray) or len(first_elem) != 8:
                        log_message(f"状态字段 {key} 数据格式错误: shape={getattr(first_elem, 'shape', 'N/A')}", "info", "LeRobot")
                        return False
                elif key == 'actions':
                    if not isinstance(first_elem, np.ndarray) or len(first_elem) != 8:
                        log_message(f"动作字段 {key} 数据格式错误: shape={getattr(first_elem, 'shape', 'N/A')}", "info", "LeRobot")
                        return False
            
            log_message(f"✅ 数据验证通过，共 {actual_length} 帧数据", "info", "LeRobot")
            return True
            
        except Exception as e:
            log_message(f"数据验证失败: {e}", "info", "LeRobot")
            import traceback
            log_message(traceback.format_exc(), "info", "LeRobot")
            return False


def test_lerobot_dataset_manager():
    """测试LeRobot数据集管理器"""
    log_message("开始测试LeRobot数据集管理器", "info", "LeRobot")
    
    # 创建测试目录
    test_dir = "test_dataset"
    manager = LeRobotDatasetManager(dataset_dir=test_dir)
    
    try:
        # 创建数据集
        if not manager.create_dataset():
            log_message("创建数据集失败", "info", "LeRobot")
            return
        
        # 开始录制episode
        if not manager.start_episode():
            log_message("开始episode失败", "info", "LeRobot")
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
                log_message(f"添加帧 {i+1} 成功", "info", "LeRobot")
            else:
                log_message(f"添加帧 {i+1} 失败", "info", "LeRobot")
        
        # 结束episode
        if manager.end_episode():
            log_message("Episode结束成功", "info", "LeRobot")
        else:
            log_message("Episode结束失败", "info", "LeRobot")
        
        # 获取数据集信息
        info = manager.get_dataset_info()
        log_message(f"数据集信息: {info}", "info", "LeRobot")
        
        log_message("✅ 测试完成", "info", "LeRobot")
        
    except Exception as e:
        log_message(f"测试过程中出错: {e}", "info", "LeRobot")
    
    finally:
        # 清理
        manager.cleanup()
        
        # 删除测试目录
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            log_message("清理测试数据", "info", "LeRobot")


if __name__ == "__main__":
    test_lerobot_dataset_manager()