import os
import json
import torch
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
import dataclasses
import tyro
from typing import Optional
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass
class RecorderConfig:
    """数据录制相关配置"""
    
    num_demos: int = dataclasses.field(
        default=5,
        metadata={"help": "要录制的演示episodes数量"}
    )
    
    dataset_dir: str = dataclasses.field(
        default="dataset",
        metadata={"help": "数据集保存的目录"}
    )
    
    record: bool = dataclasses.field(
        default=False,
        metadata={"help": "是否启用数据录制"}
    )
    
    resume: bool = dataclasses.field(
        default=False,
        metadata={"help": "是否从现有数据集继续录制"}
    )
    
    local_files_only: bool = dataclasses.field(
        default=False,
        metadata={"help": "是否只使用本地文件（不从HuggingFace Hub下载）"}
    )


class DataRecorder:
    """数据录制工具类，用于管理IsaacLab机器人数据集的创建、配置和录制。"""
    
    def __init__(self, args, env, device):
        """
        初始化数据录制器
        
        参数:
            args: 命令行参数，包含record、dataset_dir、resume等配置
            env: IsaacLab环境实例
            device: 设备(cuda/cpu)
        """
        self.args = args
        self.env = env
        self.device = device
        self.dataset = None
        self.episode_idx = 0
        self.target_episodes = 0
        
        # 只有启用录制时才初始化数据集
        if self.args.record:
            self._init_dataset()
            # 设置目标episode数量
            self.target_episodes = self.episode_idx + self.args.num_demos
    
    def _init_dataset(self):
        """初始化数据集，处理增量录制逻辑"""
        ds_root = Path(self.args.dataset_dir).expanduser()
        fps = 1.0 / (self.env.unwrapped.cfg.sim.dt * self.env.unwrapped.cfg.decimation)
        
        # 处理增量录制逻辑
        existing_episodes = 0
        
        # 判断是否为增量录制模式
        if self.args.resume and ds_root.exists():
            # 在增量模式下，获取已有episode数量
            existing_episodes = self._get_episode_count(ds_root)
            print(f"增量录制模式：检测到 {existing_episodes} 个已有episode，将从 {existing_episodes} 开始继续录制")
            self.episode_idx = existing_episodes
            
            try:
                # 尝试载入现有数据集
                self.dataset = LeRobotDataset(
                    repo_id="isaaclab/Franka-Coffee-Gamepad-Control-Direct-v0",
                    root=ds_root
                )
                print(f"成功加载现有数据集，将继续在其上录制。")
            except Exception as e:
                print(f"无法加载现有数据集: {e}")
                print("将创建新数据集...")
                # 如果增量录制失败，创建备份
                if ds_root.exists():
                    backup_path = f"{ds_root}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    print(f"为防止数据丢失，正在创建备份：{backup_path}")
                    shutil.copytree(ds_root, backup_path)
                    
                # 由于备份已完成，可以安全删除原目录
                if ds_root.exists():
                    shutil.rmtree(ds_root)
                    
                # 创建新数据集
                self.dataset = None  # 重置，下面会重新创建
        else:
            # 非增量模式，如果目录存在则删除
            if ds_root.exists():
                print(f"非增量模式：删除已有数据集文件夹 {ds_root}")
                shutil.rmtree(ds_root)
            self.episode_idx = 0
        
        # 如果dataset为None，则创建新数据集
        if self.dataset is None:
            try:
                self.dataset = LeRobotDataset.create(
                    repo_id="isaaclab/Franka-Coffee-Gamepad-Control-Direct-v0",
                    root=ds_root,
                    fps=fps,
                    features={
                        # 摄像头
                        "observation.images.ee_cam":        {"shape": (3, 384, 384), "dtype": "image"},
                        "observation.images.rgb_rs_0": {"shape": (3, 384, 384), "dtype": "image"},
                        "observation.images.rgb_rs_1": {"shape": (3, 384, 384), "dtype": "image"},
                        # 关节 / 动作
                        "observation.state":  {"shape": (17,), "dtype": "float32"}, # 修改为observation.state
                        "actions": {"shape": (8,),  "dtype": "float32"},
                        # "task": {"shape": (1,), "dtype": "str"}
                    },
                )
                print(f"成功创建新数据集")
            except Exception as e:
                print(f"创建数据集失败: {e}")
                return
        
        # 确保meta/modality.json存在
        self._create_modality_file(ds_root)
    
    def _create_modality_file(self, ds_root):
        """创建或更新modality.json文件"""
        modality_path = ds_root / "meta/modality.json"
        if not modality_path.exists():
            modality_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump({
                "observation.state": {  # 修改1: 使用observation.state替代state
                    "franka_qpos": {"start": 0,  "end": 9}, # 关节位置 9 维
                    "franka_qvel": {"start": 9,  "end": 17} # 关节速度 8 维
                },
                "actions": {
                    "joint_target": {"start": 0, "end": 8, "absolute": True}
                },
            }, open(modality_path, "w"), indent=2)
        
    def _get_episode_count(self, dataset_dir):
        """获取数据集中已有的episode数量"""
        try:
            ds_path = Path(dataset_dir).expanduser()
            if not ds_path.exists():
                return 0
            
            # 方法1: 直接读取meta/info.json (最准确)
            info_path = ds_path / "meta" / "info.json"
            if info_path.exists():
                with open(info_path, "r") as f:
                    info = json.load(f)
                    return info.get("total_episodes", 0)
            
            # 方法2: 统计所有chunk目录下的episode文件
            episode_count = 0
            data_path = ds_path / "data"
            if data_path.exists():
                for chunk_dir in data_path.glob("chunk-*"):
                    episode_files = list(chunk_dir.glob("episode_*.parquet"))
                    episode_count += len(episode_files)
                
                if episode_count > 0:
                    return episode_count
                
            # 备选方法3: 检查可能的视频文件
            video_dirs = list(ds_path.glob("observation.images.*"))
            if video_dirs:
                for video_dir in video_dirs:
                    episode_videos = list(video_dir.glob("episode_*.mp4"))
                    if episode_videos:
                        return len(episode_videos)
            
            return 0
        except Exception as e:
            print(f"获取episode数量时出错: {e}")
            return 0
    
    def record_frame(self, obs, actions, dones):
        """记录当前帧数据并处理episode保存逻辑"""
        if not self.args.record or self.dataset is None:
            return False
        
        # 拼 observation.state
        # 当num_envs=1单环境时，位置向量里只有1个手指角度（对称运动被合并了）；多环境时，位置向量里有2个手指角度。
        if obs["policy"]["joint_pos"][0].shape[0] == 8:  # 如果只有8维
            # 复制夹爪值以使其成为9维   
            joint_pos = torch.cat([
                obs["policy"]["joint_pos"][0], 
                obs["policy"]["joint_pos"][0][-1].unsqueeze(0)
            ], dim=0)
        else:
            joint_pos = obs["policy"]["joint_pos"][0]
            
        state_vec = torch.cat([joint_pos, obs["policy"]["joint_vel"][0]], dim=0).cpu().numpy().astype(np.float32)

        # ---------- 构建帧数据并添加 ----------
        frame = {
            # === 摄像头画面 ===
            "observation.images.ee_cam":        obs["policy"]["ee_camera"][0].cpu().numpy().astype(np.uint8),
            "observation.images.rgb_rs_0": obs["policy"]["rgb_rs_0"][0].cpu().numpy().astype(np.uint8),
            "observation.images.rgb_rs_1": obs["policy"]["rgb_rs_1"][0].cpu().numpy().astype(np.uint8),
            # === 关节状态 ===
            "observation.state": state_vec,  # 修改为observation.state
            # === 动作 ===
            "actions": actions[0].cpu().numpy().astype(np.float32),
            # === 任务标识 ===
            "task": "move the coffee cup to the coffee machine"
        }
        
        try:
            self.dataset.add_frame(frame)
        except Exception as e:
            print(f"添加帧数据失败: {e}")

        # 处理episode结束逻辑
        if dones.any():
            # 保存当前episode
            print(f"保存episode_{self.episode_idx}...")
            try:
                self.dataset.save_episode()
                print(f"Episode_{self.episode_idx} 保存成功!")
            except Exception as e:
                print(f"保存Episode_{self.episode_idx}失败: {e}")
            
            self.episode_idx += 1
            return True  # 表示一个episode已完成
        
        return False  # 表示当前episode还在进行中
    
    def is_complete(self):
        """检查是否已完成目标录制数量"""
        if not self.args.record:
            return False
        return self.episode_idx >= self.target_episodes
    
    def get_target_episodes(self):
        """获取目标episode数量"""
        return self.target_episodes
    
    def get_current_episode(self):
        """获取当前episode索引"""
        return self.episode_idx

# 移除旧的add_recorder_args函数，现在使用tyro自动处理参数 