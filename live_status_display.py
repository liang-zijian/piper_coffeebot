#!/usr/bin/env python3
"""
实时状态显示模块
使用rich库的Live面板显示系统状态
"""

import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import numpy as np

console = Console()

class LiveStatusDisplay:
    """实时状态显示器"""
    
    def __init__(self, refresh_rate: float = 10.0):
        """
        初始化状态显示器
        
        Args:
            refresh_rate: 刷新率（Hz）
        """
        self.refresh_rate = refresh_rate
        self.live = None
        self.running = False
        
        # 状态数据
        self.status_data = {
            "system": {
                "start_time": time.time(),
                "fps": 0.0,
                "frame_count": 0,
                "is_recording": False,
                "episode_count": 0,
                "current_episode_frames": 0
            },
            "robot": {
                "position": np.zeros(8),
                "velocity": np.zeros(8),
                "joint5_angle": 0.0,
                "joint6_angle": 0.0,
                "gripper_open": True,
                "connected": False
            },
            "cameras": {
                "ee_cam": {"connected": False, "fps": 0.0},
                "rgb_rs_0": {"connected": False, "fps": 0.0},
                "rgb_rs_1": {"connected": False, "fps": 0.0}
            },
            "gamepad": {
                "connected": False,
                "delta_pos": np.zeros(3),
                "last_command": "无"
            },
            "dataset": {
                "directory": "未设置",
                "total_episodes": 0,
                "current_size_mb": 0.0
            },
            "messages": []  # 最近的消息列表
        }
        
        self.lock = threading.Lock()
    
    def create_layout(self) -> Layout:
        """创建布局"""
        layout = Layout()
        
        # 分割布局
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=6)
        )
        
        # 主区域分为左右两列
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # 左列分为上下两部分
        layout["left"].split_column(
            Layout(name="system_status"),
            Layout(name="robot_status")
        )
        
        # 右列分为上下两部分
        layout["right"].split_column(
            Layout(name="camera_status"),
            Layout(name="dataset_status")
        )
        
        return layout
    
    def create_header(self) -> Panel:
        """创建标题面板"""
        runtime = time.time() - self.status_data["system"]["start_time"]
        runtime_str = f"{int(runtime//3600):02d}:{int((runtime%3600)//60):02d}:{int(runtime%60):02d}"
        
        title_text = Text("Piper 机械臂实时数据录制系统", style="bold cyan")
        subtitle_text = Text(f"运行时间: {runtime_str} | 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                           style="dim")
        
        return Panel(
            Text.assemble(title_text, "\n", subtitle_text),
            style="cyan",
            padding=(0, 1)
        )
    
    def create_system_status(self) -> Panel:
        """创建系统状态面板"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("项目", style="cyan")
        table.add_column("状态")
        
        # 录制状态
        recording_status = "🔴 录制中" if self.status_data["system"]["is_recording"] else "⭕ 待机"
        recording_style = "red" if self.status_data["system"]["is_recording"] else "yellow"
        
        table.add_row("录制状态", Text(recording_status, style=recording_style))
        table.add_row("系统FPS", f"{self.status_data['system']['fps']:.1f}")
        table.add_row("总帧数", f"{self.status_data['system']['frame_count']}")
        table.add_row("Episode数", f"{self.status_data['system']['episode_count']}")
        table.add_row("当前Episode帧数", f"{self.status_data['system']['current_episode_frames']}")
        
        return Panel(table, title="[bold green]系统状态", border_style="green")
    
    def create_robot_status(self) -> Panel:
        """创建机械臂状态面板"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("关节", style="cyan")
        table.add_column("位置", style="white")
        table.add_column("速度", style="yellow")
        
        # 连接状态
        connection_status = "✅ 已连接" if self.status_data["robot"]["connected"] else "❌ 未连接"
        connection_style = "green" if self.status_data["robot"]["connected"] else "red"
        
        for i in range(6):  # 显示前6个关节
            if i < len(self.status_data["robot"]["position"]):
                pos = self.status_data["robot"]["position"][i]
                vel = self.status_data["robot"]["velocity"][i] if i < len(self.status_data["robot"]["velocity"]) else 0.0
                table.add_row(f"Joint{i+1}", f"{pos:7.3f}", f"{vel:7.3f}")
        
        # 夹爪状态
        gripper_status = "🖐️ 张开" if self.status_data["robot"]["gripper_open"] else "✊ 闭合"
        table.add_row("夹爪", gripper_status, "")
        
        title = f"[bold blue]机械臂状态 ({Text(connection_status, style=connection_style)})"
        return Panel(table, title=title, border_style="blue")
    
    def create_camera_status(self) -> Panel:
        """创建相机状态面板"""
        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("相机", style="cyan")
        table.add_column("状态")
        table.add_column("FPS", style="yellow")
        
        for camera_name, camera_info in self.status_data["cameras"].items():
            status = "✅ 连接" if camera_info["connected"] else "❌ 断开"
            status_style = "green" if camera_info["connected"] else "red"
            fps_text = f"{camera_info['fps']:.1f}" if camera_info["connected"] else "0.0"
            
            display_name = {
                "ee_cam": "腕部相机",
                "rgb_rs_0": "侧视相机1", 
                "rgb_rs_1": "侧视相机2"
            }.get(camera_name, camera_name)
            
            table.add_row(display_name, Text(status, style=status_style), fps_text)
        
        return Panel(table, title="[bold magenta]相机状态", border_style="magenta")
    
    def create_dataset_status(self) -> Panel:
        """创建数据集状态面板"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("项目", style="cyan")
        table.add_column("值")
        
        table.add_row("数据集目录", str(self.status_data["dataset"]["directory"]))
        table.add_row("总Episode数", f"{self.status_data['dataset']['total_episodes']}")
        table.add_row("数据集大小", f"{self.status_data['dataset']['current_size_mb']:.1f} MB")
        
        return Panel(table, title="[bold yellow]数据集状态", border_style="yellow")
    
    def create_gamepad_status(self) -> Panel:
        """创建手柄状态面板"""
        connection_status = "✅ 已连接" if self.status_data["gamepad"]["connected"] else "❌ 未连接"
        connection_style = "green" if self.status_data["gamepad"]["connected"] else "red"
        
        delta_pos = self.status_data["gamepad"]["delta_pos"]
        pos_text = f"X:{delta_pos[0]:6.3f} Y:{delta_pos[1]:6.3f} Z:{delta_pos[2]:6.3f}"
        
        content = Text.assemble(
            Text(connection_status, style=connection_style), "\n",
            f"位置增量: {pos_text}\n",
            f"最后指令: {self.status_data['gamepad']['last_command']}"
        )
        
        return Panel(content, title="[bold red]手柄状态", border_style="red")
    
    def create_messages_panel(self) -> Panel:
        """创建消息面板"""
        messages = self.status_data["messages"][-10:]  # 显示最近10条消息
        
        if not messages:
            content = Text("暂无消息", style="dim")
        else:
            content = Text()
            for i, msg in enumerate(messages):
                timestamp = msg.get("timestamp", "")
                text = msg.get("text", "")
                level = msg.get("level", "info")
                
                style_map = {
                    "info": "white",
                    "warning": "yellow", 
                    "error": "red",
                    "success": "green"
                }
                style = style_map.get(level, "white")
                
                content.append(f"[{timestamp}] {text}", style=style)
                if i < len(messages) - 1:
                    content.append("\n")
        
        return Panel(content, title="[bold white]系统消息", border_style="white")
    
    def create_footer(self) -> Layout:
        """创建底部布局"""
        footer_layout = Layout()
        footer_layout.split_row(
            Layout(self.create_gamepad_status(), name="gamepad"),
            Layout(self.create_messages_panel(), name="messages")
        )
        return footer_layout
    
    def update_display(self) -> Layout:
        """更新显示内容"""
        with self.lock:
            layout = self.create_layout()
            
            layout["header"].update(self.create_header())
            layout["system_status"].update(self.create_system_status())
            layout["robot_status"].update(self.create_robot_status())
            layout["camera_status"].update(self.create_camera_status())
            layout["dataset_status"].update(self.create_dataset_status())
            layout["footer"].update(self.create_footer())
            
            return layout
    
    def start(self):
        """启动实时显示"""
        self.running = True
        console.clear()
        
        with Live(self.update_display(), console=console, refresh_per_second=self.refresh_rate) as live:
            self.live = live
            while self.running:
                live.update(self.update_display())
                time.sleep(1.0 / self.refresh_rate)
    
    def stop(self):
        """停止实时显示"""
        self.running = False
    
    # 状态更新方法
    def update_system_status(self, **kwargs):
        """更新系统状态"""
        with self.lock:
            self.status_data["system"].update(kwargs)
    
    def update_robot_status(self, **kwargs):
        """更新机械臂状态"""
        with self.lock:
            self.status_data["robot"].update(kwargs)
    
    def update_camera_status(self, camera_name: str, **kwargs):
        """更新相机状态"""
        with self.lock:
            if camera_name in self.status_data["cameras"]:
                self.status_data["cameras"][camera_name].update(kwargs)
    
    def update_gamepad_status(self, **kwargs):
        """更新手柄状态"""
        with self.lock:
            self.status_data["gamepad"].update(kwargs)
    
    def update_dataset_status(self, **kwargs):
        """更新数据集状态"""
        with self.lock:
            self.status_data["dataset"].update(kwargs)
    
    def add_message(self, text: str, level: str = "info"):
        """添加消息"""
        with self.lock:
            message = {
                "text": text,
                "level": level,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            self.status_data["messages"].append(message)
            
            # 保持消息列表大小
            if len(self.status_data["messages"]) > 50:
                self.status_data["messages"] = self.status_data["messages"][-50:]


def test_live_status_display():
    """测试实时状态显示"""
    display = LiveStatusDisplay()
    
    def update_test_data():
        """更新测试数据"""
        import random
        
        while display.running:
            # 模拟更新数据
            display.update_system_status(
                fps=random.uniform(25, 35),
                frame_count=display.status_data["system"]["frame_count"] + 1,
                is_recording=random.choice([True, False])
            )
            
            display.update_robot_status(
                position=np.random.rand(8) * 2 - 1,
                velocity=np.random.rand(8) * 0.1 - 0.05,
                connected=True,
                gripper_open=random.choice([True, False])
            )
            
            display.update_camera_status("ee_cam", connected=True, fps=random.uniform(28, 32))
            display.update_camera_status("rgb_rs_0", connected=True, fps=random.uniform(28, 32))
            display.update_camera_status("rgb_rs_1", connected=random.choice([True, False]), fps=random.uniform(28, 32))
            
            display.update_gamepad_status(
                connected=True,
                delta_pos=np.random.rand(3) * 0.02 - 0.01,
                last_command=random.choice(["移动", "旋转", "夹爪", "录制"])
            )
            
            display.update_dataset_status(
                directory="test_dataset",
                total_episodes=random.randint(0, 10),
                current_size_mb=random.uniform(10, 1000)
            )
            
            # 随机添加消息
            if random.random() < 0.1:
                messages = [
                    ("系统启动完成", "success"),
                    ("相机连接成功", "info"),
                    ("录制开始", "info"),
                    ("检测到错误", "error"),
                    ("警告：帧率下降", "warning")
                ]
                msg, level = random.choice(messages)
                display.add_message(msg, level)
            
            time.sleep(0.5)
    
    # 启动测试数据更新线程
    test_thread = threading.Thread(target=update_test_data, daemon=True)
    test_thread.start()
    
    try:
        # 启动显示
        display.start()
    except KeyboardInterrupt:
        display.stop()
        print("\n测试结束")


if __name__ == "__main__":
    test_live_status_display()