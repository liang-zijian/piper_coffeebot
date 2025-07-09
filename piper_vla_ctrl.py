from multi_realsense_cameras import MultiRealSenseManager
from openpi_client import websocket_client_policy
import numpy as np
from piper_base_ctrl import PiperBaseController
import time
import tyro
import cv2
import os
import threading
from collections import deque
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from datetime import datetime
import signal
import sys

camera_configs = {
        "ee_cam": {"width": 640, "height": 480, "fps": 30},
        "rgb_rs_0": {"width": 640, "height": 480, "fps": 30},
        "rgb_rs_1": {"width": 640, "height": 480, "fps": 30}
    }
multi_realsense_manager = MultiRealSenseManager(camera_configs)

class PiperVlaController(PiperBaseController):
    def __init__(self):
        super().__init__()
        # realsense cameras manager
        # 保存上一次action chunk的最后一个动作
        self.last_action = None

    def build_obs_dict(self) -> dict:
        rgbs = multi_realsense_manager.get_all_frames()

        if self.last_action is not None:
            state_vec = np.array(self.last_action, dtype=np.float32)
        else:
            state_vec = np.zeros(8, dtype=np.float32)

        return {
            "images": {
                "rgb_rs_0": rgbs.get("rgb_rs_0"),
                "rgb_rs_1": rgbs.get("rgb_rs_1"),
                "ee_cam": rgbs.get("ee_cam"),
            },
            "state": state_vec,
            "prompt": "move the coffee cup to the coffee machine",
        }

    def update_last_action(self, action: np.ndarray):
        self.last_action = action
    
    def update_state(self):
        # print(self.piper_real.GetArmJointMsgs())
        # print(self.piper_real.GetArmGripperMsgs())
        # print(self.last_action)
        pass

class RTCController:
    """Real-Time Chunking Controller implementing the RTC algorithm."""
    
    def __init__(self, policy_client, piper_controller, H: int = 50, s_min: int = 10, 
                 d_init: int = 3, b: int = 10, beta: float = 5.0, 
                 prefix_attention_schedule: str = "linear", max_guidance_weight: float = 5.0):
        """
        Initialize RTC Controller.
        
        Args:
            policy_client: WebSocket client for policy inference
            piper_controller: Piper robot controller
            H: prediction horizon (action chunk size)
            s_min: minimum execution range
            d_init: initial delay estimate
            b: delay buffer size
            beta: guidance weight clipping value (deprecated, use max_guidance_weight)
            prefix_attention_schedule: schedule for prefix attention weights ("linear", "exp", "ones", "zeros")
            max_guidance_weight: maximum guidance weight clipping value
        """
        self.policy_client = policy_client
        self.piper_controller = piper_controller
        self.H = H
        self.s_min = s_min
        self.beta = beta  # kept for compatibility
        self.max_guidance_weight = max_guidance_weight
        self.prefix_attention_schedule = prefix_attention_schedule
        
        # Shared state (protected by mutex)
        self.mutex = threading.Lock()
        self.condition = threading.Condition(self.mutex)
        self.t = 0  # current time step
        self.A_cur = None  # current action chunk
        self.o_cur = None  # current observation
        self.running = False
        
        # Delay estimation buffer
        self.delay_buffer = deque([d_init], maxlen=b)
        
        # Inference thread
        self.inference_thread = None
        
    def start(self):
        """Start the RTC controller."""
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
    def stop(self):
        """Stop the RTC controller."""
        with self.condition:
            self.running = False
            self.condition.notify_all()

        if self.inference_thread:
            self.inference_thread.join()
            
    def get_action(self, o_next: dict) -> Optional[np.ndarray]:
        """Controller interface - called every Δt to get next action."""
        with self.condition:
            self.t += 1
            self.o_cur = o_next
            self.condition.notify()  # Notify inference thread
            
            if self.A_cur is not None and self.t <= len(self.A_cur):
                return self.A_cur[self.t - 1]
            else:
                # Fallback - return zero action if no chunk available
                return np.zeros(8, dtype=np.float32)
                
    def get_future_actions(self) -> Optional[np.ndarray]:
        """返回当前 action chunk 中尚未执行且未被冻结的部分 (H - s - d)。"""
        with self.condition:
            if self.A_cur is None:
                return None
            # 当前已执行动作数量 s 由 self.t 给出
            s_now = self.t
            # 估计的推理延迟 d 取延迟缓冲区中的最大值（与算法一致）
            d_now = max(self.delay_buffer) if len(self.delay_buffer) > 0 else 0
            start_idx = s_now + d_now
            if start_idx >= len(self.A_cur):
                return None
            return self.A_cur[start_idx:].copy()

    def _inference_loop(self):
        """Background inference loop (runs in专用线程)."""
        while self.running:
            with self.condition:
                self.condition.wait_for(lambda: self.t >= self.s_min or not self.running)
                if not self.running:
                    break

                # 已执行的动作数量 s
                s = self.t

                # 读取剩余动作（如果有）
                A_prev = np.zeros((self.H, 8), dtype=np.float32)
                if self.A_cur is not None and s < len(self.A_cur):
                    remaining = self.A_cur[s:]
                    n_remain = len(remaining)
                    if n_remain > 0:
                        A_prev[:n_remain] = remaining

                # 当前观测以及延迟估计
                o = self.o_cur
                #d = max(self.delay_buffer) if len(self.delay_buffer) > 0 else 0
                d = max(self.delay_buffer) if len(self.delay_buffer) > 0 else 0

                # 2) 锁外进行推理，避免阻塞控制线程
                print(f"estimated_delay: {d}")
                A_new = self._guided_inference(o, A_prev, d, s)

                self.A_cur = A_new
                self.t = self.t - s
                self.delay_buffer.append(self.t)


    def _guided_inference(self, obs: dict, A_prev: np.ndarray, d: int, s: int) -> np.ndarray:
        """Perform guided inference using the policy client."""
        try:
            if hasattr(self.policy_client, 'infer_guided'):
                # Use guided inference if available
                result = self.policy_client.infer_guided(
                    obs, A_prev, d, s, 
                    prefix_attention_schedule=self.prefix_attention_schedule,
                    max_guidance_weight=self.max_guidance_weight
                )
            else:
                # Fallback to regular inference
                result = self.policy_client.infer(obs)
            
            return result["actions"]
            
        except Exception as e:
            print(f"Inference error: {e}")
            # Return previous actions as fallback
            if A_prev is not None:
                return A_prev
            else:
                return np.zeros((self.H, 8), dtype=np.float32)

class ActionPlotter:
    """实时绘制和保存机器人关节角度的类"""
    
    def __init__(self, max_history_length: int = 1000):
        self.action_history = []
        self.max_history_length = max_history_length
        self.joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Gripper1', 'Gripper2']
        self.update_counter = 0
        self.update_interval = 10  # 每10次更新一次图表，减少频率
        
        # 创建图表，使用Agg后端不会显示窗口
        self.fig, self.axes = plt.subplots(2, 4, figsize=(16, 8))
        self.axes = self.axes.flatten()
        
        # 初始化每个子图
        self.lines = []
        self.future_lines = []  # 未来动作虚线
        self.last_future_actions = None  # 保存最近一次非 None 的 future
        # 保存所有 future action 段 (start_idx, ndarray[steps,8])
        self.future_segments: list[tuple[int, np.ndarray]] = []
        for i in range(8):
            self.axes[i].set_title(f'{self.joint_names[i]}')
            self.axes[i].set_xlabel('Time Step')
            self.axes[i].set_ylabel('Angle (rad)')
            self.axes[i].grid(True, alpha=0.3)
            line, = self.axes[i].plot([], [], 'b-', linewidth=2, alpha=0.7)
            future_line, = self.axes[i].plot([], [], 'r--', linewidth=1, alpha=0.5)
            self.lines.append(line)
            self.future_lines.append(future_line)
        
        plt.tight_layout()
    
    def add_action(self, action: np.ndarray, future_actions: Optional[np.ndarray] = None):
        """添加一个动作到历史记录并更新图表，同时可选地记录未来动作段。"""
        # 1. 追加当前执行动作到历史
        self.action_history.append(action.copy())

        # 2. 若有新的 future_actions，则存储其段信息，便于后续累计绘制
        if future_actions is not None and len(future_actions) > 0:
            start_idx = len(self.action_history)  # 当前 step 之后即为 future 起点
            self.future_segments.append((start_idx, future_actions.copy()))
        
        # 限制历史记录长度
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)
        
        # 降低更新频率，减少计算负担
        self.update_counter += 1
        if self.update_counter % self.update_interval == 0:
            self._update_plots()
    
    def _update_plots(self):
        """更新所有子图"""
        if len(self.action_history) < 2:
            return
            
        time_steps = list(range(len(self.action_history)))
        # 计算 future_lines 需要的聚合长度
        total_future_len = sum(seg[1].shape[0] for seg in self.future_segments)
        
        try:
            for i in range(8):
                joint_values = [act[i] for act in self.action_history]
                self.lines[i].set_data(time_steps, joint_values)
                # 聚合所有历史 future 段为连续折线（用 NaN 分割可断开虚线）
                if self.future_segments:
                    xs: list[int|float] = []
                    ys: list[float] = []
                    for start_idx, fut in self.future_segments:
                        xs.extend(range(start_idx, start_idx + fut.shape[0]))
                        ys.extend([fut[j, i] for j in range(fut.shape[0])])
                        xs.append(np.nan)  # 分割不同段
                        ys.append(np.nan)
                    self.future_lines[i].set_data(xs, ys)
                else:
                    self.future_lines[i].set_data([], [])
                
                # 自动调整y轴范围（考虑未来动作范围）
                combined_vals = joint_values.copy()
                for _, fut in self.future_segments:
                    combined_vals.extend([fut[j, i] for j in range(fut.shape[0])])
                if combined_vals:
                    y_min, y_max = min(combined_vals), max(combined_vals)
                    y_range = y_max - y_min
                    if y_range > 0:
                        self.axes[i].set_ylim(y_min - 0.5 * y_range, y_max + 0.5 * y_range)
                    else:
                        self.axes[i].set_ylim(y_min - 0.5, y_max + 0.5)

                # 调整x轴范围应始终执行
                max_x_local = len(self.action_history)
                if self.future_segments:
                    max_future_x = max(start + fut.shape[0] for start, fut in self.future_segments)
                    max_x_local = max(max_x_local, max_future_x)
                self.axes[i].set_xlim(0, max_x_local)
            

            self.axes[5].set_ylim(-1.2, 1.2)
            self.axes[6].set_ylim(-1.2, 1.2)
            self.axes[7].set_ylim(-1.2, 1.2)


        except Exception as e:
            # 静默处理绘图错误
            pass
    
    def save_plots(self, save_dir: str = "action_plots"):
        """保存图表到文件"""
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 在保存前确保使用最新数据更新一次图表
            self._update_plots()
            
            # 生成时间戳文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = os.path.join(save_dir, f"joint_actions_{timestamp}.png")
            
            # 保存图表
            self.fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            
            # 保存原始数据
            if self.action_history:
                data_filename = os.path.join(save_dir, f"joint_actions_data_{timestamp}.npy")
                np.save(data_filename, np.array(self.action_history))
        except Exception as e:
            print(f"Error: {e}")
    
    def close(self):
        """关闭图表"""
        try:
            plt.close(self.fig)
        except:
            pass

def signal_handler(signum, frame):
    global should_exit
    should_exit = True

def main(chunk_size: int = 50, use_rtc: bool = True, execution_range: int = 25,
         prefix_attention_schedule: str = "exp", max_guidance_weight: float = 2) -> None:
    """
    Args:
        chunk_size: how many actions to execute in one chunk (H parameter)
        use_rtc: whether to use RTC algorithm or original serial execution
        execution_range: how many actions to execute from each chunk (s parameter)
        prefix_attention_schedule: schedule for prefix attention weights ("linear", "exp", "ones", "zeros")
        max_guidance_weight: maximum guidance weight clipping value
    """
    # 设置信号处理
    global should_exit
    should_exit = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # vla client
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8123)
    piper_vla_controller = PiperVlaController()

    # warm up
    time.sleep(2)
    
    # 初始化动作绘图器
    action_plotter = ActionPlotter()
    
    try:
        if use_rtc:
            # Use RTC algorithm
            rtc_controller = RTCController(
                policy_client=client,
                piper_controller=piper_vla_controller,
                H=chunk_size,
                s_min=execution_range,
                prefix_attention_schedule=prefix_attention_schedule,
                max_guidance_weight=max_guidance_weight
            )
            rtc_controller.start()
            
            while not should_exit:
                obs_dict = piper_vla_controller.build_obs_dict()
                action = rtc_controller.get_action(obs_dict)
                future_actions = rtc_controller.get_future_actions()

                if action is not None:
                    # 绘制动作（历史 + 未来）
                    action_plotter.add_action(action, future_actions)
                    piper_vla_controller.apply_action(action, only_update_sim=False)
                    piper_vla_controller.update_last_action(action)
                piper_vla_controller.update_state()
                # Control loop frequency
                time.sleep(0.06)
            
            rtc_controller.stop()
                
        else:
            # Original serial execution
            while not should_exit:
                obs_dict = piper_vla_controller.build_obs_dict()
                # get actions from pi0 policy
                action_chunk = client.infer(obs_dict)["actions"]
                
                # execute previous chunk_size actions in the chunk
                for i in range(chunk_size):
                    if should_exit:
                        break
                        
                    action = action_chunk[i]
                    
                    # 绘制动作
                    action_plotter.add_action(action)
                    
                    piper_vla_controller.apply_action(action, only_update_sim=False)
                    # make sure the action is executed
                    piper_vla_controller.update_state()
                    time.sleep(0.06)
                
                # 更新上一次执行的action chunk的最后一个动作
                piper_vla_controller.update_last_action(action_chunk[chunk_size - 1])

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 清理资源
        piper_vla_controller.reset_to_initial_positions()
        # 保存并关闭图表
        action_plotter.save_plots()
        action_plotter.close()
        print("Program exited")

if __name__ == "__main__":
    tyro.cli(main)