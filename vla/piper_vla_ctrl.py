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
        "ee_cam": {"serial": "317222073629", "width": 640, "height": 480, "fps": 30},
        "rgb_rs_0": {"serial": "153122078525", "width": 640, "height": 480, "fps": 30},
        "rgb_rs_1": {"serial": "310222078614", "width": 640, "height": 480, "fps": 30}
    }
multi_realsense_manager = MultiRealSenseManager(camera_configs)

class PiperVlaController(PiperBaseController):
    def __init__(self):
        super().__init__()
        # realsense cameras manager
        # save the last action of the action chunk
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

                # number of executed actions s
                s = self.t

                # read remaining actions (if any)
                A_prev = np.zeros((self.H, 8), dtype=np.float32)
                if self.A_cur is not None and s < len(self.A_cur):
                    remaining = self.A_cur[s:]
                    n_remain = len(remaining)
                    if n_remain > 0:
                        A_prev[:n_remain] = remaining

                # current observation and delay estimation
                o = self.o_cur
                d = max(self.delay_buffer) if len(self.delay_buffer) > 0 else 0
                d = 4

                # 2) inference outside the lock, avoid blocking the control thread
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
    """Plotter for robot joint actions"""
    
    def __init__(self, max_history_length: int = 1000):
        self.action_history = []
        self.max_history_length = max_history_length
        self.joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Gripper1', 'Gripper2']
        self.update_counter = 0
        self.update_interval = 10 
        
        self.fig, self.axes = plt.subplots(2, 4, figsize=(16, 8))
        self.axes = self.axes.flatten()
        
        self.lines = []
        self.future_lines = [] 
        self.last_future_actions = None 
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
        """Add an action to the history and update the plots, optionally recording future action segments."""
        self.action_history.append(action.copy())

        if future_actions is not None and len(future_actions) > 0:
            start_idx = len(self.action_history) 
            self.future_segments.append((start_idx, future_actions.copy()))
        
        if len(self.action_history) > self.max_history_length:
            self.action_history.pop(0)
        
        self.update_counter += 1
        if self.update_counter % self.update_interval == 0:
            self._update_plots()
    
    def _update_plots(self):
        """Update all subplots"""
        if len(self.action_history) < 2:
            return
            
        time_steps = list(range(len(self.action_history)))
        total_future_len = sum(seg[1].shape[0] for seg in self.future_segments)
        
        try:
            for i in range(8):
                joint_values = [act[i] for act in self.action_history]
                self.lines[i].set_data(time_steps, joint_values)
                if self.future_segments:
                    xs: list[int|float] = []
                    ys: list[float] = []
                    for start_idx, fut in self.future_segments:
                        xs.extend(range(start_idx, start_idx + fut.shape[0]))
                        ys.extend([fut[j, i] for j in range(fut.shape[0])])
                        xs.append(np.nan) 
                        ys.append(np.nan)
                    self.future_lines[i].set_data(xs, ys)
                else:
                    self.future_lines[i].set_data([], [])
                
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

                max_x_local = len(self.action_history)
                if self.future_segments:
                    max_future_x = max(start + fut.shape[0] for start, fut in self.future_segments)
                    max_x_local = max(max_x_local, max_future_x)
                self.axes[i].set_xlim(0, max_x_local)
            

            self.axes[5].set_ylim(-1.2, 1.2)
            self.axes[6].set_ylim(-1.2, 1.2)
            self.axes[7].set_ylim(-1.2, 1.2)


        except Exception as e:
            pass
    
    def save_plots(self, save_dir: str = "action_plots"):
        """Save plots to file"""
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            self._update_plots()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = os.path.join(save_dir, f"joint_actions_{timestamp}.png")
            
            self.fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
        except Exception as e:
            print(f"Error: {e}")
    
    def close(self):
        try:
            plt.close(self.fig)
        except:
            pass

def signal_handler(signum, frame):
    global should_exit
    should_exit = True

def main(chunk_size: int = 50, use_rtc: bool = False, execution_range: int = 25,
         prefix_attention_schedule: str = "linear", max_guidance_weight: float = 2) -> None:
    """
    Args:
        chunk_size: how many actions to execute in one chunk (H parameter)
        use_rtc: whether to use RTC algorithm or original serial execution
        execution_range: how many actions to execute from each chunk (s parameter)
        prefix_attention_schedule: schedule for prefix attention weights ("linear", "exp", "ones", "zeros")
        max_guidance_weight: maximum guidance weight clipping value
    """
    # set signal handler
    global should_exit
    should_exit = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # vla client policy
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8123)
    piper_vla_controller = PiperVlaController()

    # warm up the robot
    time.sleep(2)

    action_plotter = ActionPlotter()
    time_start = time.time()
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

            # For action smoothing
            action_history = deque(maxlen=4)
            ACTION_MASK = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.float32)
            
            while not should_exit:
                obs_dict = piper_vla_controller.build_obs_dict()
                action = rtc_controller.get_action(obs_dict)
                future_actions = rtc_controller.get_future_actions()
                if action is not None:
                    # --- Action Smoothing ---
                    if len(action_history) > 0:
                        # Calculate weighted average of historical actions
                        history_weights = np.exp(-np.arange(len(action_history)))
                        history_weights /= history_weights.sum()
                        
                        historical_actions_np = np.array(action_history)
                        avg_historical_action = np.sum(historical_actions_np * history_weights[:, np.newaxis], axis=0)
                        # Combine with current action
                        smoothed_action = (
                            0.9 * action + 
                            0.1 * avg_historical_action
                        )
                        # Apply mask
                        action = (
                            action * (1 - ACTION_MASK) + 
                            smoothed_action * ACTION_MASK
                        )
                    action_history.append(action)
                    # --- End of Action Smoothing ---
                    action_plotter.add_action(action, future_actions)
                    piper_vla_controller.apply_action(action, only_update_sim=False)
                    piper_vla_controller.update_last_action(action)
                piper_vla_controller.update_state()
                # Control loop frequency
                time.sleep(0.03)
                # Check if we should reset to initial positions
                time_now = time.time()
                current_positions = piper_vla_controller.piper_real.GetArmJointMsgs()
                back_to_initial = (abs(current_positions.joint_state.joint_1))/1000 +(abs(current_positions.joint_state.joint_3))/2000 < 3
                #print(current_positions)
                if time_now - time_start > 12 and back_to_initial:
                    action_plotter.save_plots()
                    action_plotter.close()
                    piper_vla_controller.reset_to_initial_positions()
                    rtc_controller.stop()
                    print("Reset to initial positions")
                    return
            # Wait for RTC controller to finish
            rtc_controller.stop()
                
        else:
            last_action = None
            gripper_flip = False
            while not should_exit:
                obs_dict = piper_vla_controller.build_obs_dict()
                # get actions from pi0 policy
                action_chunk = client.infer(obs_dict)["actions"]
                # execute previous chunk_size actions in the chunk
                for i in range(chunk_size):
                    if should_exit:
                        break
                    action = action_chunk[i]

                    if last_action is not None and (action[6] - last_action[6] >= 0.1): # only from open to close
                        gripper_flip = True
                        print("gripper state changed")
                    else:
                        gripper_flip = False
                    action_plotter.add_action(action)
                    piper_vla_controller.apply_action(action, only_update_sim=False)
                    # make sure the action is executed
                    piper_vla_controller.update_state()
                    last_action = action
                    if gripper_flip:
                        time.sleep(0.03)
                    else:
                        if i == chunk_size - 1:
                            continue
                        time.sleep(0.03)
                
                time_now = time.time()
                current_positions = piper_vla_controller.piper_real.GetArmJointMsgs()
                back_to_initial = (abs(current_positions.joint_state.joint_1)) + (abs(current_positions.joint_state.joint_3))/2000 < 3
                #print(current_positions)
                if time_now - time_start > 12 and back_to_initial:
                    action_plotter.save_plots()
                    action_plotter.close()
                    piper_vla_controller.reset_to_initial_positions()
                    print("Reset to initial positions")
                    return
                piper_vla_controller.update_last_action(action_chunk[chunk_size - 1])

    except Exception as e:
        print(f"Error: {e}")
    finally:
        piper_vla_controller.reset_to_initial_positions()
        action_plotter.save_plots()
        action_plotter.close()
        print("Program exited")

if __name__ == "__main__":
    tyro.cli(main)