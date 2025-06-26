from multi_realsense_cameras import MultiRealSenseManager
from openpi_client import websocket_client_policy
import numpy as np
from piper_base_ctrl import PiperBaseController
import time
import tyro
import cv2
import os

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
        """
            {
                "images": {
                    "rgb_rs_0": HxWx3 uint8,
                    "rgb_rs_1": HxWx3 uint8,
                    "ee_cam":   HxWx3 uint8,
                },
                "state": np.ndarray(8,), 
            }
        """
        # ---- 1) 处理三路相机 rgb0, rgb1, wrist--------------------------------------------------
        rgbs = multi_realsense_manager.get_all_frames()
        
        # ---- 2) 8维 state ---------------------------------------------
        # 使用上一次执行的action chunk里面的最后一个动作
        if self.last_action is not None:
            state_vec = np.array(self.last_action, dtype=np.float32)
        else:
            # 如果还没有执行过action，使用当前的DOF位置
            state_vec = np.array([0]*8, dtype=np.float32)

        # ---- 3) 返回符合 PiperInputs 的字典 -------------------------------
        obs_dict = {
            "images": {
                "rgb_rs_0": rgbs["rgb_rs_0"],
                "rgb_rs_1": rgbs["rgb_rs_1"],
                "ee_cam":   rgbs["ee_cam"],
            },
            "state": state_vec,
            "prompt": "move the coffee cup to the coffee machine"
        }
        return obs_dict
    
    def update_last_action(self, action):
        """更新上一次执行的动作"""
        self.last_action = action

def main(chunk_size: int = 50) -> None:
    """
    Args:
        chunk_size: how many actions to execute in one chunk
    """
    # vla client
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8123)
    piper_vla_controller = PiperVlaController()

    # warm up
    time.sleep(2)
    try:
        while True:
            obs_dict = piper_vla_controller.build_obs_dict()
            # get actions from pi0 policy
            action_chunk = client.infer(obs_dict)["actions"]
            #print(f"\n{'='*30}\n[Get Action Chunk] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n{'='*30}")
            # os.system('cls' if os.name == 'nt' else 'clear')
            # print(piper_vla_controller.piper_real.GetArmJointMsgs())
            # print(piper_vla_controller.piper_real.GetArmGripperMsgs())
            # execute previous chunk_size actions in the chunk
            for i in range(chunk_size):
                piper_vla_controller.apply_action(action_chunk[i], only_update_sim=False)
                # make sure the action is executed
                time.sleep(0.02)
            
            # 更新上一次执行的action chunk的最后一个动作
            piper_vla_controller.update_last_action(action_chunk[chunk_size - 1])

    except KeyboardInterrupt:
        piper_vla_controller.reset_to_initial_positions()
        print("Reset complete, exiting program.")


if __name__ == "__main__":
    tyro.cli(main)