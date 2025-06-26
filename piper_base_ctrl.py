import numpy as np
import genesis as gs
from piper_sdk import *
import time

class PiperBaseController:
    def __init__(self):
        # --------- sim init ---------
        gs.init(backend=gs.gpu)
        self.viewer_options = gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
            max_FPS=60,
        )
        self.scene = gs.Scene(
            viewer_options=self.viewer_options,
            rigid_options=gs.options.RigidOptions(dt=0.01),
            show_viewer=True,
            show_FPS=False
        )
        self.JOINT_NAMES = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7', 'joint8'
        ]
        piper_xml_path = "/home/ubuntu/workspace/piper_ws/piper_teleop/agilex_piper/piper.xml"
        self.piper_sim = self.scene.add_entity(gs.morphs.MJCF(file=piper_xml_path))
        self.scene.build()  
        self.dofs_idx = [self.piper_sim.get_joint(name).dof_idx_local for name in self.JOINT_NAMES]
        self.piper_sim.control_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0.04, -0.04]), self.dofs_idx)
        self.piper_sim.set_dofs_kp(np.array([6000, 6000, 5000, 5000, 3000, 3000, 200, 200]))
        self.piper_sim.set_dofs_kv(np.array([150, 150, 120, 120, 80, 80, 10, 10]))
        self.piper_sim.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 100, 100]),
        )
        # --------- real init ---------
        self.piper_real = C_PiperInterface_V2("can0")
        print("piper_real init")
        self.reset_to_initial_positions()
        self.piper_real.ConnectPort()
        self.enable_arm()
    
    def update_dofs_position(self, target_qpos, dofs_idx):
        """update dofs position in sim"""
        # gripper control
        # if target_qpos[-1] == 1:
        #     target_qpos[-2:] = 0.0
        # else:
        #     target_qpos[-2] = 0.04
        #     target_qpos[-1] = -0.04
        #self.target_qpos = target_qpos.clone()
        self.piper_sim.control_dofs_position(target_qpos[:-2], dofs_idx[:-2])
    
    def reset_to_initial_positions(self):
        print("reset to initial positions")
        initial_pos = np.array([0, 0, 0, 0, 0, 0, 0.04, -0.04])
        self.apply_action(initial_pos)
        self.piper_real.GripperCtrl(0, 1000, 0x01, 0)

    def apply_action(self, actions: np.ndarray, only_update_sim: bool = False):
        if only_update_sim:
            self.update_dofs_position(actions, self.dofs_idx) 
            return
        
        factor = 57295.7795
        joints = [0] * 6
        for i in range(0, 6):
            joints[i] = round(actions[i] * factor)
        self.piper_real.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        # arm actions
        self.piper_real.JointCtrl(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])
        # gripper actions, actions[6] and actions[7] share the same value
        if actions[6] >= 0.5:
            self.piper_real.GripperCtrl(5000 * 1000, 1000, 0x01, 0)
        else:
            self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
    
    def enable_arm(self):
        """enable piper robot"""
        enable_flag = False
        timeout = 5
        start_time = time.time()
        
        while not enable_flag:
            elapsed_time = time.time() - start_time
            
            # 获取使能状态
            arm_info = self.piper_real.GetArmLowSpdInfoMsgs()
            motor_status = [
                arm_info.motor_1.foc_status.driver_enable_status,
                arm_info.motor_2.foc_status.driver_enable_status,
                arm_info.motor_3.foc_status.driver_enable_status,
                arm_info.motor_4.foc_status.driver_enable_status,
                arm_info.motor_5.foc_status.driver_enable_status,
                arm_info.motor_6.foc_status.driver_enable_status
            ]
            
            enable_flag = all(motor_status)
            
            self.piper_real.EnableArm(7)
            self.piper_real.GripperCtrl(0, 1000, 0x01, 0)
            
            if elapsed_time > timeout:
                time.sleep(2)
                print("enable arm failed")
                exit(0)
            
            time.sleep(0.5)
