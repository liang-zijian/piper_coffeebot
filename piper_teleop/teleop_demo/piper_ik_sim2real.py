import argparse
import numpy as np
import genesis as gs
from piper_controller import PiperController
import threading
import time

lock = threading.Lock()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    ########################## 初始化场景 ##########################
    gs.init(backend=gs.gpu)

    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(dt=0.01),
        show_viewer=True,
        show_FPS=False
    )

    ########################## 添加实体 ##########################
    #plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    cam_0 = scene.add_camera(res=(1280, 960), pos=(0, 0.5, 1.5), lookat=(0.5, 0.5, 0.0), fov=55, GUI=False)
    cam_1 = scene.add_camera(res=(1280, 960), pos=(1.5, -0.5, 1.5), lookat=(0.5, 0.5, 0), fov=55, GUI=False)
    cam_2 = scene.add_camera(res=(1280, 960), pos=(0.5, 1.0, 1.5), lookat=(0.5, 0.5, 0.0), fov=55, GUI=False)

    # 加载 Piper （仿真）
    piper_robot = scene.add_entity(gs.morphs.MJCF(file="/home/ubuntu/workspace/piper_ws/piper_teleop/agilex_piper/piper.xml"))

    ########################## 构建场景 ##########################
    scene.build()

    # 设置初始姿态
    JOINT_NAMES = [
        'joint1', 'joint2', 'joint3', 'joint4',
        'joint5', 'joint6', 'joint7', 'joint8'
    ]
    dofs_idx = [piper_robot.get_joint(name).dof_idx_local for name in JOINT_NAMES]

    piper_robot.control_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0.04, -0.04]), dofs_idx)
    piper_robot.set_dofs_kv(np.array([750, 750, 750, 750, 700, 700, 10, 10]))
    piper_robot.set_dofs_kp(np.array([6000, 6000, 6000, 6000, 6000, 6000, 200, 200]))
    # piper_robot.set_dofs_force_range(
    #     np.full(8, -100.0, dtype=np.float32),
    #     np.full(8,  100.0, dtype=np.float32),
    # )

    cams = [cam_0, cam_1, cam_2]

    controller = PiperController(piper=piper_robot, scene=scene, cams=cams, lock=lock)
    controller.start()

    try:
        # 设置目标位置和姿态
        controller.target_pos = np.array([0.06, 0.002, 0.25], dtype=np.float32)  
        controller.target_quat = np.array([ 0.707, 0,  0.707, 0], dtype=np.float32) # 执行器水平位姿
        time.sleep(6)
        controller.target_pos = np.array([0.2, 0.002, 0.20], dtype=np.float32)  
        time.sleep(6)
        controller.target_pos = np.array([0.2, 0.1, 0.20], dtype=np.float32) 
        time.sleep(6) 
        controller.target_pos = np.array([0.2, -0.1, 0.20], dtype=np.float32) 
        time.sleep(30)
    except KeyboardInterrupt:
        print("exiting main loop")
    finally:
        controller.reset_to_initial_positions()
        print("reset arm")

if __name__ == "__main__":
    main()