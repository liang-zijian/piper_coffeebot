import argparse
import numpy as np
import genesis as gs
# from piper_ps4 import PiperController
# from acc_grasp import PiperController
from piper_xiaoji_real import PiperController
import threading
import time

# 全局锁
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
        show_viewer=args.vis,
        show_FPS=False
    )

    ########################## 添加实体 ##########################
    #plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    # 摄像机设置
    cam_0 = scene.add_camera(res=(1280, 960), pos=(0, 0.5, 1.5), lookat=(0.5, 0.5, 0.0), fov=55, GUI=False)
    cam_1 = scene.add_camera(res=(1280, 960), pos=(1.5, -0.5, 1.5), lookat=(0.5, 0.5, 0), fov=55, GUI=False)
    cam_2 = scene.add_camera(res=(1280, 960), pos=(0.5, 1.0, 1.5), lookat=(0.5, 0.5, 0.0), fov=55, GUI=False)

    # 加载 Piper 机器人
    piper_robot = scene.add_entity(gs.morphs.MJCF(file="/home/ubuntu/workspace/piper_ws/agilex_piper/piper.xml"))

    ########################## 构建场景 ##########################
    scene.build()

    # 设置初始姿态
    JOINT_NAMES = [
        'joint1', 'joint2', 'joint3', 'joint4',
        'joint5', 'joint6', 'joint7', 'joint8'
    ]
    dofs_idx = [piper_robot.get_joint(name).dof_idx_local for name in JOINT_NAMES]

    piper_robot.control_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0.04, -0.04]), dofs_idx)
    # piper_robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 100, 100]))
    # piper_robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 10, 10]))
    piper_robot.set_dofs_kp(np.array([6000, 6000, 5000, 5000, 3000, 3000, 200, 200]))
    piper_robot.set_dofs_kv(np.array([150, 150, 120, 120, 80, 80, 10, 10]))
    piper_robot.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 100, 100]),
    )

    cams = [cam_0, cam_1, cam_2]

    # 创建控制器（使用手柄控制）
    controller = PiperController(piper=piper_robot, scene=scene, cams=cams, lock=lock)
    controller.start()


    print("开始运行主循环")

    # 运行主循环
    try:
        while controller.running:
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("退出程序")
    finally:
        controller.reset_to_initial_positions()
        print("真机归位")



if __name__ == "__main__":
    main()