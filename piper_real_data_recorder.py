#!/usr/bin/env python3
"""
Piper机械臂真实环境数据录制主脚本
整合所有模块实现完整的数据录制功能
"""

import argparse
import time
import threading
import numpy as np
from pathlib import Path
import signal
import sys
from typing import Optional

# 导入Genesis和Piper相关模块
import genesis as gs
from piper_xiaoji_real import PiperController

# 导入自定义模块
from multi_realsense_cameras import MultiRealSenseManager
from robot_state_recorder import RobotStateRecorder
from lerobot_dataset_manager import LeRobotDatasetManager
from gamepad_controller import ThreadedGamepadController
from live_status_display import LiveStatusDisplay

# 导入rich用于日志
from rich.console import Console
from rich.logging import RichHandler
import logging

# 配置日志
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("PiperDataRecorder")

class PiperRealDataRecorder:
    """Piper机械臂真实环境数据录制器"""
    
    def __init__(self, args):
        """初始化录制器"""
        self.args = args
        self.running = False
        self.recording = False
        
        # 组件实例
        self.scene = None
        self.piper_robot = None
        self.piper_controller = None
        self.camera_manager = None
        self.robot_recorder = None
        self.dataset_manager = None
        self.gamepad_controller = None
        self.status_display = None
        
        # 状态变量
        self.frame_count = 0
        self.episode_count = 0
        self.start_time = time.time()
        
        # 线程和锁
        self.main_lock = threading.Lock()
        self.status_thread = None
        self.recording_thread = None
        
        # 统计信息
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        logger.info("Piper data recorder initialized")
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info("Received exit signal, shutting down...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def init_genesis_scene(self):
        """初始化Genesis仿真场景"""
        try:
            logger.info("Initializing Genesis scene...")
            
            gs.init(backend=gs.gpu)
            
            viewer_options = gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=30,
                max_FPS=60,
            )
            
            self.scene = gs.Scene(
                viewer_options=viewer_options,
                rigid_options=gs.options.RigidOptions(dt=0.01),
                show_viewer=self.args.vis,
                show_FPS=False
            )
            
            # 加载Piper机器人
            piper_xml_path = "/home/ubuntu/workspace/piper_ws/piper_teleop/agilex_piper/piper.xml"
            self.piper_robot = self.scene.add_entity(gs.morphs.MJCF(file=piper_xml_path))
            
            # 构建场景
            self.scene.build()
            
            # 设置初始姿态和控制参数
            JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7', 'joint8']
            dofs_idx = [self.piper_robot.get_joint(name).dof_idx_local for name in JOINT_NAMES]
            
            self.piper_robot.control_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0.04, -0.04]), dofs_idx)
            self.piper_robot.set_dofs_kp(np.array([6000, 6000, 5000, 5000, 3000, 3000, 200, 200]))
            self.piper_robot.set_dofs_kv(np.array([150, 150, 120, 120, 80, 80, 10, 10]))
            self.piper_robot.set_dofs_force_range(
                np.array([-87, -87, -87, -87, -12, -12, -100, -100]),
                np.array([87, 87, 87, 87, 12, 12, 100, 100]),
            )
            
            logger.info("✅ Genesis scene initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Genesis scene initialization failed: {e}")
            return False
    
    def init_cameras(self):
        """初始化相机"""
        try:
            logger.info("Initializing RealSense cameras...")
            
            # 相机配置
            camera_configs = {
                "ee_cam": {"width": 640, "height": 480, "fps": 30},
                "rgb_rs_0": {"width": 640, "height": 480, "fps": 30},
                "rgb_rs_1": {"width": 640, "height": 480, "fps": 30}
            }
            
            self.camera_manager = MultiRealSenseManager(camera_configs)
            
            if self.camera_manager.get_camera_count() == 0:
                logger.warning("⚠️ No RealSense cameras detected, using simulated data")
                return False
            
            logger.info(f"✅ Successfully initialized {self.camera_manager.get_camera_count()} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def init_robot_recorder(self):
        """初始化机械臂状态记录器"""
        try:
            logger.info("Initializing robot state recorder...")
            self.robot_recorder = RobotStateRecorder(self.piper_robot)
            logger.info("Robot state recorder initialized")
            return True
            
        except Exception as e:
            logger.error(f"Robot state recorder initialization failed: {e}")
            return False
    
    def init_dataset_manager(self):
        """初始化数据集管理器"""
        try:
            logger.info("初始化数据集管理器...")
            
            self.dataset_manager = LeRobotDatasetManager(
                dataset_dir=self.args.dataset_dir,
                repo_id=self.args.repo_id,
                fps=self.args.fps,
                task_description=self.args.task_description
            )
            
            # 创建数据集
            if not self.dataset_manager.create_dataset(resume=self.args.resume):
                logger.error("Dataset creation failed")
                return False
            
            logger.info("✅ Dataset manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dataset manager initialization failed: {e}")
            return False
    
    def init_piper_controller(self):
        """初始化Piper控制器"""
        try:
            logger.info("Initializing Piper controller...")
            
            # 创建虚拟相机（用于兼容原始代码）
            cams = []
            
            self.piper_controller = PiperController(
                piper=self.piper_robot,
                scene=self.scene,
                cams=cams,
                lock=self.main_lock,
                use_external_gamepad=True  # 使用外部手柄控制器
            )
            
            self.piper_controller.start()
            logger.info("✅ Piper controller initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Piper controller initialization failed: {e}")
            return False
    
    def init_gamepad_controller(self):
        """初始化手柄控制器"""
        try:
            logger.info("初始化手柄控制器...")
            
            self.gamepad_controller = ThreadedGamepadController()
            
            # 设置回调函数
            self.gamepad_controller.set_record_start_callback(self.start_recording)
            self.gamepad_controller.set_record_stop_callback(self.stop_recording)
            self.gamepad_controller.set_movement_callback(self.handle_gamepad_movement)
            self.gamepad_controller.set_grip_callback(self.handle_gamepad_grip)
            
            if not self.gamepad_controller.start_threaded():
                logger.warning("⚠️ Gamepad controller initialization failed, using keyboard control only")
                return False
            
            logger.info("✅ Gamepad controller initialized")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Gamepad controller initialization failed: {e}")
            return False
    
    def init_status_display(self):
        """初始化状态显示"""
        try:
            logger.info("Initializing status display...")
            
            self.status_display = LiveStatusDisplay(refresh_rate=10.0)
            
            # 启动状态显示线程
            self.status_thread = threading.Thread(target=self.run_status_display, daemon=True)
            self.status_thread.start()
            
            logger.info("Status display initialized")
            return True
            
        except Exception as e:
            logger.error(f"Status display initialization failed: {e}")
            return False
    
    def run_status_display(self):
        """运行状态显示"""
        try:
            self.status_display.start()
        except Exception as e:
            logger.error(f"Status display running failed: {e}")
    
    def update_status_display(self):
        """更新状态显示数据"""
        if not self.status_display:
            return
        
        # 更新系统状态
        self.status_display.update_system_status(
            fps=self.current_fps,
            frame_count=self.frame_count,
            is_recording=self.recording,
            episode_count=self.episode_count
        )
        
        # 更新机械臂状态
        if self.robot_recorder and self.robot_recorder.current_position is not None:
            self.status_display.update_robot_status(
                position=self.robot_recorder.current_position,
                velocity=self.robot_recorder.current_velocity,
                connected=True,
                gripper_open=not getattr(self.piper_controller, 'is_grasp', False)
            )
        
        # 更新相机状态
        if self.camera_manager:
            for camera_name in self.camera_manager.get_camera_names():
                self.status_display.update_camera_status(camera_name, connected=True, fps=30.0)
        
        # 更新手柄状态
        if self.gamepad_controller:
            gamepad_input = self.gamepad_controller.get_current_input_safe()
            self.status_display.update_gamepad_status(
                connected=True,
                delta_pos=gamepad_input.get("delta_pos", np.zeros(3)),
                last_command="Normal"
            )
        
        # 更新数据集状态
        if self.dataset_manager:
            dataset_info = self.dataset_manager.get_dataset_info()
            self.status_display.update_dataset_status(
                directory=dataset_info.get("dataset_dir", "未设置"),
                total_episodes=dataset_info.get("total_episodes", 0)
            )
    
    def start_recording(self):
        """开始录制"""
        if self.recording:
            logger.warning("Already recording")
            return
        
        if not self.dataset_manager:
            logger.error("Dataset manager not initialized")
            return
        
        logger.info("🎬 Start recording episode...")
        
        if self.dataset_manager.start_episode():
            self.recording = True
            self.frame_count = 0
            self.status_display.add_message("Start recording new episode", "success")
        else:
            logger.error("Start recording failed")
            self.status_display.add_message("Start recording failed", "error")
    
    def stop_recording(self):
        """停止录制"""
        if not self.recording:
            logger.warning("Not recording")
            if self.status_display:
                self.status_display.add_message("Not recording", "warning")
            return
        
        if not self.dataset_manager:
            logger.error("Dataset manager not initialized")
            if self.status_display:
                self.status_display.add_message("Dataset manager not initialized", "error")
            return
        
        logger.info("="*60)
        logger.info(f"🎬 Start stopping recording process")
        logger.info(f"⏹️ Stop recording episode {self.episode_count}...")
        logger.info(f"Current recorded frames: {self.frame_count}")
        logger.info("="*60)
        
        # 检查是否有录制的帧数据
        if self.frame_count == 0:
            logger.warning("⚠️ 当前episode没有录制任何帧数据")
            if self.status_display:
                self.status_display.add_message("Episode has no data, skip saving", "warning")
            
            # 仍然调用end_episode以正确重置状态
            logger.info("Call dataset_manager.end_episode() to reset state...")
            if self.dataset_manager.end_episode():
                self.recording = False
                logger.info("✅ State reset successfully")
                return
            else:
                logger.error("❌ State reset failed")
                self.recording = False
                return
        
        try:
            # 记录保存开始时间
            import time
            save_start_time = time.time()
            logger.info(f"🔄 Start saving episode data...")
            
            # 尝试结束episode
            save_result = self.dataset_manager.end_episode()
            save_duration = time.time() - save_start_time
            
            logger.info(f"💾 Save operation took: {save_duration:.2f} seconds")
            logger.info(f"💾 Save result: {'Success' if save_result else 'Failed'}")
            
            if save_result:
                self.recording = False
                self.episode_count += 1
                
                success_msg = f"Episode {self.episode_count} Done ({self.frame_count} frames)"
                logger.info(f"✅ {success_msg}")
                logger.info("="*60)
                
                if self.status_display:
                    self.status_display.add_message(success_msg, "success")
                
                # 重置帧计数器
                self.frame_count = 0
                
            else:
                error_msg = "Episode save failed"
                logger.error(f"❌ {error_msg}")
                logger.error("="*60)
                
                if self.status_display:
                    self.status_display.add_message(error_msg, "error")
                    
                # 即使保存失败，也要重置录制状态以避免卡住
                self.recording = False
                self.frame_count = 0
                
                
        except Exception as e:
            error_msg = f"Stop recording failed: {e}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("="*60)
            
            # 打印完整错误堆栈
            import traceback
            logger.error("Full error stack:")
            logger.error(traceback.format_exc())
            
            if self.status_display:
                self.status_display.add_message("Stop recording failed", "error")
            
            # 强制重置状态
            self.recording = False
            self.frame_count = 0
            
    def handle_gamepad_movement(self, delta_pos, joint5_delta, joint6_delta):
        """处理手柄移动输入"""
        if self.piper_controller:
            self.piper_controller.handle_external_gamepad_input(
                delta_pos, joint5_delta, joint6_delta, False
            )
    
    def handle_gamepad_grip(self, grip_toggle):
        """处理手柄夹爪控制"""
        if self.piper_controller:
            self.piper_controller.handle_external_gamepad_input(
                np.zeros(3), 0.0, 0.0, grip_toggle
            )
    
    def record_frame(self):
        """录制一帧数据"""
        if not self.recording or not self.dataset_manager:
            return False
        
        try:
            # 获取相机图像
            camera_images = {}
            if self.camera_manager:
                try:
                    camera_images = self.camera_manager.get_color_frames_for_lerobot()
                except Exception as cam_e:
                    logger.warning(f"Camera data acquisition failed: {cam_e}, using simulated data")
                    camera_images = {}
            
            # 确保所有必需的相机数据都存在
            required_cameras = ["observation.images.ee_cam", "observation.images.rgb_rs_0", "observation.images.rgb_rs_1"]
            for camera_key in required_cameras:
                if camera_key not in camera_images or camera_images[camera_key] is None:
                    # 如果任何一个相机数据缺失，使用黑屏图像替代(BGR HWC格式)
                    if self.frame_count % 50 == 0:  # 每50帧提示一次，避免日志过多
                        logger.warning(f"Camera data missing: {camera_key}, using black screen image")
                    camera_images[camera_key] = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # 验证图像尺寸并修复(期望BGR HWC格式)
                img = camera_images[camera_key]
                if not isinstance(img, np.ndarray) or img.shape != (480, 640, 3):
                    if self.frame_count % 50 == 0:  # 每50帧提示一次
                        logger.warning(f"Image {camera_key} format incorrect, recreate")
                    camera_images[camera_key] = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 获取机械臂状态和动作
            try:
                frame_data = self.robot_recorder.get_frame_data_for_lerobot("absolute_position")
                if not frame_data:
                    logger.warning("Robot state data acquisition failed, skip this frame")
                    return False
                
                robot_state = frame_data.get("observation.state")
                actions = frame_data.get("actions")
                
            except Exception as robot_e:
                logger.warning(f"Robot data acquisition exception: {robot_e}, skip this frame")
                return False
            
            # 验证机械臂数据完整性
            if robot_state is None or not isinstance(robot_state, np.ndarray) or len(robot_state) != 8:
                if self.frame_count % 20 == 0:  # 每20帧提示一次
                    logger.warning(f"Robot state data invalid (type: {type(robot_state)}, length: {len(robot_state) if robot_state is not None else 0})")
                return False
                
            if actions is None or not isinstance(actions, np.ndarray) or len(actions) != 8:
                if self.frame_count % 20 == 0:  # 每20帧提示一次
                    logger.warning(f"Action data invalid (type: {type(actions)}, length: {len(actions) if actions is not None else 0})")
                return False
            
            # ------- 同步夹爪状态到 robot_state 与 actions -------
            try:
                if self.piper_controller is not None:
                    gripper_open = not self.piper_controller.is_grasp  # True 表示张开
                    open_val = np.int32(1 if gripper_open else 0)

                    # 为了安全，先复制一份再修改，避免意外共享引用
                    actions = actions.copy()
                    robot_state = robot_state.copy()

                    actions[-2:] = open_val
                    robot_state[-2:] = open_val
            except Exception as grip_e:
                if self.frame_count % 50 == 0:
                    logger.warning(f"更新夹爪开合状态失败: {grip_e}")
            # ------- END -------

            # 所有数据验证通过，添加帧到数据集
            try:
                success = self.dataset_manager.add_frame(
                    camera_images=camera_images,
                    robot_state=robot_state,
                    actions=actions,
                    task=self.args.task_description
                )
                
                if success:
                    self.frame_count += 1
                    
                    # 每20帧输出一次详细信息
                    if self.frame_count % 20 == 0:
                        logger.info(f"✅ Recorded {self.frame_count} frames (Episode {self.episode_count})")
                    
                    return True
                else:
                    # 数据集添加失败，但不要过于频繁地记录错误
                    if self.frame_count % 10 == 0:
                        logger.warning(f"Failed to add frame data to dataset (Recorded {self.frame_count} frames)")
                    return False
                    
            except Exception as dataset_e:
                logger.error(f"Dataset add frame exception: {dataset_e}")
                return False
                
        except Exception as e:
            logger.error(f"Record frame unknown exception: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            return False
    
    def calculate_fps(self):
        """计算FPS"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = 30.0 / elapsed
            self.fps_start_time = current_time
    
    def run_main_loop(self):
        """运行主循环"""
        logger.info("🚀 Start running main loop")
        self.running = True
        
        while self.running:
            try:
                # 运行仿真步骤
                if self.scene:
                    self.scene.step()
                
                # 录制帧（如果正在录制）
                if self.recording:
                    self.record_frame()
                
                # 计算FPS
                self.calculate_fps()
                
                # 更新状态显示
                self.update_status_display()
                
                # 控制循环频率
                time.sleep(1.0 / self.args.fps)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, exiting...")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                break
    
    def shutdown(self):
        """安全关闭"""
        logger.info("Shutting down system...")
        
        self.running = False
        
        # 停止录制
        if self.recording:
            self.stop_recording()
        
        # 停止各个组件
        if self.status_display:
            self.status_display.stop()
        
        if self.gamepad_controller:
            self.gamepad_controller.stop_threaded()
        
        if self.piper_controller:
            self.piper_controller.running = False
            self.piper_controller.reset_to_initial_positions()
        
        if self.camera_manager:
            self.camera_manager.stop_all()
        
        if self.dataset_manager:
            self.dataset_manager.cleanup()
        
        logger.info("✅ System shut down")
    
    def run(self):
        """运行录制器"""
        logger.info("Start Piper robot data recording system")
        
        # 设置信号处理器
        self.setup_signal_handlers()
        
        # 初始化各个组件
        init_steps = [
            ("Genesis scene", self.init_genesis_scene),
            ("Camera system", self.init_cameras),
            ("Robot recorder", self.init_robot_recorder),
            ("Dataset manager", self.init_dataset_manager),
            ("Piper controller", self.init_piper_controller),
            ("Gamepad controller", self.init_gamepad_controller),
            ("Status display", self.init_status_display)
        ]
        
        for step_name, init_func in init_steps:
            logger.info(f"Initializing {step_name}...")
            if not init_func():
                logger.error(f"{step_name} initialization failed")
                # 对于某些非关键组件，可以继续运行
                if step_name in ["Camera system", "Gamepad controller"]:
                    logger.warning(f"Skipping {step_name}, continue running...")
                    continue
                else:
                    self.shutdown()
                    return False
        
        logger.info("🎉 All components initialized, start running...")
        
        try:
            # 运行主循环
            self.run_main_loop()
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.shutdown()
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Piper robot data recording system")
    
    # 基本参数
    parser.add_argument("--dataset_dir", type=str, default="piper_real_dataset",
                       help="Dataset save directory")
    parser.add_argument("--repo_id", type=str, default="piper/real-manipulation",
                       help="Dataset repository ID")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Recording frame rate")
    parser.add_argument("--task_description", type=str, default="Real robot manipulation with Piper arm",
                       help="Task description")
    
    # 控制参数
    parser.add_argument("--vis", action="store_true", default=True,
                       help="Whether to show simulation scene")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Whether to resume from existing dataset")
    
    args = parser.parse_args()
    
    # 创建录制器并运行
    recorder = PiperRealDataRecorder(args)
    
    try:
        success = recorder.run()
        if success:
            logger.info("🎉 Data recording completed")
        else:
            logger.error("❌ Data recording failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Program failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()