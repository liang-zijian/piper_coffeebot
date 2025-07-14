#!/usr/bin/env python3
"""
多相机RealSense数据获取模块
支持同时连接三个RealSense D435i相机（腕部相机、侧视相机1、侧视相机2）
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
from collections import deque
from typing import Dict, Tuple, Optional, List
# 导入全局日志管理器
from global_logger import log_message, log_info, log_warning, log_error, log_success

class MultiRealSenseManager:
    """多RealSense相机管理器"""
    
    def __init__(self, camera_configs: Dict[str, Dict] = None, queue_size: int = 10):
        """
        初始化多相机管理器
        
        Args:
            camera_configs: 相机配置字典，格式如下：
            {
                "ee_cam": {"serial": "123456789", "width": 640, "height": 480, "fps": 30},
                "side_cam_0": {"serial": "987654321", "width": 640, "height": 480, "fps": 30},
                "side_cam_1": {"serial": "555444333", "width": 640, "height": 480, "fps": 30}
            }
            queue_size: 每个相机的帧队列大小
        """
        self.cameras = {}
        self.pipelines = {}
        self.configs = {}
        self.align_objects = {}
        
        # 线程和队列管理
        self.frame_queues = {}  # 每个相机的帧队列
        self.camera_threads = {}  # 每个相机的线程
        self.thread_locks = {}  # 每个相机的锁
        self.running = False  # 线程运行标志
        self.queue_size = queue_size
        
        # 默认配置
        if camera_configs is None:
            camera_configs = {
                "ee_cam": {"width": 640, "height": 480, "fps": 30},
                "rgb_rs_0": {"width": 640, "height": 480, "fps": 30},
                "rgb_rs_1": {"width": 640, "height": 480, "fps": 30}
            }
        
        self.camera_configs = camera_configs
        self._init_cameras()
        self._start_frame_threads()
    
    def _init_cameras(self):
        """初始化所有相机"""
        # 获取连接的设备
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError("未检测到RealSense设备")
        
        log_message(f"检测到 {len(devices)} 个RealSense设备", "info", "Camera")
        
        # 打印设备信息
        available_serials = []
        for i, device in enumerate(devices):
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            available_serials.append(serial)
            log_message(f"设备 {i}: {name} (序列号: {serial})", "info", "Camera")
        
        # 如果配置中没有指定序列号，自动分配
        camera_names = list(self.camera_configs.keys())
        for i, (camera_name, config) in enumerate(self.camera_configs.items()):
            if "serial" not in config and i < len(available_serials):
                config["serial"] = available_serials[i]
                log_message(f"自动分配相机 {camera_name} 到设备 {available_serials[i]}", "info", "Camera")
        
        # 初始化每个相机
        for camera_name, config in self.camera_configs.items():
            if "serial" not in config:
                log_message(f"跳过相机 {camera_name}：没有可用的设备", "warning", "Camera")
                continue
                
            try:
                self._init_single_camera(camera_name, config)
                log_message(f"✅ 相机 {camera_name} 初始化成功", "success", "Camera")
            except Exception as e:
                log_message(f"❌ 相机 {camera_name} 初始化失败: {e}", "error", "Camera")
    
    def _init_single_camera(self, camera_name: str, config: Dict):
        """初始化单个相机"""
        # 创建管道和配置
        pipeline = rs.pipeline()
        rs_config = rs.config()
        
        # 指定设备序列号
        rs_config.enable_device(config["serial"])
        
        # 配置流
        width = config.get("width", 640)
        height = config.get("height", 480)
        fps = config.get("fps", 30)
        
        rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # 启动管道
        profile = pipeline.start(rs_config)
        
        # 相机初始化后等待一段时间，避免多相机冲突
        time.sleep(0.2)
        
        # 获取设备信息
        device = profile.get_device()
        
        # 初始化帧队列和锁
        self.frame_queues[camera_name] = deque(maxlen=self.queue_size)
        self.thread_locks[camera_name] = threading.Lock()
        
        # 保存到字典
        self.pipelines[camera_name] = pipeline
        self.configs[camera_name] = rs_config
        self.cameras[camera_name] = {
            "profile": profile,
            "device": device,
            "config": config
        }
    
    def _start_frame_threads(self):
        """启动所有相机的帧获取线程"""
        self.running = True
        for camera_name in self.pipelines.keys():
            thread = threading.Thread(
                target=self._camera_frame_thread, 
                args=(camera_name,), 
                daemon=True,
                name=f"Camera-{camera_name}"
            )
            self.camera_threads[camera_name] = thread
            thread.start()
            log_message(f"启动相机 {camera_name} 帧获取线程", "info", "Camera")
    
    def _camera_frame_thread(self, camera_name: str):
        """单个相机的帧获取线程"""
        pipeline = self.pipelines[camera_name]
        
        log_message(f"相机 {camera_name} 线程开始运行", "info", "Camera")
        
        while self.running:
            try:
                # 使用poll_for_frames非阻塞获取帧
                frames = pipeline.poll_for_frames()
                
                if frames:
                    # 获取彩色帧
                    color_frame = frames.get_color_frame()
                    
                    if color_frame:
                        # 转换为numpy数组
                        color_image = np.asanyarray(color_frame.get_data())
                        
                        # 线程安全地添加到队列
                        with self.thread_locks[camera_name]:
                            self.frame_queues[camera_name].append(color_image)
                
                # 短暂休眠避免过度占用CPU
                time.sleep(0.001)
                
            except Exception as e:
                if self.running:  # 只在运行时记录错误
                    log_message(f"相机 {camera_name} 线程错误: {e}", "warning", "Camera")
                time.sleep(0.01)
        
        log_message(f"相机 {camera_name} 线程结束", "info", "Camera")
    
    def get_frame(self, camera_name: str) -> Optional[np.ndarray]:
        """
        从队列获取指定相机的最新彩色帧
        
        Args:
            camera_name: 相机名称
            
        Returns:
            color_image 或 None
        """
        if camera_name not in self.frame_queues:
            return None
        
        try:
            # 线程安全地从队列获取最新帧
            with self.thread_locks[camera_name]:
                if len(self.frame_queues[camera_name]) > 0:
                    # 获取最新的帧（队列右端）
                    return self.frame_queues[camera_name][-1]
                else:
                    return None
            
        except Exception as e:
            return None
    
    def get_all_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """
        获取所有相机的帧数据
        
        Returns:
            字典，key为相机名称，value为color_image
        """
        results = {}
        for camera_name in self.pipelines.keys():
            results[camera_name] = self.get_frame(camera_name)
        return results
    
    def get_color_frames_for_lerobot(self) -> Dict[str, Optional[np.ndarray]]:
        """
        获取所有相机的彩色帧，格式适配lerobot数据集
        录制时只存储原始BGR图像，避免耗时的颜色转换
        
        Returns:
            字典，key为lerobot格式的名称，value为原始BGR图像（HWC格式）
        """
        results = {}
        
        # 映射相机名称到lerobot格式
        name_mapping = {
            "ee_cam": "observation.images.ee_cam",
            "rgb_rs_0": "observation.images.rgb_rs_0", 
            "rgb_rs_1": "observation.images.rgb_rs_1"
        }
        
        for camera_name in self.pipelines.keys():
            color_image = self.get_frame(camera_name)
            if color_image is not None:
                # 录制时只存储原始BGR图像，避免耗时的转换
                # 保存时会批量处理BGR→RGB→CHW转换
                lerobot_key = name_mapping.get(camera_name, f"observation.images.{camera_name}")
                results[lerobot_key] = color_image  # 直接存储BGR HWC格式
            else:
                lerobot_key = name_mapping.get(camera_name, f"observation.images.{camera_name}")
                results[lerobot_key] = None
        
        return results
    
    @staticmethod
    def batch_convert_bgr_to_rgb_chw(bgr_images: Dict[str, list]) -> Dict[str, list]:
        """
        批量处理BGR→RGB→CHW转换
        
        Args:
            bgr_images: 字典，key为相机名称，value为BGR图像列表(HWC格式)
            
        Returns:
            处理后的图像字典，RGB格式(CHW)
        """
        results = {}
        
        for camera_key, image_list in bgr_images.items():
            if not image_list:
                results[camera_key] = []
                continue
                
            converted_images = []
            for bgr_image in image_list:
                if bgr_image is not None:
                    # BGR→RGB转换
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    # HWC→CHW转换
                    rgb_image_chw = np.transpose(rgb_image, (2, 0, 1))
                    converted_images.append(rgb_image_chw)
                else:
                    converted_images.append(None)
            
            results[camera_key] = converted_images
        
        return results
    
    def get_intrinsics(self, camera_name: str) -> Optional[Dict]:
        """获取指定相机的内参"""
        if camera_name not in self.cameras:
            return None
        
        try:
            profile = self.cameras[camera_name]["profile"]
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            return {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.ppx,
                'cy': intrinsics.ppy,
                'model': intrinsics.model,
                'coeffs': intrinsics.coeffs
            }
        except Exception as e:
            log_message(f"获取相机 {camera_name} 内参失败: {e}", "warning", "Camera")
        
        return None
    
    def save_frames(self, prefix: str = "capture"):
        """保存所有相机的当前帧"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        for camera_name in self.pipelines.keys():
            color_image = self.get_frame(camera_name)
            if color_image is not None:
                cv2.imwrite(f"{prefix}_{camera_name}_color_{timestamp}.png", color_image)
        
        log_message(f"💾 所有相机帧已保存: {prefix}_*_{timestamp}", "success", "Camera")
    
    def get_camera_count(self) -> int:
        """获取成功初始化的相机数量"""
        return len(self.pipelines)
    
    def get_camera_names(self) -> List[str]:
        """获取所有相机名称"""
        return list(self.pipelines.keys())
    
    def get_queue_status(self) -> Dict[str, int]:
        """获取各相机队列中的帧数"""
        status = {}
        for camera_name in self.frame_queues.keys():
            with self.thread_locks[camera_name]:
                status[camera_name] = len(self.frame_queues[camera_name])
        return status
    
    def stop_all(self):
        """停止所有相机和线程"""
        # 首先停止线程
        self.running = False
        log_message("正在停止所有相机线程...", "info", "Camera")
        
        # 等待所有线程结束
        for camera_name, thread in self.camera_threads.items():
            if thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    log_message(f"相机 {camera_name} 线程未能正常结束", "warning", "Camera")
                else:
                    log_message(f"相机 {camera_name} 线程已结束", "info", "Camera")
        
        # 停止所有管道
        for camera_name, pipeline in self.pipelines.items():
            try:
                pipeline.stop()
                log_message(f"相机 {camera_name} 管道已停止", "info", "Camera")
            except Exception as e:
                log_message(f"停止相机 {camera_name} 管道失败: {e}", "warning", "Camera")
        
        # 清理所有资源
        self.pipelines.clear()
        self.configs.clear()
        self.cameras.clear()
        self.frame_queues.clear()
        self.camera_threads.clear()
        self.thread_locks.clear()


def test_multi_cameras():
    """测试多相机功能"""
    try:
        # 创建多相机管理器
        manager = MultiRealSenseManager()
        
        if manager.get_camera_count() == 0:
            log_message("没有成功初始化任何相机", "error", "Camera")
            return
        
        log_message(f"成功初始化 {manager.get_camera_count()} 个相机", "success", "Camera")
        log_message(f"相机列表: {manager.get_camera_names()}", "info", "Camera")
        
        # 创建显示窗口
        camera_names = manager.get_camera_names()
        for camera_name in camera_names:
            cv2.namedWindow(f'{camera_name}', cv2.WINDOW_AUTOSIZE)
        
        fps_count = 0
        start_time = time.time()
        
        while True:
            # 获取所有相机的帧
            all_frames = manager.get_all_frames()
            
            # 统计有效帧数
            valid_frames = 0
            
            # 显示每个相机的图像
            for camera_name, color_image in all_frames.items():
                if color_image is not None:
                    valid_frames += 1
                    # 显示彩色图像
                    cv2.imshow(f'{camera_name}', color_image)
                else:
                    # 如果没有获取到帧，显示黑屏
                    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_img, f'{camera_name} - No Frame', (50, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(f'{camera_name}', blank_img)
            
            # 计算FPS
            fps_count += 1
            if fps_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                queue_status = manager.get_queue_status()
                queue_info = ", ".join([f"{name}:{count}" for name, count in queue_status.items()])
                log_message(f"FPS: {fps:.2f}, 有效帧: {valid_frames}/{len(camera_names)}, 队列: [{queue_info}]", "info", "Camera")
                start_time = time.time()
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q或ESC退出
                break
            elif key == ord('s'):  # s保存图像
                manager.save_frames()
        
    except Exception as e:
        log_message(f"测试失败: {e}", "error", "Camera")
    finally:
        # 清理资源
        if 'manager' in locals():
            manager.stop_all()
        cv2.destroyAllWindows()
        log_message("测试结束", "info", "Camera")


if __name__ == "__main__":
    test_multi_cameras()