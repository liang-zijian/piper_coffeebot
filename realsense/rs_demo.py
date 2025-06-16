#!/usr/bin/env python3
"""
RealSense D435i 图像获取示例
不依赖ROS,使用纯SDK获取RGB和深度图像
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
from rich.console import Console
from rich.logging import RichHandler
import logging
from rich.live import Live
from rich.panel import Panel

# 配置rich日志
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("RealSense")

class RealSenseCamera:
    def __init__(self):
        # 创建RealSense管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置深度和彩色流
        # D435i支持的分辨率：1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 启动管道
        self.profile = self.pipeline.start(self.config)
        
        # 获取深度传感器并设置参数
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        # 创建对齐对象（将深度图像对齐到彩色图像）
        self.align = rs.align(rs.stream.color)
        
        logger.info(f"✅ 相机已初始化，深度比例: [bold cyan]{self.depth_scale}[/bold cyan]")
        
    def get_frames(self):
        """获取对齐后的彩色和深度帧"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        # 获取对齐后的深度帧和彩色帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            return None, None
            
        # 转换为numpy数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image
    
    def get_intrinsics(self):
        """获取相机内参"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if color_frame:
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
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
        return None
    
    def pixel_to_point(self, x, y, depth_image):
        """将像素坐标转换为3D点"""
        depth = depth_image[y, x] * self.depth_scale  # 转换为米
        
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # 使用内参将像素坐标转换为3D坐标
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        return point
    
    def save_images(self, color_image, depth_image, prefix="capture"):
        """保存彩色和深度图像"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{prefix}_color_{timestamp}.png", color_image)
        cv2.imwrite(f"{prefix}_depth_{timestamp}.png", depth_image)
        
        # 保存深度数据为numpy文件（保留原始深度值）
        np.save(f"{prefix}_depth_{timestamp}.npy", depth_image)
        logger.info(f"💾 image saved: [bold green]{prefix}_*_{timestamp}[/bold green]")
    
    def stop(self):
        """停止相机"""
        self.pipeline.stop()

def main():
    # 清屏并显示启动信息
    console.clear()
    
    # 创建相机对象
    camera = RealSenseCamera()
    
    # 创建窗口
    cv2.namedWindow('RealSense D435i - Color', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('RealSense D435i - Depth', cv2.WINDOW_AUTOSIZE)
    
    # 用于计算FPS
    fps_count = 0
    fps = 0.0  # 初始化fps变量
    start_time = time.time()
    
    # 用于保存所有状态信息
    status_msgs = {
        "fps": "[cyan]FPS: --[/cyan]",
        "hint": "[green]按键: s-保存, q/ESC-退出, 点击深度图查坐标[/green]",
        "last_point": "[magenta]点击像素: --[/magenta]"
    }

    # 鼠标回调函数
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_image = param['depth_image']
            if depth_image is not None:
                point_3d = camera.pixel_to_point(x, y, depth_image)
                depth_mm = depth_image[y, x]
                status_msgs["last_point"] = f"[magenta]点击像素: ({x}, {y})  深度: {depth_mm}mm ({depth_mm/1000:.3f}m)  3D: X={point_3d[0]:.3f} Y={point_3d[1]:.3f} Z={point_3d[2]:.3f}[/magenta]"

    # 设置鼠标回调
    mouse_params = {'depth_image': None}
    cv2.setMouseCallback('RealSense D435i - Depth', mouse_callback, mouse_params)

    def get_panel():
        # 拼接所有状态信息
        content = f"{status_msgs['fps']}\n{status_msgs['hint']}\n{status_msgs['last_point']}"
        return Panel(content, title="[bold cyan]RealSense 状态面板", border_style="cyan")

    try:
        with Live(get_panel(), console=console, refresh_per_second=10) as live:
            while True:
                # 获取图像
                color_image, depth_image = camera.get_frames()
                
                if color_image is None or depth_image is None:
                    continue
                
                # 更新鼠标回调参数
                mouse_params['depth_image'] = depth_image
                
                # 将深度图像转换为彩色图像用于显示
                # 获取有效深度值范围进行自适应映射
                depth_valid = depth_image[depth_image > 0]  # 只考虑有效深度值
                if len(depth_valid) > 0:
                    depth_min = np.percentile(depth_valid, 5)   # 使用5%分位数作为最小值
                    depth_max = np.percentile(depth_valid, 95)  # 使用95%分位数作为最大值
                    # 将深度值映射到0-255范围，然后反转（近处=红色，远处=蓝色）
                    depth_normalized = np.clip((depth_image - depth_min) / (depth_max - depth_min) * 255, 0, 255)
                    depth_inverted = 255 - depth_normalized  # 反转：近处变红色，远处变蓝色
                    depth_colormap = cv2.applyColorMap(depth_inverted.astype(np.uint8), cv2.COLORMAP_JET)
                else:
                    # 如果没有有效深度值，使用原来的方法
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                
                # 计算并显示FPS
                fps_count += 1
                if fps_count % 30 == 0:
                    end_time = time.time()
                    fps = 30 / (end_time - start_time)
                    status_msgs["fps"] = f"[cyan]FPS: {fps:.2f}[/cyan]"
                    start_time = time.time()
                
                # 在图像上添加FPS信息
                cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow('RealSense D435i - Color', color_image)
                cv2.imshow('RealSense D435i - Depth', depth_colormap)
                
                # 刷新Live面板
                live.update(get_panel())
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q或ESC退出
                    break
                elif key == ord('s'):  # s保存图像
                    camera.save_images(color_image, depth_image)
                    status_msgs["hint"] = "[green]已保存当前帧！s-保存, q/ESC-退出, 点击深度图查坐标[/green]"
                else:
                    status_msgs["hint"] = "[green]按键: s-保存, q/ESC-退出, 点击深度图查坐标[/green]"
    except Exception as e:
        logger.error(f"error: [bold red]{e}[/bold red]")
        
    finally:
        # 清理资源
        camera.stop()
        cv2.destroyAllWindows()
        logger.info("👋 [bold green]quit[/bold green]")

if __name__ == "__main__":
    main()