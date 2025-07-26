"""
RealSense YOLO 杯子检测示例

本示例演示了如何使用Intel RealSense深度相机和YOLOv8模型进行实时杯子检测和3D定位。
主要功能包括：
1. 初始化RealSense相机并获取彩色和深度图像
2. 使用YOLOv8模型检测图像中的杯子
3. 计算检测到的杯子在相机坐标系中的3D位置
4. 实时显示检测结果和3D坐标

使用方法：
1. 确保已安装必要的依赖（pyrealsense2, ultralytics, opencv-python, numpy）
2. 连接Intel RealSense相机
3. 运行脚本：python realsense_yolo_cup_detection.py
4. 按'q'键退出程序

注意：
- 本示例使用COCO数据集预训练的YOLOv8模型，其中杯子的类别ID为41
- 3D位置计算基于深度相机的深度信息和相机内参
- 确保相机已正确安装并授予相应权限
"""


import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

class RealSenseYOLOCupDetector:
    def __init__(self, model_name='yolov8m', conf_threshold=0.5):
        """
        初始化RealSense相机和YOLO模型。
        
        参数:
            model_name (str): YOLO模型名称 (默认: 'yolov8m')
            conf_threshold (float): 检测的置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.cup_class_id = 41  # COCO数据集中'cup'类别的ID
        
        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 开始视频流
        self.profile = self.pipeline.start(config)
        
        # 获取深度传感器的深度比例
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # 创建一个对齐对象，用于将深度帧与彩色帧对齐
        self.align = rs.align(rs.stream.color)
        
        # 加载YOLO模型
        self.model = YOLO(f'{model_name}.pt')  # Loads official YOLOv8 model
        self.conf_threshold = conf_threshold
        
        # 创建颜色映射对象，用于可视化深度数据
        self.colorizer = rs.colorizer()
        # self.colorizer.set_option(rs.option.visual_preset, 0) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
        # self.colorizer.set_option(rs.option.min_distance, 0) # 单位：米
        # self.colorizer.set_option(rs.option.max_distance, 16) # 单位：米
    
    def get_frames(self):
        """
        从RealSense获取对齐的彩色和深度帧。
        """
        # 等待获取一对连贯的帧
        frames = self.pipeline.wait_for_frames()
        
        # 将深度帧与彩色帧对齐
        aligned_frames = self.align.process(frames)
        
        # 获取对齐后的帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
        
        # 将图像转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 为深度帧上色以便可视化
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        
        return color_image, depth_image, depth_colormap
    
    def detect_cups(self, color_image):
        """
        使用YOLO在彩色图像中检测杯子。
        
        参数:
            color_image: 输入的彩色图像(BGR格式)
            
        返回:
            list: 检测到的杯子的边界框列表 [x1, y1, x2, y2, 置信度, 类别ID]
        """
        # 运行YOLO推理(Ultralytics YOLO默认期望BGR格式)
        results = self.model(color_image, conf=self.conf_threshold)
        
        # 处理检测结果
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # 合并边界框、置信度和类别ID
            for box, conf, class_id in zip(boxes, confs, class_ids):
                if class_id == self.cup_class_id:
                    detections.append([*box, conf, class_id])
        
        return detections
    
    def get_3d_position(self, bbox, depth_image):
        """
        计算边界框中心点在相机坐标系中的3D位置。
        
        参数:
            bbox: 边界框 [x1, y1, x2, y2, 置信度, 类别ID]
            depth_image: 来自RealSense的深度图像
            
        返回:
            tuple: 相机坐标系中的(x, y, z)坐标，单位为米

        注意：在 RealSense 相机的坐标系中：
            原点（Origin）：位于相机的光心（optical center）
            Z 轴：从相机指向正前方（与光轴重合）
            X 轴：向右（从相机视角看）
            Y 轴：向下（从相机视角看）
        """
        # 1. 获取图像尺寸
        h, w = depth_image.shape[:2]
        
        # 2. 确保边界框在图像范围内
        x1 = max(0, min(int(bbox[0]), w-1))
        y1 = max(0, min(int(bbox[1]), h-1))
        x2 = max(0, min(int(bbox[2]), w-1))
        y2 = max(0, min(int(bbox[3]), h-1))
        
        # 3. 计算中心点（使用浮点运算）
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # 4. 获取中心点周围区域的中值深度（更鲁棒）
        half_size = 5  # 可以调整
        x_start = max(0, int(center_x) - half_size)
        y_start = max(0, int(center_y) - half_size)
        x_end = min(w, int(center_x) + half_size + 1)
        y_end = min(h, int(center_y) + half_size + 1)
        
        roi = depth_image[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            return None
        
        # 使用非零深度值的中位数
        valid_depths = roi[roi > 0]
        if valid_depths.size == 0:
            return None
        
        depth = np.median(valid_depths) * self.depth_scale  # 转换为米
        
        # 5. 获取相机内参
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        
        # 6. 将像素坐标转换为相机坐标
        x = (center_x - intrinsics.ppx) / intrinsics.fx * depth
        y = (center_y - intrinsics.ppy) / intrinsics.fy * depth
        z = depth
        
        return (x, y, z)
    
    def visualize(self, color_image, depth_colormap, detections, positions):
        """
        可视化检测结果。
        
        参数:
            color_image: 彩色图像(BGR格式)
            depth_colormap: 上色后的深度图像
            detections: 检测结果列表
            positions: 与检测结果对应的3D位置列表
        """
        if detections and positions:
            # 在彩色图像上绘制检测结果
            for i, (det, pos) in enumerate(zip(detections, positions)):
                if det is None or pos is None:
                    continue
                x1, y1, x2, y2, conf, _ = map(int, det[:6])
                
                # 绘制边界框
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # 显示3D位置
                pos_text = f'({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m'
                cv2.putText(color_image, pos_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 水平堆叠图像
        images = np.hstack((color_image, depth_colormap))
        
        # 显示结果
        cv2.imshow('RealSense YOLO Cup Detection', images)
    
    def run(self):
        """
        检测和可视化的主循环。
        """
        try:
            while True:
                # 获取帧
                color_image, depth_image, depth_colormap = self.get_frames()
                if color_image is None or depth_image is None:
                    continue
                
                # 检测杯子
                detections = self.detect_cups(color_image)
                
                # 计算3D位置
                positions = []
                for det in detections:
                    pos = self.get_3d_position(det, depth_image)
                    positions.append(pos)
                    print(f"Cup detected at 3D position (x, y, z): {pos} meters")
                
                # 可视化结果
                self.visualize(color_image.copy(), depth_colormap, detections, positions)
                
                # 按'q'键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # 停止视频流
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # 初始化检测器
    detector = RealSenseYOLOCupDetector(model_name='yolov8m', conf_threshold=0.5)
    
    # 运行检测
    detector.run()
