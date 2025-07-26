from ultralytics import YOLO
import numpy as np
import time


if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    start_time = time.perf_counter()
    results = model(img)
    end_time = time.perf_counter()
    print(f"YOLO model test successful, cost time is {end_time - start_time} s")
    
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    start_time = time.perf_counter()
    results = model(img)
    end_time = time.perf_counter()
    print(f"YOLO model test successful2, cost time is {end_time - start_time} s")


