- datarecord/recorder_example.py是在isaaclab仿真环境中录制lerobot格式的数据集的脚本
- realsense/rs_demo.py 是realsense d435i 相机获取图像的demo
- piper_teleop/piper_grasp_real.py 是手柄控制piper机械臂的demo
    其中会调用piper_teleop/piper_xiaoji_real.py 手柄控制piper机械臂末端执行器，然后解算piper机械臂的关节位置（除了joint5和joint6单独用两个键位直接控制电机位置）

    piper_teleop/piper_xiaoji_real.py还包含了获取当前机械臂各关节位置的函数使用示例：

```python
    joints_position = self.piper.get_dofs_position().cpu().numpy()
```

你要实现的功能是：
我会插入三个realsense d435i 相机，一个在腕部，两个是侧视。你需要实现：
- 手柄控制机械臂移动夹取物体
- 实时获取三个摄像头数据，和所有joint的position，以及当前的控制actions（暂时保存位置增量）保存在lerobot数据集中，参考datarecord/recorder_example.py（！注意这个脚本是从isaaclab仿真环境中拿出来的，里面很多东西不适宜用在真实环境，请仔细甄别）
- 支持增量录制，即从当前数据集继续录制
- 结束录制按钮是手柄上的X按键，对应self.joystick.get_button(1)
- terminal界面的log使用rich库的Live面板显示

lerobot数据集每个frame内容：
```python
        frame = {
            # === 摄像头画面 ===
            "observation.images.ee_cam":      
            "observation.images.rgb_rs_0": 
            "observation.images.rgb_rs_1": 
            # === 关节状态 ===
            "observation.state": state_vec,  
            # === 动作 ===
            "actions": 
            # === 任务标识 ===
            "task": "move the coffee cup to the coffee machine"
        }
```

**REMINDER**
每个函数尽可能实现专一的功能，不要把多个功能糅合在一起，注意程序解耦和模块化








