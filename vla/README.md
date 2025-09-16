

# 硬件连接
1. 连接3个realsense相机到USB3.0接口，并确保相机连接正常(通过realsense-viewer)
2. 连接piper机械臂到工作站，并确保机械臂连接正常（按照[piper sdk](https://github.com/agilexrobotics/piper_sdk)安装和激活can总线）

# 依赖安装
1. 安装genesis-world仿真引擎
```bash
pip install genesis-world==2.0.2
```
2. 安装piper sdk: follow [piper sdk](https://github.com/agilexrobotics/piper_sdk)
3. 安装openpi: 
- 安装uv: follow [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
- 进入vla/openpi目录，执行以下命令：
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

# 运行命令
开启server：
```bash
uv run vla/openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi0_base_piper --policy.dir=YOUR_CHECKPOINT_PATH
```
新开一个终端，开启client：
```bash
python vla/piper_vla_ctrl.py --no-use-rtc
```
ps：RTC来源于论文[Real time chunking](https://www.physicalintelligence.company/research/real_time_chunking)，可以不使用RTC，直接使用原始的serial执行

# 模型checkpoint
piper机械臂：[模型checkpoint下载](https://modelscope.cn/models/Bits9600/coffeecup-pp-200)