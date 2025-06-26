#!/usr/bin/env python3
"""
Upload selected parquet shards to ModelScope Hub.

Prerequisites
-------------
pip install -U modelscope   # SDK ≥ 1.22
export MODELSCOPE_TOKEN=*** # 具有写权限的访问令牌
#   令牌获取路径：ModelScope官网 -> 个人中心 -> 令牌管理
"""

import os
from pathlib import Path
from modelscope.hub.api import HubApi

# ---------- 可按需修改的参数 ----------
REPO_ID = "Bits9600/piper_coffee_data"         # 目标仓库
LOCAL_DIR = Path("/home/ubuntu/workspace/piper_ws/piper_real_dataset/data/chunk-000")
DEST_DIR_IN_REPO = "data/chunk-000"            # 仓库内子目录
START_IDX, END_IDX = 20, 91                    # episode_000020 ～ episode_000091
REPO_TYPE = "dataset"                          # 如果是模型仓可改为 "model"
# --------------------------------------

def main() -> None:
    token = "fd72c640-4d5f-4922-a67d-cfd3df7490dd"
    api = HubApi()
    api.login(token)

    for idx in range(START_IDX, END_IDX + 1):
        fname = f"episode_{idx:06d}.parquet"
        local_path = LOCAL_DIR / fname
        if not local_path.exists():
            print(f"[WARN] 本地不存在 {local_path}，跳过")
            continue

        dest_path = f"{DEST_DIR_IN_REPO}/{fname}"
        print(f"→ 正在上传 {fname} 到 {REPO_ID}:{dest_path}")

        # 单文件上传；大文件会自动分块断点续传
        api.upload_file(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_or_fileobj=str(local_path),
            path_in_repo=dest_path,
            commit_message=f"Add {dest_path}"
        )

    print("全部文件处理完毕 ✅")

if __name__ == "__main__":
    #main()
    token = "fd72c640-4d5f-4922-a67d-cfd3df7490dd"
    api = HubApi()
    api.login(token)
    api.upload_folder(
        repo_id="Bits9600/piper_coffee_data",
        repo_type="dataset",
        folder_path="/home/ubuntu/workspace/piper_ws/piper_real_dataset/meta",
        path_in_repo="meta",
        commit_message="Add meta"
    )
