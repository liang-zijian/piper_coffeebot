# -*- coding: utf-8 -*-
import dataclasses, einops, numpy as np
from openpi import transforms
from openpi.models import model as _model

def _parse(image):
    """
    将图像转换为 uint8 格式，并调整形状为 (H,W,C)
    """
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating): # 如果图像是浮点数, 转换为 uint8
        img = (255 * img).astype(np.uint8)
    if img.shape[0] == 3:                 # C,H,W → H,W,C
        img = einops.rearrange(img, "c h w -> h w c")
    return img

@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    action_dim: int 
    model_type: _model.ModelType = _model.ModelType.PI0_FAST

    def __call__(self, data: dict) -> dict:
        # 1) 关节+夹爪拼 state，并 pad 到 action_dim
        STATE_DIM = 32
        state = transforms.pad_to_dim(data["state"], STATE_DIM)
        # 2) 三路相机
        in_images = data["images"]
        base = _parse(in_images["rgb_rs_0"])
        base2 = _parse(in_images["rgb_rs_1"])
        wrist = _parse(in_images["ee_cam"])

        if self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            masks = (True, True, True)          # FAST 模型不需要 padding mask
        else:                                   # 纯 π₀
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            masks = (True, True, True)       
            #base2 = np.zeros_like(base)

        inputs = {
            "state": state,
            "image": dict(zip(names, (base, base2, wrist), strict=True)),
            "image_mask": dict(zip(names, masks, strict=True)),
        }
        if "actions" in data:  
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
            #inputs["actions"] = data["actions"]
        
        inputs["prompt"]  = "Move the green tape roll into the box, place it properly, and then return to the original position."

        return inputs

@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # π₀-FAST 输出 (B,8)；若用 π₀ 也只保留前 8 维
        return {"actions": np.asarray(data["actions"][:, :8])}
