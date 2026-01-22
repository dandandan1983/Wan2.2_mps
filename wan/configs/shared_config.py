# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

# Import device utilities
from ..utils.device import is_mps_available, get_device_type

#------------------------ Wan shared config ------------------------#
wan_shared_cfg = EasyDict()

# Determine optimal dtype based on device
# MPS doesn't fully support bfloat16, use float16 or float32 instead
def get_optimal_dtype():
    """Get the optimal dtype for the current device."""
    if is_mps_available() and get_device_type() == 'mps':
        # MPS works better with float16 or float32
        return torch.float16
    return torch.bfloat16

# t5
wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = get_optimal_dtype()
wan_shared_cfg.text_len = 512

# transformer
wan_shared_cfg.param_dtype = get_optimal_dtype()

# inference
wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
wan_shared_cfg.frame_num = 81
