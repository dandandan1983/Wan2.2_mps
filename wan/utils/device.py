# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Device utilities for cross-platform GPU support (CUDA, MPS, CPU).
"""
import logging
import torch

__all__ = [
    'get_device',
    'get_device_type',
    'is_mps_available',
    'is_cuda_available',
    'get_best_device',
    'to_device',
]


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) is available on macOS."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def get_device_type() -> str:
    """
    Get the best available device type string.
    
    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    if is_cuda_available():
        return 'cuda'
    elif is_mps_available():
        return 'mps'
    else:
        return 'cpu'


def get_best_device(device_id: int = 0) -> torch.device:
    """
    Get the best available device.
    
    Args:
        device_id: CUDA device ID (ignored for MPS/CPU)
        
    Returns:
        torch.device: The best available device
    """
    if is_cuda_available():
        return torch.device(f'cuda:{device_id}')
    elif is_mps_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_device(device_id: int = 0, force_cpu: bool = False) -> torch.device:
    """
    Get device based on availability and configuration.
    
    Args:
        device_id: CUDA device ID (ignored for MPS)
        force_cpu: If True, always return CPU device
        
    Returns:
        torch.device: The selected device
    """
    if force_cpu:
        return torch.device('cpu')
    return get_best_device(device_id)


def to_device(tensor_or_model, device: torch.device = None, device_id: int = 0):
    """
    Move a tensor or model to the appropriate device.
    
    Args:
        tensor_or_model: PyTorch tensor or model to move
        device: Target device (if None, uses best available)
        device_id: CUDA device ID if device is None
        
    Returns:
        The tensor/model on the target device
    """
    if device is None:
        device = get_best_device(device_id)
    return tensor_or_model.to(device)


def get_autocast_device_type() -> str:
    """
    Get the device type string for torch.autocast.
    
    Returns:
        str: Device type for autocast ('cuda', 'mps', or 'cpu')
    """
    return get_device_type()


def empty_cache():
    """Clear GPU memory cache for the available device."""
    if is_cuda_available():
        torch.cuda.empty_cache()
    elif is_mps_available():
        # MPS doesn't have explicit cache clearing, but we can synchronize
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


def synchronize():
    """Synchronize the current device."""
    if is_cuda_available():
        torch.cuda.synchronize()
    elif is_mps_available():
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()


def log_device_info():
    """Log information about available devices."""
    device_type = get_device_type()
    if device_type == 'cuda':
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"Using CUDA device: {device_name}")
    elif device_type == 'mps':
        logging.info("Using Apple MPS (Metal Performance Shaders) device")
    else:
        logging.info("Using CPU device")
