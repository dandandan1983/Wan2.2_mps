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
    'get_autocast_device_type',
    'autocast_context',
    'autocast_decorator',
    'autocast',
    'get_device_type_from_tensor',
    'empty_cache',
    'synchronize',
    'get_optimal_dtype',
    'get_generator',
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


def autocast_context(dtype=None, enabled=True):
    """
    Create a device-agnostic autocast context manager.
    
    Args:
        dtype: Data type for autocast (e.g., torch.float16, torch.bfloat16)
        enabled: Whether autocast is enabled
        
    Returns:
        torch.amp.autocast context manager
    """
    device_type = get_device_type()
    if dtype is not None:
        return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    else:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)


def autocast_decorator(enabled=True):
    """
    Device-agnostic autocast decorator for functions.
    
    Args:
        enabled: Whether autocast is enabled
        
    Returns:
        Decorator function
    """
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device_type = get_device_type()
            with torch.amp.autocast(device_type=device_type, enabled=enabled):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_device_type_from_tensor(tensor):
    """
    Get the device type string from a tensor.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        str: Device type ('cuda', 'mps', or 'cpu')
    """
    return tensor.device.type


def autocast(dtype=torch.float32, device_type=None):
    """
    Device-agnostic autocast context manager.
    Uses the best available device if device_type is not specified.
    
    Args:
        dtype: Data type for autocast
        device_type: Optional device type string ('cuda', 'mps', 'cpu')
        
    Returns:
        torch.amp.autocast context manager
    """
    if device_type is None:
        device_type = get_device_type()
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


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


def get_optimal_dtype(device: torch.device = None) -> torch.dtype:
    """
    Get the optimal dtype for the given device.
    
    MPS doesn't fully support bfloat16, so use float16 or float32 instead.
    CUDA works best with bfloat16 on supported hardware.
    
    Args:
        device: Target device (if None, uses best available)
        
    Returns:
        torch.dtype: The optimal dtype for the device
    """
    if device is None:
        device_type = get_device_type()
    else:
        device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
    
    if device_type == 'mps':
        # MPS has limited bfloat16 support, use float16
        return torch.float16
    elif device_type == 'cuda':
        # Check if CUDA device supports bfloat16
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:  # Ampere and newer
                return torch.bfloat16
            else:
                return torch.float16
        return torch.bfloat16
    else:
        # CPU - bfloat16 is fine
        return torch.bfloat16


def get_generator(device: torch.device = None, seed: int = None) -> torch.Generator:
    """
    Create a random number generator for the specified device.
    
    Note: MPS generator support varies by PyTorch version.
    
    Args:
        device: Target device
        seed: Random seed (optional)
        
    Returns:
        torch.Generator: A generator for the device
    """
    if device is None:
        device = get_best_device()
    
    device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
    
    # Create generator - MPS may need CPU generator for some operations
    if device_type == 'mps':
        # MPS generator support was added in PyTorch 2.0
        # For older versions, fall back to CPU generator
        try:
            gen = torch.Generator(device=device)
        except RuntimeError:
            gen = torch.Generator(device='cpu')
    else:
        gen = torch.Generator(device=device)
    
    if seed is not None:
        gen.manual_seed(seed)
    
    return gen
