# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Device utilities for cross-platform GPU support (CUDA, MPS, CPU).
"""
import gc
import logging
import os
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
    'get_high_precision_dtype',
    'to_high_precision',
    'get_mps_memory_limit',
    'set_mps_memory_limit',
    'mps_memory_efficient_mode',
    'aggressive_memory_cleanup',
    'get_mps_recommended_settings',
    'log_mps_memory_info',
]

# MPS configuration - can be overridden via environment variables
MPS_MEMORY_LIMIT_GB = float(os.environ.get('WAN_MPS_MEMORY_LIMIT_GB', '8.0'))
MPS_MEMORY_EFFICIENT = os.environ.get('WAN_MPS_MEMORY_EFFICIENT', '1') == '1'
MPS_LOW_MEMORY_MODE = os.environ.get('WAN_MPS_LOW_MEMORY', '0') == '1'


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


def autocast(dtype=None, device_type=None):
    """
    Device-agnostic autocast context manager.
    Uses the best available device if device_type is not specified.
    
    Args:
        dtype: Data type for autocast. If None, uses the optimal dtype for the device.
        device_type: Optional device type string ('cuda', 'mps', 'cpu')
        
    Returns:
        torch.amp.autocast context manager
    """
    if device_type is None:
        device_type = get_device_type()
    
    # If no dtype specified, use the optimal dtype for the device to ensure consistency
    if dtype is None:
        dtype = get_optimal_dtype()
    
    # For MPS, ensure we use float32 for stability in matrix operations
    if device_type == 'mps' and dtype == torch.bfloat16:
        dtype = torch.float32
    
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
    
    MPS requires float32 for stable matrix operations - float16/bfloat16
    can cause dtype mismatches in matmul accumulator.
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
        # MPS requires float32 for stable matrix operations
        # float16 can cause dtype mismatches in matmul accumulator
        return torch.float32
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


def get_high_precision_dtype(device: torch.device = None) -> torch.dtype:
    """
    Get the highest precision dtype supported by the device.
    
    MPS doesn't support float64, so we use float32 instead.
    CUDA and CPU support float64.
    
    Args:
        device: Target device (if None, uses best available)
        
    Returns:
        torch.dtype: float64 for CUDA/CPU, float32 for MPS
    """
    if device is None:
        device_type = get_device_type()
    else:
        device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]
    
    if device_type == 'mps':
        return torch.float32
    else:
        return torch.float64


def to_high_precision(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to highest precision supported by its device.
    
    MPS tensors are converted to float32, others to float64.
    
    Args:
        tensor: Input tensor
        
    Returns:
        torch.Tensor: Tensor in highest supported precision
    """
    if tensor.device.type == 'mps':
        return tensor.to(torch.float32)
    else:
        return tensor.to(torch.float64)


def get_mps_memory_limit() -> float:
    """
    Get the current MPS memory limit in GB.
    
    This is used by chunked attention to determine chunk sizes.
    Can be configured via WAN_MPS_MEMORY_LIMIT_GB environment variable.
    
    Returns:
        float: Memory limit in GB
    """
    return MPS_MEMORY_LIMIT_GB


def set_mps_memory_limit(limit_gb: float):
    """
    Set the MPS memory limit in GB.
    
    This controls the maximum buffer size for attention operations.
    Lower values will result in more chunking but lower memory usage.
    
    Args:
        limit_gb: Memory limit in GB (recommended: 2.0-8.0)
    """
    global MPS_MEMORY_LIMIT_GB
    MPS_MEMORY_LIMIT_GB = limit_gb


def mps_memory_efficient_mode(enabled: bool = True):
    """
    Enable or disable memory efficient mode for MPS.
    
    When enabled:
    - Uses chunked attention for large sequences
    - More aggressive cache clearing
    - May be slower but uses less memory
    
    Args:
        enabled: Whether to enable memory efficient mode
    """
    global MPS_MEMORY_EFFICIENT
    MPS_MEMORY_EFFICIENT = enabled
    
    if enabled:
        # Set a conservative memory limit
        set_mps_memory_limit(2.0)
    else:
        # Use a more generous limit
        set_mps_memory_limit(4.0)


def get_mps_fallback_device() -> torch.device:
    """
    Get the fallback device for operations not supported on MPS.
    
    Returns:
        torch.device: CPU device for fallback operations
    """
    return torch.device('cpu')


def is_mps_memory_efficient() -> bool:
    """
    Check if MPS memory efficient mode is enabled.
    
    Returns:
        bool: True if memory efficient mode is enabled
    """
    return MPS_MEMORY_EFFICIENT


def aggressive_memory_cleanup():
    """
    Perform aggressive memory cleanup for MPS devices.
    
    This is essential for running large models on limited GPU memory.
    Should be called between major operations (e.g., after each diffusion step).
    """
    # Force Python garbage collection
    gc.collect()
    
    # Device-specific cleanup
    if is_cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif is_mps_available():
        # Synchronize to ensure all operations are complete
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        # Clear MPS cache
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        # Additional garbage collection after MPS cleanup
        gc.collect()


def get_mps_recommended_settings(memory_gb: float = None) -> dict:
    """
    Get recommended settings for MPS based on available memory.
    
    Args:
        memory_gb: Available memory in GB. If None, attempts to detect.
        
    Returns:
        dict: Recommended settings for MPS generation
    """
    if memory_gb is None:
        # Default conservative estimate for Mac
        # Most Macs with MPS have 8-128GB unified memory
        memory_gb = 16.0  # Conservative default
    
    if memory_gb <= 16:
        return {
            'offload_model': True,
            't5_cpu': True,
            'frame_num': 33,  # Reduced frames
            'memory_limit_gb': 4.0,
            'chunk_size': 256,
            'low_memory': True,
        }
    elif memory_gb <= 32:
        return {
            'offload_model': True,
            't5_cpu': True,
            'frame_num': 49,
            'memory_limit_gb': 8.0,
            'chunk_size': 512,
            'low_memory': False,
        }
    elif memory_gb <= 64:
        return {
            'offload_model': True,
            't5_cpu': False,
            'frame_num': 81,
            'memory_limit_gb': 16.0,
            'chunk_size': 1024,
            'low_memory': False,
        }
    else:
        return {
            'offload_model': False,
            't5_cpu': False,
            'frame_num': 121,
            'memory_limit_gb': 32.0,
            'chunk_size': 2048,
            'low_memory': False,
        }


def log_mps_memory_info():
    """
    Log MPS memory information and recommendations.
    """
    if not is_mps_available():
        logging.info("MPS is not available on this system.")
        return
    
    logging.info("=" * 60)
    logging.info("Apple MPS (Metal Performance Shaders) Configuration")
    logging.info("=" * 60)
    logging.info(f"MPS Memory Limit: {MPS_MEMORY_LIMIT_GB:.1f} GB")
    logging.info(f"Memory Efficient Mode: {MPS_MEMORY_EFFICIENT}")
    logging.info(f"Low Memory Mode: {MPS_LOW_MEMORY_MODE}")
    logging.info("")
    logging.info("Environment Variables for Tuning:")
    logging.info("  WAN_MPS_MEMORY_LIMIT_GB=<float>  - Max buffer size (default: 8.0)")
    logging.info("  WAN_MPS_MEMORY_EFFICIENT=1       - Enable chunked attention")
    logging.info("  WAN_MPS_LOW_MEMORY=1             - Enable aggressive memory saving")
    logging.info("")
    logging.info("Tips for Mac users:")
    logging.info("  - Use --offload_model to move models to CPU between steps")
    logging.info("  - Use --t5_cpu to keep T5 on CPU")
    logging.info("  - Reduce --frame_num for less memory usage")
    logging.info("  - Close other apps to free unified memory")
    logging.info("=" * 60)


def is_mps_low_memory_mode() -> bool:
    """
    Check if MPS low memory mode is enabled.
    
    Returns:
        bool: True if low memory mode is enabled
    """
    return MPS_LOW_MEMORY_MODE


def set_mps_low_memory_mode(enabled: bool = True):
    """
    Enable or disable MPS low memory mode.
    
    When enabled:
    - More aggressive offloading
    - Smaller chunk sizes
    - More frequent cache clearing
    
    Args:
        enabled: Whether to enable low memory mode
    """
    global MPS_LOW_MEMORY_MODE, MPS_MEMORY_LIMIT_GB
    MPS_LOW_MEMORY_MODE = enabled
    
    if enabled:
        MPS_MEMORY_LIMIT_GB = 2.0
        mps_memory_efficient_mode(True)
    else:
        MPS_MEMORY_LIMIT_GB = 8.0
