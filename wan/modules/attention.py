# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import math

from ..utils.device import get_device_type, is_mps_available, is_cuda_available, empty_cache, get_mps_memory_limit

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'sdp_attention',
    'mps_chunked_attention',
]

# MPS memory management constants
# These control chunking to avoid "Invalid buffer size" errors on MPS
MPS_CHUNK_SIZE = 1024  # Default chunk size for sequence dimension


def get_mps_max_buffer_gb():
    """Get the maximum buffer size in GB for MPS operations."""
    return get_mps_memory_limit()


def estimate_attention_memory(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, dtype):
    """
    Estimate memory required for attention computation.
    Returns size in bytes.
    """
    # Attention scores: [B, N, Lq, Lk] 
    # Plus softmax output: [B, N, Lq, Lk]
    # Plus output: [B, N, Lq, C]
    bytes_per_elem = 4 if dtype in (torch.float32,) else 2
    
    # Main memory consumers: Q*K^T scores and softmax
    attention_scores_size = batch_size * num_heads * seq_len_q * seq_len_k * bytes_per_elem
    # We need this twice (scores + softmax)
    total_size = attention_scores_size * 2
    # Add output buffer
    total_size += batch_size * num_heads * seq_len_q * head_dim * bytes_per_elem
    
    return total_size


def get_optimal_chunk_size(batch_size, num_heads, seq_len_q, seq_len_k, head_dim, dtype, max_memory_gb=None):
    """
    Calculate optimal chunk size for MPS to avoid memory errors.
    """
    if max_memory_gb is None:
        max_memory_gb = get_mps_max_buffer_gb()
    
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
    
    # Try progressively smaller chunk sizes
    for chunk_size in [seq_len_q, 2048, 1024, 512, 256, 128, 64, 32]:
        if chunk_size > seq_len_q:
            continue
        
        estimated_mem = estimate_attention_memory(
            batch_size, num_heads, chunk_size, seq_len_k, head_dim, dtype
        )
        
        if estimated_mem < max_memory_bytes:
            return chunk_size
    
    return 32  # Minimum chunk size


def mps_chunked_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    chunk_size=None,
):
    """
    Chunked attention implementation for MPS devices to handle memory limitations.
    Processes attention in chunks along the query sequence dimension.
    
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    q_scale:        float. Scaling for query.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    chunk_size:     int. Size of chunks for processing. Auto-calculated if None.
    """
    device = q.device
    out_dtype = q.dtype
    compute_dtype = torch.float32  # MPS requires float32 for stability
    
    b, lq, n, c = q.shape
    lk = k.shape[1]
    
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using chunked attention. '
            'It can have a significant impact on performance.'
        )
    
    if q_scale is not None:
        q = q * q_scale
    
    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        chunk_size = get_optimal_chunk_size(b, n, lq, lk, c, compute_dtype)
    
    # Ensure chunk_size doesn't exceed sequence length
    chunk_size = min(chunk_size, lq)
    
    # Transpose for attention: [B, Lq, Nq, C] -> [B, Nq, Lq, C]
    q = q.transpose(1, 2).to(compute_dtype)
    k = k.transpose(1, 2).to(compute_dtype)
    v = v.transpose(1, 2).to(compute_dtype)
    
    # Pre-allocate output tensor
    out = torch.empty(b, n, lq, v.shape[-1], dtype=compute_dtype, device=device)
    
    # Process in chunks
    num_chunks = math.ceil(lq / chunk_size)
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, lq)
        
        q_chunk = q[:, :, start_idx:end_idx, :]
        
        # For causal attention, we only need keys/values up to current position
        if causal:
            k_chunk = k[:, :, :end_idx, :]
            v_chunk = v[:, :, :end_idx, :]
        else:
            k_chunk = k
            v_chunk = v
        
        # Compute attention for this chunk
        chunk_out = torch.nn.functional.scaled_dot_product_attention(
            q_chunk, k_chunk, v_chunk,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
            scale=softmax_scale
        )
        
        out[:, :, start_idx:end_idx, :] = chunk_out
        
        # Clear intermediate tensors and sync MPS
        del q_chunk, chunk_out
        if causal:
            del k_chunk, v_chunk
        
        # Periodically clear MPS cache to prevent memory buildup
        if chunk_idx % 4 == 0:
            empty_cache()
    
    # Transpose back: [B, Nq, Lq, C] -> [B, Lq, Nq, C]
    out = out.transpose(1, 2).contiguous()
    
    return out.to(out_dtype)


def sdp_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.float32,
):
    """
    Scaled dot-product attention for MPS and CPU devices.
    Uses PyTorch's native scaled_dot_product_attention.
    
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    device_type = q.device.type
    out_dtype = q.dtype
    
    # For MPS, use float32 for better stability; for CPU/CUDA use the specified dtype
    if device_type == 'mps':
        compute_dtype = torch.float32
    else:
        compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. '
            'It can have a significant impact on performance.'
        )
    
    if q_scale is not None:
        q = q * q_scale
    
    # Transpose for attention: [B, Lq, Nq, C] -> [B, Nq, Lq, C]
    q = q.transpose(1, 2).to(compute_dtype)
    k = k.transpose(1, 2).to(compute_dtype)
    v = v.transpose(1, 2).to(compute_dtype)
    
    # Apply scaled dot product attention
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        is_causal=causal,
        dropout_p=dropout_p,
        scale=softmax_scale
    )
    
    # Transpose back: [B, Nq, Lq, C] -> [B, Lq, Nq, C]
    out = out.transpose(1, 2).contiguous()
    return out.to(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Flash attention for CUDA devices. Falls back to chunked/SDP attention for MPS/CPU.
    
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    # For MPS devices, check if we need chunked attention to avoid memory errors
    if q.device.type == 'mps':
        b, lq, n, c = q.shape
        lk = k.shape[1]
        
        # Estimate memory requirement
        estimated_mem = estimate_attention_memory(b, n, lq, lk, c, torch.float32)
        max_mem_bytes = get_mps_max_buffer_gb() * 1024 * 1024 * 1024
        
        # Use chunked attention if memory requirement is too high
        if estimated_mem > max_mem_bytes:
            return mps_chunked_attention(
                q=q, k=k, v=v,
                q_lens=q_lens, k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
        else:
            # For smaller sequences, use standard SDP attention
            return sdp_attention(
                q=q, k=k, v=v,
                q_lens=q_lens, k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=torch.float32,
            )
    
    # For CPU, use SDP attention
    if q.device.type == 'cpu':
        return sdp_attention(
            q=q, k=k, v=v,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
        )
    
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    device_type = q.device.type
    
    # For MPS devices, check memory and use chunked attention if needed
    if device_type == 'mps':
        b, lq, n, c = q.shape
        lk = k.shape[1]
        
        # Estimate memory requirement
        estimated_mem = estimate_attention_memory(b, n, lq, lk, c, torch.float32)
        max_mem_bytes = get_mps_max_buffer_gb() * 1024 * 1024 * 1024
        
        # Use chunked attention if memory requirement is too high
        if estimated_mem > max_mem_bytes:
            return mps_chunked_attention(
                q=q, k=k, v=v,
                q_lens=q_lens, k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
        else:
            return sdp_attention(
                q=q, k=k, v=v,
                q_lens=q_lens, k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=torch.float32,
            )
    
    # For CUDA with flash attention available
    if device_type == 'cuda' and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    
    # Fallback to SDP attention for CPU and CUDA without flash attention
    return sdp_attention(
        q=q, k=k, v=v,
        q_lens=q_lens, k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
    )
