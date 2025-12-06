"""
Qwen2-VL specific handling and utilities.
"""

from typing import Optional
import torch


def get_optimal_dtype() -> torch.dtype:
    """Get the optimal dtype for the current hardware."""
    if torch.cuda.is_available():
        # Check if bfloat16 is supported
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_device_map(vram_gb: Optional[float] = None) -> str:
    """Get the optimal device map based on available VRAM."""
    if not torch.cuda.is_available():
        return "cpu"

    if vram_gb is None:
        # Auto-detect VRAM
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Qwen2-VL-7B needs ~16GB in FP16
    if vram_gb >= 20:
        return "cuda:0"
    elif vram_gb >= 12:
        # Could use int8 quantization
        return "auto"
    else:
        # Need CPU offloading
        return "auto"


def estimate_memory_usage(model_size_b: float, dtype: torch.dtype) -> float:
    """Estimate VRAM usage in GB for a model."""
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }

    param_bytes = model_size_b * bytes_per_param.get(dtype, 2)
    # Add ~20% overhead for activations, KV cache, etc.
    total_bytes = param_bytes * 1.2

    return total_bytes / (1024**3)
