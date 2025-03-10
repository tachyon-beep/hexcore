"""
Memory management utilities for efficient model loading.
"""

import os
import gc
import torch
import logging
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

# Set environment variables at the module level to ensure they're applied early
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"


def clear_gpu_memory(device_ids: Optional[Union[int, list]] = None):
    """
    Clear CUDA cache and run garbage collection to free up GPU memory.

    Args:
        device_ids: Specific device ID(s) to clear. If None, clears all devices.
    """
    # Run Python garbage collection first
    gc.collect()

    # Clear CUDA cache
    if device_ids is None:
        # Clear all devices
        torch.cuda.empty_cache()
        logger.debug("Cleared memory on all CUDA devices")
    else:
        # Convert single ID to list if needed
        if isinstance(device_ids, int):
            device_ids = [device_ids]

        # Clear specific devices
        for device in device_ids:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            logger.debug(f"Cleared memory on CUDA device {device}")


def log_gpu_memory_stats():
    """
    Log detailed GPU memory statistics for all available devices.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, cannot log GPU memory stats")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)

        # Get allocated and reserved memory
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)

        logger.info(f"GPU {i} ({props.name}) Memory Stats:")
        logger.info(f"  Total: {total_memory:.2f} GB")
        logger.info(
            f"  Allocated: {allocated:.2f} GB ({(allocated/total_memory)*100:.1f}%)"
        )
        logger.info(
            f"  Reserved: {reserved:.2f} GB ({(reserved/total_memory)*100:.1f}%)"
        )
        logger.info(
            f"  Free: {total_memory - allocated:.2f} GB ({((total_memory - allocated)/total_memory)*100:.1f}%)"
        )


def setup_optimal_memory_config():
    """
    Configure the environment for optimal memory usage with large models.
    """
    # Enable memory efficient features in PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:64,expandable_segments:True"
    )

    # Explicitly controlling OMP threads can help with memory usage
    os.environ["OMP_NUM_THREADS"] = "1"

    # Optimize NumPy thread usage
    os.environ["MKL_NUM_THREADS"] = "1"

    # Run a memory cleanup
    clear_gpu_memory()

    # Log current settings and memory status
    logger.info("Memory optimization configuration applied:")
    logger.info(
        f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}"
    )
    log_gpu_memory_stats()

    return True


def optimize_for_inference():
    """
    Apply optimizations specifically for inference mode.

    This includes settings that reduce memory usage during inference
    at the potential cost of slightly slower initialization.
    """
    # Apply general memory optimizations
    setup_optimal_memory_config()

    # Set inference-specific optimizations

    # Disable gradient computation
    torch.set_grad_enabled(False)

    # If using CUDA >= 11.6, enable memory efficient attention
    try:
        if (
            hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "enable_flash_sdp")
            and torch.cuda.get_device_capability()[0] >= 8
        ):
            torch.backends.cuda.enable_flash_sdp(True)
            logger.info(
                "Enabled Flash Scaled Dot Product Attention for efficient inference"
            )
    except Exception as e:
        logger.warning(f"Failed to enable Flash attention: {e}")

    return True


def load_with_progressive_offloading(load_func, *args, **kwargs):
    """
    Wrapper function that manages memory carefully during model loading.

    This function provides a progressive loading approach that offloads
    components temporarily and manages memory more aggressively during loading.

    Args:
        load_func: The original loading function
        *args, **kwargs: Arguments to pass to the loading function

    Returns:
        The loaded model and other return values from load_func
    """
    # Set initial optimization
    optimize_for_inference()

    # Set the PYTORCH_CUDA_ALLOC_CONF environment variable to enable expandable segments
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:64,expandable_segments:True"
    )

    # Log initial memory state
    logger.info("Initial memory state before loading:")
    log_gpu_memory_stats()

    # Clear GPU memory before loading
    clear_gpu_memory()

    try:
        # Run the loading function with provided arguments
        result = load_func(*args, **kwargs)

        # Clean up after loading
        clear_gpu_memory()

        # Log final memory state
        logger.info("Final memory state after loading:")
        log_gpu_memory_stats()

        return result
    except Exception as e:
        logger.error(f"Error during progressive loading: {e}")

        # Try to clean up memory before re-raising
        clear_gpu_memory()

        # Re-raise the exception
        raise
