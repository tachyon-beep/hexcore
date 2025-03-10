# src/utils/gpu_utils.py

import torch
import psutil
import os
import gc
from typing import Dict, List, Union, Optional, Any


def get_gpu_info() -> Dict[str, Union[List[Dict[str, Any]], str]]:
    """
    Get detailed information about available GPUs.

    Returns:
        Dictionary with GPU details including memory, compute capability, etc.
        Contains 'devices' key with a list of device info dictionaries.
        May contain 'error' key with error message if CUDA is not available.
    """
    # Initialize with an explicit list for 'devices' to avoid type confusion
    devices_list: List[Dict[str, Any]] = []
    gpu_info: Dict[str, Union[List[Dict[str, Any]], str]] = {"devices": devices_list}

    if not torch.cuda.is_available():
        return {"devices": [], "error": "CUDA not available"}

    device_count = torch.cuda.device_count()

    for i in range(device_count):
        device_properties = torch.cuda.get_device_properties(i)
        # Now we're appending to our explicit list variable
        devices_list.append(
            {
                "id": i,
                "name": device_properties.name,
                "total_memory": device_properties.total_memory / (1024**3),  # GB
                "compute_capability": f"{device_properties.major}.{device_properties.minor}",
                "multi_processor_count": device_properties.multi_processor_count,
            }
        )

    return gpu_info


def monitor_memory(device: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """
    Monitor current memory usage for CPU and specified GPU.

    Args:
        device: GPU device ID to monitor, or None to monitor all devices

    Returns:
        Dictionary with nested memory statistics for CPU and GPUs
        Format: {'cpu': {'total': float, 'available': float, 'used_percent': float},
                'gpu_0': {'total': float, 'reserved': float, 'allocated': float}}
    """
    memory_stats: Dict[str, Dict[str, float]] = {
        "cpu": {
            "total": psutil.virtual_memory().total / (1024**3),
            "available": psutil.virtual_memory().available / (1024**3),
            "used_percent": psutil.virtual_memory().percent,
        }
    }

    if torch.cuda.is_available():
        if device is not None:
            torch.cuda.synchronize(device)
            memory_stats[f"gpu_{device}"] = {
                "total": torch.cuda.get_device_properties(device).total_memory
                / (1024**3),
                "reserved": torch.cuda.memory_reserved(device) / (1024**3),
                "allocated": torch.cuda.memory_allocated(device) / (1024**3),
            }
        else:
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
                memory_stats[f"gpu_{i}"] = {
                    "total": torch.cuda.get_device_properties(i).total_memory
                    / (1024**3),
                    "reserved": torch.cuda.memory_reserved(i) / (1024**3),
                    "allocated": torch.cuda.memory_allocated(i) / (1024**3),
                }

    return memory_stats


def optimize_memory() -> None:
    """
    Perform memory optimization operations.
    Clears PyTorch cache and runs garbage collection.
    """
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "memory_summary"):
        for i in range(torch.cuda.device_count()):
            print(f"Memory summary for GPU {i}:")
            print(torch.cuda.memory_summary(device=i, abbreviated=True))


def device_mapping(num_experts: int, active_experts: List[int]) -> Dict[int, str]:
    """
    Create a device mapping for experts based on available GPUs.

    Args:
        num_experts: Total number of experts in the model
        active_experts: List of expert indices that will be activated

    Returns:
        Dictionary mapping expert indices to device strings
    """
    if not torch.cuda.is_available():
        return {i: "cpu" for i in range(num_experts)}

    num_gpus = torch.cuda.device_count()
    device_map = {}

    # Place active experts first, distributing evenly across GPUs
    gpu_loads = [0] * num_gpus
    for expert_idx in active_experts:
        # Find GPU with lowest current load
        target_gpu = gpu_loads.index(min(gpu_loads))
        device_map[expert_idx] = f"cuda:{target_gpu}"
        gpu_loads[target_gpu] += 1

    # Place remaining experts
    for expert_idx in range(num_experts):
        if expert_idx not in device_map:
            if max(gpu_loads) >= num_experts / num_gpus * 1.5:
                # If GPUs are heavily loaded, put on CPU
                device_map[expert_idx] = "cpu"
            else:
                # Otherwise distribute across GPUs
                target_gpu = gpu_loads.index(min(gpu_loads))
                device_map[expert_idx] = f"cuda:{target_gpu}"
                gpu_loads[target_gpu] += 1

    return device_map
