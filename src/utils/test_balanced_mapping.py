#!/usr/bin/env python
# Test script to validate the improved balanced device mapping

import os
import sys
import torch
import gc
from typing import Dict, Optional

# Add memory fragmentation mitigation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import necessary modules
from src.utils.device_mapping import DeviceMapper
from src.utils.gpu_memory_tracker import track_memory, log_memory_usage


def test_device_mapping():
    """Test and validate the balanced device mapping implementation."""

    print("\n\n" + "=" * 100)
    print("TESTING BALANCED DEVICE MAPPING")
    print("=" * 100)

    # Clear memory before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Log initial memory state
    print("\nInitial Memory State:")
    log_memory_usage("Before mapping")

    # Create a device mapper with default Mixtral settings (8 experts, 32 layers)
    mapper = DeviceMapper(num_experts=8, num_layers=32)

    # Test with different quantization settings
    test_quantization_settings(mapper)

    print("\nTest complete!")


def test_quantization_settings(mapper: DeviceMapper):
    """Test device mapping with different quantization settings."""

    quantization_settings = [None, 8, 4]
    for bits in quantization_settings:
        print("\n" + "=" * 80)
        print(f"TESTING WITH {'NO' if bits is None else f'{bits}-BIT'} QUANTIZATION")
        print("=" * 80)

        # Create device map
        bits_str = f"{bits}bit" if bits else "no"
        with track_memory(
            f"Device Mapping - {bits_str} quantization", log_to_console=True
        ):
            device_map = mapper.create_mixtral_device_map(quantization_bits=bits)

        # Validate and analyze the generated map
        analyze_device_map(device_map)

        # Print memory estimates
        print("\nEstimated memory with this mapping:")
        _ = mapper.trace_memory_usage(
            device_map, quantization=bits_str if bits else None
        )

        # Log current memory usage
        print("\nCurrent system memory usage:")
        log_memory_usage(f"After {bits_str} quantization mapping")


def analyze_device_map(device_map: Dict[str, str]):
    """Analyze a device map to count components per device."""

    # Count components per device
    device_counts = {}
    for module, device in device_map.items():
        if device not in device_counts:
            device_counts[device] = {"layers": 0, "experts": 0, "other": 0}

        if (
            "model.layers." in module
            and ".block_sparse_moe.experts." not in module
            and ".block_sparse_moe.gate" not in module
        ):
            device_counts[device]["layers"] += 1
        elif ".block_sparse_moe.experts." in module:
            device_counts[device]["experts"] += 1
        elif "embed_tokens" in module or "norm" in module or "lm_head" in module:
            device_counts[device]["other"] += 1

    # Print statistics
    print("\nDevice Map Statistics:")
    for device, counts in device_counts.items():
        print(f"\n  {device}:")
        if counts["layers"] > 0:
            print(f"    Layers: {counts['layers']}")
        if counts["experts"] > 0:
            print(f"    Experts: {counts['experts']}")
        if counts["other"] > 0:
            print(f"    Other components: {counts['other']}")

    # Calculate expert ratio for cuda:0 vs cuda:1
    total_experts = sum(counts["experts"] for device, counts in device_counts.items())
    if "cuda:0" in device_counts and "cuda:1" in device_counts:
        experts_gpu0 = device_counts["cuda:0"].get("experts", 0)
        experts_gpu1 = device_counts["cuda:1"].get("experts", 0)

        if total_experts > 0:
            print(f"\nExpert Distribution:")
            print(
                f"  GPU 0: {experts_gpu0} experts ({experts_gpu0/total_experts*100:.1f}%)"
            )
            print(
                f"  GPU 1: {experts_gpu1} experts ({experts_gpu1/total_experts*100:.1f}%)"
            )
            print(
                f"  Ratio: {experts_gpu0/(experts_gpu0+experts_gpu1)*100:.1f}% : {experts_gpu1/(experts_gpu0+experts_gpu1)*100:.1f}%"
            )

            # Check if balanced
            if 40 <= (experts_gpu0 / (experts_gpu0 + experts_gpu1) * 100) <= 60:
                print("  ✅ Expert distribution is well-balanced")
            else:
                print("  ⚠️ Expert distribution is unbalanced")


if __name__ == "__main__":
    test_device_mapping()
