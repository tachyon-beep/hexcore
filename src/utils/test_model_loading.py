#!/usr/bin/env python
# Test script for model loading with balanced device mapping

import os
import sys
import torch
import gc
import argparse
from pathlib import Path

# Add memory fragmentation mitigation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import necessary modules
try:
    from transformers import AutoConfig
except ImportError:
    print(
        "Warning: transformers library not installed. This test requires Hugging Face Transformers."
    )
    AutoConfig = None

from src.utils.device_mapping import DeviceMapper
from src.utils.gpu_memory_tracker import track_memory, log_memory_usage


def simulate_model_loading(
    model_name="mistralai/Mixtral-8x7B-v0.1", quantization_bits=4
):
    """
    Simulate model loading with balanced device mapping.

    This function doesn't actually load the model (which could be very large),
    but simulates the memory allocation to verify the mapping works correctly.

    Args:
        model_name: Model identifier to use
        quantization_bits: Quantization level (4 or 8) or None for FP16
    """
    print("\n" + "=" * 80)
    print(f"SIMULATING MODEL LOADING: {model_name}")
    print(
        f"QUANTIZATION: {'None' if quantization_bits is None else f'{quantization_bits}-bit'}"
    )
    print("=" * 80)

    # Clear memory before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Log initial memory state
    print("\nInitial Memory State:")
    log_memory_usage("Before model loading")

    # Get model configuration from Hugging Face
    if AutoConfig is None:
        print(
            "Error: transformers library is not installed. Cannot load model configuration."
        )
        print("Using default Mixtral parameters instead (8 experts, 32 layers).")
        num_experts = 8
        num_layers = 32
    else:
        try:
            config = AutoConfig.from_pretrained(model_name)
            print(f"\nLoaded configuration for {model_name}")
            print(f"  Hidden size: {config.hidden_size}")
            print(f"  Vocabulary size: {config.vocab_size}")
            print(f"  Number of layers: {config.num_hidden_layers}")
            print(f"  Number of experts: {config.num_local_experts}")
            num_experts = config.num_local_experts
            num_layers = config.num_hidden_layers
        except Exception as e:
            print(f"Error loading model configuration: {e}")
            print("Using default Mixtral parameters instead (8 experts, 32 layers).")
            num_experts = 8
            num_layers = 32

    # Create a device mapper with model configuration
    mapper = DeviceMapper(num_experts=num_experts, num_layers=num_layers)

    # Get device map using our balanced approach
    print("\nGenerating balanced device map:")
    try:
        with track_memory("Device Mapping Generation", log_to_console=True):
            device_map = mapper.create_mixtral_device_map(
                quantization_bits=quantization_bits
            )

        # Log memory state after mapping
        print("\nMemory State After Mapping Generation:")
        log_memory_usage("After device mapping")

        # Analyze the generated map
        analyze_device_map(device_map)

        return device_map
    except Exception as e:
        print(f"Error generating device map: {e}")
        return None


def analyze_device_map(device_map):
    """Analyze device map to validate balancing."""
    # Count components per device
    gpu0_components = 0
    gpu1_components = 0
    gpu0_experts = 0
    gpu1_experts = 0

    for module, device in device_map.items():
        if device == "cuda:0":
            gpu0_components += 1
            if ".block_sparse_moe.experts." in module:
                gpu0_experts += 1
        elif device == "cuda:1":
            gpu1_components += 1
            if ".block_sparse_moe.experts." in module:
                gpu1_experts += 1

    # Print summary
    print("\nDevice Map Analysis:")
    print(f"  Total components mapped: {gpu0_components + gpu1_components}")
    print(f"  Components on GPU 0: {gpu0_components}")
    print(f"  Components on GPU 1: {gpu1_components}")
    print(f"  Experts on GPU 0: {gpu0_experts}")
    print(f"  Experts on GPU 1: {gpu1_experts}")

    # Calculate balance ratio
    total_experts = gpu0_experts + gpu1_experts
    if total_experts > 0:
        print(f"\nExpert Distribution:")
        gpu0_ratio = gpu0_experts / total_experts * 100
        gpu1_ratio = gpu1_experts / total_experts * 100
        print(f"  GPU 0: {gpu0_experts} experts ({gpu0_ratio:.1f}%)")
        print(f"  GPU 1: {gpu1_experts} experts ({gpu1_ratio:.1f}%)")

        # Check if balanced
        if 40 <= gpu0_ratio <= 60:
            print("  ✅ Expert distribution is well-balanced")
        else:
            print("  ⚠️ Expert distribution is unbalanced")


def compare_device_maps(
    old_approach=True, new_approach=True, model_name="mistralai/Mixtral-8x7B-v0.1"
):
    """Compare old and new device mapping approaches."""
    if old_approach:
        print("\n\n" + "=" * 100)
        print("TESTING ORIGINAL IMBALANCED DEVICE MAPPING APPROACH")
        print("=" * 100)

        # Save current implementation
        current_create_dual_gpu_map = DeviceMapper._create_dual_gpu_map

        # Replace with a mock of the old implementation that skips actual execution
        # but prints what would happen with the old strategy
        def mock_old_mapping(self, quantization_bits=None):
            print("\nOLD MAPPING STRATEGY WOULD RESULT IN:")
            print("  GPU 0: 12 layers, ~50 experts (significantly underutilized)")
            print("  GPU 1: 20 layers, ~206 experts (severely overloaded)")
            print(
                "  This leads to GPU 1 using 15.36GB out of 15.60GB total (98.5% utilized)"
            )
            print(
                "  🔴 Would likely cause OOM error when loading additional 112MB during model loading"
            )

            # Return a placeholder device map
            placeholder = {"placeholder": "This is a simulated run of the old mapping"}
            return placeholder

        # Apply mock for comparison
        DeviceMapper._create_dual_gpu_map = mock_old_mapping

        # Run with old approach (mocked)
        old_map = simulate_model_loading(model_name=model_name)

        # Restore the current (new) implementation
        DeviceMapper._create_dual_gpu_map = current_create_dual_gpu_map

    if new_approach:
        print("\n\n" + "=" * 100)
        print("TESTING NEW BALANCED DEVICE MAPPING APPROACH")
        print("=" * 100)

        # Run with new balanced approach
        new_map = simulate_model_loading(model_name=model_name)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Test balanced device mapping for model loading"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model to use for testing",
    )
    parser.add_argument(
        "--quantization",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits (4 or 8)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare old and new mapping approaches"
    )

    args = parser.parse_args()

    if args.compare:
        compare_device_maps(model_name=args.model)
    else:
        simulate_model_loading(
            model_name=args.model, quantization_bits=args.quantization
        )


if __name__ == "__main__":
    main()
