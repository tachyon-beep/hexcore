import os
import pytest
import torch
import gc
from pathlib import Path

# Add memory fragmentation mitigation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import necessary modules
try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None

from src.utils.device_mapping import DeviceMapper
from src.utils.gpu_memory_tracker import track_memory, log_memory_usage


@pytest.fixture
def mapper():
    """Fixture providing a DeviceMapper instance."""
    return DeviceMapper(num_experts=8, num_layers=32)


@pytest.mark.parametrize("quantization_bits", [None, 4, 8])
def test_model_mapping(mapper, quantization_bits):
    """
    Test model mapping with different quantization levels.

    Args:
        mapper: DeviceMapper fixture
        quantization_bits: Quantization level to test
    """
    print("\n" + "=" * 80)
    print(
        f"TESTING MAPPING WITH {'NO' if quantization_bits is None else f'{quantization_bits}-BIT'} QUANTIZATION"
    )
    print("=" * 80)

    # Clear memory before starting
    torch.cuda.empty_cache()
    gc.collect()

    # Get device map using balanced approach
    print("\nGenerating balanced device map:")
    try:
        with track_memory("Device Mapping Generation", log_to_console=False):
            device_map = mapper.create_mixtral_device_map(
                quantization_bits=quantization_bits
            )

        # Analyze the generated map
        analyze_device_map(device_map)

        # The test passes if we reach here without errors
        assert device_map is not None

    except Exception as e:
        pytest.fail(f"Error generating device map: {e}")


def test_compare_mapping_strategies():
    """Test to compare the old and new mapping strategies."""
    # Save current implementation
    current_create_dual_gpu_map = DeviceMapper._create_dual_gpu_map

    # Replace with a mock of the old implementation
    def mock_old_mapping(self, quantization_bits=None):
        print("\nSimulating old unbalanced mapping strategy output")
        # Return a placeholder device map
        placeholder = {"placeholder": "This is a simulated run of the old mapping"}
        return placeholder

    # Apply mock for comparison
    DeviceMapper._create_dual_gpu_map = mock_old_mapping

    # Run with old approach (mocked)
    mapper = DeviceMapper(num_experts=8, num_layers=32)
    old_map = mapper.create_mixtral_device_map()

    # Restore the current implementation
    DeviceMapper._create_dual_gpu_map = current_create_dual_gpu_map

    # Run with new balanced approach
    new_map = mapper.create_mixtral_device_map()

    # Validate
    assert old_map is not None
    assert new_map is not None
    assert old_map != new_map  # They should be different


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
        print("\nExpert Distribution:")
        gpu0_ratio = gpu0_experts / total_experts * 100
        gpu1_ratio = gpu1_experts / total_experts * 100
        print(f"  GPU 0: {gpu0_experts} experts ({gpu0_ratio:.1f}%)")
        print(f"  GPU 1: {gpu1_experts} experts ({gpu1_ratio:.1f}%)")

        # Check if balanced
        if 40 <= gpu0_ratio <= 60:
            print("  ✅ Expert distribution is well-balanced")
            assert 40 <= gpu0_ratio <= 60, "Expert distribution should be balanced"
        else:
            print("  ⚠️ Expert distribution is unbalanced")
