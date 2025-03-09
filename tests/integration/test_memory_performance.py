#!/usr/bin/env python
# tests/integration/test_memory_performance.py
"""Integration tests for memory usage and performance in the MTG AI Assistant."""

import pytest
import sys
import torch
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the shared fixture
from tests.integration import prepared_pipeline
from src.utils.gpu_memory_tracker import GPUMemoryTracker
from src.utils.kv_cache_manager import KVCacheManager


@pytest.mark.integration
@pytest.mark.performance
def test_memory_usage(prepared_pipeline):
    """Test memory usage during inference with a simpler query."""
    # Skip if GPU memory tracking is not available
    try:
        memory_tracker = GPUMemoryTracker()
    except Exception as e:
        pytest.skip(f"GPU memory tracking not available: {e}")

    # Check total GPU memory across all devices
    total_gpu_memory = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        total_gpu_memory += gpu_mem

    # Check if we have enough total GPU memory for dual GPU setup (2x16GB = 32GB)
    # Using 24GB as threshold to account for system reserved memory
    available_memory = total_gpu_memory
    required_memory = 24.0
    if available_memory < required_memory:
        pytest.xfail(
            f"Test requires at least {required_memory}GB total GPU memory (available: {available_memory:.1f}GB)"
        )

    # Empty the CUDA cache
    torch.cuda.empty_cache()

    # Start memory tracking
    memory_tracker.start_monitoring()

    # Process a simpler query to reduce memory usage
    simple_query = "What does Lightning Bolt do?"

    # Configure a memory-efficient KV cache manager
    kv_cache_manager = KVCacheManager(
        max_memory_percentage=0.2,  # Conservative memory usage
        sliding_window=256,  # Limit context size
        auto_clear=True,  # Aggressively clear caches
    )

    # Attach to pipeline
    prepared_pipeline.kv_cache_manager = kv_cache_manager

    try:
        # Use a single expert to reduce memory pressure
        # Ensure device consistency is enabled to prevent device mismatch errors
        result = prepared_pipeline.generate_response(
            simple_query,
            use_multiple_experts=False,
            max_new_tokens=256,  # Limit the output size
            ensure_device_consistency=True,
        )

        # Stop tracking and get memory stats
        memory_tracker.stop_monitoring()
        max_gpu_usage, max_cpu_usage = memory_tracker.get_max_memory_usage()

        # Print memory usage for debugging
        print(f"Max GPU memory usage: {max_gpu_usage}")
        print(f"Max CPU memory usage: {max_cpu_usage}MB")

        # Ensure we got a valid response
        assert result is not None
        assert "response" in result
        assert len(result["response"]) > 0

    except Exception as e:
        # Ensure we stop tracking in case of errors
        memory_tracker.stop_monitoring()
        # Re-raise the exception to fail the test
        raise e
    finally:
        # Clean up memory
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.performance
def test_large_context_memory_usage(prepared_pipeline):
    """Test memory usage with a moderate context window query."""
    # Skip if GPU memory tracking is not available
    try:
        memory_tracker = GPUMemoryTracker()
    except Exception as e:
        pytest.skip(f"GPU memory tracking not available: {e}")

    # Check if we have multiple GPUs
    if torch.cuda.device_count() < 2:
        pytest.skip("This test requires at least 2 GPUs")

    # Check total GPU memory across all devices
    total_gpu_memory = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        total_gpu_memory += gpu_mem

    # Check if we have enough total GPU memory (30GB threshold for dual 16GB GPUs)
    # Using slightly lower threshold to account for system reserved memory
    available_memory = total_gpu_memory
    required_memory = 30.0
    if available_memory < required_memory:
        pytest.xfail(
            f"Test requires at least {required_memory}GB total GPU memory (available: {available_memory:.1f}GB). "
            f"This test needs two 16GB GPUs."
        )

    # Empty CUDA cache
    torch.cuda.empty_cache()

    # Start memory tracking
    memory_tracker.start_monitoring()

    # Simplified query with fewer card references
    moderate_context_query = (
        "Compare Lightning Bolt and Lava Spike for a Modern Burn deck. "
        "Which is better and why?"
    )

    try:
        # Use a single expert for this test with device consistency enforced
        result = prepared_pipeline.generate_response(
            moderate_context_query,
            use_multiple_experts=False,  # Use single expert to reduce memory
            max_new_tokens=256,  # Limit the output size
            ensure_device_consistency=True,  # Prevent device mismatch errors
        )

        # Stop tracking and get memory stats
        memory_tracker.stop_monitoring()
        max_gpu_usage, max_cpu_usage = memory_tracker.get_max_memory_usage()

        # Log memory usage
        print(f"Context test - Max GPU memory usage: {max_gpu_usage}")
        print(f"Context test - Max CPU memory usage: {max_cpu_usage}MB")

        # Verify response quality
        assert result is not None
        assert "response" in result
        assert len(result["response"]) > 0
        assert "expert_types" in result

    except Exception as e:
        # Ensure we stop tracking in case of errors
        memory_tracker.stop_monitoring()
        # Re-raise the exception
        raise e
    finally:
        # Clean up
        torch.cuda.empty_cache()


# Add a new test to monitor memory usage with multiple experts but smaller context
@pytest.mark.integration
@pytest.mark.performance
def test_memory_usage_with_experts(prepared_pipeline):
    """Test memory usage when using multiple experts with a small query."""
    # Skip if GPU memory tracking is not available
    try:
        memory_tracker = GPUMemoryTracker()
    except Exception as e:
        pytest.skip(f"GPU memory tracking not available: {e}")

    # Check if we have multiple GPUs
    if torch.cuda.device_count() < 2:
        pytest.skip("This test requires at least 2 GPUs")

    # Check total GPU memory across all devices
    total_gpu_memory = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        total_gpu_memory += gpu_mem

    # Check if we have enough total GPU memory (30GB threshold for dual 16GB GPUs)
    # This test uses multiple experts, so it needs more memory
    available_memory = total_gpu_memory
    required_memory = 30.0
    if available_memory < required_memory:
        pytest.xfail(
            f"Test requires at least {required_memory}GB total GPU memory (available: {available_memory:.1f}GB). "
            f"This test needs dual GPUs with at least 16GB each."
        )

    # Empty cache
    torch.cuda.empty_cache()

    # Start memory tracking
    memory_tracker.start_monitoring()

    # Very simple query to minimize context size
    simple_query = "Explain the difference between sorcery and instant speed in MTG."

    try:
        # Use multiple experts but with a simple query
        result = prepared_pipeline.generate_response(
            simple_query,
            use_multiple_experts=True,  # Test with multiple experts
            max_new_tokens=128,  # Keep output very small
            ensure_device_consistency=True,  # Critical for multi-expert tests
        )

        # Stop tracking and get memory stats
        memory_tracker.stop_monitoring()
        max_gpu_usage, max_cpu_usage = memory_tracker.get_max_memory_usage()

        # Log memory usage
        print(f"Multi-expert - Max GPU memory usage: {max_gpu_usage}")
        print(f"Multi-expert - Max CPU memory usage: {max_cpu_usage}MB")

        # Verify response
        assert result is not None
        assert "response" in result
        assert "expert_types" in result
        # With multiple experts, we should have at least one expert type
        assert len(result["expert_types"]) >= 1

    except Exception as e:
        # Ensure tracking stops
        memory_tracker.stop_monitoring()
        raise e
    finally:
        # Clean up
        torch.cuda.empty_cache()
