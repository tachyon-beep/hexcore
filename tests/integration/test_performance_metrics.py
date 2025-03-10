#!/usr/bin/env python
# tests/integration/test_performance_metrics.py
"""Integration tests for performance metrics and timing in the MTG AI Assistant."""

import pytest
import sys
import time
import torch
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the shared fixture
from tests.integration import prepared_pipeline


@pytest.mark.integration
@pytest.mark.performance
def test_response_time_performance(prepared_pipeline):
    """Test response time performance for simple queries."""
    # Check total GPU memory across all devices
    total_gpu_memory = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        total_gpu_memory += gpu_mem

    # Check if we have enough total GPU memory
    available_memory = total_gpu_memory
    required_memory = 24.0
    if available_memory < required_memory:
        pytest.xfail(
            f"Test requires at least {required_memory}GB total GPU memory (available: {available_memory:.1f}GB)"
        )

    # Clear GPU cache
    torch.cuda.empty_cache()

    try:
        start_time = time.time()

        # Process a simple query with limited output size
        result = prepared_pipeline.generate_response(
            "What is Magic: The Gathering?",
            use_multiple_experts=False,
            max_new_tokens=256,  # Limit output size
            ensure_device_consistency=True,  # Prevent device mismatch errors
        )

        # Measure actual processing time
        total_time = time.time() - start_time
        print(f"Simple query - Total response time: {total_time:.2f}s")

        # Check that response time metrics match actual time (within reason)
        assert (
            abs(result["metrics"]["total_time"] - total_time) < 1.0
        ), "Reported time metrics should be close to actual time"

        # Verify basic response quality
        assert result is not None
        assert "response" in result
        assert len(result["response"]) > 0  # Should have content

    finally:
        # Ensure GPU memory is cleared
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.performance
def test_response_time_complex_query(prepared_pipeline):
    """Test response time performance for complex rules queries."""
    # Check total GPU memory across all devices
    total_gpu_memory = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        total_gpu_memory += gpu_mem

    # Check if we have enough total GPU memory
    available_memory = total_gpu_memory
    required_memory = 24.0
    if available_memory < required_memory:
        pytest.xfail(
            f"Test requires at least {required_memory}GB total GPU memory (available: {available_memory:.1f}GB)"
        )

    # Clear GPU cache
    torch.cuda.empty_cache()

    try:
        start_time = time.time()

        # Process a less complex query
        complex_query = "How does Lightning Bolt interact with Hexproof?"
        result = prepared_pipeline.generate_response(
            complex_query,
            use_multiple_experts=False,
            max_new_tokens=256,  # Limit output size
            ensure_device_consistency=True,  # Prevent device mismatch errors
        )

        # Measure actual processing time
        total_time = time.time() - start_time
        print(f"Complex query - Total response time: {total_time:.2f}s")

        # Verify metrics structure
        assert "metrics" in result
        assert "classification_time" in result["metrics"]
        assert "retrieval_time" in result["metrics"]
        assert "generation_time" in result["metrics"]
        assert "total_time" in result["metrics"]

        # Verify individual steps are accounted for in total time
        components_sum = (
            result["metrics"]["classification_time"]
            + result["metrics"]["retrieval_time"]
            + result["metrics"]["generation_time"]
        )

        # Allow small differences due to overhead not counted in components
        assert (
            abs(result["metrics"]["total_time"] - components_sum) < 0.5
        ), "Sum of component times should approximately equal total time"

    finally:
        # Ensure GPU memory is cleared
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.performance
def test_expert_routing_overhead(prepared_pipeline):
    """Test overhead of expert routing compared to single expert mode."""
    # Check if we have enough GPUs for this test
    if torch.cuda.device_count() < 2:
        pytest.skip("This test requires at least 2 GPUs")

    # Check total GPU memory across all devices
    total_gpu_memory = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        total_gpu_memory += gpu_mem

    # Check if we have enough total GPU memory (30GB threshold for dual 16GB GPUs)
    available_memory = total_gpu_memory
    required_memory = 30.0
    if available_memory < required_memory:
        pytest.xfail(
            f"Test requires at least {required_memory}GB total GPU memory (available: {available_memory:.1f}GB). "
            f"This test needs dual GPUs with at least 16GB each."
        )

    # Clear GPU cache
    torch.cuda.empty_cache()

    try:
        # Use a very simple query to reduce memory usage
        query = "What is a mana cost?"

        # Test with single expert
        start_time = time.time()
        single_result = prepared_pipeline.generate_response(
            query,
            use_multiple_experts=False,
            max_new_tokens=128,  # Keep response very short
            ensure_device_consistency=True,  # Prevent device mismatch errors
        )
        single_expert_time = time.time() - start_time

        # Clear cache between tests
        torch.cuda.empty_cache()

        # Test with multiple experts
        start_time = time.time()
        multi_result = prepared_pipeline.generate_response(
            query,
            use_multiple_experts=True,
            max_new_tokens=128,  # Keep response very short
            ensure_device_consistency=True,  # Critical for multi-expert operations
        )
        multi_expert_time = time.time() - start_time

        # Log timing for comparison
        print(f"Single expert time: {single_expert_time:.2f}s")
        print(f"Multi-expert time: {multi_expert_time:.2f}s")
        print(f"Overhead: {multi_expert_time - single_expert_time:.2f}s")

        # Verify that both approaches produced valid responses
        assert single_result is not None and multi_result is not None
        assert "response" in single_result and "response" in multi_result
        assert len(single_result["response"]) > 0
        assert len(multi_result["response"]) > 0

        # Ensure both methods report metrics properly
        assert "metrics" in single_result
        assert "metrics" in multi_result

    finally:
        # Ensure GPU memory is cleared
        torch.cuda.empty_cache()
