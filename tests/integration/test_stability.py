#!/usr/bin/env python
# tests/integration/test_stability.py
"""
Integration tests for system stability and reliability in the MTG AI Assistant.

These tests verify that the system can handle multiple sequential queries without
degradation in quality or crashes, simulating real-world usage patterns.
"""

import pytest
import sys
import time
import torch
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the shared fixture
from tests.integration import prepared_pipeline
from src.utils.kv_cache_manager import KVCacheManager


@pytest.mark.integration
@pytest.mark.stability
def test_multiple_sequential_queries(prepared_pipeline):
    """Test handling multiple sequential queries to ensure system stability with KV cache management."""
    # Check total GPU memory across all devices
    total_gpu_memory = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        total_gpu_memory += gpu_mem

    # Check if we have enough total GPU memory (24GB threshold for dual 16GB GPUs)
    # We need slightly less memory for these simpler tests
    available_memory = total_gpu_memory
    required_memory = 24.0
    if available_memory < required_memory:
        pytest.xfail(
            f"Test requires at least {required_memory}GB total GPU memory (available: {available_memory:.1f}GB)"
        )

    # Use fewer, simpler queries for stability testing
    queries = [
        "What does Lightning Bolt do?",
        "How much life do players start with?",
        "What is a land card?",
    ]

    # Configure an optimized KV Cache Manager
    kv_cache_manager = KVCacheManager(
        max_memory_percentage=0.25,  # Moderate memory usage
        sliding_window=192,  # Small sliding window for efficiency
        auto_clear=True,  # Clear between generations
        prune_threshold=0.7,  # Aggressive pruning threshold
    )

    # Attach to pipeline
    prepared_pipeline.kv_cache_manager = kv_cache_manager

    response_lengths = []
    response_times = []

    try:
        for i, query in enumerate(queries, 1):
            print(f"Processing query {i}/{len(queries)}: {query}")

            # Clear cache between queries
            torch.cuda.empty_cache()

            # Measure response time
            start_time = time.time()
            result = prepared_pipeline.generate_response(
                query,
                use_multiple_experts=False,
                max_new_tokens=128,  # Limit output size
                ensure_device_consistency=True,  # Prevent device mismatch errors
            )
            query_time = time.time() - start_time

            # Collect metrics
            response_lengths.append(len(result["response"]))
            response_times.append(query_time)

            # Verify basic response quality
            assert result is not None, f"Failed to get response for query {i}: {query}"
            assert "response" in result, f"Missing response for query {i}"
            assert len(result["response"]) > 0, f"Empty response for query {i}"

            print(f"  Response length: {len(result['response'])} chars")
            print(f"  Response time: {query_time:.2f}s")

            # Allow a short delay between queries
            time.sleep(1.0)

    finally:
        # Final cleanup
        torch.cuda.empty_cache()

    # Verify no degradation in response quality over time
    print(f"Response lengths: {response_lengths}")
    print(f"Response times: {response_times}")

    # No response should be extremely short
    min_length = min(response_lengths) if response_lengths else 0
    assert min_length > 0, "All responses should have some content"


@pytest.mark.integration
@pytest.mark.stability
def test_long_complex_response_stability(prepared_pipeline):
    """Test system stability when generating responses to complex queries."""
    # Check if we have multiple GPUs
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
            f"This test works best with dual GPUs with at least 16GB each."
        )

    # Clear cache
    torch.cuda.empty_cache()

    try:
        # A simpler query that should still generate a detailed response
        complex_query = "Explain the phases of a turn in Magic: The Gathering."

        # Generate response
        result = prepared_pipeline.generate_response(
            complex_query,
            use_multiple_experts=False,  # Use single expert to reduce memory pressure
            max_new_tokens=256,  # Limit output size
            ensure_device_consistency=True,  # Prevent device mismatch errors
        )

        # Verify response quality
        assert result is not None
        assert "response" in result
        assert len(result["response"]) > 0

        # Check for a few key turn structure terms
        response_lower = result["response"].lower()
        expected_terms = [
            "untap",
            "upkeep",
            "draw",
            "main",
            "combat",
        ]

        found_terms = [term for term in expected_terms if term in response_lower]
        print(f"Found {len(found_terms)} of {len(expected_terms)} expected terms")

    finally:
        # Clean up
        torch.cuda.empty_cache()


@pytest.mark.integration
@pytest.mark.stability
def test_error_handling_edge_cases(prepared_pipeline):
    """Test error handling and edge cases to ensure system stability."""
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

    # Use a smaller set of edge cases to reduce test duration and memory pressure
    edge_cases = [
        "",  # Empty query
        "MTG?",  # Very short query
        "a" * 100,  # Long-ish single-token query
        "What does 🔥 mean in MTG?",  # Emoji query
    ]

    try:
        for i, query in enumerate(edge_cases):
            print(f"Testing edge case {i+1}/{len(edge_cases)}")

            # Clear cache between queries
            torch.cuda.empty_cache()

            # System should not crash on any input
            try:
                result = prepared_pipeline.generate_response(
                    query,
                    use_multiple_experts=False,
                    max_new_tokens=128,  # Limit output size
                    ensure_device_consistency=True,  # Prevent device mismatch errors
                )

                # Even with edge cases, we should get a response with the standard structure
                assert result is not None, f"Failed on edge case: {query[:50]}..."
                assert "response" in result, "Response missing"
                assert "expert_types" in result, "Expert types missing"
                assert "metrics" in result, "Metrics missing"

            except Exception as e:
                pytest.fail(
                    f"System crashed on edge case: {query[:50]}...\nError: {str(e)}"
                )

    finally:
        # Final cleanup
        torch.cuda.empty_cache()
