#!/usr/bin/env python
"""
Demonstration script for the KV Cache Manager.

This script creates a simplified model of KV caching behavior and demonstrates
how our KV Cache Manager optimizes memory usage through sliding window attention.
"""

import torch
import time
from src.utils.kv_cache_manager import KVCacheManager
from src.utils.gpu_memory_tracker import GPUMemoryTracker, log_memory_usage


# Create mock past_key_values similar to what a model would produce
def create_mock_kv_cache(
    batch_size=1, num_heads=8, seq_length=512, head_dim=64, num_layers=4
):
    """Create mock past_key_values similar to transformer model outputs."""
    past_key_values = []
    for _ in range(num_layers):
        # Create key tensor: [batch_size, num_heads, seq_length, head_dim]
        key = torch.randn(batch_size, num_heads, seq_length, head_dim)
        # Create value tensor: [batch_size, num_heads, seq_length, head_dim]
        value = torch.randn(batch_size, num_heads, seq_length, head_dim)
        past_key_values.append((key, value))

    return tuple(past_key_values)


# Function to simulate generation with KV cache growth
def simulate_generation(
    initial_seq_length=128, final_seq_length=2048, kv_cache_manager=None
):
    """Simulate text generation with growing KV cache."""
    print("\n=== Simulating generation with KV cache ===")

    # Start memory tracking
    memory_tracker = GPUMemoryTracker(log_to_console=False)
    memory_tracker.start_monitoring()

    # Initial KV cache
    kv_cache = create_mock_kv_cache(seq_length=initial_seq_length)
    cache_sizes = [initial_seq_length]
    memory_usages = []

    # Log initial memory
    log_memory_usage("Initial KV cache")
    memory_stats = GPUMemoryTracker.memory_stats()
    if "gpu" in memory_stats and 0 in memory_stats["gpu"]:
        memory_usages.append(memory_stats["gpu"][0]["allocated_memory_gb"])

    # Simulate generation with increasing KV cache size
    for i in range(10):  # 10 steps of generation
        # Calculate new sequence length (linear growth toward final length)
        new_seq_length = min(
            initial_seq_length
            + int((final_seq_length - initial_seq_length) * (i + 1) / 10),
            final_seq_length,
        )

        # Create new kv_cache with increased sequence length
        kv_cache = create_mock_kv_cache(seq_length=new_seq_length)

        # Apply cache manager pruning if available
        if kv_cache_manager:
            print(f"Before pruning: sequence length = {new_seq_length}")
            kv_cache = kv_cache_manager.prune_cache(kv_cache)
            pruned_seq_length = kv_cache[0][0].size(2)
            print(f"After pruning: sequence length = {pruned_seq_length}")
            cache_sizes.append(pruned_seq_length)
        else:
            print(f"KV cache size: {new_seq_length} tokens")
            cache_sizes.append(new_seq_length)

        # Log memory usage
        log_memory_usage(f"Step {i+1}")
        memory_stats = GPUMemoryTracker.memory_stats()
        if "gpu" in memory_stats and 0 in memory_stats["gpu"]:
            memory_usages.append(memory_stats["gpu"][0]["allocated_memory_gb"])

        # Small delay to simulate generation time
        time.sleep(0.5)

    # Stop memory tracking
    memory_tracker.stop_monitoring()
    max_gpu_usage, max_cpu_usage = memory_tracker.get_max_memory_usage()

    # Print summary
    print("\n=== Memory Usage Summary ===")
    print(f"Maximum GPU usage: {max_gpu_usage}")
    print(f"Cache sizes: {cache_sizes}")

    return cache_sizes, memory_usages


def main():
    # Clear CUDA cache
    torch.cuda.empty_cache()

    print("Starting demonstration of KV Cache Manager...")

    # Run without cache manager
    print("\n=== WITHOUT CACHE MANAGER ===")
    unmanaged_sizes, unmanaged_memory = simulate_generation()

    # Clear CUDA cache between runs
    torch.cuda.empty_cache()

    # Create KV Cache Manager with sliding window
    kv_cache_manager = KVCacheManager(
        max_memory_percentage=0.3,
        sliding_window=512,  # Limit to 512 tokens
        auto_clear=True,
    )

    # Run with cache manager
    print("\n=== WITH CACHE MANAGER (512 token sliding window) ===")
    managed_sizes, managed_memory = simulate_generation(
        kv_cache_manager=kv_cache_manager
    )

    # Print comparison
    print("\n=== COMPARISON ===")
    print("KV Cache sizes:")
    print(f"Without manager: {unmanaged_sizes[-1]} tokens")
    print(f"With manager:    {managed_sizes[-1]} tokens")
    print(f"Reduction:       {(1 - managed_sizes[-1]/unmanaged_sizes[-1])*100:.1f}%")

    if len(unmanaged_memory) > 0 and len(managed_memory) > 0:
        print("\nMemory usage (GPU 0):")
        print(f"Without manager: {unmanaged_memory[-1]:.2f} GB")
        print(f"With manager:    {managed_memory[-1]:.2f} GB")
        if unmanaged_memory[-1] > 0:
            print(
                f"Reduction:       {(1 - managed_memory[-1]/unmanaged_memory[-1])*100:.1f}%"
            )


if __name__ == "__main__":
    main()
