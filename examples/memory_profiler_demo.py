"""
Demonstration of the memory profiler functionality.

This example shows how to:
1. Use the memory profiler to track memory usage
2. Register and monitor components
3. Detect memory leaks
4. Generate memory reports
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add the project to the path so we can import it
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.memory_profiler import MemoryProfiler, profile_memory


def demo_basic_monitoring():
    """Demonstrate basic memory monitoring."""
    print("\n=== Basic Memory Monitoring ===")

    # Create a memory profiler
    profiler = MemoryProfiler(
        alert_threshold=0.8,  # Alert at 80% memory utilization
        sampling_interval=0.5,  # Take snapshots every 0.5 seconds
    )

    # Start monitoring
    profiler.start_monitoring()
    print("Started memory monitoring...")

    # Allocate some memory
    print("Allocating tensors...")
    tensors = []
    for i in range(3):
        tensors.append(torch.ones((1000, 1000)))
        time.sleep(1)

    # Get memory summary
    summary = profiler.get_memory_usage_summary()
    print(f"\nMemory usage summary:")
    print(f"  GPU utilization: {summary.get('gpu_utilization_percent', 0):.1f}%")
    print(f"  CPU utilization: {summary.get('cpu_utilization_percent', 0):.1f}%")

    # Clean up
    print("\nReleasing tensors...")
    del tensors
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(1)

    # Stop monitoring
    profiler.stop_monitoring()
    print("Stopped memory monitoring")


def demo_component_tracking():
    """Demonstrate component-level memory tracking."""
    print("\n=== Component-Level Memory Tracking ===")

    # Create a profiler
    profiler = MemoryProfiler(sampling_interval=0.5)
    profiler.start_monitoring()

    # Create some components to track
    class ModelComponent(torch.nn.Module):
        def __init__(self, name, size):
            super().__init__()
            self.name = name
            self.linear1 = torch.nn.Linear(size, size)
            self.linear2 = torch.nn.Linear(size, size)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    # Register components
    print("Registering components...")
    model1 = ModelComponent("small_model", 100)
    model2 = ModelComponent("large_model", 1000)

    profiler.register_component("small_model", model1)
    profiler.register_component("large_model", model2)

    # Use the components to allocate memory
    print("Using components...")
    x1 = torch.randn(10, 100)
    x2 = torch.randn(10, 1000)

    y1 = model1(x1)
    time.sleep(1)

    y2 = model2(x2)
    time.sleep(1)

    # Get component stats
    print("\nComponent memory stats:")
    stats = profiler.get_component_memory_stats()
    for name, component_stats in stats.items():
        print(
            f"  {name}: {component_stats['current_mb']:.2f}MB (peak: {component_stats['peak_mb']:.2f}MB)"
        )

    # Clean up and stop monitoring
    del model1, model2, x1, x2, y1, y2
    profiler.stop_monitoring()


def demo_context_manager():
    """Demonstrate the context manager for easy profiling."""
    print("\n=== Using the Context Manager ===")

    # Ensure the logs directory exists
    os.makedirs("logs/memory", exist_ok=True)

    # Use the context manager
    with profile_memory(
        description="Matrix Multiplication",
        alert_threshold=0.9,
        save_report=True,
        report_path="logs/memory/matrix_mult_report.txt",
    ) as profiler:
        print("Running matrix multiplication operations...")

        # Register a component
        profiler.register_component("matrix_operations")

        # Run some operations
        for i in range(3):
            size = 1000 * (i + 1)
            print(f"  Creating {size}x{size} matrices...")
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            c = a @ b  # Matrix multiplication
            time.sleep(1)
            del a, b, c

    print("\nProfile completed and report saved.")
    print("Check logs/memory/matrix_mult_report.txt for the detailed report.")


def demo_leak_detection():
    """Demonstrate memory leak detection."""
    print("\n=== Memory Leak Detection ===")

    # Create a list to simulate a leak
    leaky_list = []

    # Start profiling
    profiler = MemoryProfiler(
        sampling_interval=0.2,
        leak_detection_window=10,  # Use 10 samples for leak detection
    )
    profiler.start_monitoring()

    # Register component that will leak
    profiler.register_component("leaky_component")

    print("Simulating a memory leak...")
    for i in range(15):
        # Add increasingly larger tensors to the list
        tensor = torch.randn(1000 * (i + 1), 100)
        leaky_list.append(tensor)

        # Update component memory stats
        profiler._update_component_memory()

        # Check for leaks every few iterations
        if i > 0 and i % 5 == 0:
            leak_result = profiler.detect_memory_leaks()
            if leak_result and leak_result.get("detected", False):
                growth_rate = leak_result.get("growth_rate_mb_per_snapshot", 0)
                component = leak_result.get("component")
                print(f"  Leak detected! Growth rate: {growth_rate:.1f}MB per snapshot")
                print(f"  Suspected component: {component}")

                # Get recommendations
                recommendations = profiler._generate_recommendations(
                    "LEAK_DETECTED", 0, growth_rate
                )
                print("\nRecommendations:")
                for i, rec in enumerate(recommendations):
                    print(f"  {i+1}. {rec}")

                break

        time.sleep(0.3)

    # Clean up
    del leaky_list
    profiler.stop_monitoring()


def main():
    """Run the memory profiler demonstrations."""
    print("Memory Profiler Demonstration")
    print("============================\n")

    # Print environment info
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available, using CPU only")
    print()

    # Run demos
    demo_basic_monitoring()
    demo_component_tracking()
    demo_context_manager()
    demo_leak_detection()

    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()
