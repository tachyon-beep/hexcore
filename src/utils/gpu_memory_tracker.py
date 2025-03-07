import os
import time
import torch
import psutil
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class MemorySnapshot:
    """Class to store a snapshot of memory usage."""

    timestamp: float
    gpu_memory_used: Dict[int, int]  # GPU ID -> memory used in MB
    gpu_memory_total: Dict[int, int]  # GPU ID -> total memory in MB
    cpu_memory_used: int  # Memory used in MB
    cpu_memory_total: int  # Total memory in MB

    @property
    def gpu_utilization(self) -> Dict[int, float]:
        """Return GPU utilization as a percentage."""
        return {
            gpu_id: used / total * 100
            for gpu_id, used in self.gpu_memory_used.items()
            for total in [self.gpu_memory_total[gpu_id]]
        }

    @property
    def cpu_utilization(self) -> float:
        """Return CPU utilization as a percentage."""
        return self.cpu_memory_used / self.cpu_memory_total * 100


class GPUMemoryTracker:
    """Track GPU and CPU memory usage over time."""

    def __init__(
        self,
        snapshot_interval: float = 0.5,
        log_to_console: bool = False,
        max_snapshots: int = 1000,
    ):
        """
        Initialize the memory tracker.

        Args:
            snapshot_interval: Time between memory snapshots in seconds
            log_to_console: Whether to log memory usage to console
            max_snapshots: Maximum number of snapshots to store (prevents memory issues during long runs)
        """
        self.snapshot_interval = snapshot_interval
        self.log_to_console = log_to_console
        self.max_snapshots = max_snapshots
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread = None

        # Check available GPUs
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            print("Warning: No GPUs detected. Only tracking CPU memory.")

    def start_monitoring(self):
        """Start monitoring memory usage in a background thread."""
        if self.monitoring:
            print("Memory monitoring already in progress.")
            return

        self.monitoring = True
        self.snapshots = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"Started memory monitoring with {self.snapshot_interval}s interval.")

    def stop_monitoring(self):
        """Stop monitoring memory usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2 * self.snapshot_interval)
        print(f"Stopped memory monitoring. Collected {len(self.snapshots)} snapshots.")

    def _monitor_loop(self):
        """Main monitoring loop that runs in a background thread."""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)

                # Limit the number of snapshots to prevent memory issues
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots = self.snapshots[-self.max_snapshots :]

                if self.log_to_console:
                    self._log_snapshot(snapshot)

                time.sleep(self.snapshot_interval)

            except Exception as e:
                print(f"Error in memory monitoring: {e}")
                self.monitoring = False
                break

    def _take_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory usage."""
        timestamp = time.time()

        # Get GPU memory info
        gpu_memory_used = {}
        gpu_memory_total = {}

        for gpu_id in range(self.num_gpus):
            memory_info = torch.cuda.get_device_properties(gpu_id).total_memory
            memory_used = torch.cuda.memory_allocated(gpu_id)
            memory_cached = torch.cuda.memory_reserved(gpu_id)

            # Convert bytes to MB
            gpu_memory_total[gpu_id] = memory_info // (1024 * 1024)
            gpu_memory_used[gpu_id] = (memory_used + memory_cached) // (1024 * 1024)

        # Get CPU memory info
        cpu_memory = psutil.virtual_memory()
        cpu_memory_used = cpu_memory.used // (1024 * 1024)  # MB
        cpu_memory_total = cpu_memory.total // (1024 * 1024)  # MB

        return MemorySnapshot(
            timestamp=timestamp,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            cpu_memory_used=cpu_memory_used,
            cpu_memory_total=cpu_memory_total,
        )

    def _log_snapshot(self, snapshot: MemorySnapshot):
        """Log a memory snapshot to console."""
        print("\n----- Memory Usage -----")
        for gpu_id, used in snapshot.gpu_memory_used.items():
            total = snapshot.gpu_memory_total[gpu_id]
            util = snapshot.gpu_utilization[gpu_id]
            print(f"GPU {gpu_id}: {used}MB / {total}MB ({util:.1f}%)")

        print(
            f"CPU: {snapshot.cpu_memory_used}MB / {snapshot.cpu_memory_total}MB ({snapshot.cpu_utilization:.1f}%)"
        )
        print("------------------------\n")

    def get_max_memory_usage(self) -> Tuple[Dict[int, int], int]:
        """Get the maximum memory usage observed for GPUs and CPU."""
        if not self.snapshots:
            return {}, 0

        max_gpu_usage = {}
        for gpu_id in range(self.num_gpus):
            max_gpu_usage[gpu_id] = max(
                snapshot.gpu_memory_used.get(gpu_id, 0) for snapshot in self.snapshots
            )

        max_cpu_usage = max(snapshot.cpu_memory_used for snapshot in self.snapshots)

        return max_gpu_usage, max_cpu_usage

    def plot_memory_usage(self, save_path: Optional[str] = None):
        """
        Plot the memory usage over time.

        Args:
            save_path: If provided, save the plot to this path instead of displaying
        """
        if not self.snapshots:
            print("No memory snapshots to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Extract timestamps and convert to relative time in seconds
        start_time = self.snapshots[0].timestamp
        timestamps = [(snapshot.timestamp - start_time) for snapshot in self.snapshots]

        # Plot GPU memory usage
        for gpu_id in range(self.num_gpus):
            memory_usage = [
                snapshot.gpu_memory_used.get(gpu_id, 0) for snapshot in self.snapshots
            ]
            memory_total = [
                snapshot.gpu_memory_total.get(gpu_id, 0) for snapshot in self.snapshots
            ]

            # Only plot if we have data for this GPU
            if any(memory_usage):
                ax1.plot(timestamps, memory_usage, label=f"GPU {gpu_id}")
                ax1.axhline(
                    y=memory_total[0], linestyle="--", alpha=0.3, color=f"C{gpu_id}"
                )

        ax1.set_title("GPU Memory Usage Over Time")
        ax1.set_ylabel("Memory Usage (MB)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot CPU memory usage
        cpu_usage = [snapshot.cpu_memory_used for snapshot in self.snapshots]
        cpu_total = self.snapshots[0].cpu_memory_total

        ax2.plot(timestamps, cpu_usage, label="CPU")
        ax2.axhline(y=cpu_total, linestyle="--", alpha=0.3, color="C0")

        ax2.set_title("CPU Memory Usage Over Time")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Memory usage plot saved to {save_path}")
        else:
            plt.show()

    def save_report(self, file_path: str):
        """
        Save a detailed memory usage report to a file.

        Args:
            file_path: Path to save the report
        """
        if not self.snapshots:
            print("No memory snapshots to report.")
            return

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        max_gpu_usage, max_cpu_usage = self.get_max_memory_usage()

        with open(file_path, "w") as f:
            f.write("=" * 50 + "\n")
            f.write("MEMORY USAGE REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of snapshots: {len(self.snapshots)}\n")
            f.write(
                f"Monitoring duration: {(self.snapshots[-1].timestamp - self.snapshots[0].timestamp):.2f} seconds\n\n"
            )

            f.write("-" * 50 + "\n")
            f.write("MAXIMUM MEMORY USAGE\n")
            f.write("-" * 50 + "\n\n")

            for gpu_id, max_usage in max_gpu_usage.items():
                total = self.snapshots[0].gpu_memory_total[gpu_id]
                utilization = max_usage / total * 100
                f.write(
                    f"GPU {gpu_id}: {max_usage}MB / {total}MB ({utilization:.1f}%)\n"
                )

            cpu_total = self.snapshots[0].cpu_memory_total
            cpu_utilization = max_cpu_usage / cpu_total * 100
            f.write(
                f"CPU: {max_cpu_usage}MB / {cpu_total}MB ({cpu_utilization:.1f}%)\n\n"
            )

            f.write("-" * 50 + "\n")
            f.write("MEMORY USAGE OVER TIME\n")
            f.write("-" * 50 + "\n\n")

            f.write("Time(s),")
            for gpu_id in range(self.num_gpus):
                f.write(f"GPU{gpu_id}(MB),")
            f.write("CPU(MB)\n")

            start_time = self.snapshots[0].timestamp
            for snapshot in self.snapshots:
                relative_time = snapshot.timestamp - start_time
                f.write(f"{relative_time:.2f},")

                for gpu_id in range(self.num_gpus):
                    f.write(f"{snapshot.gpu_memory_used.get(gpu_id, 0)},")

                f.write(f"{snapshot.cpu_memory_used}\n")

            print(f"Memory usage report saved to {file_path}")

    @staticmethod
    def memory_stats() -> Dict[str, any]:
        """
        Get current memory statistics (static method for quick access).

        Returns:
            Dictionary with current memory usage information
        """
        stats = {}

        # Get CPU memory info
        cpu = psutil.virtual_memory()
        stats["cpu_used_gb"] = cpu.used / (1024**3)
        stats["cpu_total_gb"] = cpu.total / (1024**3)
        stats["cpu_percent"] = cpu.percent

        # Get GPU info if available
        gpu_count = torch.cuda.device_count()
        stats["gpu_count"] = gpu_count

        if gpu_count > 0:
            stats["gpu"] = {}
            for i in range(gpu_count):
                gpu_stats = {}
                props = torch.cuda.get_device_properties(i)
                gpu_stats["name"] = props.name
                gpu_stats["total_memory_gb"] = props.total_memory / (1024**3)
                gpu_stats["allocated_memory_gb"] = torch.cuda.memory_allocated(i) / (
                    1024**3
                )
                gpu_stats["reserved_memory_gb"] = torch.cuda.memory_reserved(i) / (
                    1024**3
                )
                gpu_stats["percent_used"] = (
                    gpu_stats["allocated_memory_gb"] / gpu_stats["total_memory_gb"]
                ) * 100

                stats["gpu"][i] = gpu_stats

        return stats


# Context manager for convenient usage
def track_memory(
    description: str = "Memory tracking",
    interval: float = 0.5,
    log_to_console: bool = False,
    save_plot: bool = True,
    save_report: bool = True,
):
    """
    Context manager for tracking memory usage during a code block.

    Args:
        description: Description for this tracking session (used for filenames)
        interval: Snapshot interval in seconds
        log_to_console: Whether to log to console
        save_plot: Whether to save a plot
        save_report: Whether to save a report

    Example:
        with track_memory("Model Loading"):
            model = load_large_model()
    """

    class MemoryTrackingContext:
        def __init__(
            self, description, interval, log_to_console, save_plot, save_report
        ):
            self.description = description
            self.interval = interval
            self.log_to_console = log_to_console
            self.save_plot = save_plot
            self.save_report = save_report
            self.tracker = None

            # Clean up description for filenames
            self.file_desc = description.lower().replace(" ", "_")
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def __enter__(self):
            self.tracker = GPUMemoryTracker(
                snapshot_interval=self.interval, log_to_console=self.log_to_console
            )
            self.tracker.start_monitoring()
            return self.tracker

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.tracker:
                self.tracker.stop_monitoring()

                if self.save_plot:
                    plot_path = f"./logs/memory/{self.timestamp}_{self.file_desc}.png"
                    self.tracker.plot_memory_usage(save_path=plot_path)

                if self.save_report:
                    report_path = f"./logs/memory/{self.timestamp}_{self.file_desc}.txt"
                    self.tracker.save_report(file_path=report_path)

                # Print summary
                max_gpu, max_cpu = self.tracker.get_max_memory_usage()
                print(f"\nMemory Usage Summary ({self.description}):")
                for gpu_id, max_usage in max_gpu.items():
                    total = list(self.tracker.snapshots[0].gpu_memory_total.values())[0]
                    print(
                        f"  GPU {gpu_id} peak: {max_usage}MB / {total}MB ({max_usage/total*100:.1f}%)"
                    )
                print(f"  CPU peak: {max_cpu}MB\n")

    return MemoryTrackingContext(
        description, interval, log_to_console, save_plot, save_report
    )
