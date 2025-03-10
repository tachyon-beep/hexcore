"""
Advanced memory profiling system with leak detection and alerts.

This module provides comprehensive memory monitoring capabilities:
- Real-time memory usage tracking across GPUs and CPU
- Memory leak detection through trend analysis
- Component-level memory attribution
- Alert system for critical memory conditions
- Optimization recommendations based on detected patterns
"""

import os
import time
import torch
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Set, Callable
from dataclasses import dataclass

from src.utils.gpu_memory_tracker import GPUMemoryTracker, MemorySnapshot

logger = logging.getLogger(__name__)


@dataclass
class MemoryAlert:
    """Class representing a memory alert."""

    type: str  # Alert type e.g., "HIGH_UTILIZATION", "LEAK_DETECTED", "FRAGMENTATION"
    gpu_id: Optional[int]  # GPU ID or None for CPU alerts
    value: float  # Relevant value (utilization percentage, leak rate, etc.)
    timestamp: float  # When the alert was triggered
    recommendations: List[str]  # List of suggested actions
    component: Optional[str] = None  # Component name if known


class MemoryProfiler:
    """
    Advanced memory profiling system with leak detection and alerts.

    This class builds on the GPUMemoryTracker to provide more advanced
    monitoring capabilities including leak detection, component-level
    attribution, and actionable recommendations.
    """

    def __init__(
        self,
        alert_threshold: float = 0.85,
        leak_detection_window: int = 20,
        sampling_interval: float = 1.0,
        auto_start: bool = True,
        log_to_console: bool = False,
    ):
        """
        Initialize the memory profiler.

        Args:
            alert_threshold: Memory utilization threshold (0-1) that triggers alerts
            leak_detection_window: Number of samples to use for trend analysis
            sampling_interval: Time between memory snapshots in seconds
            auto_start: Whether to start monitoring automatically
            log_to_console: Whether to log memory statistics to console
        """
        self.tracker = GPUMemoryTracker(
            snapshot_interval=sampling_interval, log_to_console=log_to_console
        )
        self.alert_threshold = alert_threshold
        self.leak_detection_window = leak_detection_window
        self.component_memory_map: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[MemoryAlert] = []
        self.alert_handlers: List[Callable[[MemoryAlert], None]] = []
        self.monitoring = False
        self.monitor_thread = None
        self.dashboard_thread = None

        # Alert throttling to prevent alert storms
        self.alert_cooldown: Dict[str, float] = {}
        self.alert_cooldown_period = 60.0  # seconds between repeated alerts

        if auto_start:
            self.start_monitoring()

    def start_monitoring(self):
        """Start the memory monitoring process."""
        if self.monitoring:
            logger.warning("Memory profiler already running")
            return

        # Start the base tracker
        self.tracker.start_monitoring()

        # Start monitoring thread for profiler-specific operations
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.info(
            f"Started memory profiler (alert threshold: {self.alert_threshold*100}%)"
        )

    def stop_monitoring(self):
        """Stop the memory monitoring process."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        # Stop the base tracker
        self.tracker.stop_monitoring()

        logger.info("Stopped memory profiler")

    def _monitoring_loop(self):
        """Main monitoring loop that runs checks periodically."""
        while self.monitoring:
            try:
                # Only run checks if we have enough data
                if len(self.tracker.snapshots) > 0:
                    # Check for alerts
                    self._check_alerts()

                    # Check for leaks
                    if len(self.tracker.snapshots) >= self.leak_detection_window:
                        leak_result = self.detect_memory_leaks()
                        if leak_result and leak_result["detected"]:
                            self._create_alert(
                                alert_type="LEAK_DETECTED",
                                gpu_id=leak_result.get("gpu_id", 0),
                                value=leak_result["growth_rate_mb_per_snapshot"],
                                component=leak_result.get("component"),
                            )

                    # Update component memory usage
                    self._update_component_memory()

                # Sleep until next check
                time.sleep(self.tracker.snapshot_interval * 2)

            except Exception as e:
                logger.error(f"Error in memory profiler monitoring loop: {e}")

    def register_component(self, component_name: str, component_obj: Any = None):
        """
        Register a component for memory tracking.

        Args:
            component_name: Name to identify this component
            component_obj: The actual component object (optional)
        """
        # Take initial snapshot
        initial_snapshot = self._take_component_snapshot(component_obj)

        self.component_memory_map[component_name] = {
            "initial_snapshot": initial_snapshot,
            "current_usage": 0,
            "peak_usage": 0,
            "history": [],
            "object_ref": component_obj,
        }

        logger.debug(f"Registered component '{component_name}' for memory tracking")

        return True

    def _take_component_snapshot(self, component_obj: Any = None) -> Dict[int, int]:
        """
        Take memory snapshot that can be attributed to a component.

        Args:
            component_obj: Component object to analyze

        Returns:
            Dictionary mapping device IDs to memory usage in bytes
        """
        # Clear cache first to get accurate reading
        torch.cuda.empty_cache()

        # Get current memory state
        memory_snapshot = {
            i: torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
        }

        # If component provided, ensure it's on GPU to get accurate reading
        if component_obj is not None and hasattr(component_obj, "parameters"):
            try:
                # Try to get device from the component's parameters
                params = list(component_obj.parameters())
                if params:
                    device = params[0].device
                    if device.type != "cuda":
                        logger.debug(f"Component is not on CUDA device: {device}")
            except Exception as e:
                logger.debug(f"Error checking component device: {e}")

        return memory_snapshot

    def _update_component_memory(self):
        """Update memory usage statistics for all registered components."""
        if not self.component_memory_map:
            return

        # Get current memory state
        current_memory = {
            i: torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())
        }

        # Update each component's stats
        timestamp = time.time()
        for component_name, stats in self.component_memory_map.items():
            if not stats["initial_snapshot"]:
                continue

            # Estimate component's current usage
            if stats["object_ref"] is not None:
                # Try to measure directly if object is available
                try:
                    if hasattr(stats["object_ref"], "parameters"):
                        # Get memory usage by summing tensor sizes
                        usage = sum(
                            p.numel() * p.element_size()
                            for p in stats["object_ref"].parameters()
                        )
                    else:
                        # Default method
                        usage = sum(current_memory.values()) - sum(
                            stats["initial_snapshot"].values()
                        )

                except Exception as e:
                    logger.debug(f"Error measuring component memory: {e}")
                    usage = sum(current_memory.values()) - sum(
                        stats["initial_snapshot"].values()
                    )
            else:
                # If no object ref, estimate based on differences
                usage = sum(current_memory.values()) - sum(
                    stats["initial_snapshot"].values()
                )

            # Update stats
            usage = max(0, usage)  # Ensure non-negative
            stats["current_usage"] = usage
            stats["peak_usage"] = max(stats["peak_usage"], usage)
            stats["history"].append((timestamp, usage))

            # Keep history to a reasonable size
            if len(stats["history"]) > 1000:
                stats["history"] = stats["history"][-1000:]

    def detect_memory_leaks(self) -> Optional[Dict[str, Any]]:
        """
        Analyze memory patterns to detect potential leaks.

        Returns:
            Dictionary with leak detection results or None if not enough data
        """
        if len(self.tracker.snapshots) < self.leak_detection_window:
            return None  # Not enough data

        # Extract the last N snapshots
        recent_snapshots = self.tracker.snapshots[-self.leak_detection_window :]

        # Check each GPU for leak patterns
        results = {}
        for gpu_id in range(torch.cuda.device_count()):
            # Extract memory usage for this GPU
            memory_usage = [
                snapshot.gpu_memory_used.get(gpu_id, 0)
                for snapshot in recent_snapshots
                if gpu_id in snapshot.gpu_memory_used
            ]

            if not memory_usage or len(memory_usage) < self.leak_detection_window:
                continue

            # Check for monotonic increases (potential leak indicator)
            increases = [
                memory_usage[i + 1] - memory_usage[i]
                for i in range(len(memory_usage) - 1)
            ]

            # Count positive increases
            positive_increases = sum(1 for inc in increases if inc > 0)

            # If more than 80% of snapshots show increasing memory, potential leak
            if positive_increases / len(increases) > 0.8:
                # Calculate rate of change
                first_mem = memory_usage[0]
                last_mem = memory_usage[-1]
                growth_rate = (last_mem - first_mem) / self.leak_detection_window

                # If steady growth exceeds threshold, likely a leak
                if growth_rate > 50:  # 50MB per snapshot threshold
                    # Try to determine which component might be leaking
                    leaking_component = self._identify_leaking_component()

                    # Calculate time to OOM
                    total_memory = recent_snapshots[-1].gpu_memory_total.get(gpu_id, 0)
                    available_memory = total_memory - last_mem
                    time_to_oom = (
                        available_memory / growth_rate
                        if growth_rate > 0
                        else float("inf")
                    )

                    results = {
                        "detected": True,
                        "gpu_id": gpu_id,
                        "growth_rate_mb_per_snapshot": growth_rate,
                        "estimated_time_to_oom_snapshots": time_to_oom,
                        "estimated_time_to_oom_seconds": time_to_oom
                        * self.tracker.snapshot_interval,
                        "component": leaking_component,
                    }

                    # One detected leak is enough to report
                    return results

        # Check for CPU leaks as well
        cpu_memory_usage = [snapshot.cpu_memory_used for snapshot in recent_snapshots]

        increases = [
            cpu_memory_usage[i + 1] - cpu_memory_usage[i]
            for i in range(len(cpu_memory_usage) - 1)
        ]

        positive_increases = sum(1 for inc in increases if inc > 0)

        if positive_increases / len(increases) > 0.8:
            first_mem = cpu_memory_usage[0]
            last_mem = cpu_memory_usage[-1]
            growth_rate = (last_mem - first_mem) / self.leak_detection_window

            if growth_rate > 100:  # 100MB per snapshot threshold for CPU
                return {
                    "detected": True,
                    "gpu_id": None,  # None indicates CPU
                    "growth_rate_mb_per_snapshot": growth_rate,
                    "component": self._identify_leaking_component(is_cpu=True),
                }

        # No leaks detected
        return {"detected": False}

    def _identify_leaking_component(self, is_cpu: bool = False) -> Optional[str]:
        """
        Try to identify which component might be leaking memory.

        Args:
            is_cpu: Whether to check for CPU memory leaks

        Returns:
            Name of the suspected leaking component or None
        """
        if not self.component_memory_map:
            return None

        # Look for components with consistent growth pattern
        suspect_components = []

        for component_name, stats in self.component_memory_map.items():
            if len(stats["history"]) < self.leak_detection_window:
                continue

            # Get memory history for this component
            history = stats["history"][-self.leak_detection_window :]
            memory_usage = [mem for _, mem in history]

            # Check for consistent growth
            increases = [
                memory_usage[i + 1] - memory_usage[i]
                for i in range(len(memory_usage) - 1)
            ]

            positive_increases = sum(1 for inc in increases if inc > 0)

            if positive_increases / len(increases) > 0.8:
                # Calculate growth rate
                growth_rate = (memory_usage[-1] - memory_usage[0]) / len(memory_usage)

                if growth_rate > 0:
                    suspect_components.append((component_name, growth_rate))

        # Return the component with the highest growth rate, if any
        if suspect_components:
            suspect_components.sort(key=lambda x: x[1], reverse=True)
            return suspect_components[0][0]

        return None

    def _check_alerts(self):
        """Check for memory conditions that should trigger alerts."""
        if not self.tracker.snapshots:
            return

        current_snapshot = self.tracker.snapshots[-1]
        now = time.time()

        # Check for high memory utilization
        for gpu_id, util in current_snapshot.gpu_utilization.items():
            alert_key = f"HIGH_UTILIZATION_GPU{gpu_id}"

            # Skip if we recently alerted for this condition
            if (
                alert_key in self.alert_cooldown
                and now - self.alert_cooldown[alert_key] < self.alert_cooldown_period
            ):
                continue

            if util > self.alert_threshold * 100:
                self._create_alert(
                    alert_type="HIGH_UTILIZATION", gpu_id=gpu_id, value=util
                )
                self.alert_cooldown[alert_key] = now

        # Check for memory fragmentation
        for gpu_id in range(torch.cuda.device_count()):
            alert_key = f"FRAGMENTATION_GPU{gpu_id}"

            # Skip if we recently alerted for this condition
            if (
                alert_key in self.alert_cooldown
                and now - self.alert_cooldown[alert_key] < self.alert_cooldown_period
            ):
                continue

            allocated = torch.cuda.memory_allocated(gpu_id)
            reserved = torch.cuda.memory_reserved(gpu_id)

            if reserved > 0:
                fragmentation = (reserved - allocated) / reserved
                if fragmentation > 0.3:  # 30% fragmentation threshold
                    self._create_alert(
                        alert_type="FRAGMENTATION",
                        gpu_id=gpu_id,
                        value=fragmentation * 100,  # Convert to percentage
                    )
                    self.alert_cooldown[alert_key] = now

    def _create_alert(
        self,
        alert_type: str,
        gpu_id: Optional[int],
        value: float,
        component: Optional[str] = None,
    ):
        """
        Create and process a memory alert.

        Args:
            alert_type: Type of alert
            gpu_id: GPU ID or None for CPU
            value: Alert value
            component: Component name if known
        """
        # Generate recommendations based on alert type
        recommendations = self._generate_recommendations(alert_type, gpu_id, value)

        # Create alert object
        alert = MemoryAlert(
            type=alert_type,
            gpu_id=gpu_id,
            value=value,
            timestamp=time.time(),
            recommendations=recommendations,
            component=component,
        )

        # Add to alerts history
        self.alerts.append(alert)

        # Log the alert
        device_str = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
        component_str = f" (component: {component})" if component else ""
        logger.warning(
            f"Memory Alert: {alert_type} on {device_str}{component_str} - {value:.1f}"
        )

        # Log recommendations
        for i, rec in enumerate(recommendations):
            logger.info(f"Recommendation {i+1}: {rec}")

        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    def _generate_recommendations(
        self, alert_type: str, gpu_id: Optional[int], value: float
    ) -> List[str]:
        """
        Generate recommendations based on the alert type.

        Args:
            alert_type: Type of alert
            gpu_id: GPU ID or None for CPU
            value: Alert value

        Returns:
            List of recommendation strings
        """
        if alert_type == "HIGH_UTILIZATION":
            return [
                "Consider offloading inactive experts to CPU",
                "Reduce batch size or sequence length",
                "Check for tensor accumulation in any loops",
                "Implement gradient checkpointing if training",
                "Clear GPU cache after major operations",
            ]
        elif alert_type == "FRAGMENTATION":
            return [
                "Call torch.cuda.empty_cache() to reclaim unused memory",
                "Restructure operations to reduce memory fragmentation",
                "If using quantization, check alignment requirements",
                "Consider using contiguous tensors where possible",
            ]
        elif alert_type == "LEAK_DETECTED":
            return [
                "Check for tensors stored in global variables",
                "Verify all forward hook references are properly released",
                "Look for accidental accumulation in lists or caches",
                "Ensure KV caches are being properly pruned",
                "Verify all references are cleared when switching experts",
            ]
        else:
            return ["Monitor memory usage", "Check for unusual patterns"]

    def register_alert_handler(self, handler: Callable[[MemoryAlert], None]):
        """
        Register a callback function to be called when alerts are generated.

        Args:
            handler: Function that takes an alert object as parameter
        """
        self.alert_handlers.append(handler)

    def get_component_memory_stats(
        self, component_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get memory statistics for components.

        Args:
            component_name: Specific component to get stats for, or None for all

        Returns:
            Dictionary with memory statistics
        """
        if component_name and component_name in self.component_memory_map:
            # Return stats for single component
            stats = self.component_memory_map[component_name]
            return {
                "name": component_name,
                "current_mb": stats["current_usage"] / (1024 * 1024),
                "peak_mb": stats["peak_usage"] / (1024 * 1024),
                "history": [(ts, mem / (1024 * 1024)) for ts, mem in stats["history"]],
            }
        elif component_name:
            # Component not found
            return {}
        else:
            # Return stats for all components
            return {
                name: {
                    "current_mb": stats["current_usage"] / (1024 * 1024),
                    "peak_mb": stats["peak_usage"] / (1024 * 1024),
                    "history_points": len(stats["history"]),
                }
                for name, stats in self.component_memory_map.items()
            }

    def get_alerts(
        self, limit: int = 100, alert_type: Optional[str] = None
    ) -> List[MemoryAlert]:
        """
        Get recent memory alerts.

        Args:
            limit: Maximum number of alerts to return
            alert_type: Filter by alert type or None for all

        Returns:
            List of alerts
        """
        if alert_type:
            filtered_alerts = [a for a in self.alerts if a.type == alert_type]
            return filtered_alerts[-limit:]
        else:
            return self.alerts[-limit:]

    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current memory usage.

        Returns:
            Dictionary with memory usage summary
        """
        if not self.tracker.snapshots:
            return {}

        last_snapshot = self.tracker.snapshots[-1]

        # Calculate total GPU memory
        total_gpu_used = sum(last_snapshot.gpu_memory_used.values())
        total_gpu_available = sum(last_snapshot.gpu_memory_total.values())

        return {
            "timestamp": last_snapshot.timestamp,
            "total_gpu_used_mb": total_gpu_used,
            "total_gpu_available_mb": total_gpu_available,
            "gpu_utilization_percent": (
                total_gpu_used / total_gpu_available * 100
                if total_gpu_available > 0
                else 0
            ),
            "cpu_used_mb": last_snapshot.cpu_memory_used,
            "cpu_total_mb": last_snapshot.cpu_memory_total,
            "cpu_utilization_percent": last_snapshot.cpu_utilization,
            "per_gpu": {
                gpu_id: {
                    "used_mb": used,
                    "total_mb": last_snapshot.gpu_memory_total[gpu_id],
                    "utilization_percent": (
                        used / last_snapshot.gpu_memory_total[gpu_id] * 100
                        if last_snapshot.gpu_memory_total[gpu_id] > 0
                        else 0
                    ),
                }
                for gpu_id, used in last_snapshot.gpu_memory_used.items()
            },
            "per_component": self.get_component_memory_stats(),
            "alert_count": len(self.alerts),
        }

    def save_memory_report(
        self,
        file_path: str,
        include_alerts: bool = True,
        include_recommendations: bool = True,
    ):
        """
        Save a comprehensive memory report to a file.

        Args:
            file_path: Path to save the report
            include_alerts: Whether to include alert history
            include_recommendations: Whether to include recommendations
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MEMORY PROFILER REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Memory usage summary
            summary = self.get_memory_usage_summary()
            if summary:
                f.write("\n" + "-" * 40 + "\n")
                f.write("MEMORY USAGE SUMMARY\n")
                f.write("-" * 40 + "\n\n")

                f.write(
                    f"Total GPU Memory: {summary['total_gpu_used_mb']:.1f}MB / {summary['total_gpu_available_mb']:.1f}MB "
                )
                f.write(f"({summary['gpu_utilization_percent']:.1f}%)\n")

                f.write(
                    f"CPU Memory: {summary['cpu_used_mb']:.1f}MB / {summary['cpu_total_mb']:.1f}MB "
                )
                f.write(f"({summary['cpu_utilization_percent']:.1f}%)\n\n")

                f.write("Per-GPU Memory:\n")
                for gpu_id, stats in summary["per_gpu"].items():
                    f.write(
                        f"  GPU {gpu_id}: {stats['used_mb']:.1f}MB / {stats['total_mb']:.1f}MB "
                    )
                    f.write(f"({stats['utilization_percent']:.1f}%)\n")

                f.write("\n")

            # Component memory usage
            if self.component_memory_map:
                f.write("\n" + "-" * 40 + "\n")
                f.write("COMPONENT MEMORY USAGE\n")
                f.write("-" * 40 + "\n\n")

                for name, stats in self.component_memory_map.items():
                    current_mb = stats["current_usage"] / (1024 * 1024)
                    peak_mb = stats["peak_usage"] / (1024 * 1024)
                    f.write(f"{name}: {current_mb:.1f}MB (peak: {peak_mb:.1f}MB)\n")

                f.write("\n")

            # Alert history
            if include_alerts and self.alerts:
                f.write("\n" + "-" * 40 + "\n")
                f.write("ALERT HISTORY\n")
                f.write("-" * 40 + "\n\n")

                for i, alert in enumerate(self.alerts[-20:]):  # Last 20 alerts
                    device = (
                        f"GPU {alert.gpu_id}" if alert.gpu_id is not None else "CPU"
                    )
                    ts = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                    f.write(f"[{ts}] {alert.type} on {device}: {alert.value:.1f}\n")

                    if include_recommendations:
                        for rec in alert.recommendations:
                            f.write(f"  - {rec}\n")

                    if i < len(self.alerts) - 1:
                        f.write("\n")

            # Leak detection analysis
            leak_result = self.detect_memory_leaks()
            if leak_result:
                f.write("\n" + "-" * 40 + "\n")
                f.write("LEAK DETECTION ANALYSIS\n")
                f.write("-" * 40 + "\n\n")

                if leak_result["detected"]:
                    device = (
                        f"GPU {leak_result['gpu_id']}"
                        if leak_result.get("gpu_id") is not None
                        else "CPU"
                    )
                    f.write(f"⚠️ Memory leak detected on {device}\n")
                    f.write(
                        f"Growth rate: {leak_result['growth_rate_mb_per_snapshot']:.1f}MB per snapshot\n"
                    )

                    if "estimated_time_to_oom_seconds" in leak_result:
                        oom_time_min = leak_result["estimated_time_to_oom_seconds"] / 60
                        f.write(f"Estimated time to OOM: {oom_time_min:.1f} minutes\n")

                    if leak_result.get("component"):
                        f.write(f"Suspected component: {leak_result['component']}\n")
                else:
                    f.write("No memory leaks detected.\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Memory profiler report saved to {file_path}")


# Context manager for convenient profiling
def profile_memory(
    description: str = "Memory profiling",
    alert_threshold: float = 0.85,
    save_report: bool = True,
    report_path: Optional[str] = None,
):
    """
    Context manager for memory profiling during a code block.

    Args:
        description: Description for this profiling session
        alert_threshold: Memory utilization threshold for alerts
        save_report: Whether to save a report after profiling
        report_path: Custom path for the report or None for auto-generated

    Example:
        with profile_memory("Model Loading"):
            model = load_large_model()
    """

    class MemoryProfilingContext:
        def __init__(self, description, alert_threshold, save_report, report_path):
            self.description = description
            self.alert_threshold = alert_threshold
            self.save_report = save_report
            self.report_path = report_path
            self.profiler = None

            # Clean up description for filenames
            self.file_desc = description.lower().replace(" ", "_")
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def __enter__(self):
            self.profiler = MemoryProfiler(alert_threshold=self.alert_threshold)
            self.profiler.start_monitoring()
            return self.profiler

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.profiler:
                self.profiler.stop_monitoring()

                if self.save_report:
                    if self.report_path:
                        report_path = self.report_path
                    else:
                        report_path = f"./logs/memory/{self.timestamp}_{self.file_desc}_report.txt"

                    # Ensure directory exists
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    self.profiler.save_memory_report(report_path)

                # Print summary
                logger.info(f"\nMemory Profiling Summary ({self.description}):")

                summary = self.profiler.get_memory_usage_summary()
                if summary:
                    logger.info(
                        f"  GPU Memory: {summary['total_gpu_used_mb']:.1f}MB / "
                        f"{summary['total_gpu_available_mb']:.1f}MB "
                        f"({summary['gpu_utilization_percent']:.1f}%)"
                    )

                    # Report any alerts
                    alerts = self.profiler.get_alerts(limit=5)
                    if alerts:
                        logger.info(f"  Alerts: {len(alerts)} memory alerts detected")

                # Report leak detection
                leak_result = self.profiler.detect_memory_leaks()
                if leak_result and leak_result["detected"]:
                    device_str = (
                        f"GPU {leak_result['gpu_id']}"
                        if leak_result.get("gpu_id") is not None
                        else "CPU"
                    )
                    time_to_oom = (
                        leak_result.get("estimated_time_to_oom_seconds", 0) / 60
                    )
                    component_str = (
                        f" (component: {leak_result['component']})"
                        if leak_result.get("component")
                        else ""
                    )

                    logger.warning(
                        f"⚠️ Memory leak detected on {device_str}{component_str}"
                    )
                    logger.warning(
                        f"  Growth rate: {leak_result['growth_rate_mb_per_snapshot']:.1f}MB per snapshot"
                    )
                    if time_to_oom > 0:
                        logger.warning(
                            f"  Estimated time to OOM: {time_to_oom:.1f} minutes"
                        )

            # Return False to allow any exceptions to propagate
            return False

    # Return the context manager instance
    return MemoryProfilingContext(
        description, alert_threshold, save_report, report_path
    )
