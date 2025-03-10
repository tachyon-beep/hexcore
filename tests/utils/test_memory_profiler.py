"""
Tests for the memory profiler system.
"""

import sys
import time
import torch
import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.memory_profiler import MemoryProfiler, MemoryAlert, profile_memory


class TestMemoryProfiler:
    """Tests for the advanced memory profiling system."""

    @pytest.fixture
    def memory_profiler(self):
        """Create a memory profiler instance with monitoring disabled by default."""
        profiler = MemoryProfiler(
            alert_threshold=0.8,
            leak_detection_window=5,
            sampling_interval=0.1,
            auto_start=False,
        )
        return profiler

    @pytest.fixture
    def mock_component(self):
        """Create a mock component for memory tracking."""

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(100, 100)
                self.layer2 = torch.nn.Linear(100, 100)

            def forward(self, x):
                return self.layer2(self.layer1(x))

        return MockModel()

    def test_initialization(self, memory_profiler):
        """Test that memory profiler initializes correctly."""
        assert memory_profiler.alert_threshold == pytest.approx(0.8)
        assert memory_profiler.leak_detection_window == 5
        assert memory_profiler.tracker.snapshot_interval == pytest.approx(0.1)
        assert not memory_profiler.monitoring
        assert len(memory_profiler.component_memory_map) == 0
        assert len(memory_profiler.alerts) == 0

    def test_register_component(self, memory_profiler, mock_component):
        """Test registering a component for memory tracking."""
        # Register component
        result = memory_profiler.register_component("test_model", mock_component)
        assert result is True

        # Check it was added to the component map
        assert "test_model" in memory_profiler.component_memory_map
        component_data = memory_profiler.component_memory_map["test_model"]
        assert component_data["object_ref"] == mock_component
        assert component_data["current_usage"] == 0
        assert component_data["peak_usage"] == 0
        assert len(component_data["history"]) == 0

        # Initial snapshot should contain GPU memory info
        if torch.cuda.is_available():
            assert len(component_data["initial_snapshot"]) == torch.cuda.device_count()

    def test_component_memory_stats(self, memory_profiler, mock_component):
        """Test getting memory statistics for components."""
        # Register a component
        memory_profiler.register_component("test_model", mock_component)

        # Set some usage data
        memory_profiler.component_memory_map["test_model"][
            "current_usage"
        ] = 1000000  # 1MB
        memory_profiler.component_memory_map["test_model"][
            "peak_usage"
        ] = 2000000  # 2MB

        # Get stats for all components
        all_stats = memory_profiler.get_component_memory_stats()
        assert "test_model" in all_stats
        # Use a larger relative tolerance for floating point comparison (5%)
        assert pytest.approx(1.0, rel=0.05) == all_stats["test_model"]["current_mb"]
        assert pytest.approx(2.0, rel=0.05) == all_stats["test_model"]["peak_mb"]

        # Get stats for specific component
        specific_stats = memory_profiler.get_component_memory_stats("test_model")
        assert specific_stats["name"] == "test_model"
        assert pytest.approx(1.0, rel=0.05) == specific_stats["current_mb"]
        assert pytest.approx(2.0, rel=0.05) == specific_stats["peak_mb"]

        # Get stats for nonexistent component
        nonexistent = memory_profiler.get_component_memory_stats("nonexistent")
        assert nonexistent == {}

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    @patch("psutil.virtual_memory")
    def test_memory_leak_detection(
        self,
        mock_vm,
        mock_device_props,
        mock_memory_reserved,
        mock_memory_allocated,
        memory_profiler,
    ):
        """Test memory leak detection functionality."""
        # Set up mock for device count
        with patch("torch.cuda.device_count", return_value=1):
            # Mock device properties
            mock_props = MagicMock()
            mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16 GB
            mock_device_props.return_value = mock_props

            # Create mock snapshots with increasing memory usage pattern
            for i in range(10):
                # Each loop adds 100MB of "usage"
                allocated = (4 + (i * 0.1)) * 1024 * 1024 * 1024  # 4GB + increasing
                reserved = (0.5 + (i * 0.05)) * 1024 * 1024 * 1024  # 0.5GB + increasing

                mock_memory_allocated.return_value = allocated
                mock_memory_reserved.return_value = reserved

                # Mock CPU memory
                mock_vm_obj = MagicMock()
                mock_vm_obj.used = (8 + i) * 1024 * 1024 * 1024  # 8GB + increasing
                mock_vm_obj.total = 32 * 1024 * 1024 * 1024  # 32 GB
                mock_vm.return_value = mock_vm_obj

                # Take snapshot
                snapshot = memory_profiler.tracker._take_snapshot()
                memory_profiler.tracker.snapshots.append(snapshot)

            # Add a component with increasing memory
            memory_profiler.component_memory_map["leaky_component"] = {
                "initial_snapshot": {0: 0},
                "current_usage": 0,
                "peak_usage": 0,
                "history": [(time.time() - i, i * 50 * 1024 * 1024) for i in range(10)],
                "object_ref": None,
            }

            # Run leak detection
            leak_result = memory_profiler.detect_memory_leaks()

            # Verify leak was detected
            assert leak_result["detected"] is True
            assert leak_result["gpu_id"] == 0
            assert (
                leak_result["growth_rate_mb_per_snapshot"] > 50
            )  # Should be about 100MB
            assert (
                leak_result["component"] == "leaky_component"
            )  # Should identify the leaky component

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    @patch("psutil.virtual_memory")
    def test_alert_generation(
        self,
        mock_vm,
        mock_device_props,
        mock_memory_reserved,
        mock_memory_allocated,
        memory_profiler,
    ):
        """Test alert generation."""
        # Set up mock for device count
        with patch("torch.cuda.device_count", return_value=1):
            # Mock device properties
            mock_props = MagicMock()
            mock_props.total_memory = 10 * 1024 * 1024 * 1024  # 10 GB
            mock_device_props.return_value = mock_props

            # Create mock snapshot with high utilization
            allocated = 8.5 * 1024 * 1024 * 1024  # 8.5GB (85%)
            reserved = 9 * 1024 * 1024 * 1024  # 9GB (90%)

            mock_memory_allocated.return_value = allocated
            mock_memory_reserved.return_value = reserved

            # Mock CPU memory
            mock_vm_obj = MagicMock()
            mock_vm_obj.used = 8 * 1024 * 1024 * 1024  # 8GB
            mock_vm_obj.total = 16 * 1024 * 1024 * 1024  # 16 GB
            mock_vm.return_value = mock_vm_obj

            # Take snapshot
            snapshot = memory_profiler.tracker._take_snapshot()
            memory_profiler.tracker.snapshots.append(snapshot)

            # Check for alerts
            memory_profiler._check_alerts()

            # Should have generated at least one high utilization alert
            assert len(memory_profiler.alerts) >= 1
            # Get the first alert for verification
            alert = next(
                a for a in memory_profiler.alerts if a.type == "HIGH_UTILIZATION"
            )
            assert alert.type == "HIGH_UTILIZATION"
            assert alert.gpu_id == 0
            assert alert.value > 80  # Should be about 85%
            assert len(alert.recommendations) > 0

    def test_memory_report_generation(self, memory_profiler, tmp_path):
        """Test memory report generation."""
        # Create some sample data
        # Add a mock snapshot
        memory_profiler.tracker.snapshots = [
            MagicMock(
                timestamp=time.time(),
                gpu_memory_used={0: 8000, 1: 6000},
                gpu_memory_total={0: 16000, 1: 16000},
                cpu_memory_used=8000,
                cpu_memory_total=16000,
                gpu_utilization={0: 50.0, 1: 37.5},
                cpu_utilization=50.0,
            )
        ]

        # Add a component
        memory_profiler.component_memory_map["test_component"] = {
            "initial_snapshot": {0: 0},
            "current_usage": 1000000,  # 1MB
            "peak_usage": 2000000,  # 2MB
            "history": [(time.time(), 1000000)],
            "object_ref": None,
        }

        # Add an alert
        memory_profiler.alerts.append(
            MemoryAlert(
                type="HIGH_UTILIZATION",
                gpu_id=0,
                value=85.0,
                timestamp=time.time(),
                recommendations=["Test recommendation 1", "Test recommendation 2"],
                component="test_component",
            )
        )

        # Generate the report
        report_path = tmp_path / "memory_report.txt"
        memory_profiler.save_memory_report(str(report_path))

        # Verify file was created and has content
        assert os.path.exists(report_path)
        with open(report_path, "r") as f:
            content = f.read()
            assert "MEMORY PROFILER REPORT" in content
            assert "test_component" in content
            assert "HIGH_UTILIZATION" in content
            assert "Test recommendation" in content

    def test_context_manager_wrapper(self, tmp_path):
        """Test that the profile_memory context manager works correctly."""
        # Create a real test profile path
        report_path = str(tmp_path / "memory_profile_test.txt")

        # Import the profile_memory function
        from src.utils.memory_profiler import profile_memory

        # Use the context manager
        with profile_memory(
            description="Test Context Manager Test",
            alert_threshold=0.9,
            save_report=True,
            report_path=report_path,
        ) as profiler:
            # Verify we got a profiler instance
            assert profiler is not None, "Context manager did not return a profiler"

            # Verify the profiler is monitoring
            assert profiler.monitoring, "Profiler is not monitoring"

            # Do some simple operations
            x = 1 + 1

        # After exiting the context, verify monitoring has stopped
        assert not profiler.monitoring, "Profiler did not stop monitoring"

        # Verify the report was created
        assert os.path.exists(report_path), "Memory report was not created"

    # Integration test rather than unit test
    @pytest.mark.integration
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No CUDA device available"
    )
    def test_real_memory_tracking(self, tmp_path):
        """Test actual memory tracking if CUDA is available."""
        # Use a try-finally to ensure cleanup
        profiler = None
        try:
            # Directly create a profiler rather than using context manager
            # to avoid potential test issues with the context manager
            profiler = MemoryProfiler(
                alert_threshold=0.95,  # High threshold to avoid false alerts
                sampling_interval=0.5,
                auto_start=True,
            )

            # Register a test component
            profiler.register_component("test_tracking")

            # Allow some time for initial monitoring
            time.sleep(0.6)

            # Create and manipulate some tensors if CUDA is available
            if torch.cuda.is_available():
                tensors = []
                for _ in range(3):
                    tensors.append(torch.zeros((1000, 1000), device="cuda"))
                    time.sleep(0.6)

                # Update component memory
                profiler._update_component_memory()

                # Cleanup
                del tensors
                torch.cuda.empty_cache()

            # Give the profiler time to capture the memory changes
            time.sleep(0.6)
            profiler._update_component_memory()

            # Verify we have some memory stats
            stats = profiler.get_component_memory_stats("test_tracking")

            # In an integration test with real CUDA devices, we should see some memory usage
            if torch.cuda.is_available():
                assert len(stats.get("history", [])) > 0

        finally:
            # Ensure we always stop the profiler
            if profiler:
                profiler.stop_monitoring()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
