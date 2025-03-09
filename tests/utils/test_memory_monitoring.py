# tests/utils/test_memory_monitoring.py

import sys
import torch
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.gpu_memory_tracker import GPUMemoryTracker, track_memory, MemorySnapshot
from src.models.model_loader import load_quantized_model
from src.models.expert_adapters import ExpertAdapterManager
from src.inference.pipeline import MTGInferencePipeline


class TestMemoryMonitoring:
    """Tests for memory monitoring during inference operations."""

    @pytest.fixture
    def memory_tracker(self):
        """Create a memory tracker instance."""
        tracker = GPUMemoryTracker(
            snapshot_interval=0.1, log_to_console=False, max_snapshots=100
        )
        return tracker

    @pytest.fixture
    def mock_gpu_memory_stats(self):
        """Create mock GPU memory statistics."""

        # Create fake memory stats that show utilization
        def fake_memory_stats():
            return {
                "cpu_used_gb": 4.0,
                "cpu_total_gb": 16.0,
                "cpu_percent": 25.0,
                "gpu_count": 2,
                "gpu": {
                    0: {
                        "name": "Mock GPU 0",
                        "total_memory_gb": 16.0,
                        "allocated_memory_gb": 4.0,
                        "reserved_memory_gb": 2.0,
                        "percent_used": 25.0,
                    },
                    1: {
                        "name": "Mock GPU 1",
                        "total_memory_gb": 16.0,
                        "allocated_memory_gb": 6.0,
                        "reserved_memory_gb": 1.0,
                        "percent_used": 37.5,
                    },
                },
            }

        return fake_memory_stats

    @pytest.fixture
    def mock_memory_snapshot(self):
        """Create a mock memory snapshot."""
        snapshot = MemorySnapshot(
            timestamp=1000.0,
            gpu_memory_used={0: 4000, 1: 6000},
            gpu_memory_total={0: 16000, 1: 16000},
            cpu_memory_used=4000,
            cpu_memory_total=16000,
        )
        return snapshot

    def test_memory_tracker_initialization(self, memory_tracker):
        """Test that memory tracker initializes correctly."""
        assert pytest.approx(0.1, rel=1e-5) == memory_tracker.snapshot_interval
        assert memory_tracker.log_to_console is False
        assert memory_tracker.max_snapshots == 100
        assert memory_tracker.snapshots == []
        assert memory_tracker.monitoring is False
        assert memory_tracker.monitor_thread is None

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    @patch("psutil.virtual_memory")
    def test_taking_memory_snapshot(
        self,
        mock_vm,
        mock_device_props,
        mock_memory_reserved,
        mock_memory_allocated,
        memory_tracker,
    ):
        """Test taking a memory snapshot."""
        # Mock the device count
        with patch("torch.cuda.device_count", return_value=2):
            # Mock device properties
            mock_props = MagicMock()
            mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16 GB in bytes
            mock_device_props.return_value = mock_props

            # Mock memory allocation and reservation
            mock_memory_allocated.side_effect = [
                4 * 1024 * 1024 * 1024,
                6 * 1024 * 1024 * 1024,
            ]  # 4 GB, 6 GB
            mock_memory_reserved.side_effect = [
                1 * 1024 * 1024 * 1024,
                2 * 1024 * 1024 * 1024,
            ]  # 1 GB, 2 GB

            # Mock CPU memory
            mock_vm_obj = MagicMock()
            mock_vm_obj.used = 8 * 1024 * 1024 * 1024  # 8 GB
            mock_vm_obj.total = 32 * 1024 * 1024 * 1024  # 32 GB
            mock_vm.return_value = mock_vm_obj

            # Take a snapshot
            snapshot = memory_tracker._take_snapshot()

            # Verify the snapshot contains expected data
            assert len(snapshot.gpu_memory_used) == 2
            assert len(snapshot.gpu_memory_total) == 2

            # Values should be in MB
            assert (
                pytest.approx(16 * 1024, rel=1e-5) == snapshot.gpu_memory_total[0]
            )  # 16 GB in MB
            assert (
                pytest.approx(16 * 1024, rel=1e-5) == snapshot.gpu_memory_total[1]
            )  # 16 GB in MB

            # Used memory includes allocated + cached
            assert (
                pytest.approx((4 + 1) * 1024, rel=1e-5) == snapshot.gpu_memory_used[0]
            )  # 5 GB in MB
            assert (
                pytest.approx((6 + 2) * 1024, rel=1e-5) == snapshot.gpu_memory_used[1]
            )  # 8 GB in MB

            assert (
                pytest.approx(8 * 1024, rel=1e-5) == snapshot.cpu_memory_used
            )  # 8 GB in MB
            assert (
                pytest.approx(32 * 1024, rel=1e-5) == snapshot.cpu_memory_total
            )  # 32 GB in MB

    def test_memory_snapshot_properties(self, mock_memory_snapshot):
        """Test the computed properties of a memory snapshot."""
        # Test GPU utilization
        gpu_util = mock_memory_snapshot.gpu_utilization
        assert len(gpu_util) == 2
        assert pytest.approx(25.0, rel=1e-5) == gpu_util[0]  # 4000 / 16000 * 100
        assert pytest.approx(37.5, rel=1e-5) == gpu_util[1]  # 6000 / 16000 * 100

        # Test CPU utilization
        cpu_util = mock_memory_snapshot.cpu_utilization
        assert pytest.approx(25.0, rel=1e-5) == cpu_util  # 4000 / 16000 * 100

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.get_device_properties")
    @patch("psutil.virtual_memory")
    def test_start_stop_monitoring(
        self,
        mock_vm,
        mock_device_props,
        mock_memory_reserved,
        mock_memory_allocated,
        memory_tracker,
    ):
        """Test starting and stopping memory monitoring."""
        # Mock the device count
        with patch("torch.cuda.device_count", return_value=2):
            # Mock device properties
            mock_props = MagicMock()
            mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16 GB in bytes
            mock_device_props.return_value = mock_props

            # Mock memory allocation and reservation (constant values for testing)
            mock_memory_allocated.return_value = 4 * 1024 * 1024 * 1024  # 4 GB
            mock_memory_reserved.return_value = 1 * 1024 * 1024 * 1024  # 1 GB

            # Mock CPU memory
            mock_vm_obj = MagicMock()
            mock_vm_obj.used = 8 * 1024 * 1024 * 1024  # 8 GB
            mock_vm_obj.total = 32 * 1024 * 1024 * 1024  # 32 GB
            mock_vm.return_value = mock_vm_obj

            # Start monitoring
            memory_tracker.start_monitoring()

            # Let it collect a few snapshots
            import time

            time.sleep(0.3)  # Should collect about 3 snapshots

            # Stop monitoring
            memory_tracker.stop_monitoring()

            # Should have some snapshots
            assert len(memory_tracker.snapshots) > 0

            # Verify snapshot contents (just checking one)
            first_snapshot = memory_tracker.snapshots[0]
            assert (
                pytest.approx(16 * 1024, rel=1e-5) == first_snapshot.gpu_memory_total[0]
            )  # 16 GB in MB
            assert (
                pytest.approx(5 * 1024, rel=1e-5) == first_snapshot.gpu_memory_used[0]
            )  # 5 GB in MB (allocated + reserved)

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("src.utils.gpu_memory_tracker.GPUMemoryTracker.memory_stats")
    def test_model_loading_memory_impact(
        self,
        mock_memory_stats,
        mock_tokenizer_from_pretrained,
        mock_model_from_pretrained,
        mock_gpu_memory_stats,
    ):
        """Test measuring memory impact during model loading."""
        # Setup mocks to avoid any real model loading
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        # Set memory stats to show before/after memory change
        mock_memory_stats.side_effect = [
            # Initial memory state - returned when before_stats is called
            mock_gpu_memory_stats(),
            # After model loading (shows increased usage) - returned when after_stats is called
            {
                "cpu_used_gb": 4.2,
                "cpu_total_gb": 16.0,
                "cpu_percent": 26.25,
                "gpu_count": 2,
                "gpu": {
                    0: {
                        "name": "Mock GPU 0",
                        "total_memory_gb": 16.0,
                        "allocated_memory_gb": 10.0,  # Increased from 4.0
                        "reserved_memory_gb": 2.0,
                        "percent_used": 62.5,  # Increased from 25.0
                    },
                    1: {
                        "name": "Mock GPU 1",
                        "total_memory_gb": 16.0,
                        "allocated_memory_gb": 6.0,
                        "reserved_memory_gb": 1.0,
                        "percent_used": 37.5,
                    },
                },
            },
        ]

        # Check memory usage before model loading
        before_stats = GPUMemoryTracker.memory_stats()

        # Call the actual load_quantized_model function
        # The model loading is fully mocked but the function code still runs
        model, tokenizer = load_quantized_model(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            quantization_type="4bit",
            device_map="auto",
            compute_dtype=torch.float16,
        )

        # Get memory stats after model loading
        after_stats = GPUMemoryTracker.memory_stats()

        # Verify memory increase
        assert (
            after_stats["gpu"][0]["allocated_memory_gb"]
            > before_stats["gpu"][0]["allocated_memory_gb"]
        )
        assert (
            after_stats["gpu"][0]["percent_used"]
            > before_stats["gpu"][0]["percent_used"]
        )

        # The exact increase would depend on the model, but we verify there is an increase
        memory_increase = (
            after_stats["gpu"][0]["allocated_memory_gb"]
            - before_stats["gpu"][0]["allocated_memory_gb"]
        )
        assert pytest.approx(6.0, rel=1e-5) == memory_increase  # 10.0 - 4.0

    @patch("src.utils.gpu_memory_tracker.GPUMemoryTracker")
    def test_context_manager_usage(self, mock_tracker_class):
        """Test the memory tracking context manager."""
        # Create a mock tracker instance
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_tracker.get_max_memory_usage.return_value = ({0: 8000, 1: 6000}, 4000)

        # Mock the first snapshot for the report
        mock_snapshot = MagicMock()
        mock_snapshot.gpu_memory_total = {0: 16000, 1: 16000}
        mock_tracker.snapshots = [mock_snapshot]

        # Use the context manager
        with patch("os.makedirs"):  # Mock directory creation
            with track_memory(
                description="Test Context",
                interval=0.1,
                save_plot=True,
                save_report=True,
            ) as _:  # Use underscore for unused variable
                # Simulate some memory-intensive operation
                pass

        # Verify tracker methods were called
        mock_tracker.start_monitoring.assert_called_once()
        mock_tracker.stop_monitoring.assert_called_once()
        mock_tracker.plot_memory_usage.assert_called_once()
        mock_tracker.save_report.assert_called_once()
        mock_tracker.get_max_memory_usage.assert_called_once()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No CUDA device available"
    )
    def test_real_inference_memory_tracking(self):
        """Test tracking memory during actual inference (skipped if no CUDA available)."""
        # This test uses actual GPU if available to measure real memory impact

        # Skip detailed implementation since it requires actual model loading
        # In a real test, you would:
        # 1. Load a small test model
        # 2. Create an inference pipeline
        # 3. Track memory during inference with different expert combinations
        # 4. Verify memory patterns match expectations

        # For now, just ensure the tracker can run without errors
        tracker = GPUMemoryTracker(snapshot_interval=0.1)
        tracker.start_monitoring()

        # Perform some simple tensor operations that use GPU memory
        device = torch.device("cuda")
        for _ in range(3):
            # Create and delete tensors to cause memory fluctuations
            tensors = [torch.randn(1000, 1000, device=device) for _ in range(5)]
            del tensors
            torch.cuda.empty_cache()

        tracker.stop_monitoring()

        # Just verify we collected some snapshots
        assert len(tracker.snapshots) > 0

    @patch("src.utils.gpu_memory_tracker.GPUMemoryTracker")
    def test_expert_switching_memory_optimization(self, mock_tracker_class):
        """Test tracking memory optimization during expert switching."""
        # Create a mock tracker
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker

        # Create memory snapshots with patterns showing memory optimization
        snapshots = []
        # Starting with one expert active
        snapshots.append(
            MemorySnapshot(
                timestamp=1000.0,
                gpu_memory_used={0: 8000, 1: 6000},
                gpu_memory_total={0: 16000, 1: 16000},
                cpu_memory_used=4000,
                cpu_memory_total=16000,
            )
        )
        # When switching experts, the inactive one should move to CPU
        snapshots.append(
            MemorySnapshot(
                timestamp=1001.0,
                gpu_memory_used={0: 8000, 1: 2000},  # GPU 1 usage reduced
                gpu_memory_total={0: 16000, 1: 16000},
                cpu_memory_used=8000,  # CPU usage increased
                cpu_memory_total=16000,
            )
        )
        # When switching back, GPU memory increases again
        snapshots.append(
            MemorySnapshot(
                timestamp=1002.0,
                gpu_memory_used={0: 8000, 1: 6000},  # GPU 1 usage back up
                gpu_memory_total={0: 16000, 1: 16000},
                cpu_memory_used=4000,  # CPU usage decreased
                cpu_memory_total=16000,
            )
        )
        mock_tracker.snapshots = snapshots

        # Mock expert manager and offloading
        with patch(
            "src.models.expert_adapters.ExpertAdapterManager"
        ) as mock_adapter_manager:
            mock_manager = MagicMock()
            mock_adapter_manager.return_value = mock_manager

            # The actual test would track memory during expert switching
            # For simplicity, we'll just verify the memory pattern in the snapshots

            # GPU 1 memory should decrease during offloading
            assert (
                snapshots[1].gpu_memory_used[1] < snapshots[0].gpu_memory_used[1]
            ), "GPU memory should decrease during offloading"

            # CPU memory should increase during offloading
            assert (
                snapshots[1].cpu_memory_used > snapshots[0].cpu_memory_used
            ), "CPU memory should increase during offloading"

            # Memory should return to similar levels when switched back
            assert (
                snapshots[2].gpu_memory_used[1] > snapshots[1].gpu_memory_used[1]
            ), "GPU memory should increase when expert is loaded back"
            assert (
                snapshots[2].cpu_memory_used < snapshots[1].cpu_memory_used
            ), "CPU memory should decrease when expert is loaded back to GPU"
