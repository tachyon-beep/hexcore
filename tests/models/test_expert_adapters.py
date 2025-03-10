# tests/models/test_expert_adapters.py

import os
import sys
import pytest
import torch
import shutil
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
from collections import OrderedDict

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.expert_adapters import ExpertAdapterManager
from peft.peft_model import PeftModel
from peft.tuners.lora import LoraConfig


class TestExpertAdapterManager:
    """Test cases for the ExpertAdapterManager class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock base model."""
        mock = MagicMock()
        # Mock the parameters method to return a generator with a mock tensor
        mock_param = MagicMock()
        mock_param.device.type = "cpu"
        mock.parameters.return_value = iter([mock_param])
        return mock

    @pytest.fixture
    def temp_adapter_dir(self, tmpdir):
        """Create a temporary directory for adapter files."""
        adapter_dir = tmpdir.mkdir("test_adapters")
        # Create subdirectories for each expert type
        for expert_type in ["reason", "explain", "teach", "predict", "retrospect"]:
            expert_dir = adapter_dir.mkdir(expert_type)
            # Create a dummy adapter_config.json file to simulate adapter presence
            config_file = expert_dir.join("adapter_config.json")
            config_file.write('{"type": "LORA"}')

        yield str(adapter_dir)
        # Clean up
        shutil.rmtree(str(adapter_dir), ignore_errors=True)

    @patch("src.models.expert_adapters.PeftModel")
    def test_init_loads_available_adapters(
        self, mock_peft_model, mock_model, temp_adapter_dir
    ):
        """Test that __init__ attempts to load available adapters."""
        # Setup
        mock_peft_model.from_pretrained.return_value = mock_model

        # Execute
        manager = ExpertAdapterManager(mock_model, adapters_dir=temp_adapter_dir)

        # Verify
        # Now only loads up to max_gpu_experts (default=2) adapters in __init__
        # to save memory, others are just noted as available
        assert mock_peft_model.from_pretrained.call_count == 2

        # Check if all expert types are initialized in the configs
        assert len(manager.expert_configs) == 5
        assert "REASON" in manager.expert_configs
        assert "EXPLAIN" in manager.expert_configs
        assert "TEACH" in manager.expert_configs
        assert "PREDICT" in manager.expert_configs
        assert "RETROSPECT" in manager.expert_configs

        # Verify each config is a LoraConfig
        for config in manager.expert_configs.values():
            assert isinstance(config, LoraConfig)

    @patch("src.models.expert_adapters.PeftModel")
    def test_apply_adapter_already_active(
        self, mock_peft_model, mock_model, temp_adapter_dir
    ):
        """Test applying an adapter that's already active."""
        # Setup
        mock_peft_model.from_pretrained.return_value = mock_model
        manager = ExpertAdapterManager(mock_model, adapters_dir=temp_adapter_dir)
        manager.current_adapter = "REASON"  # Manually set current adapter

        # Execute
        result = manager.apply_adapter("REASON")

        # Verify
        assert result is True  # Should succeed
        assert manager.current_adapter == "REASON"  # Should remain the same
        # No new adapter should be loaded since it's already active
        assert mock_peft_model.from_pretrained.call_count == 2  # Only from init

    @patch("src.models.expert_adapters.PeftModel")
    @patch("src.models.expert_adapters.get_peft_model")
    def test_apply_adapter_new_adapter(
        self, mock_get_peft, mock_peft_model, mock_model, temp_adapter_dir
    ):
        """Test applying a different adapter."""
        # Setup
        mock_peft_model.from_pretrained.return_value = mock_model
        mock_get_peft.return_value = mock_model

        # Make the mock model correctly handle device placement
        mock_model.to.return_value = mock_model  # Return self from to() method

        manager = ExpertAdapterManager(mock_model, adapters_dir=temp_adapter_dir)
        manager.current_adapter = "REASON"  # Set initial adapter
        manager.expert_adapters = {
            "REASON": mock_model,
            "EXPLAIN": mock_model,
        }  # Add some loaded adapters

        # Mock the _verify_device_consistency method to do nothing
        manager._verify_device_consistency = MagicMock()

        # Execute - Apply a different loaded adapter
        result = manager.apply_adapter("EXPLAIN")

        # Verify
        assert result is True  # Should succeed
        assert manager.current_adapter == "EXPLAIN"  # Should be updated
        # Since we mocked .to() to return the same object, base_model should be the same
        assert manager.base_model is mock_model  # Should use the loaded adapter

    @patch("src.models.expert_adapters.PeftModel")
    @patch("src.models.expert_adapters.get_peft_model")
    @patch("src.models.expert_adapters.isinstance")
    def test_apply_adapter_create_new(
        self,
        mock_isinstance,
        mock_get_peft,
        mock_peft_model,
        mock_model,
        temp_adapter_dir,
    ):
        """Test creating and applying a new adapter that isn't loaded."""
        # Setup
        mock_peft_model.from_pretrained.return_value = mock_model
        new_model = MagicMock()
        mock_get_peft.return_value = new_model

        # Make the isinstance check pass
        mock_isinstance.return_value = True

        manager = ExpertAdapterManager(mock_model, adapters_dir=temp_adapter_dir)
        manager.current_adapter = "REASON"
        # Clear the loaded adapters so we force creating a new one
        manager.expert_adapters = {"REASON": mock_model}

        # Execute - Apply an adapter that isn't loaded but has a config
        result = manager.apply_adapter("TEACH")

        # Verify
        assert result is True  # Should succeed
        assert manager.current_adapter == "TEACH"  # Should be updated
        assert manager.base_model == new_model  # Should use the new model
        assert "TEACH" in manager.expert_adapters  # Should be added to adapters
        assert manager.expert_adapters["TEACH"] == new_model  # Should store new model

        # Verify the isinstance check was called
        mock_isinstance.assert_called()

    @patch("src.models.expert_adapters.PeftModel")
    def test_apply_adapter_unknown_type(
        self, mock_peft_model, mock_model, temp_adapter_dir
    ):
        """Test applying an adapter of unknown type."""
        # Setup
        mock_peft_model.from_pretrained.return_value = mock_model
        manager = ExpertAdapterManager(mock_model, adapters_dir=temp_adapter_dir)

        # Execute - Apply an adapter type that doesn't exist
        result = manager.apply_adapter("UNKNOWN_TYPE")

        # Verify
        assert result is False  # Should fail
        assert manager.current_adapter is None  # Should not change

    @patch("src.models.expert_adapters.PeftModel")
    @patch("torch.cuda.empty_cache")
    @patch("gc.collect")
    @patch("torch.cuda.memory_allocated")
    def test_offload_inactive_experts(
        self,
        mock_cuda_memory,
        mock_gc,
        mock_cuda_empty,
        mock_peft_model,
        mock_model,
        temp_adapter_dir,
    ):
        """Test offloading inactive experts to CPU with LRU caching."""
        # Setup memory measurement mocks
        mock_cuda_memory.return_value = 1024 * 1024 * 1024  # 1GB

        # Setup
        mock_peft_model.from_pretrained.return_value = mock_model

        # Create models with device info for active and inactive experts
        active_model = MagicMock()
        active_param = MagicMock()
        active_param.device.type = "cuda"
        active_model.parameters.return_value = iter([active_param])

        inactive_model1 = MagicMock()
        inactive_param1 = MagicMock()
        inactive_param1.device.type = "cuda"
        inactive_model1.parameters.return_value = iter([inactive_param1])

        inactive_model2 = MagicMock()
        inactive_param2 = MagicMock()
        inactive_param2.device.type = "cuda"  # Also on CUDA
        inactive_model2.parameters.return_value = iter([inactive_param2])

        inactive_model3 = MagicMock()
        inactive_param3 = MagicMock()
        inactive_param3.device.type = "cpu"  # Already on CPU
        inactive_model3.parameters.return_value = iter([inactive_param3])

        manager = ExpertAdapterManager(
            mock_model, adapters_dir=temp_adapter_dir, max_gpu_experts=2
        )
        # Set up some expert adapters with different devices
        manager.expert_adapters = {
            "REASON": active_model,
            "EXPLAIN": inactive_model1,
            "TEACH": inactive_model2,
            "PREDICT": inactive_model3,
        }

        # Set up LRU tracking - REASON is active, EXPLAIN was used recently
        manager.expert_usage = OrderedDict(
            [
                ("TEACH", True),
                ("EXPLAIN", True),
                ("REASON", True),
            ]
        )

        # Execute with keep_recent=1 (should keep EXPLAIN on GPU as it's most recent after REASON)
        manager.offload_inactive_experts("REASON", keep_recent=1)

        # Verify
        # Should move inactive_model2 (TEACH) to CPU as it's least recently used
        inactive_model2.to.assert_called_once_with("cpu")
        # EXPLAIN should stay on GPU as it's recently used (with keep_recent=1)
        assert inactive_model1.to.call_count == 0
        # PREDICT already on CPU, shouldn't be moved
        assert inactive_model3.to.call_count == 0
        # Active model (REASON) should not be moved
        assert active_model.to.call_count == 0

        # Should call garbage collection
        mock_gc.assert_called_once()
        # Should call CUDA cache emptying
        mock_cuda_empty.assert_called_once()

        # Now test force offloading (offload everything except active expert)
        manager.offload_inactive_experts("REASON", force_offload=True)

        # Now EXPLAIN should be moved to CPU too
        inactive_model1.to.assert_called_once_with("cpu")

    @patch("src.models.expert_adapters.PeftModel")
    def test_prefetch_expert(self, mock_peft_model, mock_model, temp_adapter_dir):
        """Test prefetching an expert to GPU memory."""
        # Setup
        mock_peft_model.from_pretrained.return_value = mock_model

        # Create mock model on CPU
        cpu_model = MagicMock()
        cpu_param = MagicMock()
        cpu_param.device.type = "cpu"
        cpu_model.parameters.return_value = iter([cpu_param])

        # Create mock model already on GPU
        gpu_model = MagicMock()
        gpu_param = MagicMock()
        gpu_param.device.type = "cuda"
        gpu_model.parameters.return_value = iter([gpu_param])

        # Create manager and setup adapters
        manager = ExpertAdapterManager(mock_model, adapters_dir=temp_adapter_dir)
        manager.expert_adapters = {
            "REASON": gpu_model,  # Already on GPU
            "EXPLAIN": cpu_model,  # On CPU, can be prefetched
        }
        manager.available_adapters = {"REASON", "EXPLAIN", "TEACH"}  # TEACH is on disk

        # Test 1: Prefetch expert that's already on GPU - should do nothing but return True
        result = manager.prefetch_expert("REASON")
        assert result is True
        assert gpu_model.to.call_count == 0  # No need to move

        # Test 2: Prefetch expert that's loaded but on CPU - should move to GPU
        result = manager.prefetch_expert("EXPLAIN")
        assert result is True
        cpu_model.to.assert_called_once()  # Should be moved to GPU

        # Test 3: Try to prefetch expert that's not loaded but available
        # Mock the new model loading
        new_model = MagicMock()
        mock_peft_model.from_pretrained.return_value = new_model

        result = manager.prefetch_expert("TEACH")
        assert result is True
        # Should have called PeftModel.from_pretrained to load the model
        assert mock_peft_model.from_pretrained.call_count > 0

        # Verify it was added to LRU tracking
        assert "TEACH" in manager.expert_usage

    @patch("src.models.expert_adapters.PeftModel")
    def test_get_memory_usage_stats(
        self, mock_peft_model, mock_model, temp_adapter_dir
    ):
        """Test retrieving memory usage statistics."""
        # Setup
        mock_peft_model.from_pretrained.return_value = mock_model

        # Create mock models on different devices
        gpu_model = MagicMock()
        gpu_param = MagicMock()
        gpu_param.device.type = "cuda"
        gpu_model.parameters.return_value = iter([gpu_param])

        cpu_model = MagicMock()
        cpu_param = MagicMock()
        cpu_param.device.type = "cpu"
        cpu_model.parameters.return_value = iter([cpu_param])

        # Create manager with some adapters and statistics
        manager = ExpertAdapterManager(mock_model, adapters_dir=temp_adapter_dir)
        manager.expert_adapters = {
            "REASON": gpu_model,
            "EXPLAIN": cpu_model,
        }
        manager.current_adapter = "REASON"
        manager.expert_memory_stats = {
            "REASON": 0.5,  # 500MB
            "EXPLAIN": 0.3,  # 300MB
        }
        manager.expert_usage = OrderedDict(
            [
                ("EXPLAIN", True),
                ("REASON", True),
            ]
        )

        # Get memory statistics
        stats = manager.get_memory_usage_stats()

        # Verify statistics
        assert stats["experts_in_gpu"] == 1
        assert stats["experts_in_cpu"] == 1
        assert stats["active_expert"] == "REASON"
        assert "expert_memory_stats" in stats
        assert stats["expert_memory_stats"]["REASON"] == pytest.approx(0.5)
        assert "lru_order" in stats
        assert stats["lru_order"] == ["EXPLAIN", "REASON"]

    @patch("src.models.expert_adapters.PeftModel")
    def test_get_active_model(self, mock_peft_model, mock_model, temp_adapter_dir):
        """Test getting the active model."""
        # Setup
        active_model = MagicMock()
        mock_peft_model.from_pretrained.return_value = active_model
        manager = ExpertAdapterManager(mock_model, adapters_dir=temp_adapter_dir)
        manager.base_model = active_model  # Set the active model

        # Execute
        result = manager.get_active_model()

        # Verify
        assert result == active_model
