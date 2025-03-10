"""
Tests for the MixedPrecisionTrainer implementation.

This module contains tests for the mixed precision training functionality,
which is critical for memory-efficient adapter training.
"""

import sys
import pytest
import torch
import math
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.mixed_precision import MixedPrecisionTrainer


class TestMixedPrecisionTrainer:
    """Test cases for the MixedPrecisionTrainer class."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer."""
        optimizer = MagicMock()
        param_group = {
            "params": [torch.nn.Parameter(torch.randn(10, 10, requires_grad=True))]
        }
        optimizer.param_groups = [param_group]
        return optimizer

    @pytest.fixture
    def mock_loss(self):
        """Create a mock loss tensor."""
        return torch.tensor(2.5, requires_grad=True)

    @patch("torch.cuda.is_available")
    def test_init_with_cuda_available(self, mock_cuda_available):
        """Test initialization when CUDA is available."""
        # Setup
        mock_cuda_available.return_value = True

        # Execute
        trainer = MixedPrecisionTrainer(use_amp=True)

        # Verify
        assert trainer.use_amp is True
        assert trainer.scaler.is_enabled() is True

    @patch("torch.cuda.is_available")
    def test_init_with_cuda_unavailable(self, mock_cuda_available):
        """Test initialization when CUDA is not available."""
        # Setup
        mock_cuda_available.return_value = False

        # Execute
        trainer = MixedPrecisionTrainer(use_amp=True)

        # Verify - should disable AMP when CUDA is unavailable
        assert trainer.use_amp is False
        assert trainer.scaler.is_enabled() is False

    def test_init_with_amp_disabled(self):
        """Test initialization with AMP explicitly disabled."""
        # Execute
        trainer = MixedPrecisionTrainer(use_amp=False)

        # Verify
        assert trainer.use_amp is False
        assert trainer.scaler.is_enabled() is False

    @patch("torch.amp.autocast_mode.autocast")
    def test_get_ctx_manager_with_amp(self, mock_autocast):
        """Test context manager retrieval with AMP enabled."""
        # Setup
        mock_autocast.return_value = "autocast_context"
        trainer = MixedPrecisionTrainer()
        trainer.use_amp = True  # Force AMP enabled

        # Execute
        ctx = trainer.get_ctx_manager()

        # Verify
        assert ctx == "autocast_context"
        mock_autocast.assert_called_once_with(device_type="cuda")

    def test_get_ctx_manager_without_amp(self):
        """Test context manager retrieval with AMP disabled."""
        # Setup
        trainer = MixedPrecisionTrainer(use_amp=False)

        # Execute
        ctx = trainer.get_ctx_manager()

        # Verify - should return nullcontext
        assert ctx.__class__.__name__ == "nullcontext"

    @patch.object(torch.amp.grad_scaler.GradScaler, "scale")
    def test_backward_with_amp(self, mock_scale, mock_loss):
        """Test backward pass with AMP enabled."""
        # Setup
        trainer = MixedPrecisionTrainer()
        trainer.use_amp = True  # Force AMP enabled
        mock_scale.return_value = mock_loss
        mock_loss.backward = MagicMock()

        # Execute
        trainer.backward(mock_loss)

        # Verify
        mock_scale.assert_called_once_with(mock_loss)
        mock_loss.backward.assert_called_once()

    def test_backward_without_amp(self, mock_loss):
        """Test backward pass with AMP disabled."""
        # Setup
        trainer = MixedPrecisionTrainer(use_amp=False)
        mock_loss.backward = MagicMock()

        # Execute
        trainer.backward(mock_loss)

        # Verify
        mock_loss.backward.assert_called_once()

    @patch.object(torch.amp.grad_scaler.GradScaler, "unscale_")
    @patch.object(torch.amp.grad_scaler.GradScaler, "step")
    @patch.object(torch.amp.grad_scaler.GradScaler, "update")
    @patch.object(torch.nn.utils, "clip_grad_norm_")
    def test_step_with_amp(
        self, mock_clip_grad, mock_update, mock_step, mock_unscale, mock_optimizer
    ):
        """Test optimizer step with AMP enabled."""
        # Setup
        trainer = MixedPrecisionTrainer()
        trainer.use_amp = True  # Force AMP enabled

        # Initialize the scaler manually to avoid unscale_ error
        trainer.scaler = MagicMock()
        trainer.scaler.unscale_ = mock_unscale
        trainer.scaler.step = mock_step
        trainer.scaler.update = mock_update

        # Mock gradient norm
        mock_clip_grad.return_value = torch.tensor(0.5)

        # Execute
        trainer.step(mock_optimizer)

        # Verify
        mock_unscale.assert_called_once_with(mock_optimizer)
        mock_clip_grad.assert_called_once()
        mock_step.assert_called_once_with(mock_optimizer)
        mock_update.assert_called_once()
        assert trainer.steps == 1
        assert len(trainer.grad_norm_history) == 1
        assert trainer.grad_norm_history[0] == pytest.approx(0.5)

    @patch.object(torch.nn.utils, "clip_grad_norm_")
    def test_step_without_amp(self, mock_clip_grad, mock_optimizer):
        """Test optimizer step with AMP disabled."""
        # Setup
        trainer = MixedPrecisionTrainer(use_amp=False)

        # Mock gradient norm
        mock_clip_grad.return_value = torch.tensor(0.5)

        # Execute
        trainer.step(mock_optimizer)

        # Verify
        mock_clip_grad.assert_called_once()
        mock_optimizer.step.assert_called_once()
        assert trainer.steps == 1
        assert len(trainer.grad_norm_history) == 1
        assert trainer.grad_norm_history[0] == pytest.approx(0.5)

    def test_register_fp32_operation(self):
        """Test registering operations that need full precision."""
        # Setup
        trainer = MixedPrecisionTrainer()

        # Execute
        trainer.register_fp32_operation("special_attention")
        trainer.register_fp32_operation("complex_norm")

        # Verify
        assert "special_attention" in trainer.fp32_operations
        assert "complex_norm" in trainer.fp32_operations
        assert len(trainer.fp32_operations) == 2

    def test_get_statistics_empty(self):
        """Test getting statistics with no training steps."""
        # Setup
        trainer = MixedPrecisionTrainer()

        # Execute
        stats = trainer.get_statistics()

        # Verify
        assert stats["status"] == "No training steps recorded"

    def test_get_statistics_with_history(self):
        """Test getting statistics with training history."""
        # Setup
        trainer = MixedPrecisionTrainer(use_amp=True)
        trainer.steps = 10
        trainer.grad_norm_history = [0.1, 0.2, 0.3, 0.4, 0.5]
        trainer.register_fp32_operation("test_op")

        # Mock scaler scale
        trainer.scaler.get_scale = MagicMock(return_value=2.0)

        # Execute
        stats = trainer.get_statistics()

        # Verify
        assert stats["amp_enabled"] is True
        assert stats["steps"] == 10
        assert stats["current_scale"] == pytest.approx(2.0)
        assert "test_op" in stats["fp32_operations"]
        assert stats["mean_grad_norm"] == pytest.approx(0.3)
        assert stats["max_grad_norm"] == pytest.approx(0.5)
        assert stats["min_grad_norm"] == pytest.approx(0.1)
        assert stats["training_stability"] == "Insufficient data"

    def test_assess_stability_insufficient_data(self):
        """Test stability assessment with insufficient data."""
        # Setup
        trainer = MixedPrecisionTrainer()
        trainer.grad_norm_history = [0.1] * 50  # Less than 100 points

        # Execute
        stability = trainer._assess_stability()

        # Verify
        assert stability == "Insufficient data"

    def test_assess_stability_with_nans(self):
        """Test stability assessment with NaN values."""
        # Setup
        trainer = MixedPrecisionTrainer()
        trainer.grad_norm_history = [0.1] * 90 + [
            float("nan")
        ] * 10  # 100 points with NaNs

        # Execute
        stability = trainer._assess_stability()

        # Verify
        assert stability == "Unstable - NaN values detected"

    def test_assess_stability_with_high_gradients(self):
        """Test stability assessment with high gradient norms."""
        # Setup
        trainer = MixedPrecisionTrainer()
        trainer.grad_norm_history = [0.1] * 80 + [
            150.0
        ] * 20  # 100 points with high gradients

        # Execute
        stability = trainer._assess_stability()

        # Verify
        assert stability == "Potentially unstable - high gradient norms"

    def test_assess_stability_with_vanishing_gradients(self):
        """Test stability assessment with vanishing gradients."""
        # Setup
        trainer = MixedPrecisionTrainer()
        trainer.grad_norm_history = [0.1] * 80 + [
            1e-7
        ] * 20  # 100 points with tiny gradients

        # Execute
        stability = trainer._assess_stability()

        # Verify
        assert stability == "Potentially unstable - vanishing gradients"

    def test_assess_stability_when_stable(self):
        """Test stability assessment with healthy gradient norms."""
        # Setup
        trainer = MixedPrecisionTrainer()
        trainer.grad_norm_history = [0.1] * 50 + [
            0.5
        ] * 50  # 100 points with reasonable gradients

        # Execute
        stability = trainer._assess_stability()

        # Verify
        assert stability == "Stable"
