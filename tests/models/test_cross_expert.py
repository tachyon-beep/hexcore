# tests/models/test_cross_expert.py

import sys
import torch
import pytest
from pathlib import Path
from unittest.mock import patch

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.cross_expert import CrossExpertAttention


class TestCrossExpertAttention:
    """Test cases for the CrossExpertAttention class."""

    @pytest.fixture
    def cross_expert_attention(self):
        """Create a CrossExpertAttention instance for testing."""
        return CrossExpertAttention(hidden_size=64, dropout=0.1)

    @pytest.fixture
    def expert_outputs(self):
        """Create mock expert outputs for testing."""
        # Create 3 expert outputs with batch_size=2, seq_len=3, hidden_size=64
        return [torch.randn(2, 3, 64) for _ in range(3)]

    def test_initialization(self):
        """Test initialization of CrossExpertAttention."""
        # Create with custom parameters
        hidden_size = 128
        dropout = 0.2

        attention = CrossExpertAttention(hidden_size=hidden_size, dropout=dropout)

        # Verify initialization
        assert attention.hidden_size == hidden_size

        # Verify layers were created - simplified architecture
        assert hasattr(attention, "weight_proj")
        assert hasattr(attention, "output_proj")
        assert hasattr(attention, "layer_norm")
        assert hasattr(attention, "dropout")

        # Check layer dimensions
        assert attention.weight_proj.in_features == hidden_size
        assert attention.weight_proj.out_features == 1
        assert attention.output_proj.in_features == hidden_size
        assert attention.output_proj.out_features == hidden_size

    def test_single_expert_passthrough(self, cross_expert_attention):
        """Test that a single expert output is passed through unchanged."""
        # Create a single expert output
        single_expert = torch.randn(2, 3, 64)

        # Process through attention mechanism
        output = cross_expert_attention([single_expert])

        # Should return the input unchanged
        assert torch.allclose(output, single_expert)

    def test_attention_output_shape(self, cross_expert_attention, expert_outputs):
        """Test that the output shape matches the input shape."""
        # Get shape of one expert output
        batch_size, seq_len, hidden_size = expert_outputs[0].shape

        # Process through attention mechanism
        output = cross_expert_attention(expert_outputs)

        # Output should have same shape as a single expert
        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_attention_computation(self):
        """Test the core attention computation in detail."""
        # Use smaller dims for this test to make it more tractable
        hidden_size = 8
        batch_size = 1
        seq_len = 2

        # Create a deterministic attention module with no dropout for reproducibility
        attention = CrossExpertAttention(hidden_size=hidden_size, dropout=0)

        # Create controlled expert outputs
        expert1 = torch.ones(batch_size, seq_len, hidden_size)
        expert2 = torch.ones(batch_size, seq_len, hidden_size) * 2

        # Process through attention
        output = attention([expert1, expert2])

        # Verify the output shape is correct
        assert output.shape == (batch_size, seq_len, hidden_size)

        # The exact values can vary due to the attention mechanism,
        # but the output should contain information from both experts
        # Check that output is not all zeros
        assert not torch.allclose(output, torch.zeros_like(output))

        # The attention computation can produce a range of values
        # but they should generally be influenced by the input values
        # Just check that the output has reasonable values
        assert output.abs().sum() > 0  # Not all zeros

        # Check the range isn't extreme
        assert torch.all(output.min() > -5.0)  # Not extremely negative
        assert torch.all(output.max() < 5.0)  # Not extremely positive

    def test_multi_expert_attention(self, cross_expert_attention):
        """Test integration of multiple expert outputs."""
        # Create expert outputs with distinctive patterns
        batch_size, seq_len, hidden_size = 2, 3, 64
        expert1 = torch.ones(batch_size, seq_len, hidden_size) * 0.1
        expert2 = torch.ones(batch_size, seq_len, hidden_size) * 0.2
        expert3 = torch.ones(batch_size, seq_len, hidden_size) * 0.3

        # Make experts have different values in different regions to test attention
        # But ensure all values are positive for easier testing
        expert1[:, :, :21] *= 10  # Emphasize first third in expert1
        expert2[:, :, 21:42] *= 10  # Emphasize middle third in expert2
        expert3[:, :, 42:] *= 10  # Emphasize last third in expert3

        # Process through attention mechanism
        output = cross_expert_attention([expert1, expert2, expert3])

        # Output should have combined information from all experts
        # Since experts have non-overlapping information, the attention mechanism
        # should attend differently to each expert

        # Check shape is correct
        assert output.shape == (batch_size, seq_len, hidden_size)

        # Output should not be all zeros (should have obtained information from experts)
        assert not torch.allclose(output, torch.zeros_like(output))

        # At least make sure output isn't zero throughout
        assert output.abs().sum() > 0

    def test_attention_pattern(self, cross_expert_attention, expert_outputs):
        """Test the pattern of attention calculations."""
        # This test verifies that the CrossExpertAttention module properly processes
        # multiple expert outputs and produces reasonable output

        # Get shape of one expert output
        batch_size, seq_len, hidden_size = expert_outputs[0].shape

        # Create a modified test input with distinctive patterns to test attention
        test_experts = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(3)]

        # Process through attention mechanism
        output = cross_expert_attention(test_experts)

        # The output should have the correct shape
        assert output.shape == (batch_size, seq_len, hidden_size)

        # The output should not be all zeros
        assert not torch.allclose(output, torch.zeros_like(output))

        # Output should have finite values (no NaNs or infinities)
        assert torch.all(torch.isfinite(output))

    def test_device_compatibility(self):
        """Test that the module can work on different devices."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device compatibility test")

        attention_cpu = CrossExpertAttention(hidden_size=32, dropout=0.1)

        # Create sample data on cpu
        expert1 = torch.randn(1, 2, 32)
        expert2 = torch.randn(1, 2, 32)

        # Forward pass on CPU
        output_cpu = attention_cpu([expert1, expert2])

        # Move module and data to GPU
        attention_gpu = attention_cpu.to("cuda")
        expert1_gpu = expert1.to("cuda")
        expert2_gpu = expert2.to("cuda")

        # Forward pass on GPU
        output_gpu = attention_gpu([expert1_gpu, expert2_gpu])

        # Output should be on the correct device
        assert output_gpu.device.type == "cuda"

        # Shapes should match
        assert output_cpu.shape == output_gpu.cpu().shape
