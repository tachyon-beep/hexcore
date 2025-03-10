import pytest
import torch
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.kv_cache_manager import KVCacheManager


@pytest.fixture
def kv_cache_manager():
    return KVCacheManager(
        max_memory_percentage=0.2,
        prune_threshold=0.8,
        auto_clear=True,
        sliding_window=512,
    )


@pytest.fixture
def mock_past_key_values():
    """Create mock past_key_values with a structure similar to what models produce"""
    num_layers = 4
    batch_size = 1
    num_heads = 8
    head_dim = 64
    seq_length = 128

    past_key_values = []
    for _ in range(num_layers):
        # Create key tensor: [batch_size, num_heads, seq_length, head_dim]
        key = torch.randn(batch_size, num_heads, seq_length, head_dim)
        # Create value tensor: [batch_size, num_heads, seq_length, head_dim]
        value = torch.randn(batch_size, num_heads, seq_length, head_dim)
        past_key_values.append((key, value))

    return tuple(past_key_values)


def test_cache_initialization(kv_cache_manager):
    """Test that the cache manager initializes correctly"""
    assert kv_cache_manager.max_memory_percentage == pytest.approx(0.2)
    assert kv_cache_manager.sliding_window == 512
    assert kv_cache_manager.current_cache_size == 0


def test_cache_stats(kv_cache_manager, mock_past_key_values):
    """Test cache statistics calculation"""
    stats = kv_cache_manager.monitor_cache_growth(mock_past_key_values)

    assert "seq_length" in stats
    assert stats["seq_length"] == 128
    assert "estimated_memory_mb" in stats
    assert stats["estimated_memory_mb"] > 0


def test_cache_pruning(kv_cache_manager, mock_past_key_values):
    """Test cache pruning with sliding window"""
    # The mock has 128 tokens, but sliding window is 512, so no pruning
    pruned = kv_cache_manager.prune_cache(mock_past_key_values)
    assert pruned[0][0].size(2) == 128  # No change in sequence length

    # Create a larger mock that exceeds the sliding window
    num_layers = 4
    batch_size = 1
    num_heads = 8
    head_dim = 64
    seq_length = 768  # > 512 sliding window

    large_past_key_values = []
    for _ in range(num_layers):
        key = torch.randn(batch_size, num_heads, seq_length, head_dim)
        value = torch.randn(batch_size, num_heads, seq_length, head_dim)
        large_past_key_values.append((key, value))

    large_past_key_values = tuple(large_past_key_values)

    # Now pruning should occur
    pruned = kv_cache_manager.prune_cache(large_past_key_values)
    assert pruned[0][0].size(2) == 512  # Should be reduced to sliding window size


def test_generation_kwargs(kv_cache_manager):
    """Test that generation kwargs are properly enhanced"""
    base_kwargs = {"max_new_tokens": 100, "temperature": 0.7}

    enhanced = kv_cache_manager.get_generation_kwargs(**base_kwargs)

    # Original kwargs should be preserved
    assert enhanced["max_new_tokens"] == 100
    assert enhanced["temperature"] == pytest.approx(0.7)

    # Our enhancements should be added
    assert "sliding_window" in enhanced
    assert enhanced["sliding_window"] == 512


def test_should_clear_cache(kv_cache_manager, mock_past_key_values):
    """Test the should_clear_cache decision function"""
    # By default with our small mock, should not need clearing
    assert not kv_cache_manager.should_clear_cache(mock_past_key_values)

    # Test with auto_clear disabled
    kv_cache_manager.auto_clear = False
    assert not kv_cache_manager.should_clear_cache(mock_past_key_values)

    # Reset for other tests
    kv_cache_manager.auto_clear = True


def test_clear_cache(kv_cache_manager):
    """Test that clearing the cache resets internal counters"""
    # Set some dummy values
    kv_cache_manager.current_cache_size = 100
    kv_cache_manager.memory_usage = {"dummy": 1000}

    # Clear cache
    kv_cache_manager.clear_cache()

    # Check counters were reset
    assert kv_cache_manager.current_cache_size == 0
    assert kv_cache_manager.memory_usage == {}
