import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import gc

logger = logging.getLogger(__name__)


class KVCacheManager:
    """
    Manages the key-value cache used in transformer generation to optimize memory usage.

    During text generation, models accumulate key-value pairs for each processed token,
    which can consume significant memory over long generations. This manager:

    1. Limits maximum memory used by the KV cache
    2. Provides pruning strategies to remove less relevant cached tokens
    3. Automatically clears cache between generations
    4. Monitors cache growth during generation
    """

    def __init__(
        self,
        max_memory_percentage: float = 0.3,
        prune_threshold: float = 0.8,
        auto_clear: bool = True,
        max_seq_length: Optional[int] = None,
        sliding_window: Optional[int] = None,
        device_map: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the KV cache manager.

        Args:
            max_memory_percentage: Maximum percentage of GPU memory to use for KV cache
            prune_threshold: Threshold at which to start pruning the cache (as percentage of max)
            auto_clear: Whether to automatically clear the cache between generations
            max_seq_length: Maximum sequence length to keep in cache (None = no limit)
            sliding_window: If set, use a sliding window attention of this size
            device_map: Optional mapping of layer indices to devices
        """
        self.max_memory_percentage = max_memory_percentage
        self.prune_threshold = prune_threshold
        self.auto_clear = auto_clear
        self.max_seq_length = max_seq_length
        self.sliding_window = sliding_window
        self.device_map = device_map or {}

        # Track current cache size (in tokens)
        self.current_cache_size = 0

        # Track memory usage
        self.memory_usage = {}
        self.max_memory = self._calculate_max_memory_per_gpu()

        logger.info(
            f"Initialized KV Cache Manager (max memory %: {max_memory_percentage*100:.1f}%)"
        )
        if sliding_window:
            logger.info(
                f"Using sliding window attention with window size: {sliding_window}"
            )

    def _calculate_max_memory_per_gpu(self) -> Dict[int, int]:
        """
        Calculate the maximum cache size based on available GPU memory and settings.

        Returns:
            Dictionary mapping GPU indices to maximum cache memory in bytes
        """
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = int(total_memory * self.max_memory_percentage)
            logger.debug(
                f"GPU {i}: Max KV cache memory: {max_memory[i] / (1024**2):.1f} MB"
            )
        return max_memory

    def clear_cache(self) -> None:
        """
        Clear the KV cache completely and run garbage collection.

        This should be called between different generation tasks.
        """
        # Force garbage collection
        gc.collect()

        # Clear PyTorch's CUDA cache
        torch.cuda.empty_cache()

        # Reset our internal counters
        self.current_cache_size = 0
        self.memory_usage = {}

        logger.debug("KV cache cleared")

    def prune_cache(self, past_key_values: Any) -> Any:
        """
        Prune the cache to stay within memory limits by implementing cache strategies.

        Strategies include:
        1. Sliding window attention - keep only the last N tokens
        2. Pruning least important tokens (not key positions like BOS/EOS)

        Args:
            past_key_values: The current past_key_values tuple from the model

        Returns:
            Pruned past_key_values
        """
        if past_key_values is None or not isinstance(past_key_values, tuple):
            return past_key_values

        # Check if we need to prune based on sequence length
        if self.max_seq_length is not None or self.sliding_window is not None:
            # Determine the current sequence length
            # This assumes the standard format for past_key_values from transformer models
            if len(past_key_values) > 0 and len(past_key_values[0]) >= 2:
                seq_length = past_key_values[0][0].size(2)  # Size of seq dimension in K

                # Apply sliding window if configured
                if self.sliding_window is not None and seq_length > self.sliding_window:
                    # Keep only the last window_size tokens
                    window_size = self.sliding_window
                    logger.debug(
                        f"Pruning KV cache using sliding window: {seq_length} -> {window_size}"
                    )

                    # Create pruned past_key_values
                    pruned_past_key_values = []
                    for layer_past in past_key_values:
                        pruned_layer_past = []
                        for i, tensor in enumerate(layer_past):
                            # For key and value tensors, keep only the last window_size tokens
                            if i < 2:  # Key and value tensors
                                pruned_tensor = tensor[:, :, -window_size:, :]
                            else:  # Any other tensors (though standard past_key_values has only K,V)
                                pruned_tensor = tensor
                            pruned_layer_past.append(pruned_tensor)
                        pruned_past_key_values.append(tuple(pruned_layer_past))

                    # Update our tracking
                    self.current_cache_size = window_size

                    return tuple(pruned_past_key_values)

                # Apply max sequence length if configured
                elif (
                    self.max_seq_length is not None and seq_length > self.max_seq_length
                ):
                    # Keep only the last max_seq_length tokens
                    keep_length = self.max_seq_length
                    logger.debug(
                        f"Pruning KV cache using max seq length: {seq_length} -> {keep_length}"
                    )

                    # Create pruned past_key_values
                    pruned_past_key_values = []
                    for layer_past in past_key_values:
                        pruned_layer_past = []
                        for i, tensor in enumerate(layer_past):
                            if i < 2:  # Key and value tensors
                                pruned_tensor = tensor[:, :, -keep_length:, :]
                            else:
                                pruned_tensor = tensor
                            pruned_layer_past.append(pruned_tensor)
                        pruned_past_key_values.append(tuple(pruned_layer_past))

                    # Update our tracking
                    self.current_cache_size = keep_length

                    return tuple(pruned_past_key_values)

        # More advanced pruning strategies could be implemented here
        # For example, pruning based on attention scores to keep tokens that are more relevant

        return past_key_values

    def get_generation_kwargs(self, **existing_kwargs) -> Dict[str, Any]:
        """
        Enhance model.generate() keyword arguments with cache optimization settings.

        Args:
            **existing_kwargs: Existing generation kwargs

        Returns:
            Updated kwargs dict with cache management settings
        """
        kwargs = existing_kwargs.copy()

        # Add sliding window attention setting if configured
        if self.sliding_window is not None:
            # Different models may use different parameter names
            for param_name in [
                "sliding_window",
                "attention_window",
                "max_cache_length",
            ]:
                kwargs[param_name] = self.sliding_window

        # Add max sequence length if configured
        if self.max_seq_length is not None:
            kwargs.setdefault("max_length", self.max_seq_length)

        return kwargs

    def monitor_cache_growth(self, past_key_values: Any) -> Dict[str, Any]:
        """
        Monitor KV cache growth during generation and return stats.

        Args:
            past_key_values: Current past_key_values from the model

        Returns:
            Dictionary with cache size statistics
        """
        stats = {}

        if past_key_values is None:
            return {"seq_length": 0, "estimated_memory_mb": 0}

        try:
            # Get sequence length from the first layer's key tensor
            if len(past_key_values) > 0 and len(past_key_values[0]) >= 2:
                seq_length = past_key_values[0][0].size(2)  # Size of seq dimension in K

                # Update tracking
                self.current_cache_size = seq_length

                # Estimate memory usage of the KV cache
                total_cache_memory = 0
                for layer_idx, layer_past in enumerate(past_key_values):
                    for tensor in layer_past:
                        # Get tensor size in bytes
                        tensor_memory = tensor.nelement() * tensor.element_size()
                        total_cache_memory += tensor_memory

                # Convert to MB for readability
                cache_memory_mb = total_cache_memory / (1024 * 1024)

                stats = {
                    "seq_length": seq_length,
                    "estimated_memory_mb": cache_memory_mb,
                    "total_cache_bytes": total_cache_memory,
                }

                # Check if we're approaching memory limits
                for gpu_idx, max_mem in self.max_memory.items():
                    if total_cache_memory > max_mem * self.prune_threshold:
                        device_str = f"cuda:{gpu_idx}"
                        if not self.device_map or any(
                            device == device_str for device in self.device_map.values()
                        ):
                            logger.warning(
                                f"KV cache approaching memory limit on GPU {gpu_idx}: {cache_memory_mb:.1f}MB vs limit {max_mem/(1024**2):.1f}MB"
                            )
                            stats["near_limit"] = True
                            break

        except Exception as e:
            logger.warning(f"Error monitoring KV cache growth: {e}")
            stats = {"error": str(e)}

        return stats

    def should_clear_cache(self, past_key_values: Any) -> bool:
        """
        Determine if cache should be cleared, based on configured thresholds.

        Args:
            past_key_values: Current past_key_values

        Returns:
            Boolean indicating whether cache should be cleared
        """
        if not self.auto_clear:
            return False

        stats = self.monitor_cache_growth(past_key_values)
        return stats.get("near_limit", False)
