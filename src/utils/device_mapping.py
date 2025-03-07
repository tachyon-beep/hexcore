import torch
import math
from typing import Dict, List, Optional, Union, Tuple


class DeviceMapper:
    """
    Utility for mapping model components across multiple GPUs.
    Implements different strategies for distributing model components
    to optimize memory usage across available devices.
    """

    def __init__(
        self,
        num_experts: int = 8,
        num_layers: int = 32,
        force_cpu_offload: bool = False,
    ):
        """
        Initialize the device mapper.

        Args:
            num_experts: Number of experts in the MoE model
            num_layers: Number of transformer layers in the model
            force_cpu_offload: Whether to force offloading some components to CPU
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.force_cpu_offload = force_cpu_offload

        # Check available GPUs
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            raise RuntimeError("No GPUs detected. Cannot perform device mapping.")

        # Get GPU memory information
        self.gpu_memory = {}
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            # Convert to GB for easier reasoning
            self.gpu_memory[i] = props.total_memory / (1024**3)

        print(f"DeviceMapper initialized with {self.num_gpus} GPUs")
        for i, mem in self.gpu_memory.items():
            print(f"  GPU {i}: {mem:.2f} GB")

    def create_mixtral_device_map(
        self, reserve_memory_gb: float = 2.0
    ) -> Dict[str, str]:
        """
        Create a device map for Mixtral 8x7B MoE model.

        This implements the strategy described in the technical document,
        distributing model components across available GPUs to optimize memory usage.

        Args:
            reserve_memory_gb: Amount of GPU memory to reserve for other operations (in GB)

        Returns:
            A dictionary mapping model components to device strings ('cuda:0', 'cuda:1', etc.)
        """
        device_map = {}

        # Handle different GPU configurations
        if self.num_gpus >= 2:
            # We can implement the dual-GPU strategy from the technical document
            return self._create_dual_gpu_map(reserve_memory_gb)
        elif self.num_gpus == 1:
            # Adapt strategy for single GPU
            return self._create_single_gpu_map(reserve_memory_gb)
        else:
            # We already checked for 0 GPUs in __init__, so this shouldn't happen
            raise RuntimeError("Unexpected error: No GPUs detected.")

    def _create_dual_gpu_map(self, reserve_memory_gb: float) -> Dict[str, str]:
        """Create a device map for dual GPU setup."""
        device_map = {}

        # Implement the strategy from the technical document for dual GPUs

        # Map embedding layers to GPU 0
        device_map["model.embed_tokens"] = "cuda:0"

        # Map first half of layers to GPU 0, second half to GPU 1
        half_layers = self.num_layers // 2
        for i in range(self.num_layers):
            if i < half_layers:
                device_map[f"model.layers.{i}"] = "cuda:0"
            else:
                device_map[f"model.layers.{i}"] = "cuda:1"

        # Map experts across GPUs - first half experts to GPU 0, second half to GPU 1
        half_experts = self.num_experts // 2
        for i in range(self.num_layers):
            layer_device = device_map[f"model.layers.{i}"]
            gpu_id = int(layer_device.split(":")[1])

            # Try to map gate to same device as layer
            device_map[f"model.layers.{i}.block_sparse_moe.gate"] = layer_device

            # Map experts
            for j in range(self.num_experts):
                expert_device = "cuda:0" if j < half_experts else "cuda:1"
                device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                    expert_device
                )

        # Map final layers to GPU 1
        device_map["model.norm"] = "cuda:1"
        device_map["lm_head"] = "cuda:1"

        return device_map

    def _create_single_gpu_map(self, reserve_memory_gb: float) -> Dict[str, str]:
        """Create a device map for single GPU setup with potential CPU offloading."""
        device_map = {}
        available_memory = self.gpu_memory[0] - reserve_memory_gb

        # Simple heuristic: if insufficient memory, offload some experts to CPU
        # We estimate needing at least 14GB for the core model components on a single GPU
        if available_memory < 14.0 or self.force_cpu_offload:
            print(
                f"Warning: Limited GPU memory ({available_memory:.2f}GB). Using CPU offloading."
            )

            # Put core components on GPU
            device_map["model.embed_tokens"] = "cuda:0"
            device_map["model.norm"] = "cuda:0"
            device_map["lm_head"] = "cuda:0"

            # Put layers on GPU, but some experts on CPU
            cpu_expert_ratio = max(0, min(0.75, 1.0 - (available_memory / 24.0)))
            cpu_experts = math.ceil(self.num_experts * cpu_expert_ratio)

            for i in range(self.num_layers):
                device_map[f"model.layers.{i}"] = "cuda:0"
                device_map[f"model.layers.{i}.block_sparse_moe.gate"] = "cuda:0"

                # Distribute experts between GPU and CPU
                for j in range(self.num_experts):
                    if j < (self.num_experts - cpu_experts):
                        device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                            "cuda:0"
                        )
                    else:
                        device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                            "cpu"
                        )
        else:
            # Enough memory - put everything on GPU
            device_map = {"": "cuda:0"}

        return device_map

    def create_auto_device_map(self, model_info: Dict[str, int]) -> Dict[str, str]:
        """
        Create a device map using automatic assignment based on module sizes.

        Args:
            model_info: Dictionary mapping model components to their sizes in bytes

        Returns:
            Device map dictionary
        """
        from accelerate.utils import get_balanced_memory

        # If only one GPU, we do simple auto-mapping
        if self.num_gpus == 1:
            return {"": "cuda:0"}

        # For multi-GPU, we calculate balanced allocation
        max_memory = {
            i: (self.gpu_memory[i] - 2.0) * (1024**3) for i in range(self.num_gpus)
        }
        max_memory["cpu"] = 2 * (1024**3)  # 2 GB CPU memory

        # Use accelerate's balanced memory utility
        return get_balanced_memory(model_info, max_memory, dtype=torch.float16)

    def trace_memory_usage(self, device_map: Dict[str, str]) -> Dict[str, float]:
        """
        Trace expected memory usage based on device map.
        This is an estimate based on model configuration and doesn't account
        for all runtime allocations.

        Args:
            device_map: The device map to analyze

        Returns:
            Dictionary with expected memory usage per device in GB
        """
        # This is a simplified implementation - a real one would need
        # knowledge of actual module sizes

        # For demonstration purposes, we'll make some assumptions
        estimated_expert_size_gb = 1.0  # ~1GB per expert in 4-bit
        estimated_layer_size_gb = 0.5  # ~0.5GB per transformer layer in 4-bit
        estimated_embedding_size_gb = 1.0  # ~1GB for embeddings
        estimated_output_size_gb = 0.5  # ~0.5GB for output layers

        # Count components per device
        device_counts = {}
        for module, device in device_map.items():
            if device not in device_counts:
                device_counts[device] = {"layers": 0, "experts": 0, "other": 0}

            if "layers" in module and ".block_sparse_moe.experts" not in module:
                device_counts[device]["layers"] += 1
            elif ".block_sparse_moe.experts" in module:
                device_counts[device]["experts"] += 1
            elif "embed_tokens" in module:
                device_counts[device]["embed"] = 1
            elif "norm" in module or "lm_head" in module:
                device_counts[device]["output"] = 1

        # Calculate memory usage
        memory_usage = {}
        for device, counts in device_counts.items():
            memory_gb = (
                counts.get("layers", 0) * estimated_layer_size_gb
                + counts.get("experts", 0) * estimated_expert_size_gb
                + counts.get("embed", 0) * estimated_embedding_size_gb
                + counts.get("output", 0) * estimated_output_size_gb
            )
            memory_usage[device] = memory_gb

        return memory_usage

    def apply_device_map(
        self, model: torch.nn.Module, device_map: Dict[str, str]
    ) -> torch.nn.Module:
        """
        Apply a device map to a model, distributing components across devices.

        Args:
            model: The PyTorch model to apply the device map to
            device_map: The device map to apply

        Returns:
            The model with components distributed across devices
        """
        # This is a very simplified implementation
        # In practice, you would use Accelerate or other tools to apply the device map

        # For demonstration purposes only:
        if not device_map or list(device_map.keys()) == [""]:
            # Simple case: move entire model to a single device
            device = device_map.get("", "cuda:0")
            return model.to(device)

        # For complex device maps, we would need to process each component
        # This is better handled by libraries like Accelerate
        from accelerate import dispatch_model

        return dispatch_model(model, device_map)
