import torch
import math
from typing import Dict, Union


class DeviceMapper:
    """
    Utility for mapping model components across multiple GPUs.
    Implements different strategies for distributing model components
    to optimise memory usage across available devices.
    """

    CUDA0 = "cuda:0"
    CUDA1 = "cuda:1"

    def __init__(
        self,
        num_experts: int = 8,
        num_layers: int = 32,
        force_cpu_offload: bool = False,
    ):
        """
        Initialise the device mapper.

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

        # Get GPU memory information (in GB)
        self.gpu_memory = {}
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            self.gpu_memory[i] = props.total_memory / (1024**3)

        print(f"DeviceMapper initialised with {self.num_gpus} GPUs")
        for i, mem in self.gpu_memory.items():
            print(f"  GPU {i}: {mem:.2f} GB")

    def create_mixtral_device_map(
        self, reserve_memory_gb: float = 2.0
    ) -> Dict[str, str]:
        """
        Create a device map for Mixtral 8x7B MoE model.

        This implements the strategy described in the technical document,
        distributing model components across available GPUs to optimise memory usage.

        Args:
            reserve_memory_gb: Amount of GPU memory to reserve for other operations (in GB)

        Returns:
            A dictionary mapping model components to device strings.
        """
        if self.num_gpus >= 2:
            return self._create_dual_gpu_map()
        elif self.num_gpus == 1:
            return self._create_single_gpu_map(reserve_memory_gb)
        else:
            raise RuntimeError("Unexpected error: No GPUs detected.")

    def _create_dual_gpu_map(self) -> Dict[str, str]:
        """Create a device map for a dual GPU setup."""
        device_map = {}

        # Map embedding layers to GPU 0
        device_map["model.embed_tokens"] = self.CUDA0

        # Map first half of layers to GPU 0, second half to GPU 1
        half_layers = self.num_layers // 2
        for i in range(self.num_layers):
            device_map[f"model.layers.{i}"] = (
                self.CUDA0 if i < half_layers else self.CUDA1
            )

        # Map experts across GPUs - first half experts to GPU 0, second half to GPU 1
        half_experts = self.num_experts // 2
        for i in range(self.num_layers):
            layer_device = device_map[f"model.layers.{i}"]
            # Map gate to the same device as the layer
            device_map[f"model.layers.{i}.block_sparse_moe.gate"] = layer_device

            # Map experts
            for j in range(self.num_experts):
                expert_device = self.CUDA0 if j < half_experts else self.CUDA1
                device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                    expert_device
                )

        # Map final layers to GPU 1
        device_map["model.norm"] = self.CUDA1
        device_map["lm_head"] = self.CUDA1

        return device_map

    def _create_single_gpu_map(self, reserve_memory_gb: float) -> Dict[str, str]:
        """Create a device map for a single GPU setup with potential CPU offloading."""
        device_map = {}
        available_memory = self.gpu_memory[0] - reserve_memory_gb

        if available_memory < 14.0 or self.force_cpu_offload:
            print(
                f"Warning: Limited GPU memory ({available_memory:.2f}GB). Using CPU offloading."
            )

            # Put core components on GPU
            device_map["model.embed_tokens"] = self.CUDA0
            device_map["model.norm"] = self.CUDA0
            device_map["lm_head"] = self.CUDA0

            # Put layers on GPU, but some experts on CPU
            cpu_expert_ratio = max(0, min(0.75, 1.0 - (available_memory / 24.0)))
            cpu_experts = math.ceil(self.num_experts * cpu_expert_ratio)

            for i in range(self.num_layers):
                device_map[f"model.layers.{i}"] = self.CUDA0
                device_map[f"model.layers.{i}.block_sparse_moe.gate"] = self.CUDA0

                for j in range(self.num_experts):
                    if j < (self.num_experts - cpu_experts):
                        device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                            self.CUDA0
                        )
                    else:
                        device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                            "cpu"
                        )
        else:
            # Enough memory – put everything on GPU.
            device_map = {"": self.CUDA0}

        return device_map

    def create_auto_device_map(self, model: torch.nn.Module) -> Dict[str, str]:
        """
        Create a device map using automatic assignment based on module sizes.

        Args:
            model: The PyTorch model to be mapped.

        Returns:
            Device map dictionary mapping module names to device strings.
        """
        from accelerate.utils import get_balanced_memory

        if self.num_gpus == 1:
            return {"": self.CUDA0}

        max_memory: Dict[Union[int, str], Union[int, str]] = {
            i: int((self.gpu_memory[i] - 2.0) * (1024**3)) for i in range(self.num_gpus)
        }
        max_memory["cpu"] = 2 * (1024**3)

        auto_map = get_balanced_memory(model, max_memory, dtype=torch.float16)
        device_map = {str(k): str(v) for k, v in auto_map.items()}
        return device_map

    def trace_memory_usage(self, device_map: Dict[str, str]) -> Dict[str, float]:
        """
        Trace expected memory usage based on the device map.

        Args:
            device_map: The device map to analyse.

        Returns:
            Dictionary with expected memory usage per device in GB.
        """
        estimated_expert_size_gb = 1.0
        estimated_layer_size_gb = 0.5
        estimated_embedding_size_gb = 1.0
        estimated_output_size_gb = 0.5

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
        self,
        model: torch.nn.Module,
        device_map: Dict[str, Union[str, int, torch.device]],
    ) -> torch.nn.Module:
        """
        Apply a device map to a model, distributing components across devices.

        Args:
            model: The PyTorch model to apply the device map to.
            device_map: The device map to apply.

        Returns:
            The model with components distributed across devices.
        """
        if not device_map or list(device_map.keys()) == [""]:
            device = device_map.get("", self.CUDA0)
            return model.to(device)

        from accelerate import dispatch_model

        return dispatch_model(model, device_map)
