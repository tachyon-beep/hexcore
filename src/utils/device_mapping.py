import torch
import math
from typing import Dict, Union, Any, Optional


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
        self, reserve_memory_gb: float = 2.0, quantization_bits: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Create a device map for Mixtral 8x7B MoE model.

        This implements the strategy described in the technical document,
        distributing model components across available GPUs to optimise memory usage.

        Args:
            reserve_memory_gb: Amount of GPU memory to reserve for other operations (in GB)
            quantization_bits: If provided (4 or 8), ensures compatible device mapping for
                               quantized model (GPU-only for 4-bit)

        Returns:
            A dictionary mapping model components to device strings.
        """
        if self.num_gpus >= 2:
            return self._create_dual_gpu_map(quantization_bits=quantization_bits)
        elif self.num_gpus == 1:
            return self._create_single_gpu_map(reserve_memory_gb)
        else:
            raise RuntimeError("Unexpected error: No GPUs detected.")

    def _create_dual_gpu_map(
        self, quantization_bits: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Create a device map for a dual GPU setup with optimized memory distribution.

        This implementation ensures embedding layers are on the same device
        as the first set of experts they interact with, preventing device
        mismatch errors during inference. It also carefully balances memory
        usage across both GPUs to maximize efficiency.

        Args:
            quantization_bits: If provided (4 or 8), ensure no CPU offloading is used
                               as quantization requires tensors to be on GPU

        Returns:
            A dictionary mapping model components to device strings
        """
        device_map = {}
        use_4bit = quantization_bits == 4

        # Map embedding layers to GPU 0 to match the first experts
        # This is critical to prevent device mismatch during embedding lookup
        device_map["model.embed_tokens"] = self.CUDA0

        # Calculate optimal layer distribution based on expert size
        # Each expert is ~0.75-1GB with 4-bit quantization,
        # while each layer is ~0.2-0.3GB
        total_layers = self.num_layers

        # For more efficient memory distribution, put fewer layers on GPU 0
        # We need to leave more room on GPU 0 for initial tensors and embedding layer
        if use_4bit:
            # With 4-bit quantization, we need much more free memory on GPU 0
            cuda0_layers = (
                total_layers // 2
            ) - 4  # Even fewer layers on GPU 0 for 4-bit
        else:
            # With 8-bit or no quantization, still reduce GPU 0 layers
            cuda0_layers = (total_layers // 2) - 3

        # Ensure cuda0_layers stays in reasonable bounds
        cuda0_layers = max(min(cuda0_layers, total_layers - 4), 4)

        # Distribute layers between GPUs
        for i in range(total_layers):
            device_map[f"model.layers.{i}"] = (
                self.CUDA0 if i < cuda0_layers else self.CUDA1
            )

        # Only keep the first layer's experts on GPU 0 to save memory
        # This is the most critical layer for preventing device mismatch
        critical_layers = 1  # Reduced from 2 to save GPU 0 memory

        # Map experts with special consideration for early layers and memory balance
        experts_on_gpu0 = 0  # Track for balancing
        experts_on_gpu1 = 0

        for i in range(total_layers):
            layer_device = device_map[f"model.layers.{i}"]
            # Map gate to the same device as the layer
            device_map[f"model.layers.{i}.block_sparse_moe.gate"] = layer_device

            for j in range(self.num_experts):
                if i < critical_layers:
                    # Keep ALL experts for early layers on same device as embeddings
                    device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                        self.CUDA0
                    )
                    experts_on_gpu0 += 1
                elif i < cuda0_layers:
                    # For layers on GPU 0, put fewer experts on GPU 0 to save memory
                    # We need to leave more room on GPU 0 for the embedding layer and activations
                    cutoff = (
                        self.num_experts // 4
                    )  # Only 25% of experts on GPU 0 to save memory
                    expert_device = self.CUDA0 if j < cutoff else self.CUDA1
                    device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                        expert_device
                    )
                    if expert_device == self.CUDA0:
                        experts_on_gpu0 += 1
                    else:
                        experts_on_gpu1 += 1
                else:
                    # For layers on GPU 1, minimize experts on GPU 0 even further
                    # GPU 1 layers should primarily use GPU 1 memory
                    cutoff = self.num_experts // 5  # Only 20% of experts on GPU 0
                    expert_device = self.CUDA0 if j < cutoff else self.CUDA1
                    device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                        expert_device
                    )
                    if expert_device == self.CUDA0:
                        experts_on_gpu0 += 1
                    else:
                        experts_on_gpu1 += 1

        # Put normalization layer on GPU 0 to avoid device transfer when computing logits
        # But keep output projection on GPU 1 to save GPU 0 memory
        device_map["model.norm"] = self.CUDA0  # Changed from CUDA1 to CUDA0
        device_map["lm_head"] = self.CUDA1  # Keep on GPU 1

        # Print memory distribution summary
        print(f"Device mapping summary:")
        print(f"  GPU 0: {cuda0_layers} layers, ~{experts_on_gpu0} experts")
        print(
            f"  GPU 1: {total_layers - cuda0_layers} layers, ~{experts_on_gpu1} experts"
        )

        # Validate no CPU placements if 4-bit quantization is used
        if use_4bit and any(device == "cpu" for device in device_map.values()):
            print(
                "WARNING: 4-bit quantization requires all tensors on GPU, but CPU placements were detected"
            )
            print("Removing CPU placements...")
            # Replace any CPU placements with GPU1 (which should have more space with our balancing)
            device_map = {
                k: self.CUDA1 if v == "cpu" else v for k, v in device_map.items()
            }

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

        # Override embedding placement to ensure it's on a specific GPU
        # This prevents device mismatch errors when using auto mapping
        if "model.embed_tokens" in device_map:
            device_map["model.embed_tokens"] = self.CUDA0

        return device_map

    def _calculate_component_sizes(
        self, hidden_size: int, vocab_size: int, quantization: Optional[str]
    ) -> Dict[str, float]:
        """
        Calculate memory requirements for different model components.

        Args:
            hidden_size: Model hidden dimension size
            vocab_size: Model vocabulary size
            quantization: Quantization level ('4bit', '8bit', or None)

        Returns:
            Dictionary with memory estimates for each component type
        """
        # Scaling factor based on hidden size relative to Mixtral 8x7B baseline
        size_scale = (hidden_size / 4096) ** 2

        # Calculate base memory estimates
        base_expert_size_gb = 0.95 * size_scale
        base_layer_size_gb = 0.45 * size_scale
        base_embedding_size_gb = (vocab_size / 32000) * hidden_size / 4096
        base_output_size_gb = 0.45 * size_scale

        # Determine memory factor based on quantization
        memory_factor = 1.0  # Default: No quantization or FP16
        if quantization == "4bit":
            memory_factor = 0.25  # 4-bit is ~1/4 the size of FP16
        elif quantization == "8bit":
            memory_factor = 0.5  # 8-bit is ~1/2 the size of FP16

        # Apply quantization factor to all components
        return {
            "expert": base_expert_size_gb * memory_factor,
            "layer": base_layer_size_gb * memory_factor,
            "embedding": base_embedding_size_gb * memory_factor,
            "output": base_output_size_gb * memory_factor,
        }

    def _count_modules_by_device(
        self, device_map: Dict[str, str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Count module types assigned to each device.

        Args:
            device_map: The device map to analyze

        Returns:
            Dictionary with counts of each module type per device
        """
        device_counts = {}

        for module, device in device_map.items():
            # Initialize counter dict for this device if needed
            if device not in device_counts:
                device_counts[device] = {"layers": 0, "experts": 0, "other": 0}

            # Classify and count module types
            if "layers" in module and ".block_sparse_moe.experts" not in module:
                device_counts[device]["layers"] += 1
            elif ".block_sparse_moe.experts" in module:
                device_counts[device]["experts"] += 1
            elif "embed_tokens" in module:
                device_counts[device]["embed"] = 1
            elif "norm" in module or "lm_head" in module:
                device_counts[device]["output"] = 1

        return device_counts

    def debug_model_memory_requirements(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        quantization_type: Optional[str] = None,
        print_details: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Provide detailed per-component memory requirements to help with device mapping.

        This method estimates memory usage for each model component with fine-grained detail,
        making it easier to understand how memory is distributed and optimize device mappings.

        Args:
            model_config: Model configuration containing hidden_size, vocab_size etc.
                          If None, defaults to Mixtral 8x7B parameters
            quantization_type: Type of quantization ('4bit', '8bit', or None)
            print_details: Whether to print detailed breakdown to console

        Returns:
            Dictionary containing memory requirements by component category and specific components
        """
        # Default model parameters (based on Mixtral 8x7B)
        hidden_size = 4096
        vocab_size = 32000
        ffn_dim = 14336  # 3.5x hidden_size for Mixtral
        num_layers = self.num_layers
        num_experts = self.num_experts

        # If model config is provided, use those parameters instead
        if model_config is not None:
            hidden_size = model_config.get("hidden_size", hidden_size)
            vocab_size = model_config.get("vocab_size", vocab_size)
            ffn_dim = model_config.get("ffn_dim", ffn_dim)
            num_layers = model_config.get("num_hidden_layers", num_layers)
            num_experts = model_config.get("num_local_experts", num_experts)

        # Determine memory factor based on quantization
        memory_factor = 1.0  # Default: No quantization (FP16)
        if quantization_type == "4bit":
            memory_factor = 0.25  # 4-bit is ~1/4 the size of FP16
        elif quantization_type == "8bit":
            memory_factor = 0.5  # 8-bit is ~1/2 the size of FP16

        quant_str = "none" if quantization_type is None else quantization_type

        # Calculate sizes with more granular breakdown
        # All values in GB

        # Embedding layer (input + output embedding weight)
        embed_size_gb = (vocab_size * hidden_size * 2 / (1024**3)) * memory_factor

        # Attention block components per layer
        attn_q_proj_gb = (hidden_size * hidden_size / (1024**3)) * memory_factor
        attn_k_proj_gb = (hidden_size * hidden_size / (1024**3)) * memory_factor
        attn_v_proj_gb = (hidden_size * hidden_size / (1024**3)) * memory_factor
        attn_o_proj_gb = (hidden_size * hidden_size / (1024**3)) * memory_factor
        attn_total_gb = (
            attn_q_proj_gb + attn_k_proj_gb + attn_v_proj_gb + attn_o_proj_gb
        )

        # MoE components per layer
        gate_size_gb = (hidden_size * num_experts / (1024**3)) * memory_factor

        # Each expert size
        expert_size_gb = (hidden_size * ffn_dim * 2 / (1024**3)) * memory_factor

        # Other layer components (layer norms, etc)
        other_layer_gb = (hidden_size * 4 / (1024**3)) * memory_factor

        # Final layer norm and output
        final_norm_gb = (hidden_size / (1024**3)) * memory_factor

        # Total per-layer size excluding experts
        layer_base_gb = attn_total_gb + gate_size_gb + other_layer_gb

        # Create detailed memory map
        memory_map = {
            "embedding": {
                "total": embed_size_gb,
                "description": "Token embeddings (input and output sharing)",
            },
            "layer_base": {
                "total": layer_base_gb,
                "per_layer": layer_base_gb / num_layers,
                "attention": {
                    "total": attn_total_gb,
                    "q_proj": attn_q_proj_gb,
                    "k_proj": attn_k_proj_gb,
                    "v_proj": attn_v_proj_gb,
                    "o_proj": attn_o_proj_gb,
                },
                "gate": gate_size_gb,
                "other": other_layer_gb,
                "description": "Base layer components excluding experts",
            },
            "experts": {
                "total": expert_size_gb * num_experts * num_layers,
                "per_expert": expert_size_gb,
                "per_layer": expert_size_gb * num_experts,
                "description": "MoE experts (FFN weights)",
            },
            "final_layer": {
                "total": final_norm_gb,
                "description": "Final layer normalization",
            },
        }

        # Calculate totals
        total_gb = (
            embed_size_gb  # Embeddings
            + layer_base_gb * num_layers  # Base layer components
            + expert_size_gb * num_experts * num_layers  # All experts
            + final_norm_gb  # Final norm
        )

        memory_map["total"] = {
            "total": total_gb,
            "description": f"Total model size with {quant_str} quantization",
        }

        if print_details:
            self._print_memory_analysis(
                memory_map, num_layers, num_experts, quantization_type
            )

        return memory_map

    def _print_memory_analysis(
        self,
        memory_map: Dict[str, Dict[str, float]],
        num_layers: int,
        num_experts: int,
        quantization_type: Optional[str] = None,
    ) -> None:
        """Print formatted memory analysis to console."""

        quant_str = "none" if quantization_type is None else quantization_type

        # Create width for bar chart (maximum 50 chars)
        max_bar_width = 50
        total_gb = memory_map["total"]["total"]
        gb_to_char = max_bar_width / total_gb if total_gb > 0 else 0

        # Initialize variables for recommendation calculation
        gpu0_memory = 0.0
        gpu1_memory = 0.0

        # Print header
        print("\n" + "=" * 80)
        print(f"Mixtral Memory Analysis ({quant_str} quantization)")
        print("=" * 80)

        # Print embedding layer
        embed_gb = memory_map["embedding"]["total"]
        embed_bar = "█" * int(embed_gb * gb_to_char)
        print(f"Embedding Layer:        {embed_gb:.2f} GB  {embed_bar}")

        # Print layer components (excluding experts)
        layer_base_gb = memory_map["layer_base"]["per_layer"]
        print(f"\nPer-Layer Components (excluding experts):")

        attn_gb = memory_map["layer_base"]["attention"]["total"] / num_layers
        attn_bar = "█" * int(attn_gb * gb_to_char)
        print(f"  Attention Block:      {attn_gb:.2f} GB  {attn_bar}")

        gate_gb = memory_map["layer_base"]["gate"] / num_layers
        gate_bar = "█" * int(gate_gb * gb_to_char)
        print(f"  MoE Gate:             {gate_gb:.2f} GB  {gate_bar}")

        other_gb = memory_map["layer_base"]["other"] / num_layers
        other_bar = "█" * int(other_gb * gb_to_char)
        print(f"  Other (LayerNorms):   {other_gb:.2f} GB  {other_bar}")

        print(f"  -----------------------")
        layer_bar = "█" * int(layer_base_gb * gb_to_char)
        print(f"  Base Layer Total:     {layer_base_gb:.2f} GB  {layer_bar}")

        # Print expert information
        expert_gb = memory_map["experts"]["per_expert"]
        expert_bar = "█" * int(expert_gb * gb_to_char)
        print(f"\nExperts:")
        print(f"  Single Expert:        {expert_gb:.2f} GB  {expert_bar}")

        experts_per_layer_gb = memory_map["experts"]["per_layer"] / num_layers
        experts_layer_bar = "█" * int(experts_per_layer_gb * gb_to_char)
        print(
            f"  All Experts (1 layer): {experts_per_layer_gb:.2f} GB  {experts_layer_bar}"
        )

        # Print final component
        final_gb = memory_map["final_layer"]["total"]
        final_bar = "█" * int(final_gb * gb_to_char)
        print(f"\nFinal Layer:            {final_gb:.2f} GB  {final_bar}")

        # Print totals and estimates by GPU
        print("\nEstimated Total Memory Requirements:")
        print("-" * 50)

        # Calculate optimal split for dual GPUs
        if self.num_gpus >= 2:
            # Rough estimate assuming balanced split
            memory_per_gpu = total_gb / 2
            print(f"Total model size:       {total_gb:.2f} GB")
            print(f"Per GPU (balanced):     {memory_per_gpu:.2f} GB")

            # Calculate what our current device mapping strategy would put on each GPU
            # This is a rough approximation
            gpu0_layers = num_layers // 2 - 3  # Our current strategy
            layer_ratio = gpu0_layers / num_layers

            # First layer experts all on GPU 0
            gpu0_experts = num_experts  # First layer

            # For middle layers on GPU 0, 25% of experts
            if gpu0_layers > 1:
                gpu0_experts += (gpu0_layers - 1) * (num_experts // 4)

            # For GPU 1 layers, 20% of experts on GPU 0
            gpu1_layers = num_layers - gpu0_layers
            gpu0_experts += gpu1_layers * (num_experts // 5)

            gpu1_experts = (num_experts * num_layers) - gpu0_experts

            # Get sizes from memory map
            embed_gb = memory_map["embedding"]["total"]
            layer_base_gb = memory_map["layer_base"]["per_layer"]
            expert_gb = memory_map["experts"]["per_expert"]
            final_gb = memory_map["final_layer"]["total"]

            # Calculate memory
            gpu0_memory = (
                embed_gb  # Embedding
                + layer_base_gb * gpu0_layers  # Base layers
                + expert_gb * gpu0_experts  # Experts
                + final_gb  # Final layer (we put it on GPU 0)
            )

            gpu1_memory = (
                layer_base_gb * gpu1_layers  # Base layers
                + expert_gb * gpu1_experts  # Experts
            )

            print("\nWith Current Mapping Strategy:")
            print(
                f"GPU 0 Memory:           {gpu0_memory:.2f} GB ({gpu0_memory/self.gpu_memory.get(0, 1.0)*100:.1f}% of available {self.gpu_memory.get(0, 1.0):.1f} GB)"
            )
            print(
                f"GPU 1 Memory:           {gpu1_memory:.2f} GB ({gpu1_memory/self.gpu_memory.get(1, 1.0)*100:.1f}% of available {self.gpu_memory.get(1, 1.0):.1f} GB)"
            )
        else:
            print(f"Total model size:       {total_gb:.2f} GB")
            if self.num_gpus >= 1 and 0 in self.gpu_memory:
                print(
                    f"GPU 0 Memory:           {total_gb:.2f} GB ({total_gb/self.gpu_memory[0]*100:.1f}% of available {self.gpu_memory[0]:.1f} GB)"
                )

        print("=" * 80)

        # Recommendations
        print("\nRecommendations:")
        if total_gb > sum(self.gpu_memory.values()):
            print("⚠️ Model is larger than available GPU memory! Consider:")
            print("  • Using 4-bit quantization instead of 8-bit")
            print("  • Offloading some components to CPU")
            print("  • Reducing context length")
        elif self.num_gpus >= 2 and max(gpu0_memory, gpu1_memory) > min(
            self.gpu_memory.values()
        ):
            print("⚠️ Unbalanced distribution exceeds individual GPU memory! Consider:")
            print("  • Reducing expert count on the larger GPU")
            print("  • Moving more base layers to the GPU with fewer experts")
            print("  • Adjusting critical_layers to save GPU memory")
        else:
            print("✓ Current configuration should fit within available GPU memory")

        print("")

    def trace_memory_usage(
        self,
        device_map: Dict[str, str],
        model_config: Optional[Dict[str, Any]] = None,
        quantization: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Trace expected memory usage based on the device map, model configuration, and quantization level.

        Args:
            device_map: The device map to analyze
            model_config: Optional dictionary containing model configuration information
                          Should contain 'hidden_size' and 'vocab_size' if provided
            quantization: Quantization level ('4bit', '8bit', or None for fp16/fp32)

        Returns:
            Dictionary with expected memory usage per device in GB
        """
        # Default model parameters (based on Mixtral 8x7B)
        hidden_size = 4096
        vocab_size = 32000

        # If model config is provided, use those parameters instead
        if model_config is not None:
            hidden_size = model_config.get("hidden_size", hidden_size)
            vocab_size = model_config.get("vocab_size", vocab_size)

        # Calculate component sizes based on model parameters and quantization
        component_sizes = self._calculate_component_sizes(
            hidden_size, vocab_size, quantization
        )

        # Log memory estimates
        quant_str = "no" if quantization is None else quantization
        print(f"Memory estimates (with {quant_str} quantization):")
        print(f"  Expert: {component_sizes['expert']:.2f} GB")
        print(f"  Layer: {component_sizes['layer']:.2f} GB")
        print(f"  Embeddings: {component_sizes['embedding']:.2f} GB")
        print(f"  Output: {component_sizes['output']:.2f} GB")

        # Count modules by device
        device_counts = self._count_modules_by_device(device_map)

        # Calculate memory usage per device
        memory_usage = {}
        for device, counts in device_counts.items():
            memory_gb = (
                counts.get("layers", 0) * component_sizes["layer"]
                + counts.get("experts", 0) * component_sizes["expert"]
                + counts.get("embed", 0) * component_sizes["embedding"]
                + counts.get("output", 0) * component_sizes["output"]
            )
            memory_usage[device] = memory_gb

        return memory_usage

    def ensure_device_consistency(
        self,
        model: torch.nn.Module,
        device_map: Dict[str, Union[str, int, torch.device]],
    ) -> torch.nn.Module:
        """
        Ensures all components are properly aligned on their assigned devices.

        This method verifies that model components are on their assigned devices
        and fixes any inconsistencies, focusing on critical components like
        embedding layers and expert modules.

        Args:
            model: The PyTorch model to verify
            device_map: The device map specifying where each component should be

        Returns:
            The model with verified device placement
        """
        import re

        # Get embedding device
        embedding_device = device_map.get("model.embed_tokens", self.CUDA0)
        embedding_device_str = str(embedding_device)

        # Function to ensure tensor is on correct device
        def ensure_device(module_name, module, assigned_device):
            assigned_device_str = str(assigned_device)

            # Check if any parameter is on the wrong device
            param_device = next(module.parameters(), None)
            if param_device is not None:
                current_device_str = str(param_device.device)
                if current_device_str != assigned_device_str:
                    print(
                        f"Moving {module_name} from {current_device_str} to {assigned_device_str}"
                    )
                    module.to(assigned_device)
                    return True
            return False

        # Analyze model structure and fix device placement
        moved_components = 0

        # Check if model has MoE structure
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Handle embeddings first
            if hasattr(model.model, "embed_tokens"):
                if ensure_device(
                    "model.embed_tokens", model.model.embed_tokens, embedding_device
                ):
                    moved_components += 1

            # Process each layer's components
            for i, layer in enumerate(model.model.layers):
                layer_key = f"model.layers.{i}"
                layer_device = device_map.get(layer_key, embedding_device)

                # Check if this layer has MoE structure
                if hasattr(layer, "block_sparse_moe"):
                    # Handle gate
                    gate_key = f"{layer_key}.block_sparse_moe.gate"
                    gate_device = device_map.get(gate_key, layer_device)
                    if ensure_device(
                        gate_key, layer.block_sparse_moe.gate, gate_device
                    ):
                        moved_components += 1

                    # Handle experts
                    if hasattr(layer.block_sparse_moe, "experts"):
                        for j, expert in enumerate(layer.block_sparse_moe.experts):
                            expert_key = f"{layer_key}.block_sparse_moe.experts.{j}"
                            expert_device = device_map.get(expert_key, embedding_device)
                            if ensure_device(expert_key, expert, expert_device):
                                moved_components += 1

        # Handle output layers
        if hasattr(model.model, "norm"):
            norm_device = device_map.get("model.norm", embedding_device)
            if ensure_device("model.norm", model.model.norm, norm_device):
                moved_components += 1

        if hasattr(model, "lm_head"):
            lm_head_device = device_map.get("lm_head", embedding_device)
            if ensure_device("lm_head", model.lm_head, lm_head_device):
                moved_components += 1

        if moved_components > 0:
            print(f"Fixed device placement for {moved_components} model components")

        return model

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

        # Use accelerate's dispatch_model to distribute model components
        model = dispatch_model(model, device_map)

        # Verify device consistency after dispatch
        model = self.ensure_device_consistency(model, device_map)

        # Ensure CUDA cache is cleared after moving components
        torch.cuda.empty_cache()

        return model
