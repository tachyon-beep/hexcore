# src/models/model_loader.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from peft.utils.other import prepare_model_for_kbit_training
import bitsandbytes as bnb
from typing import Dict, Tuple, Optional, Any, Union
import os
import gc
import logging
import time

# Import our memory management utilities
from src.utils.memory_management import (
    clear_gpu_memory,
    log_gpu_memory_stats,
    optimize_for_inference,
    load_with_progressive_offloading,
)

logger = logging.getLogger(__name__)

# Set memory optimization environment variables at the module level
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"


def force_complete_gpu_reset():
    """Force a complete GPU memory reset and garbage collection."""
    import gc
    import torch
    import time

    # Run multiple garbage collection cycles
    for _ in range(3):
        gc.collect()

    # Reset CUDA for each device
    if torch.cuda.is_available():
        # Synchronize all devices to ensure pending operations complete
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

        # Empty cache multiple times
        for _ in range(2):
            torch.cuda.empty_cache()

        # Reset statistics tracking
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.reset_accumulated_memory_stats(i)
            except AttributeError:
                pass  # Not all PyTorch versions have these

        # Small pause to let OS catch up with memory release
        time.sleep(0.5)

    # Set aggressive memory management parameters
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:32,expandable_segments:True"
    )

    # One final garbage collection
    gc.collect()


def _load_model_with_memory_optimizations(
    model_id: str,
    model_kwargs: Dict[str, Any],
    use_progressive_loading: bool = True,
) -> torch.nn.Module:
    """
    Internal function to load a model with optimized memory handling.

    Args:
        model_id: Hugging Face model ID
        model_kwargs: Dictionary of keyword arguments for model loading
        use_progressive_loading: Whether to use progressive loading with memory management

    Returns:
        Loaded model
    """
    # Set up memory optimizations
    optimize_for_inference()

    # Make sure expandable segments are enabled in PyTorch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:64,expandable_segments:True"
    )

    # Log memory state before loading
    logger.info("Memory state before model loading:")
    log_gpu_memory_stats()

    # Clear GPU memory before loading
    clear_gpu_memory()

    if use_progressive_loading:
        # Define internal loading function for progressive loading
        def _internal_load():
            return AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        # Use progressive loading with memory management
        try:
            logger.info("Using progressive loading with memory optimizations")
            model = load_with_progressive_offloading(_internal_load)
        except Exception as e:
            logger.error(f"Progressive loading failed: {e}")
            # Fall back to regular loading as a last resort
            logger.info("Falling back to regular loading")
            clear_gpu_memory()
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    else:
        # Use regular loading with basic memory optimizations
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    # Log memory state after loading
    logger.info("Memory state after model loading:")
    log_gpu_memory_stats()

    return model


def load_quantized_model(
    model_id: str = "mistralai/Mixtral-8x7B-v0.1",
    device_map: Union[str, Dict[str, Any]] = "auto",
    quantization_type: str = "4bit",
    compute_dtype: torch.dtype = torch.bfloat16,
    use_safetensors: bool = True,
    offload_folder: Optional[str] = None,
    force_device_map: bool = False,
    use_memory_optimizations: bool = True,
) -> Tuple[torch.nn.Module, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Load a quantized version of the specified model with memory optimizations.

    Args:
        model_id: Hugging Face model ID
        device_map: How to map model across devices ('auto', 'balanced', etc.)
        quantization_type: Type of quantization ('4bit', '8bit', etc.)
        compute_dtype: Data type for computation
        use_safetensors: Whether to use safetensors for loading
        offload_folder: Folder to offload model weights to, if needed
        force_device_map: Force the use of the supplied device_map even if it's incompatible
        use_memory_optimizations: Whether to use enhanced memory optimizations

    Returns:
        Tuple of (model, tokenizer)
    """
    # Start with a complete memory reset
    force_complete_gpu_reset()

    logger.info(f"Loading model {model_id} with {quantization_type} quantization")

    # Load tokenizer first (lightweight)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set up conservative memory usage limits from the start
    # Calculate safe memory limits for each device
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024**3)
        # Leave a safe buffer of 1GB on each device
        safe_mem_gb = total_mem_gb - 1.0
        # Ensure we don't go negative
        safe_mem_gb = max(safe_mem_gb, total_mem_gb * 0.7)
        max_memory[i] = f"{safe_mem_gb:.1f}GiB"

    # Always allow CPU offloading as a safety valve
    max_memory["cpu"] = "4GiB"

    # Build model kwargs with conservative settings
    model_kwargs = {
        "device_map": "auto",  # Start with auto mapping for safety
        "max_memory": max_memory,
        "torch_dtype": compute_dtype if compute_dtype else torch.float16,
        "use_safetensors": use_safetensors,
        "low_cpu_mem_usage": True,
    }

    # Apply quantization settings
    if quantization_type == "4bit":
        logger.info("Using 4-bit quantization with conservative memory settings")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype or torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_type == "8bit":
        logger.info("Using 8-bit quantization with conservative memory settings")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_use_double_quant=True,
        )

    # If a custom device map was provided and forced, use it instead
    if isinstance(device_map, dict) and force_device_map:
        logger.info("Using forced custom device map")
        model_kwargs["device_map"] = device_map
        # Remove max_memory constraint when using custom mapping
        if "max_memory" in model_kwargs:
            del model_kwargs["max_memory"]

    # Convert quantization type to bits for DeviceMapper
    quantization_bits = None
    if quantization_type == "4bit":
        quantization_bits = 4
    elif quantization_type == "8bit":
        quantization_bits = 8

    # Set up optimized device mapping if needed
    if device_map in ["auto", "balanced"] and torch.cuda.device_count() >= 2:
        logger.info("Using optimized device mapping for multi-GPU setup")
        from transformers import AutoConfig
        from src.utils.device_mapping import DeviceMapper

        # Get model config to determine layer count and expert count
        model_config = AutoConfig.from_pretrained(model_id)
        num_layers = getattr(model_config, "num_hidden_layers", 32)
        num_experts = getattr(model_config, "num_local_experts", 8)

        # Create device mapper and generate optimized mapping
        device_mapper = DeviceMapper(num_experts=num_experts, num_layers=num_layers)
        device_map = device_mapper.create_mixtral_device_map(
            quantization_bits=quantization_bits
        )
        logger.info("Created optimized device map with memory balancing")
        model_kwargs["device_map"] = device_map

    # Attempt to load the model with a single, controlled strategy
    logger.info(f"Loading model with kwargs: {model_kwargs}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Clean up before raising the exception
        force_complete_gpu_reset()
        raise

    return model, tokenizer


def _distribute_layers(
    device_map: Dict[str, Union[int, str]], num_layers: int
) -> Dict[str, Union[int, str]]:
    """
    Distribute model layers across GPUs.

    Args:
        device_map: Existing device map to update
        num_layers: Number of transformer layers in the model

    Returns:
        Updated device map with layer assignments
    """
    middle_point = num_layers // 2

    for i in range(num_layers):
        device_map[f"model.layers.{i}"] = 0 if i < middle_point else 1

    return device_map


def _distribute_experts(
    device_map: Dict[str, Union[int, str]],
    num_layers: int,
    num_experts: int,
    has_router: bool,
) -> Dict[str, Union[int, str]]:
    """
    Distribute MoE experts across GPUs.

    Args:
        device_map: Existing device map to update
        num_layers: Number of transformer layers
        num_experts: Number of experts per layer
        has_router: Whether the model has a router component

    Returns:
        Updated device map with expert assignments
    """
    middle_point = num_layers // 2

    # Critical early layers that should keep all experts on same device as embeddings
    # to minimize cross-device operations during the critical first passes
    critical_layers = 3  # First 3 layers are critical

    for i in range(num_layers):
        layer_device = 0 if i < middle_point else 1

        # Router stays with the layer's GPU
        if has_router:
            device_map[f"model.layers.{i}.block_sparse_moe.gate"] = layer_device

        # Distribute experts - with special handling for early layers
        for j in range(num_experts):
            if i < critical_layers:
                # For early layers, keep ALL experts on same device as embeddings (GPU 0)
                # This is critical to prevent device mismatch errors during early passes
                device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = 0
            else:
                # For other layers, balance experts between GPUs
                expert_device = 0 if j < num_experts // 2 else 1
                device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = (
                    expert_device
                )

    return device_map


def create_optimized_device_map(
    model_id: str = "mistralai/Mixtral-8x7B-v0.1",
    quantization_type: Optional[str] = None,
) -> Dict[str, Union[int, str]]:
    """
    Create an optimized device map for the Mixtral model on dual GPUs.

    This function is a wrapper around the more sophisticated DeviceMapper class.

    Args:
        model_id: Hugging Face model ID
        quantization_type: Type of quantization ('4bit', '8bit', or None)

    Returns:
        Device mapping dictionary
    """
    from transformers import AutoConfig
    import torch
    from src.utils.device_mapping import DeviceMapper

    # Convert quantization type to bits for DeviceMapper
    quantization_bits = None
    if quantization_type == "4bit":
        quantization_bits = 4
    elif quantization_type == "8bit":
        quantization_bits = 8

    # Check if we have enough GPUs
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        logger.warning(
            f"Only {gpu_count} GPU(s) available. Dual GPU optimization requires 2 GPUs."
        )
        # Return a simplified map for single GPU
        return {"": 0} if gpu_count > 0 else {"": "cpu"}

    # Log GPU information
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            f"GPU {i}: {props.name} with {props.total_memory / (1024**3):.1f}GB memory"
        )

    # Load model config to get layer structure
    model_config = AutoConfig.from_pretrained(model_id)
    num_layers = getattr(model_config, "num_hidden_layers", 32)
    num_experts = getattr(model_config, "num_local_experts", 8)

    # Use DeviceMapper for optimized distribution
    device_mapper = DeviceMapper(num_experts=num_experts, num_layers=num_layers)
    device_map_str = device_mapper.create_mixtral_device_map(
        quantization_bits=quantization_bits
    )

    # Convert string device map to int for compatibility with HF Accelerate
    device_map = {}
    for k, v in device_map_str.items():
        if v == "cuda:0":
            device_map[k] = 0
        elif v == "cuda:1":
            device_map[k] = 1
        else:
            device_map[k] = v  # Keep 'cpu' as string

    # Count components per device for logging
    gpu0_count = sum(1 for device in device_map.values() if device == 0)
    gpu1_count = sum(1 for device in device_map.values() if device == 1)
    cpu_count = sum(1 for device in device_map.values() if device == "cpu")

    logger.info(f"Created optimized device map for {model_id} across {gpu_count} GPUs")
    logger.info(
        f"Device distribution: {gpu0_count} components on GPU 0, {gpu1_count} components on GPU 1"
    )

    if cpu_count > 0:
        logger.info(f"CPU offloading: {cpu_count} components on CPU")
        if quantization_type == "4bit":
            logger.warning(
                "CPU offloading with 4-bit quantization may cause errors. "
                "Consider using 8-bit quantization instead."
            )

    return device_map
