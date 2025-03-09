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
import logging

logger = logging.getLogger(__name__)


def load_quantized_model(
    model_id: str = "mistralai/Mixtral-8x7B-v0.1",
    device_map: Union[str, Dict[str, Any]] = "auto",
    quantization_type: str = "4bit",
    compute_dtype: torch.dtype = torch.bfloat16,
    use_safetensors: bool = True,
    offload_folder: Optional[str] = None,
    force_device_map: bool = False,
) -> Tuple[torch.nn.Module, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Load a quantized version of the specified model.

    Args:
        model_id: Hugging Face model ID
        device_map: How to map model across devices ('auto', 'balanced', etc.)
        quantization_type: Type of quantization ('4bit', '8bit', etc.)
        compute_dtype: Data type for computation
        use_safetensors: Whether to use safetensors for loading
        offload_folder: Folder to offload model weights to, if needed
        force_device_map: Force the use of the supplied device_map even if it's incompatible

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model {model_id} with {quantization_type} quantization")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Configure loading parameters based on quantization type
    model_kwargs = {}

    # Set default compute dtype if None is specified
    if compute_dtype is None:
        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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

    # CPU offload should be automatically avoided with our device map in 4-bit mode
    # but let's verify anyway
    has_cpu_offload = False
    if isinstance(device_map, dict):
        has_cpu_offload = any(
            str(device).lower() == "cpu" for device in device_map.values()
        )

        # Warn if CPU offloading with 4-bit quantization (which can cause errors)
        if has_cpu_offload and quantization_type == "4bit" and not force_device_map:
            logger.warning(
                "CPU offloading detected with 4-bit quantization, which is incompatible. "
                "Switching to optimized device mapping."
            )
            from src.utils.device_mapping import DeviceMapper

            # Create device mapper with correct settings for model type
            from transformers import AutoConfig

            model_config = AutoConfig.from_pretrained(model_id)
            num_layers = getattr(model_config, "num_hidden_layers", 32)
            num_experts = getattr(model_config, "num_local_experts", 8)

            device_mapper = DeviceMapper(num_experts=num_experts, num_layers=num_layers)
            device_map = device_mapper.create_mixtral_device_map(
                quantization_bits=quantization_bits
            )
            has_cpu_offload = False  # Updated - no CPU offload in optimized map

    # Set the device map
    model_kwargs["device_map"] = device_map

    # Add safetensors option
    model_kwargs["use_safetensors"] = use_safetensors

    # Add offload folder if specified
    if offload_folder:
        model_kwargs["offload_folder"] = offload_folder

    # Set memory-efficient configuration via environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:128,expandable_segments:True"
    )

    # Handle quantization
    if quantization_type == "4bit":
        logger.info("Using 4-bit quantization")
        # Set up 4-bit quantization config with explicit dtype settings
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NF4 provides better quality than int4 for LLMs
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,  # Double quantization for additional memory savings
            # Disable CPU offloading for 4-bit as it's problematic
            bnb_4bit_enable_cpu_offload=False,
        )

        # Always enable low CPU memory usage for large models
        model_kwargs["low_cpu_mem_usage"] = True

    elif quantization_type == "8bit":
        logger.info("Using 8-bit quantization")
        # Set up 8-bit quantization config with explicit CPU offloading setting
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            # Use FP32 for CPU operations, important for quantization
            llm_int8_enable_fp32_cpu_offload=has_cpu_offload,
            llm_int8_skip_modules=None,  # Don't skip any modules
            llm_int8_threshold=6.0,  # Default threshold
            llm_int8_has_fp16_weight=False,  # Don't use FP16 weights
            llm_int8_use_double_quant=True,  # Enable double quantization for 8-bit too
        )

        # Always enable low CPU memory usage for large models
        model_kwargs["low_cpu_mem_usage"] = True
    else:
        logger.info(f"Using no quantization, with dtype {compute_dtype}")
        # For no quantization, just set dtype explicitly
        model_kwargs["torch_dtype"] = compute_dtype

    # Load model with appropriate parameters
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        logger.info("Model loaded successfully")
    except RuntimeError as e:
        # Handle potential tensor device mismatch errors
        if "Tensor for" in str(e) and "was on device" in str(e):
            logger.error(f"Device mismatch error during loading: {e}")
            if not force_device_map:
                logger.info("Trying again with simplified device map")
                # Simplify to balanced device map as fallback
                model_kwargs["device_map"] = "balanced"
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                logger.info("Model loaded successfully with simplified device map")
            else:
                # Re-raise the error if forcing the provided device map
                raise
        else:
            # Re-raise other errors
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
