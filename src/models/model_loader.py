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
) -> Tuple[torch.nn.Module, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Load a quantized version of the specified model.

    Args:
        model_id: Hugging Face model ID
        device_map: How to map model across devices ('auto', 'balanced', etc.)
        quantization_type: Type of quantization ('4bit', '8bit', etc.)
        compute_dtype: Data type for computation

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model {model_id} with {quantization_type} quantization")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Configure quantization parameters
    quantization_config = None
    load_in_4bit = False
    load_in_8bit = False

    if quantization_type == "4bit":
        load_in_4bit = True
        # Create a BitsAndBytesConfig object here
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normal float 4-bit
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,  # Double quantization for further compression
        )
    elif quantization_type == "8bit":
        load_in_8bit = True

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )

    logger.info("Model loaded successfully")

    return model, tokenizer


def _distribute_layers(device_map: Dict[str, int], num_layers: int) -> Dict[str, int]:
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
    device_map: Dict[str, int], num_layers: int, num_experts: int, has_router: bool
) -> Dict[str, int]:
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

    for i in range(num_layers):
        layer_device = 0 if i < middle_point else 1

        # Distribute experts
        for j in range(num_experts):
            expert_device = 0 if j < num_experts // 2 else 1
            device_map[f"model.layers.{i}.block_sparse_moe.experts.{j}"] = expert_device

        # Router stays with the layer's GPU
        if has_router:
            device_map[f"model.layers.{i}.block_sparse_moe.gate"] = layer_device

    return device_map


def create_optimized_device_map(
    model_id: str = "mistralai/Mixtral-8x7B-v0.1",
) -> Dict[str, int]:
    """
    Create an optimized device map for the Mixtral model on dual GPUs.

    Args:
        model_id: Hugging Face model ID

    Returns:
        Device mapping dictionary
    """
    from transformers import AutoConfig

    # Load model config to get layer structure
    model_config = AutoConfig.from_pretrained(model_id)

    # Custom device map that distributes layers across GPUs
    device_map: Dict[str, int] = {
        "model.embed_tokens": 0,
        "model.norm": 1,
        "lm_head": 1,
    }

    # Divide the layers between GPUs
    num_layers = getattr(model_config, "num_hidden_layers", 32)
    device_map = _distribute_layers(device_map, num_layers)

    # For MoE models, distribute experts across GPUs
    if hasattr(model_config, "num_local_experts"):
        num_experts = model_config.num_local_experts
        has_router = hasattr(model_config, "router_aux_loss_coef")
        device_map = _distribute_experts(
            device_map, num_layers, num_experts, has_router
        )

    return device_map
