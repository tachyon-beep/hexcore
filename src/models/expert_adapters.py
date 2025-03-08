# src/models/expert_adapters.py

from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
import torch
import os
import logging
from transformers import PreTrainedModel
from typing import Dict, Any, Optional, List, Union, cast

logger = logging.getLogger(__name__)


class ExpertAdapterManager:
    """
    Manages LoRA adapters for different experts in the MTG AI Assistant.
    """

    def __init__(self, base_model, adapters_dir: str = "adapters"):
        """
        Initialize the expert adapter manager.

        Args:
            base_model: Base language model to apply adapters to
            adapters_dir: Directory where adapter weights are stored
        """
        self.base_model = base_model
        self.adapters_dir = adapters_dir
        self.current_adapter = None
        self.expert_configs = self._get_expert_configs()

        # Map of expert types to their loaded adapter models
        self.expert_adapters = {}

        # Load all available adapters
        self._load_available_adapters()

    def _get_expert_configs(self) -> Dict[str, LoraConfig]:
        """Define LoRA configurations for each expert type."""
        return {
            "REASON": LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            "EXPLAIN": LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.10,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            "TEACH": LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            "PREDICT": LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            "RETROSPECT": LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        }

    def _load_available_adapters(self):
        """Load all available adapters from the adapters directory."""
        os.makedirs(self.adapters_dir, exist_ok=True)

        for expert_type in self.expert_configs.keys():
            adapter_path = os.path.join(self.adapters_dir, expert_type.lower())
            if os.path.exists(adapter_path):
                try:
                    logger.info(f"Loading adapter for {expert_type}")
                    self.expert_adapters[expert_type] = PeftModel.from_pretrained(
                        self.base_model, adapter_path
                    )
                    logger.info(f"Successfully loaded adapter for {expert_type}")
                except Exception as e:
                    logger.error(f"Failed to load adapter for {expert_type}: {str(e)}")

    def apply_adapter(self, expert_type: str) -> bool:
        """
        Apply the adapter for the specified expert type.

        Args:
            expert_type: The expert type to activate

        Returns:
            True if adapter was successfully applied, False otherwise
        """
        # If this expert is already active, do nothing
        if self.current_adapter == expert_type:
            return True

        # First offload inactive experts to save memory
        self.offload_inactive_experts(expert_type)

        # Check if adapter is loaded
        if expert_type in self.expert_adapters:
            # Use the pre-loaded adapter
            model = self.expert_adapters[expert_type]

            # Replace the active model
            self.base_model = model
            self.current_adapter = expert_type
            return True
        else:
            # Try to create a new adapter if we have the config
            if expert_type in self.expert_configs:
                try:
                    config = self.expert_configs[expert_type]

                    # Get the original base model if current model is a PeftModel
                    base_model = self.base_model
                    if isinstance(base_model, PeftModel):
                        # Extract the base model to avoid type issues
                        base_model = base_model.get_base_model()

                    # Create new adapter with the proper base model
                    adapted_model = get_peft_model(
                        cast(PreTrainedModel, base_model), config
                    )
                    self.expert_adapters[expert_type] = adapted_model
                    self.base_model = adapted_model
                    self.current_adapter = expert_type
                    return True
                except Exception as e:
                    logger.error(
                        f"Failed to create adapter for {expert_type}: {str(e)}"
                    )
                    return False
            else:
                logger.warning(f"No configuration for expert type {expert_type}")
                return False

    def get_active_model(self):
        """Get the currently active model with applied adapter."""
        return self.base_model

    def create_adapter(self, expert_type: str, training_data: Any) -> bool:
        """
        Create and train a new adapter for the specified expert type.

        Args:
            expert_type: The expert type to create an adapter for
            train_data: Training data for the adapter

        Returns:
            True if adapter was successfully created, False otherwise
        """
        # Implementation of adapter training
        # This would typically involve fine-tuning on expert-specific data
        # and saving the resulting adapter weights

        logger.info(f"Training adapter for {expert_type} - Not implemented yet")
        return False  # Return False since it's not implemented

    def offload_inactive_experts(self, active_expert_type):
        """
        Offload all inactive experts to CPU to save GPU memory.

        Args:
            active_expert_type: Type of the currently active expert
        """
        logger.info(
            f"Offloading inactive experts (keeping {active_expert_type} active)"
        )

        for expert_type, adapter_model in self.expert_adapters.items():
            if expert_type != active_expert_type:
                # Move inactive expert to CPU
                try:
                    # Only move if it's currently on GPU
                    if next(adapter_model.parameters()).device.type == "cuda":
                        adapter_model.to("cpu")
                        logger.debug(f"Expert {expert_type} moved to CPU")

                        # Force garbage collection to free GPU memory
                        import gc

                        gc.collect()
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error moving expert {expert_type} to CPU: {str(e)}")
