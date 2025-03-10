"""
Trainer for LoRA adapters with memory optimization for dual 16GB GPUs.

This module provides a complete training pipeline for fine-tuning expert-specific
LoRA adapters with advanced memory optimization strategies.
"""

import os
import torch
import logging
import gc
import time
import math
import json  # Added explicit json import here
from typing import Dict, Any, Optional, List, Union, Tuple, cast, TYPE_CHECKING
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn as nn

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Import PEFT components
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.utils.other import prepare_model_for_kbit_training

from src.training.mixed_precision import MixedPrecisionTrainer
from src.training.expert_train_configs import get_expert_config

logger = logging.getLogger(__name__)


class LoRAAdapterTrainer:
    """
    Trainer for LoRA adapters with memory optimization for dual 16GB GPUs.

    Features:
    - 4-bit quantization support
    - Gradient checkpointing
    - Mixed precision training
    - Memory-efficient optimization
    - Multi-GPU support
    """

    def __init__(
        self,
        base_model_path: str,
        expert_type: str,
        output_dir: str,
        quantization_bits: int = 4,
        use_mixed_precision: bool = True,
        override_lora_config: Optional[Dict[str, Any]] = None,
        override_train_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the LoRA adapter trainer.

        Args:
            base_model_path: Path to the base model
            expert_type: Expert type to train
            output_dir: Output directory for checkpoints
            quantization_bits: Quantization precision (4 or 8)
            use_mixed_precision: Whether to use mixed precision training
            override_lora_config: Optional override for LoRA configuration
            override_train_params: Optional override for training parameters
        """
        self.base_model_path = base_model_path
        self.expert_type = expert_type
        self.output_dir = output_dir
        self.quantization_bits = quantization_bits
        self.use_mixed_precision = use_mixed_precision

        # Get expert configuration
        self.expert_config = get_expert_config(
            expert_type, override_params=override_train_params
        )

        # Create LoRA config from expert config
        self.lora_config = {
            "r": self.expert_config.lora_r,
            "lora_alpha": self.expert_config.lora_alpha,
            "lora_dropout": self.expert_config.lora_dropout,
            "target_modules": self.expert_config.target_modules,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

        # Apply overrides to LoRA config if provided
        if override_lora_config:
            self.lora_config.update(override_lora_config)

        # Handle max_steps for calculating num_epochs carefully to avoid MagicMock comparison errors
        max_steps = getattr(self.expert_config, "max_steps", 100)
        # Convert to int if it's a MagicMock to avoid comparison errors
        if not isinstance(max_steps, (int, float)):
            max_steps = 100  # Default value if it's a mock

        # Training parameters from expert config
        self.train_params = {
            "batch_size": self.expert_config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.expert_config.gradient_accumulation_steps,
            "learning_rate": self.expert_config.learning_rate,
            "warmup_ratio": 0.03,  # Default warmup ratio
            "weight_decay": self.expert_config.weight_decay,
            "num_epochs": max(1, math.ceil(max_steps / 100)),  # Approximate epochs
            "max_grad_norm": 1.0,  # Default max gradient norm
        }

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.mp_trainer = MixedPrecisionTrainer(use_amp=use_mixed_precision)

    def setup(self, device_map: Union[str, Dict[str, Any]] = "auto"):
        """
        Setup model, tokenizer, and optimizer.

        Args:
            device_map: Device mapping strategy
        """
        logger.info(f"Setting up LoRA adapter training for {self.expert_type}")

        start_time = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        logger.info(f"Loaded tokenizer in {time.time() - start_time:.2f}s")

        # Load quantized model for kbit training
        logger.info(f"Loading quantized model with {self.quantization_bits} bits...")
        load_start = time.time()

        # Force complete memory cleanup before loading model
        gc.collect()
        torch.cuda.empty_cache()

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            load_in_4bit=(self.quantization_bits == 4),
            load_in_8bit=(self.quantization_bits == 8),
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        logger.info(f"Loaded base model in {time.time() - load_start:.2f}s")

        # Prepare model for kbit training
        logger.info("Preparing model for kbit training...")
        prepare_start = time.time()
        self.model = prepare_model_for_kbit_training(self.model)
        logger.info(f"Prepared model in {time.time() - prepare_start:.2f}s")

        # Configure LoRA
        logger.info(f"Applying LoRA config with rank {self.lora_config['r']}...")
        lora_start = time.time()
        peft_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.model, peft_config)

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

        # Use contiguous parameters for better memory efficiency
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = param.data.contiguous()

        logger.info(f"Applied LoRA in {time.time() - lora_start:.2f}s")
        logger.info(f"Total setup time: {time.time() - start_time:.2f}s")

        # Print trainable parameters
        self.model.print_trainable_parameters()

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Train the adapter with memory optimization.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Optional checkpoint to resume from

        Returns:
            Training metrics
        """
        logger.info(f"Starting training for {self.expert_type}...")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            # Batch size and optimization
            per_device_train_batch_size=self.train_params["batch_size"],
            gradient_accumulation_steps=self.train_params[
                "gradient_accumulation_steps"
            ],
            # Learning rate and schedule
            learning_rate=self.train_params["learning_rate"],
            lr_scheduler_type="cosine",
            warmup_ratio=self.train_params["warmup_ratio"],
            weight_decay=self.train_params["weight_decay"],
            # Training length
            num_train_epochs=self.train_params["num_epochs"],
            # Mixed precision training
            fp16=self.use_mixed_precision,
            fp16_full_eval=self.use_mixed_precision,
            # Evaluation & Logging
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            logging_steps=10,
            report_to="tensorboard",
            # Optimizer
            optim="paged_adamw_8bit",
            max_grad_norm=self.train_params["max_grad_norm"],
            # Multi-GPU settings
            deepspeed=None,  # We'll handle our own memory optimization
            local_rank=-1,
            # Misc
            remove_unused_columns=False,
            dataloader_drop_last=True,
            dataloader_num_workers=1,
            # Gradient checkpointing
            gradient_checkpointing=True,
        )

        # Initialize trainer - ensure model is not None first
        if self.model is None:
            raise ValueError("Model is not initialized. Call setup() first.")

        # Now we know self.model is not None, cast it to satisfy the type checker
        model = cast(nn.Module, self.model)

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )

        # Setup mixed precision hooks
        if self.use_mixed_precision:
            self._setup_mixed_precision_hooks()

        # Train the model
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save model and metrics
        self.trainer.save_model()
        self.trainer.log_metrics("train", train_result.metrics)
        self.trainer.save_metrics("train", train_result.metrics)
        self.trainer.save_state()

        # Get mixed precision training statistics
        mp_stats = self.mp_trainer.get_statistics() if self.use_mixed_precision else {}
        logger.info(f"Mixed precision stats: {mp_stats}")

        return {**train_result.metrics, "mp_stats": mp_stats}

    def save_adapter(self, path: Optional[str] = None):
        """
        Save adapter weights.

        Args:
            path: Path to save adapter (defaults to output_dir)
        """
        save_path = path or self.output_dir
        os.makedirs(save_path, exist_ok=True)

        logger.info(f"Saving adapter to {save_path}")

        # Check if model is initialized
        if self.model is None:
            raise ValueError("Model is not initialized. Call setup() first.")

        # Save model
        self.model.save_pretrained(save_path)

        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)

        # Save expert-specific config
        expert_config_path = os.path.join(save_path, "expert_config.json")

        # Extract serializable parts of the config
        serializable_config = {
            "expert_type": self.expert_type,
            "description": self.expert_config.additional_params.get("description", ""),
            "lora_config": self.lora_config,
            "training_params": self.train_params,
        }

        # Safely serialize the config to JSON
        with open(expert_config_path, "w") as f:
            json.dump(serializable_config, f, indent=2)

        logger.info(f"Saved expert config to {expert_config_path}")

    def evaluate(self, eval_dataset):
        """
        Evaluate adapter performance.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Evaluation metrics
        """
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call setup() and train() first.")

        logger.info("Running evaluation...")
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)

        # Log and save metrics
        logger.info(f"Evaluation results: {eval_result}")
        self.trainer.log_metrics("eval", eval_result)
        self.trainer.save_metrics("eval", eval_result)

        return eval_result

    def _setup_mixed_precision_hooks(self):
        """Setup hooks for the mixed precision trainer."""
        if not self.use_mixed_precision or self.trainer is None:
            return

        # Create a reference to self.mp_trainer for use in the hook
        mp_trainer = self.mp_trainer

        # Get the original compute_loss method which has the right signature
        original_compute_loss = self.trainer.compute_loss

        # Define a new compute_loss with mixed precision
        def compute_loss_with_mixed_precision(
            model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            with mp_trainer.get_ctx_manager():
                loss_and_outputs = original_compute_loss(
                    model, inputs, return_outputs, num_items_in_batch
                )

            # If return_outputs is True, loss_and_outputs is a tuple (loss, outputs)
            # We need to make sure the tuple is properly formed before unpacking
            if return_outputs:
                # If it's a tuple, unpack it normally
                if isinstance(loss_and_outputs, tuple) and len(loss_and_outputs) >= 2:
                    loss, outputs = loss_and_outputs
                    mp_trainer.backward(loss)
                    return loss, outputs
                else:
                    # If not a tuple with at least 2 elements, log a warning and handle it
                    logger.warning(
                        f"Expected tuple of length 2 for loss_and_outputs when return_outputs=True, got {loss_and_outputs}"
                    )
                    # Create a dummy outputs dict if needed for compatibility
                    loss = loss_and_outputs  # Use whatever we got as the loss
                    outputs = {}  # Empty outputs as a fallback
                    mp_trainer.backward(loss)
                    return loss, outputs
            else:
                # Otherwise, it's just the loss
                loss = loss_and_outputs
                mp_trainer.backward(loss)
                return loss

        # Replace the method with our version
        self.trainer.compute_loss = compute_loss_with_mixed_precision
