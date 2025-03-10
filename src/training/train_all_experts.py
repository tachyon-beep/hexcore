#!/usr/bin/env python
"""
Script to train all expert adapters for the MTG AI Reasoning Assistant.

This script trains LoRA adapters for all reasoning modes (REASON, EXPLAIN,
TEACH, PREDICT, RETROSPECT) using the specialized training data and configs.
"""

import os
import argparse
import logging
import json
import torch
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional

from transformers import AutoTokenizer

from src.training.adapter_trainer import LoRAAdapterTrainer
from src.training.adapter_dataset import create_expert_dataset_from_jsonl
from src.training.expert_train_configs import get_all_expert_configs, get_expert_config
from src.utils.gpu_memory_tracker import log_memory_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training_log.txt")],
)
logger = logging.getLogger(__name__)


def setup_training_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MTG expert adapters")

    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Base model to fine-tune",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="adapters",
        help="Directory to save adapter weights",
    )

    parser.add_argument(
        "--experts",
        type=str,
        nargs="+",
        default=["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"],
        help="Expert types to train (space-separated)",
    )

    parser.add_argument(
        "--quantization",
        type=int,
        default=4,
        choices=[4, 8, 16],
        help="Quantization bits (4, 8, or 16)",
    )

    parser.add_argument(
        "--mixed-precision", action="store_true", help="Use mixed precision training"
    )

    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy (auto, cpu, cuda:0, etc.)",
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Portion of data to use for validation",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint if available",
    )

    return parser.parse_args()


def train_expert(
    expert_type: str,
    base_model_path: str,
    output_dir: str,
    quantization_bits: int,
    use_mixed_precision: bool,
    device_map: str,
    validation_split: float,
    resume_from_checkpoint: bool,
) -> Dict[str, Any]:
    """
    Train a specific expert adapter.

    Args:
        expert_type: Type of expert to train
        base_model_path: Path to base model
        output_dir: Directory to save adapter
        quantization_bits: Quantization precision
        use_mixed_precision: Whether to use mixed precision
        device_map: Device mapping strategy
        validation_split: Portion of data for validation
        resume_from_checkpoint: Whether to resume from checkpoint

    Returns:
        Training metrics
    """
    logger.info(f"=== Training {expert_type} Expert ===")

    # Get expert config
    expert_config = get_expert_config(expert_type)

    # Set up paths
    expert_output_dir = os.path.join(output_dir, expert_type.lower())
    os.makedirs(expert_output_dir, exist_ok=True)

    # Clean up before loading model
    gc.collect()
    torch.cuda.empty_cache()
    log_memory_usage(f"Before loading model for {expert_type}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create datasets
    data_path = expert_config.to_dict().get("training_data_path")
    # Ensure data_path is a string
    if not data_path or not isinstance(data_path, str):
        data_path = f"data/training/{expert_type.lower()}_examples.jsonl"

    logger.info(f"Creating dataset from {data_path}")
    train_dataset, eval_dataset = create_expert_dataset_from_jsonl(
        expert_type=expert_type,
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=expert_config.max_seq_length,
        validation_split=validation_split,
    )

    # Helper function to safely get dataset size with proper type handling
    def get_safe_dataset_size(dataset):
        """Safely get the size of a dataset, handling various dataset types."""
        if dataset is None:
            return 0

        # Try getting size via len
        if hasattr(dataset, "__len__"):
            try:
                from typing import cast, Sized

                return len(cast(Sized, dataset))
            except Exception:
                pass

        # Try getting size via num_rows
        if hasattr(dataset, "num_rows"):
            try:
                # Access attribute safely
                return dataset.num_rows
            except Exception:
                pass

        # Try getting size via shape
        if hasattr(dataset, "shape"):
            try:
                return dataset.shape[0]
            except Exception:
                pass

        return "unknown"  # Return "unknown" if size cannot be determined

    # Get dataset sizes using the helper function
    train_size = get_safe_dataset_size(train_dataset)
    eval_size = get_safe_dataset_size(eval_dataset)
    logger.info(f"Created datasets - Train: {train_size}, Eval: {eval_size}")

    # Initialize trainer
    trainer = LoRAAdapterTrainer(
        base_model_path=base_model_path,
        expert_type=expert_type,
        output_dir=expert_output_dir,
        quantization_bits=quantization_bits,
        use_mixed_precision=use_mixed_precision,
    )

    # Set up model, tokenizer, and optimizer
    trainer.setup(device_map=device_map)

    # Determine checkpoint path if resuming
    checkpoint_path = None
    if resume_from_checkpoint:
        checkpoint_glob = "checkpoint-*"
        # Use the expert_output_dir as the base path for glob
        checkpoints = sorted(
            Path(expert_output_dir).glob(checkpoint_glob), key=os.path.getctime
        )
        if checkpoints:
            checkpoint_path = str(checkpoints[-1])
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            logger.warning(f"No checkpoints found in {expert_output_dir}")

    # Train the model
    logger.info("Starting training...")
    train_metrics = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_from_checkpoint=checkpoint_path,
    )

    # Save the final adapter
    trainer.save_adapter()

    # Log final memory usage
    log_memory_usage(f"After training {expert_type}")

    return train_metrics


def main():
    """Main training function."""
    args = setup_training_args()

    logger.info(f"Starting training with args: {args}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train each expert
    metrics = {}
    for expert_type in args.experts:
        if expert_type not in ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"]:
            logger.warning(f"Unknown expert type: {expert_type}, skipping")
            continue

        try:
            expert_metrics = train_expert(
                expert_type=expert_type,
                base_model_path=args.base_model,
                output_dir=args.output_dir,
                quantization_bits=args.quantization,
                use_mixed_precision=args.mixed_precision,
                device_map=args.device_map,
                validation_split=args.validation_split,
                resume_from_checkpoint=args.resume,
            )
            metrics[expert_type] = expert_metrics

            # Save metrics to file
            with open(
                os.path.join(args.output_dir, f"{expert_type.lower()}_metrics.json"),
                "w",
            ) as f:
                json.dump(expert_metrics, f, indent=2)

            logger.info(f"Completed training {expert_type} expert")

        except Exception as e:
            logger.error(
                f"Error training {expert_type} expert: {str(e)}", exc_info=True
            )

    logger.info("Training complete for all experts")


if __name__ == "__main__":
    main()
