#!/usr/bin/env python
"""
Minimal training script for experts with optimized memory usage.
This script is designed to train on limited GPU resources by using aggressive
quantization, sequence length reduction, and gradient checkpointing.
"""

import os
import argparse
import logging
import torch
import gc
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from transformers import Trainer, default_data_collator
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft.utils.other import prepare_model_for_kbit_training

from src.training.adapter_dataset import create_expert_dataset_from_jsonl
from src.training.expert_train_configs import get_expert_config
from src.utils.gpu_memory_tracker import log_memory_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_minimal")


def setup_training_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MTG expert adapters (minimal version)"
    )

    parser.add_argument(
        "--expert",
        type=str,
        default="REASON",
        choices=["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"],
        help="Expert type to train",
    )

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
        "--seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length (reduce to save memory)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (use 1 for limited memory)",
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,  # Reduced from 16 for memory savings
        help="LoRA rank parameter",
    )

    parser.add_argument(
        "--steps", type=int, default=200, help="Number of training steps"
    )

    parser.add_argument(
        "--save-steps", type=int, default=100, help="Steps between checkpoint saves"
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Portion of data to use for validation",
    )

    parser.add_argument(
        "--fp16", action="store_true", help="Use FP16 mixed precision training"
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision training (preferred if supported by GPU)",
    )

    return parser.parse_args()


def train_expert(args):
    """Train a single expert with memory-optimized settings."""
    torch.cuda.empty_cache()
    gc.collect()
    log_memory_usage(f"Before loading model for {args.expert}")

    expert_type = args.expert.upper()
    expert_config = get_expert_config(expert_type)
    output_dir = os.path.join(args.output_dir, expert_type.lower())
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load tokenizer
    logger.info(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Create dataset
    data_path = expert_config.to_dict().get(
        "training_data_path", f"data/training/{expert_type.lower()}_examples.jsonl"
    )

    logger.info(f"Creating dataset from {data_path}")
    train_dataset, eval_dataset = create_expert_dataset_from_jsonl(
        expert_type=expert_type,
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=args.seq_length,  # Reduced sequence length
        validation_split=args.validation_split,
    )

    # Use string representation for safe size reporting
    train_size = str(train_dataset)
    eval_size = str(eval_dataset) if eval_dataset else "None"
    logger.info(
        f"Dataset created with {train_size} training examples and {eval_size} validation examples"
    )

    # 3. Configure BitsAndBytes with nf4 quantization (newer and better than 4bit)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NF4 data type is more accurate
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,  # Double quantization for more memory savings
        bnb_4bit_quant_storage=torch.uint8,  # Storage type for weights
    )

    # 4. Load model with advanced quantization
    logger.info("Loading model with optimized 4-bit settings")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",  # Let transformers handle device mapping automatically
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    # 5. Apply memory optimizations

    # Prepare model for QLoRA
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,  # Required parameter
        gradient_checkpointing_kwargs=None,  # Optional parameter
    )
    model.config.use_cache = False  # Critical for gradient checkpointing
    model.gradient_checkpointing_enable()

    # 6. Configure LoRA
    logger.info(f"Preparing model with LoRA (rank={args.lora_r})")

    # The target modules for Mixtral are different from smaller models
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=args.lora_r,  # Lower rank
        lora_alpha=args.lora_r * 2,  # Alpha is typically 2x rank
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. Setup training arguments
    precision_type = "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32"
    logger.info(f"Setting up training with {precision_type} precision")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=2e-4,  # Lower learning rate for stability
        num_train_epochs=1,
        max_steps=args.steps,
        # Precision settings
        fp16=args.fp16,
        bf16=args.bf16,
        # Memory optimizations
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        # Reporting and evaluation
        logging_steps=10,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        # Misc
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to=["tensorboard"],
        disable_tqdm=False,
    )

    # 8. Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # 9. Train and save
    logger.info(f"Starting training for {args.steps} steps")
    train_result = trainer.train()

    logger.info(f"Training complete. Saving model to {output_dir}")
    model.save_pretrained(output_dir)

    # 10. Save metrics
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    return train_result.metrics


def main():
    """Main function."""
    args = setup_training_args()

    # Print system info
    logger.info(f"Starting minimal training for {args.expert} expert")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f}GB total memory"
            )

    try:
        metrics = train_expert(args)
        logger.info(f"Training successful with metrics: {metrics}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

    logger.info("Training complete")


if __name__ == "__main__":
    main()
