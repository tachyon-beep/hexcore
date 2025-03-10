# MTG AI Expert Training Guide

This guide explains the training pipeline for the MTG AI Reasoning Assistant's expert adapters.

## Overview

The MTG AI project uses a Mixtral 8x7B Mixture-of-Experts model with fine-tuned LoRA adapters for different reasoning modes (REASON, EXPLAIN, TEACH, PREDICT, RETROSPECT). Each expert requires specialized training on domain-specific data.

## Training Environment Requirements

- CUDA-compatible GPU(s) with at least 16GB VRAM (two 8GB GPUs can work with our optimized scripts)
- Python 3.8+ with PyTorch, Transformers, PEFT, and BitsAndBytes libraries
- Training data organized in JSONL format (samples in `data/training/`)

## Memory-Optimized Training

We've created specialized scripts to enable training on limited GPU resources:

1. **src/training/train_minimal.py**: Core training script with memory optimizations

   - 4-bit NF4 quantization (better quality than standard 4-bit)
   - Gradient checkpointing
   - Parameter-efficient fine-tuning (LoRA)
   - Sequence length reduction
   - Low batch size with gradient accumulation
   - 8-bit optimizers

2. **run_train.sh**: Convenience wrapper with optimal settings

### Memory Usage Optimizations

The standard training would require 40+ GB VRAM, but our optimizations reduce this to ~16GB:

- **4-bit NF4 quantization**: 8x smaller model (40GB → 5GB)
- **LoRA adapters**: Only train 0.1% of parameters (all in smaller matrices)
- **Gradient checkpointing**: Trade computation for memory (2-3x savings)
- **Small batch size with gradient accumulation**: Control memory spikes
- **Automatic mixed precision**: Using BF16 where supported

## Training an Expert

To train an expert adapter:

```bash
# Train with default settings (REASON expert)
./run_train.sh

# Train a specific expert
./run_train.sh EXPLAIN
```

Training will:

1. Load the Mixtral 8x7B model with 4-bit quantization
2. Prepare the model with LoRA configuration
3. Load and process the expert-specific training data
4. Fine-tune the model with optimized memory settings
5. Save the adapter weights to the `adapters/[expert_name]` directory

## Expert Configurations

Each expert has custom configurations in `src/training/expert_train_configs.py`:

- **REASON**: Logical reasoning for MTG rules and interactions (rank 16)
- **EXPLAIN**: Clear rule explanations for players (rank 16 with higher dropout)
- **TEACH**: Educational content for learning the game (rank 8)
- **PREDICT**: Game state and probability analysis (rank 16)
- **RETROSPECT**: Game analysis and decision review (rank 8)

## Debugging and Monitoring

- Training logs will be displayed in the console and written to `training_log.txt`
- TensorBoard logs are saved to `adapters/[expert_name]/runs/`
- Memory usage is logged for monitoring GPU constraints

## Production Training Pipeline

For production training, we recommend:

1. Start with a test run on a small subset of data
2. Monitor memory usage to validate settings
3. Run full training in sequence for all experts (total time: ~5-8 hours)
4. Verify adapters with example inferences

## Data Format

Training data should be in JSONL format with input/output pairs:

```json
{ "input": "MTG rules question...", "output": "Detailed expert answer..." }
```

Each expert type has dedicated data files in `data/training/[expert_type]_examples.jsonl`.
