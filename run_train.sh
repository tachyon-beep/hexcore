#!/bin/bash
# Training script for MTG experts with optimal memory settings
# This wrapper handles the model training with optimized parameters

# Enable error handling
set -e

# Parse arguments
EXPERT=${1:-REASON}  # Default to REASON if not provided

# Default settings optimized for memory efficiency
MODEL_PATH="mistralai/Mixtral-8x7B-v0.1"
SEQ_LENGTH=1024
BATCH_SIZE=1
GRAD_ACCUM=4
LORA_RANK=8
STEPS=200
SAVE_STEPS=100

echo "Starting training for $EXPERT expert"
echo "Using model: $MODEL_PATH"
echo "Memory-optimized settings: Seq Length=$SEQ_LENGTH, Batch=$BATCH_SIZE, Grad Accum=$GRAD_ACCUM"

# Detect GPU capability for best precision mode
if python -c "import torch; print(torch.cuda.is_bf16_supported())" | grep -q "True"; then
    echo "Using BF16 precision (detected support)"
    PRECISION_FLAG="--bf16"
else
    echo "Using FP16 precision"
    PRECISION_FLAG="--fp16"
fi

# Run the training script
python -m src.training.train_minimal \
    --expert $EXPERT \
    --base-model $MODEL_PATH \
    --seq-length $SEQ_LENGTH \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation $GRAD_ACCUM \
    --lora-r $LORA_RANK \
    --steps $STEPS \
    --save-steps $SAVE_STEPS \
    $PRECISION_FLAG

echo "Training complete for $EXPERT"
