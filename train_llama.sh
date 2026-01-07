#!/bin/bash
# Training script for Coconut with Llama models on GSM8k
# Based on the paper's experimental setup:
# - c = 1 (one continuous thought token)
# - Stage 0: 3 epochs
# - Subsequent stages: 1 epoch each

set -e

# Configuration
NUM_GPUS=4  # Adjust based on your available GPUs
MASTER_PORT=29500

# Model selection (uncomment the one you want to train)
# MODEL="llama3.2-1b"
MODEL="llama3.2-3b"
# MODEL="llama3.1-8b"

echo "=========================================="
echo "Training Coconut on GSM8k with ${MODEL}"
echo "=========================================="

# Set the configuration file based on model selection
case $MODEL in
    "llama3.2-1b")
        CONFIG_FILE="args/gsm_coconut_llama3.2_1b.yaml"
        ;;
    "llama3.2-3b")
        CONFIG_FILE="args/gsm_coconut_llama3.2_3b.yaml"
        ;;
    "llama3.1-8b")
        CONFIG_FILE="args/gsm_coconut_llama3.1_8b.yaml"
        ;;
    *)
        echo "Invalid model selection: $MODEL"
        exit 1
        ;;
esac

echo "Using config file: ${CONFIG_FILE}"
echo "Number of GPUs: ${NUM_GPUS}"
echo ""

# Check if data files exist
if [ ! -f "data/gsm_train.json" ] || [ ! -f "data/gsm_valid.json" ]; then
    echo "Error: GSM8k data files not found!"
    echo "Please ensure data/gsm_train.json and data/gsm_valid.json exist."
    echo "You may need to run preprocessing scripts first."
    exit 1
fi

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Run distributed training
echo "Starting training..."
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    run.py ${CONFIG_FILE}

echo ""
echo "Training completed!"
echo "Checkpoints saved in: checkpoints/gsm-coconut-${MODEL}/"
