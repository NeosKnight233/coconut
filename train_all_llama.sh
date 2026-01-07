#!/bin/bash
# Training script for all three Llama models sequentially
# This will train Llama-3.2-1B, Llama-3.2-3B, and Llama-3.1-8B one after another

set -e

NUM_GPUS=4  # Adjust based on your available GPUs
MASTER_PORT=29500

echo "=========================================="
echo "Sequential Training of All Llama Models"
echo "=========================================="
echo ""

# Array of models to train
MODELS=("llama3.2-1b" "llama3.2-3b" "llama3.1-8b")
CONFIG_FILES=(
    "args/gsm_coconut_llama3.2_1b.yaml"
    "args/gsm_coconut_llama3.2_3b.yaml"
    "args/gsm_coconut_llama3.1_8b.yaml"
)

# Check if data files exist
if [ ! -f "data/gsm_train.json" ] || [ ! -f "data/gsm_valid.json" ]; then
    echo "Error: GSM8k data files not found!"
    echo "Please ensure data/gsm_train.json and data/gsm_valid.json exist."
    exit 1
fi

# Create checkpoints directory
mkdir -p checkpoints

# Train each model
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    CONFIG_FILE="${CONFIG_FILES[$i]}"
    
    echo ""
    echo "=========================================="
    echo "Training Model $((i+1))/3: ${MODEL}"
    echo "Config: ${CONFIG_FILE}"
    echo "=========================================="
    echo ""
    
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        run.py ${CONFIG_FILE}
    
    echo ""
    echo "Completed training ${MODEL}"
    echo "Checkpoints saved in: checkpoints/gsm-coconut-${MODEL}/"
    echo ""
done

echo ""
echo "=========================================="
echo "All models training completed!"
echo "=========================================="
