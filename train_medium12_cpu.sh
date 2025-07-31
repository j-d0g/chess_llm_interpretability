#!/bin/bash
# CPU job for medium-12 model on login node
# No SLURM directives - run directly on login node

# Create logs directory if it doesn't exist
mkdir -p logs

# Set output files
LOG_FILE="logs/medium12_probes_$(date +%Y%m%d_%H%M%S).out"
ERR_FILE="logs/medium12_probes_$(date +%Y%m%d_%H%M%S).err"

echo "Starting medium-12 probe training on login node at $(date)"

# Run probe training for medium-12 model
python train_test_chess.py \
    --mode train \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_medium-12-600k_iters \
    --n_layers 12 \
    --first_layer 0 \
    --last_layer 11 \
    > "$LOG_FILE" 2> "$ERR_FILE"

echo "Medium-12 probe training completed at $(date)"
echo "Output logged to: $LOG_FILE"
echo "Errors logged to: $ERR_FILE"