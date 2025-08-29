#!/bin/bash
#SBATCH --job-name=small36_probes
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/small36_probes_%j.out
#SBATCH --error=logs/small36_probes_%j.err

# Load modules and activate environment
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run probe training for small-36 model
python train_test_chess.py \
    --mode train \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_small-36-600k_iters \
    --n_layers 36 \
    --first_layer 0 \
    --last_layer 35

echo "Small-36 probe training completed"