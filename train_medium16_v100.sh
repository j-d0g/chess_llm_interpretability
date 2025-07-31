#!/bin/bash
#SBATCH --job-name=medium16_probes
#SBATCH --partition=gpuV
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/medium16_probes_%j.out
#SBATCH --error=logs/medium16_probes_%j.err

# Load modules and activate environment
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run probe training for medium-16 model
python train_test_chess.py \
    --mode train \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_medium-16-600K_iters \
    --n_layers 16 \
    --first_layer 0 \
    --last_layer 15

echo "Medium-16 probe training completed"