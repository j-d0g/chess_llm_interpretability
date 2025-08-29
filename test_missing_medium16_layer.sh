#!/bin/bash
#SBATCH --job-name=test_medium16_missing
#SBATCH --output=logs/test_medium16_missing_%j.out
#SBATCH --error=logs/test_medium16_missing_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Load environment
cd /mnt/iusers01/fse-ugpgt01/compsci01/j74739jt/scratch/chess_llm_interpretability
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# Test missing layer 15 for Medium-16
python train_test_chess.py \
    --mode test \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_medium-16-600K_iters \
    --n_layers 16 \
    --first_layer 15 \
    --last_layer 15

echo "Medium-16 missing layer testing completed!"