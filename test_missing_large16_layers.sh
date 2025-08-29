#!/bin/bash
#SBATCH --job-name=test_large16_missing
#SBATCH --output=logs/test_large16_missing_%j.out
#SBATCH --error=logs/test_large16_missing_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Load environment
cd /mnt/iusers01/fse-ugpgt01/compsci01/j74739jt/scratch/chess_llm_interpretability
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# Test missing layers 13-15 for Large-16
python train_test_chess.py \
    --mode test \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_large-16-600K_iters \
    --n_layers 16 \
    --first_layer 13 \
    --last_layer 15

echo "Large-16 missing layers testing completed!"