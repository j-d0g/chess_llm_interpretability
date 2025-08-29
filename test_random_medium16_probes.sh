#!/bin/bash
#SBATCH --job-name=test_random_medium16
#SBATCH --output=logs/test_random_medium16_%j.out
#SBATCH --error=logs/test_random_medium16_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load environment
cd /mnt/iusers01/fse-ugpgt01/compsci01/j74739jt/scratch/chess_llm_interpretability
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# Test trained probes on RANDOM Medium-16 model (baseline comparison)
python train_test_chess.py \
    --mode test \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_medium-16_RANDOM \
    --n_layers 16 \
    --first_layer 0 \
    --last_layer 15

echo "Random Medium-16 baseline testing completed!"