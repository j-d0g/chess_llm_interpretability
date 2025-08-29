#!/bin/bash
#SBATCH --job-name=test_random_small24
#SBATCH --output=logs/test_random_small24_%j.out
#SBATCH --error=logs/test_random_small24_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpuV
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load environment
cd /mnt/iusers01/fse-ugpgt01/compsci01/j74739jt/scratch/chess_llm_interpretability
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# Test trained probes on RANDOM Small-24 model (baseline comparison)
python train_test_chess.py \
    --mode test \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_small-24_RANDOM \
    --n_layers 24 \
    --first_layer 0 \
    --last_layer 23

echo "Random Small-24 baseline testing completed!"