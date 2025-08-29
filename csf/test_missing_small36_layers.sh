#!/bin/bash
#SBATCH --job-name=test_small36_missing
#SBATCH --output=logs/test_small36_missing_%j.out
#SBATCH --error=logs/test_small36_missing_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpuV
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load environment
cd /mnt/iusers01/fse-ugpgt01/compsci01/j74739jt/scratch/chess_llm_interpretability
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# Test missing layers 23-35 for Small-36 (INCLUDES CRITICAL PEAK LAYERS 25-28!)
python train_test_chess.py \
    --mode test \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_small-36-600k_iters \
    --n_layers 36 \
    --first_layer 23 \
    --last_layer 35

echo "Small-36 missing layers testing completed! Peak layers 25-28 included!"