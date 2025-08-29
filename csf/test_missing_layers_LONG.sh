#!/bin/bash
#SBATCH --job-name=test_missing_layers
#SBATCH --partition=gpuV
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/test_missing_layers_%j.out
#SBATCH --error=logs/test_missing_layers_%j.err

# Long-duration job to complete ALL missing test layers
# 12 hours should be more than enough for testing

module load Anaconda3/2022.05
source activate base
cd /mnt/iusers01/fse-ugpgt01/compsci01/j74739jt/scratch/chess_llm_interpretability

echo "=== COMPLETING MISSING TEST LAYERS ==="
echo "Large-16: layers 13-15"
echo "Medium-16: layer 15" 
echo "Small-36: layers 23-35 (CRITICAL PEAK LAYERS)"

# Large-16 missing layers (13-15)
echo "Testing Large-16 layers 13-15..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ \
    --model_name tf_lens_large-16-600K_iters --n_layers 16 --first_layer 13 --last_layer 15

# Medium-16 missing layer (15)
echo "Testing Medium-16 layer 15..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ \
    --model_name tf_lens_medium-16-600K_iters --n_layers 16 --first_layer 15 --last_layer 15

# Small-36 missing layers (23-35) - THE CRITICAL PEAK LAYERS!
echo "Testing Small-36 layers 23-35 (PEAK PERFORMANCE LAYERS)..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ \
    --model_name tf_lens_small-36-600k_iters --n_layers 36 --first_layer 23 --last_layer 35

echo "=== ALL MISSING LAYERS COMPLETE ==="