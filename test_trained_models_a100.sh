#!/bin/bash
#SBATCH --job-name=test_trained_models
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/test_trained_models_%j.out
#SBATCH --error=logs/test_trained_models_%j.err

# Load modules and activate environment
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

echo "🧪 Testing TRAINED models on A100"
echo "Starting at $(date)"

# Test small-16 trained model
echo "🔍 Testing small-16 trained model..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ --model_name tf_lens_small-16-600k_iters --n_layers 16 --first_layer 0 --last_layer 15

# Test medium-16 trained model  
echo "🔍 Testing medium-16 trained model..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ --model_name tf_lens_medium-16-600K_iters --n_layers 16 --first_layer 0 --last_layer 15

# Test large-16 trained model
echo "🔍 Testing large-16 trained model..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ --model_name tf_lens_large-16-600K_iters --n_layers 16 --first_layer 0 --last_layer 15

echo "✅ Trained models testing completed at $(date)"