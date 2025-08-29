#!/bin/bash
#SBATCH --job-name=test_random_models
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/test_random_models_%j.out
#SBATCH --error=logs/test_random_models_%j.err

# Load modules and activate environment
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

echo "🧪 Testing RANDOM baseline models on A100"
echo "Starting at $(date)"

# Test random small-16 model
echo "🔍 Testing random small-16 model..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ --model_name tf_lens_small-16_RANDOM --n_layers 16 --first_layer 0 --last_layer 15

# Test random medium-16 model  
echo "🔍 Testing random medium-16 model..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ --model_name tf_lens_medium-16_RANDOM --n_layers 16 --first_layer 0 --last_layer 15

# Test random large-16 model
echo "🔍 Testing random large-16 model..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ --model_name tf_lens_large-16_RANDOM --n_layers 16 --first_layer 0 --last_layer 15

echo "✅ Random models testing completed at $(date)"