#!/bin/bash
#SBATCH --job-name=test_trained_small
#SBATCH --partition=gpuV
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/test_trained_small_%j.out
#SBATCH --error=logs/test_trained_small_%j.err

# Load modules and activate environment
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

echo "🧪 Testing TRAINED small models on V100"
echo "Starting at $(date)"

# Test small-24 trained model
echo "🔍 Testing small-24 trained model..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ --model_name tf_lens_small-24-600K_iters --n_layers 24 --first_layer 0 --last_layer 23

# Test small-36 trained model
echo "🔍 Testing small-36 trained model..."
python train_test_chess.py --mode test --probe piece --dataset_prefix stockfish_ --model_name tf_lens_small-36-600k_iters --n_layers 36 --first_layer 0 --last_layer 35

echo "✅ Trained small models testing completed at $(date)"