#!/bin/bash
#SBATCH --job-name=random_medium16
#SBATCH --partition=gpuV
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/random_medium16_%j.out
#SBATCH --error=logs/random_medium16_%j.err

# Load modules and activate environment
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🎲 Training probes on RANDOM WEIGHTS medium-16 model for baseline comparison"

# Run probe training on randomized medium-16 model
python -c "
import torch
import os
from pathlib import Path

# Create randomized version of medium-16 model
print('🎲 Creating randomized medium-16 model...')
original_path = 'models/tf_lens_medium-16-600K_iters.pth'
random_path = 'models/tf_lens_medium-16_RANDOM.pth'

state_dict = torch.load(original_path, map_location='cpu')

# Randomize all weights and biases
for key, tensor in state_dict.items():
    if 'weight' in key or 'bias' in key:
        if 'weight' in key:
            torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
        elif 'bias' in key:
            torch.nn.init.zeros_(tensor)

torch.save(state_dict, random_path)
print(f'✅ Saved randomized model to {random_path}')
"

# Train probes on the randomized model
python train_test_chess.py \
    --mode train \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_medium-16_RANDOM \
    --n_layers 16 \
    --first_layer 0 \
    --last_layer 15

echo "✅ Random baseline medium-16 probe training completed"