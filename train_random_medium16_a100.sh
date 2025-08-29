#!/bin/bash
#SBATCH --job-name=random_medium16
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
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

echo "🎲 Creating and training probes on RANDOM WEIGHTS medium-16 model"
echo "Starting at $(date)"

# Create properly randomized version of medium-16 model
python -c "
import torch
import numpy as np

# Set seed for reproducibility
torch.manual_seed(16)
np.random.seed(16)

# Load original model
original_path = 'models/tf_lens_medium-16-600K_iters.pth'
random_path = 'models/tf_lens_medium-16_RANDOM.pth'

print('🎲 Loading original medium-16 model...')
state_dict = torch.load(original_path, map_location='cpu')

print('🎲 Randomizing weights...')
randomized_count = 0
total_params = 0

for key, tensor in state_dict.items():
    param_count = tensor.numel()
    total_params += param_count
    
    if any(x in key for x in ['weight', 'bias', 'W_', 'b_']):
        # Create completely new random tensor with same shape and dtype
        if 'weight' in key or 'W_' in key:
            # Use Xavier/Glorot initialization for weights
            fan_in = tensor.shape[-1] if len(tensor.shape) > 1 else tensor.shape[0]
            fan_out = tensor.shape[0] if len(tensor.shape) > 1 else tensor.shape[0]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            state_dict[key] = torch.randn_like(tensor) * std
        else:
            # Zero initialization for biases
            state_dict[key] = torch.zeros_like(tensor)
        randomized_count += param_count

print(f'✅ Randomized {randomized_count:,} out of {total_params:,} parameters ({100*randomized_count/total_params:.1f}%)')

# Save the randomized model
torch.save(state_dict, random_path)
print(f'✅ Saved randomized model to {random_path}')

# Verify randomization worked
original = torch.load(original_path, map_location='cpu')
randomized = torch.load(random_path, map_location='cpu')
are_identical = torch.equal(original['embed.W_E'], randomized['embed.W_E'])
print(f'🔍 Verification - Are embed weights identical? {are_identical}')
if are_identical:
    raise RuntimeError('❌ RANDOMIZATION FAILED - weights are identical!')
else:
    print('✅ Randomization verified - weights are different')
"

echo "🧪 Training probes on randomized medium-16 model..."

# Train probes on the randomized model
python train_test_chess.py \
    --mode train \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_medium-16_RANDOM \
    --n_layers 16 \
    --first_layer 0 \
    --last_layer 15

echo "✅ Random medium-16 probe training completed at $(date)"