#!/bin/bash
#SBATCH --job-name=verify_randomization
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_randomization_%j.out
#SBATCH --error=logs/verify_randomization_%j.err

# Load modules and activate environment
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

echo "🔍 FINAL VERIFICATION: Checking all randomized models"
echo "Starting at $(date)"
echo "=" * 60

# Run comprehensive verification
python verify_randomization.py

echo ""
echo "✅ Verification completed at $(date)"