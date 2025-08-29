#!/bin/bash
#SBATCH --job-name=test_medium16_probes
#SBATCH --partition=gpuA
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_medium16_probes_%j.out
#SBATCH --error=logs/test_medium16_probes_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU Info:"
nvidia-smi
echo "==================="

echo "Testing Medium-16 probes..."
cd /mnt/iusers01/fse-ugpgt01/compsci01/j74739jt/scratch/chess_llm_interpretability

python train_test_chess.py \
    --mode test \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_medium-16-600K_iters \
    --n_layers 16 \
    --first_layer 0 \
    --last_layer 15

echo "=== Test Complete ==="