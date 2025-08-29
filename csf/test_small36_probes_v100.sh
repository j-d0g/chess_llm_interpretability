#!/bin/bash
#SBATCH --job-name=test_small36_probes_v100
#SBATCH --partition=gpuV
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/test_small36_probes_v100_%j.out
#SBATCH --error=logs/test_small36_probes_v100_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU Info:"
nvidia-smi
echo "==================="

echo "Testing Small-36 probes on V100..."
cd /mnt/iusers01/fse-ugpgt01/compsci01/j74739jt/scratch/chess_llm_interpretability

python train_test_chess.py \
    --mode test \
    --probe piece \
    --dataset_prefix stockfish_ \
    --model_name tf_lens_small-36-600k_iters \
    --n_layers 36 \
    --first_layer 0 \
    --last_layer 35

echo "=== Test Complete ==="