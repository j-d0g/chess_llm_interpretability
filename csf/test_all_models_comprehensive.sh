#!/bin/bash
#SBATCH --job-name=test_all_comprehensive
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/test_all_comprehensive_%j.out
#SBATCH --error=logs/test_all_comprehensive_%j.err

# Load modules and activate environment
module load Python/3.10.8-GCCcore-12.2.0
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🧪 COMPREHENSIVE TEST EVALUATION - All Models vs Random Baselines"
echo "Starting at $(date)"
echo "=========================================="

# Function to test a model if probes exist
test_model() {
    local model_name=$1
    local n_layers=$2
    local model_type=$3
    
    echo ""
    echo "🔍 Testing $model_type: $model_name ($n_layers layers)"
    echo "----------------------------------------"
    
    # Check if any probe files exist for this model
    probe_count=$(ls linear_probes/${model_name}_chess_piece_probe_layer_*.pth 2>/dev/null | wc -l)
    
    if [ $probe_count -gt 0 ]; then
        echo "✅ Found $probe_count probe files for $model_name"
        echo "🧪 Running test evaluation..."
        
        python train_test_chess.py \
            --mode test \
            --probe piece \
            --dataset_prefix stockfish_ \
            --model_name $model_name \
            --n_layers $n_layers \
            --first_layer 0 \
            --last_layer $((n_layers-1))
            
        echo "✅ Test completed for $model_name"
    else
        echo "⚠️  No probe files found for $model_name - skipping"
    fi
}

echo "🎯 TESTING TRAINED MODELS"
echo "========================="

# Test all trained models
test_model "tf_lens_small-16-600k_iters" 16 "TRAINED"
test_model "tf_lens_small-24-600K_iters" 24 "TRAINED" 
test_model "tf_lens_small-36-600k_iters" 36 "TRAINED"
test_model "tf_lens_medium-16-600K_iters" 16 "TRAINED"
test_model "tf_lens_large-16-600K_iters" 16 "TRAINED"

echo ""
echo "🎲 TESTING RANDOM BASELINE MODELS"
echo "================================="

# Test all random baseline models
test_model "tf_lens_small-16_RANDOM" 16 "RANDOM BASELINE"
test_model "tf_lens_small-24_RANDOM" 24 "RANDOM BASELINE"
test_model "tf_lens_small-36_RANDOM" 36 "RANDOM BASELINE" 
test_model "tf_lens_medium-16_RANDOM" 16 "RANDOM BASELINE"
test_model "tf_lens_large-16_RANDOM" 16 "RANDOM BASELINE"

echo ""
echo "🎉 COMPREHENSIVE TESTING COMPLETED!"
echo "==================================="
echo "Finished at $(date)"

# Generate summary report
echo ""
echo "📊 GENERATING SUMMARY REPORT..."
python -c "
import os
import glob
import re

print('\\n' + '='*60)
print('📊 COMPREHENSIVE TEST RESULTS SUMMARY')
print('='*60)

# Find all test result files
test_files = glob.glob('linear_probes/test_data/*.pkl')

if test_files:
    print(f'\\n✅ Found {len(test_files)} test result files:')
    for f in sorted(test_files):
        filename = os.path.basename(f)
        print(f'  - {filename}')
else:
    print('\\n⚠️  No test result files found yet')

print('\\n' + '='*60)
print('🔍 Check individual log files in logs/ directory for detailed results')
print('='*60)
"

echo "✅ All testing completed successfully at $(date)"