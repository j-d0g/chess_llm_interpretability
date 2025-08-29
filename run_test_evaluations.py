#!/usr/bin/env python3
"""
Run test evaluations for all trained probe models.
This script runs on the login node and evaluates trained probes on the test set.
"""

import subprocess
import os
import glob
from pathlib import Path
import re

def run_test_evaluation(model_name, n_layers, dataset_prefix="stockfish_"):
    """Run test evaluation for a specific model."""
    print(f"\n🧪 Testing {model_name} ({n_layers} layers)...")
    
    cmd = [
        "python", "train_test_chess.py",
        "--mode", "test",
        "--probe", "piece", 
        "--dataset_prefix", dataset_prefix,
        "--model_name", model_name,
        "--n_layers", str(n_layers)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode == 0:
            print(f"✅ {model_name} test completed successfully")
            return True
        else:
            print(f"❌ {model_name} test failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name} test timed out")
        return False
    except Exception as e:
        print(f"💥 {model_name} test error: {e}")
        return False

def main():
    print("🧪 Running test evaluations for all trained models...")
    
    # Models to test (in order of completion)
    models_to_test = [
        ("tf_lens_large-16-600K_iters", 16),
        ("tf_lens_medium-16-600K_iters", 16),
        ("tf_lens_small-36-600k_iters", 36),
        ("tf_lens_small-24-600K_iters", 24),  # if completed
    ]
    
    results = {}
    
    for model_name, n_layers in models_to_test:
        # Check if we have trained probes for this model
        probe_pattern = f"linear_probes/{model_name}_chess_piece_probe_layer_*.pth"
        probe_files = glob.glob(probe_pattern)
        
        if probe_files:
            print(f"📊 Found {len(probe_files)} probe files for {model_name}")
            success = run_test_evaluation(model_name, n_layers)
            results[model_name] = success
        else:
            print(f"⚠️  No probe files found for {model_name} (pattern: {probe_pattern})")
            results[model_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST EVALUATION SUMMARY:")
    print("="*60)
    
    for model_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model_name}: {status}")
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    print(f"\n🎯 Overall: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All test evaluations completed successfully!")
    else:
        print("⚠️  Some test evaluations failed - check logs above")

if __name__ == "__main__":
    main()