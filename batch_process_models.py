#!/usr/bin/env python3
"""
Batch Processing Script for Multiple Chess Models
Automatically converts and evaluates all models in the models directory
"""

import os
import subprocess
from pathlib import Path
import time

# Your 7 models (from the attached folder)
MODELS = [
    "ChessGPT_50m_lmm_lr1e-4_gac8.pt",
    "ChessGPT_50m_lmm_lr1e-4_gac16.pt", 
    "ChessGPT_50m_lmm_lr3e-4_gac8.pt",
    "ChessGPT_50m_lmm_lr3e-4_gac16.pt",
    "ChessGPT_50m_lmm_lr5e-4_gac8.pt",
    "ChessGPT_50m_lmm_lr5e-4_gac16.pt",
    "ChessGPT_100m_lmm_lr3e-4_gac8.pt"
]

def convert_model(model_name):
    """Convert nanoGPT model to TransformerLens format"""
    print(f"\n🔄 Converting {model_name}...")
    
    # Update model_setup.py for this specific model
    model_path = f"models/{model_name}"
    
    # Run conversion
    cmd = f"python model_setup.py --model_path {model_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {model_name} converted successfully")
        return True
    else:
        print(f"❌ Failed to convert {model_name}: {result.stderr}")
        return False

def train_probes(model_name):
    """Train piece and skill probes for a model"""
    print(f"\n🧠 Training probes for {model_name}...")
    
    tf_model_name = model_name.replace('.pt', '.pth')
    
    # Train piece probe
    print(f"   Training piece probe...")
    cmd = f"python train_test_chess.py --probe piece --model_name {tf_model_name}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Piece probe failed: {result.stderr}")
        return False
    
    # Train skill probe  
    print(f"   Training skill probe...")
    cmd = f"python train_test_chess.py --probe skill --model_name {tf_model_name}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Skill probe failed: {result.stderr}")
        return False
        
    print(f"✅ Probes completed for {model_name}")
    return True

def main():
    print("🏁 Starting batch processing of 7 Stockfish-trained models...")
    print("📊 Pipeline: Model Conversion → Piece Probe → Skill Probe")
    
    results = {}
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n{'='*60}")
        print(f"🎯 Processing Model {i}/7: {model}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Step 1: Convert model
        if not convert_model(model):
            results[model] = "FAILED - Conversion"
            continue
            
        # Step 2: Train probes
        if not train_probes(model):
            results[model] = "FAILED - Probes"
            continue
            
        elapsed = time.time() - start_time
        results[model] = f"SUCCESS - {elapsed:.1f}s"
        print(f"⏱️  Model {i} completed in {elapsed:.1f} seconds")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📋 FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for model, status in results.items():
        status_emoji = "✅" if "SUCCESS" in status else "❌"
        print(f"{status_emoji} {model}: {status}")
    
    successful = sum(1 for s in results.values() if "SUCCESS" in s)
    print(f"\n🎉 {successful}/7 models processed successfully!")

if __name__ == "__main__":
    main() §