#!/usr/bin/env python3
"""
Script to verify that randomized models are truly random and not copies.
"""
import torch
import numpy as np
import os

def verify_randomization(original_path, random_path, model_name):
    """Verify that a randomized model is truly different from original."""
    print(f"\n🔍 Verifying randomization for {model_name}")
    print("=" * 50)
    
    if not os.path.exists(original_path):
        print(f"❌ Original model not found: {original_path}")
        return False
        
    if not os.path.exists(random_path):
        print(f"❌ Random model not found: {random_path}")
        return False
    
    # Load both models
    original = torch.load(original_path, map_location='cpu')
    randomized = torch.load(random_path, map_location='cpu')
    
    # Check key structure is identical
    if set(original.keys()) != set(randomized.keys()):
        print(f"❌ Key structure mismatch!")
        return False
    
    # Check specific weight tensors
    test_keys = ['embed.W_E', 'blocks.0.attn.W_Q', 'blocks.0.mlp.W_in', 'unembed.W_U']
    
    identical_count = 0
    different_count = 0
    
    for key in test_keys:
        if key in original:
            are_identical = torch.equal(original[key], randomized[key])
            if are_identical:
                print(f"❌ {key}: IDENTICAL (not randomized!)")
                identical_count += 1
            else:
                # Check they're actually different (not just floating point errors)
                diff = torch.abs(original[key] - randomized[key]).mean()
                print(f"✅ {key}: DIFFERENT (mean diff: {diff:.6f})")
                different_count += 1
    
    # Overall assessment
    if identical_count == 0:
        print(f"✅ {model_name}: PROPERLY RANDOMIZED")
        return True
    else:
        print(f"❌ {model_name}: NOT RANDOMIZED ({identical_count}/{len(test_keys)} identical)")
        return False

def main():
    print("🎲 RANDOMIZATION VERIFICATION")
    print("=" * 60)
    
    models_to_check = [
        ("models/tf_lens_small-16-600k_iters.pth", "models/tf_lens_small-16_RANDOM.pth", "small-16"),
        ("models/tf_lens_small-24-600K_iters.pth", "models/tf_lens_small-24_RANDOM.pth", "small-24"),
        ("models/tf_lens_small-36-600k_iters.pth", "models/tf_lens_small-36_RANDOM.pth", "small-36"),
        ("models/tf_lens_medium-16-600K_iters.pth", "models/tf_lens_medium-16_RANDOM.pth", "medium-16"),
        ("models/tf_lens_large-16-600K_iters.pth", "models/tf_lens_large-16_RANDOM.pth", "large-16"),
    ]
    
    results = []
    for original_path, random_path, model_name in models_to_check:
        result = verify_randomization(original_path, random_path, model_name)
        results.append((model_name, result))
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    for model_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{model_name:12}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} models properly randomized")
    
    if passed == len(results):
        print("🎉 All randomized models are verified!")
    else:
        print("⚠️  Some models need proper randomization!")

if __name__ == "__main__":
    main()