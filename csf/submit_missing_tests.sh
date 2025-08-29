#!/bin/bash

echo "Submitting missing test layer jobs with generous buffer times..."

echo "1. Large-16 missing layers (13-15) - 2 hours allocated"
sbatch test_missing_large16_layers.sh

echo "2. Medium-16 missing layer (15) - 1 hour allocated"  
sbatch test_missing_medium16_layer.sh

echo "3. Small-36 missing layers (23-35) - 4 hours allocated"
echo "   ⚠️  CRITICAL: Includes peak layers 25-28 with ~99.5% expected accuracy!"
sbatch test_missing_small36_layers.sh

echo ""
echo "✅ All missing test jobs submitted with generous time allocations"
echo "📊 This will complete our comprehensive layer-by-layer analysis"
echo "🎯 Expected results: 99%+ accuracy on all peak layers"