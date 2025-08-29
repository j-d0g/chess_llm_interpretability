#!/bin/bash

echo "🎯 Submitting RANDOM BASELINE test jobs with generous time allocations..."
echo "These will test trained probes on randomized models for comparison"
echo ""

echo "1. Random Large-16 baseline - 6 hours allocated"
sbatch test_random_large16_probes.sh

echo "2. Random Medium-16 baseline - 6 hours allocated"  
sbatch test_random_medium16_probes.sh

echo "3. Random Small-24 baseline - 8 hours allocated"
sbatch test_random_small24_probes.sh

echo "4. Random Small-36 baseline - 12 hours allocated"
sbatch test_random_small36_probes.sh

echo "5. Random Small-16 baseline - 4 hours allocated"
sbatch test_random_small16_probes.sh

echo ""
echo "✅ All random baseline test jobs submitted!"
echo "📊 Expected results: ~65-70% accuracy (data pattern baseline)"
echo "🎯 Comparison: Trained models achieve 99%+ (30-35% gap = genuine learning)"