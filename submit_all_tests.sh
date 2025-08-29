#!/bin/bash
# Submit all probe test jobs

echo "Submitting all probe test jobs..."

echo "1. Submitting Large-16 test..."
sbatch test_large16_probes.sh

echo "2. Submitting Medium-16 test..."
sbatch test_medium16_probes.sh

echo "3. Submitting Small-24 test..."
sbatch test_small24_probes.sh

echo "4. Submitting Small-36 test..."
sbatch test_small36_probes.sh

echo "All test jobs submitted!"
echo "Check status with: squeue -u \$USER"