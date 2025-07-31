# CSF3 Setup Guide - University of Manchester

**CSF3 cluster setup guide** for running chess model evaluation. For project overview and current workflows, see the [main README](../README.md).

This guide covers how to get started with the CSF3 (Computational Shared Facility) system at the University of Manchester, specifically for GPU-accelerated machine learning projects.

## System Overview

CSF3 is a high-performance computing cluster with the following GPU resources:
- **gpuV partition**: 68 NVIDIA V100 GPUs
- **gpuA partition**: 76 NVIDIA A100 80GB GPUs
- Maximum job time: 4 days
- Access via SSH: `ssh username@csf3.itservices.manchester.ac.uk`

## Getting GPU Access

### Interactive Sessions

For development and testing:

```bash
# A100 GPU sessions (85GB memory, fastest)
srun --partition=gpuA --gres=gpu:a100_80g:1 --time=1:00:00 --pty bash
srun --partition=gpuA --gres=gpu:a100_80g:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash
srun --partition=gpuA --gres=gpu:a100_80g:1 --cpus-per-task=16 --mem=64G --time=4:00:00 --pty bash

# V100 GPU sessions (16GB memory, readily available)  
srun --partition=gpuV --gres=gpu:v100:1 --time=1:00:00 --pty bash
srun --partition=gpuV --gres=gpu:v100:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash
```

**Resource Guidelines:**
- **A100**: 85GB GPU memory, 48 CPUs, ~515GB system memory per node
- **V100**: 16GB GPU memory, 32 CPUs, ~191GB system memory per node
- Start with conservative resources (8 CPUs, 32GB) and increase as needed

### Batch Jobs

Create a job script for unattended runs:

```bash
#!/bin/bash
#SBATCH --job-name=my_experiment
#SBATCH --partition=gpuA                # or gpuV for V100
#SBATCH --gres=gpu:a100_80g:1          # or gpu:v100:1 for V100
#SBATCH --cpus-per-task=8              # CPU cores
#SBATCH --mem=32G                      # System memory
#SBATCH --time=4:00:00                 # Max 4 days
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Create logs directory
mkdir -p logs

# Load required modules
module load cuda/12.6.2

# Your commands here
cd /path/to/your/project
python your_script.py
```

**Batch Job Examples:**
```bash
# Submit job
sbatch your_job_script.sh

# Monitor jobs
squeue --me
squeue -p gpuA    # Check A100 queue
squeue -p gpuV    # Check V100 queue

# Cancel job
scancel JOBID
```

## Environment Setup

### 1. Load Essential Modules

```bash
# CUDA for GPU support
module load cuda/12.6.2

# Python environment management
module load miniforge/24.11.2
```

### 2. Check GPU Availability

```bash
# Verify GPU access
nvidia-smi

# Check CUDA version
nvcc --version
```

### 3. Python Dependencies

```bash
# Upgrade pip first (important for some packages)
pip install --upgrade pip

# Install your requirements
pip install -r requirements.txt

# Or install individual packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Common Workflows

### Starting a New Session

```bash
# 1. SSH into CSF3
ssh your_username@csf3.itservices.manchester.ac.uk

# 2. Navigate to your scratch space (recommended for large files)
cd /mnt/iusers01/fse-ugpgt01/compsci01/your_username/scratch/

# 3. Request GPU (choose A100 or V100)
srun --partition=gpuA --gres=gpu:a100_80g:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash
# OR for V100:
# srun --partition=gpuV --gres=gpu:v100:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash

# 4. Load modules
module load cuda/12.6.2

# 5. Test GPU access
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 6. Run your code
python your_script.py
```

### Testing GPU Setup

Before running heavy workloads, validate your environment:

```bash
# Quick GPU test script
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ CUDA not available - check you are on a GPU node')
"
```

### Monitoring Jobs

```bash
# Check your running jobs
squeue -u your_username

# Check detailed job info
scontrol show job JOBID

# Cancel a job
scancel JOBID
```

## File Management

### Storage Locations

- **Home directory**: `/home/your_username` (limited space, backed up)
- **Scratch space**: `/mnt/iusers01/fse-ugpgt01/compsci01/your_username/scratch/` (large space, not backed up)

### Best Practices

- Use scratch space for large datasets and model files
- Keep code in home directory (backed up)
- Clean up scratch space regularly
- Transfer results to local machine when done

## Troubleshooting

### Common Issues

1. **Module not found**: Ensure you've loaded the correct modules in the right order
2. **CUDA errors**: Check that you're on a GPU node with `nvidia-smi`
3. **Out of memory**: Monitor GPU memory usage, consider smaller batch sizes
4. **Import errors**: Verify all dependencies are installed in the current environment
5. **File not found**: Check file paths - absolute paths often work better than relative
6. **`srun: fatal: No command given to execute`**: This error occurs when using `srun` without specifying what to run. See below for the difference between `srun` and `sbatch`.

### srun vs sbatch Usage

**Use `sbatch` for batch scripts** (recommended for most jobs):
```bash
# Submit a job script with #SBATCH directives
sbatch your_script.sh
```

**Use `srun` for interactive sessions or direct commands**:
```bash
# Interactive shell (note the --pty bash at the end)
srun --partition=gpuA --gres=gpu:a100_80g:1 --time=2:00:00 --pty bash

# Run a specific command directly
srun --partition=gpuA --gres=gpu:a100_80g:1 --time=2:00:00 python your_script.py

# Run a batch script interactively
srun --partition=gpuA --gres=gpu:a100_80g:1 --time=2:00:00 bash your_script.sh
```

**Key differences**:
- **`sbatch`**: Submits jobs to queue, runs in background, output goes to log files
- **`srun`**: Runs interactively, you see output immediately, requires a command to execute

### Performance Tips

- **GPU Utilization**: Monitor with `nvidia-smi` to ensure GPU is being used
- **File I/O**: Use scratch space for better I/O performance
- **Dependencies**: Some packages (like tiktoken) may need Rust compiler - upgrade pip first
- **Batch Size**: Start small and increase gradually to find optimal GPU memory usage

### Debugging Commands

```bash
# Check current directory and files
pwd && ls -la

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Check loaded modules
module list

# Check Python path and packages
which python
pip list | grep torch
```

## Chess LLM Interpretability Workflow

**Tested setup for chess linear probe training on transformers.**

### 1. Environment Setup

```bash
# SSH to CSF3
ssh your_username@csf3.itservices.manchester.ac.uk
cd /mnt/iusers01/fse-ugpgt01/compsci01/your_username/scratch/chess_llm_interpretability

# Install dependencies
pip install -r requirements.txt

# Test environment
python -c "import transformer_lens, torch, chess_utils; print('✅ All imports successful')"
```

### 2. Model Conversion (nanoGPT → TransformerLens)

```bash
# Convert your models to TransformerLens format
python model_setup.py --convert_all

# Verify conversion
ls -la models/tf_lens_*.pth
```

### 3. Data Processing

```bash
# Download games data from HuggingFace
python download_games_data.py --model small-36  # or your model name

# Process into train/test format
python process_small36_data.py  # adapt for your model
```

### 4. GPU Testing

```bash
# Test A100 access
sbatch test_a100.sh
squeue --me

# Test V100 access  
sbatch test_v100.sh
squeue --me

# Check results
cat logs/small36_*_*.out
```

### 5. Probe Training

```bash
# Interactive session for testing
srun --partition=gpuA --gres=gpu:a100_80g:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash
module load cuda/12.6.2

# Small test (few layers)
python train_small36_probes.py  # or your training script

# Full batch job for all layers
sbatch train_probes_full.sh
```

**Verified working on:**
- ✅ **A100**: NVIDIA A100-SXM4-80GB (85GB memory) - node851
- ✅ **V100**: Tesla V100-SXM2-16GB (17GB memory) - node806  
- ✅ **PyTorch**: 2.2.2+cu121
- ✅ **TransformerLens**: 1.10.0

## Getting Help

- CSF3 Documentation: [IT Services website]
- Check job limits: `sinfo`
- System status: Check IT Services status page
- For technical issues: Contact IT Services helpdesk

## Notes

- Jobs are killed if they exceed time limits
- GPU nodes may have different architectures (check with `nvidia-smi`)
- Some software may need specific CUDA versions
- Always test with short interactive sessions before submitting long batch jobs 