# ChessGPT Board Probes

This repository contains linear probes trained to predict chess piece positions from the internal representations of various chess language models.

## Overview

These probes were trained as part of interpretability research on chess LLMs, investigating how board-state representations develop across different model architectures and layers.

## Models Analyzed

- **Small-16** (512 dim, 16 layers): All layers 0-15
- **Small-24** (512 dim, 24 layers): All layers 0-23  
- **Small-36** (512 dim, 36 layers): Layers 0-23 (layers 24-35 pending)
- **Medium-16** (768 dim, 16 layers): All layers 0-15
- **Large-16** (1024 dim, 16 layers): All layers 0-15

## Probe Types

### Trained Model Probes
Linear classifiers trained on activations from models trained on chess games.
- Format: `tf_lens_{model_name}_chess_piece_probe_layer_{N}.pth`
- Example: `tf_lens_large-16-600K_iters_chess_piece_probe_layer_8.pth`

### Random Baseline Probes  
Linear classifiers trained on activations from models with randomized weights, used as experimental controls.
- Format: `tf_lens_{model_name}_RANDOM_chess_piece_probe_layer_{N}.pth`
- Example: `tf_lens_large-16_RANDOM_chess_piece_probe_layer_8.pth`

## Probe Details

- **Task**: Predict the piece type on each of the 64 chess board squares
- **Input**: Model activations at specific sequence positions (after move notation dots)
- **Output**: 13-class classification per square (empty, 6 white pieces, 6 black pieces)
- **Architecture**: Single linear layer (no hidden layers)
- **Training**: Cross-entropy loss, trained on Stockfish games

## Key Findings

- **Trained models**: Show clear learning progression, with later layers achieving 75-99% accuracy
- **Random baselines**: Consistently lower performance (65-71%), validating experimental design
- **Layer progression**: Earlier layers show lower accuracy, later layers show higher accuracy
- **Model scaling**: Larger models tend to develop better board representations

## File Naming Convention

```
tf_lens_{model_size}-{layers}[-{training_iters}][_RANDOM]_chess_piece_probe_layer_{layer_num}.pth
```

Where:
- `model_size`: small, medium, large
- `layers`: 16, 24, 36
- `training_iters`: 600K_iters, 600k_iters
- `RANDOM`: Present for randomized baseline models
- `layer_num`: 0 to (layers-1)

## Usage

Load probes using PyTorch:

```python
import torch

# Load a trained probe
probe = torch.load('tf_lens_large-16-600K_iters_chess_piece_probe_layer_8.pth')

# The probe is a linear layer: torch.nn.Linear(d_model, 64*13)
# where d_model depends on the model (512/768/1024)
# and 64*13 represents 64 squares × 13 piece classes
```

## Research Context

This work is part of mechanistic interpretability research on chess language models, investigating:
- How board-state representations emerge during training
- Scaling laws for internal representations
- Layer-wise development of chess understanding
- Comparison between trained and random baselines

## Citation

If you use these probes in your research, please cite the original work:

```
@misc{chessgpt-board-probes-2024,
  title={ChessGPT Board State Probes},
  author={[Author Name]},
  year={2024},
  url={https://huggingface.co/jd0g/chessgpt-board-probes}
}
```

## License

MIT License - See LICENSE file for details.