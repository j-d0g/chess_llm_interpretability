### Chess LLM Interpretability: Linear Probes on Stockfish-Trained Models

This project evaluates transformer LLMs trained on PGN chess games using linear probes to decode internal board state (piece-on-square classification; 13 classes per square). We extend prior work on Lichess-trained models by constructing a compatible Stockfish evaluation dataset and systematically validating results with randomized-weight baselines.

Key outcomes:
- Trained probes across `small-16`, `small-24`, `small-36`, `medium-16`, `large-16` models (TransformerLens format)
- 10k-game test accuracies per layer for trained models; comprehensive `.pkl` traces saved for reproducibility
- Randomized baselines for each architecture/layer with robust re-initialization and verification
- Detailed write-ups in `docs/thesis/blog.md` and `docs/thesis/probes_progress.md`

### Setup

Create a Python environment with Python 3.10 or 3.11 (I'm using 3.11).
```
pip install -r requirements.txt
python model_setup.py
```

The `train_test_chess.py` script trains/tests linear probes on board-state (and optionally skill). We run on SLURM (CSF3 `gpuA`/`gpuV`) with helper scripts in the repo.

Command line arguments:

--mode: Specifies `train`  or `test`. Optional, defaults to `train`.

--probe: Determines the type of probe to be used. `piece` probes for the piece type on each square, `skill` probes the skill level of the White player. Optional, defaults to `piece`.


Examples (Stockfish dataset, board-state):

- Train small-36 (all layers):
```
python train_test_chess.py --mode train --probe piece \
  --dataset_prefix stockfish_ --model_name tf_lens_small-36-600k_iters \
  --n_layers 36 --first_layer 0 --last_layer 35
```

- Test large-16 (all layers):
```
python train_test_chess.py --mode test --probe piece \
  --dataset_prefix stockfish_ --model_name tf_lens_large-16-600K_iters \
  --n_layers 16 --first_layer 0 --last_layer 15
```

See all options: `python train_test_chess.py -h`

To add new functions, refer to `utils/custom_functions_guide.md`.

All experiments in this repo can be done with less than 1 GB of VRAM. Training probes on the 8 layer model takes about 10 minutes on my RTX 3050.

### Dataset construction and filtering

- Adam’s repo ships Lichess evaluation data only. We constructed a Stockfish-based evaluation set compatible with our Stockfish-trained models by filtering self-play logs (`Stockfish 9 vs Stockfish 0–9`) and model-vs-Stockfish games. Final CSVs:
  - `data/stockfish_train.csv`, `data/stockfish_test.csv` (10,000 games eval)
  - Uniform transcript length per row; PGN normalized to `1.e4` format (no spaces after move numbers)
- Filtering goals:
  - Ensure constant-length transcripts for batching
  - Remove malformed/short games; retain standard token ranges used during training
  - Keep data compatible with the meta tokenizer (`models/meta.pkl`)
- Token positions: probes use `find_dots_indices` to select indices at `.` in PGN (`1.e4`, `2.Nf3`, ...). After fixing PGN formatting, these indices correspond to valid semantic positions across the move sequence.

### Probe training methodology

- Architecture autodetection: `train_test_chess.py` loads model state dict and derives `d_model`, `n_heads`, `d_head`, `d_mlp`, and `d_vocab` dynamically before constructing `HookedTransformerConfig`.
- Data pipeline:
  - Load CSV → assert fixed-length `transcript`
  - Encode with `meta.pkl` into integer sequences
  - Compute custom indices via `find_dots_indices` over strings
  - Build one-hot board state stacks per index
  - Cache and index `resid_post` activations per layer at those positions
- Training: single linear tensor per layer with cross-entropy over 13-way per square; average over positions and squares. Light-weight B=2 for high-throughput testing.

### Validation and randomized baselines

- We randomized model weights to establish baselines and detect artifacts:
  - Initialization: for each weight tensor, sampled `torch.randn_like` scaled by Xavier/Glorot std; biases set to zeros
  - Produced `_RANDOM` checkpoints per model; verified difference vs originals
  - Ran full-layers tests with randomized probes; typical accuracies ~0.64–0.71
- Result integrity:
  - Earlier anomaly (near-100% across all layers) traced to data/tokenization mismatch and fixed by PGN normalization and correct indexing strategy
  - Trained models outperform random baselines by ~30–35% at peak layers, confirming genuine learned board-state representations

### Results overview

- Full per-layer test accuracies and coverage are in `docs/thesis/probes_progress.md` (tables per model).
- Random baselines summarized there as well (~0.64–0.71 across models).
- `.pkl` traces of test runs saved under `linear_probes/test_data/` (ignored in git; retained locally for reproducibility).

### SLURM usage (CSF3)

- Partitions: `gpuA` (A100 80G), `gpuV` (V100)
- Scripts: `train_*`, `test_*`, `submit_*` provide parallelized runs and afterany dependencies
- Environment on nodes: `module load Python/3.10.8-GCCcore-12.2.0 && source venv/bin/activate`

### Uploads

- Probes will be uploaded to HuggingFace under `jd0g/chessgpt-board-probes` using `upload_probes_to_hf.py` and `HF_README.md`.
- Model checkpoints remain outside the repo (`models/**` ignored) and will be uploaded separately.

This repo can also be used for training linear probes on OthelloGPT. Refer to `utils/othello_data_filtering.ipynb`.

### Interventions and Othello

To perform board state interventions on one layer, run `python board_state_interventions.py`. It will record JSON results in `intervention_logs/`. To get better results, train a set of 8 (one per layer) board state probes using `train_test_chess.py` and rerun.

To perform skill interventions, you can train a set of 8 skill probes using `train_test_chess.py` or generate a set of 8 contrastive activations using `caa.py`. Note that contrastive activations tend to work a little better. If you want to use probe derived interventions, use this script to create activation files from the probes: `utils/create_skill_intervention_from_skill_probe.ipynb`.

Then, follow these directions to use them to perform skill interventions: https://github.com/adamkarvonen/chess_gpt_eval/tree/master/nanogpt

### Shape annotations

I've been using this tip from Noam Shazeer:

Dimension key (from https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):

M = modes

l  = seq length before indexing

L  = seq length after indexing

B = batch_size

R = rows (or cols)

C = classes for one hot encoding

D = d_model of the GPT (512)

For example

```
probe_out_MBLRRC = einsum(
    "batch pos d_model, modes d_model rows cols options -> modes batch pos rows cols options",
    resid_post_BLD,
    linear_probe_MDRRC,
)
```

### Useful links

All code, models, and datasets are open source.

To play the nanoGPT model against Stockfish, please visit: https://github.com/adamkarvonen/chess_gpt_eval/tree/master/nanogpt

To train a Chess-GPT from scratch, please visit: https://github.com/adamkarvonen/nanoGPT

All pretrained models are available here: https://huggingface.co/adamkarvonen/chess_llms

All datasets are available here: https://huggingface.co/datasets/adamkarvonen/chess_games

Wandb training loss curves and model configs can be viewed here: https://api.wandb.ai/links/adam-karvonen/u783xspb

### Testing

To run the end to end test suite, run `pytest -s` from the root directory. This will first train and test probes end to end on the 8 layer model, including comparing expected accuracy to actual accuracy within some tolerance. Then it will test out board state interventions and caa creation. It takes around 14 minutes. The `-s` flag is so you can see the training updates and gauge progress.

# References

Much of my linear probing was developed using Neel Nanda's linear probing code as a reference. Here are the main references I used:

https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Othello_GPT.ipynb
https://colab.research.google.com/github/likenneth/othello_world/blob/master/Othello_GPT_Circuits.ipynb
https://www.neelnanda.io/mechanistic-interpretability/othello
https://github.com/likenneth/othello_world/tree/master/mechanistic_interpretability