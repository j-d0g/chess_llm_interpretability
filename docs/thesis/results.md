### Linear Probe Experiments: Coverage and Results (Stockfish dataset)

This file summarizes what we trained, what we tested, key findings, and what (if anything) is still missing before upload.

- Dataset: `data/stockfish_train.csv`, `data/stockfish_test.csv` (10,000 games eval)
- Task: Board-state (13 classes) from intermediate activations
- Probing positions: `find_dots_indices` on corrected PGN format (e.g., `1.e4`)
- Models (TransformerLens format): `small-16`, `small-24`, `small-36`, `medium-16`, `large-16`

---

### Coverage summary

| Model | Layers | Trained probes saved | Randomized probes saved | Tested (trained) | Tested (random) |
|---|---:|---|---|---|---|
| small-16 | 16 | Yes (L0–L15) | Yes (L0–L15) | Yes (L0–L15) | Yes (L0–L15) |
| small-24 | 24 | Yes (L0–L23) | Yes (L0–L23) | Yes (L0–L23) | Yes (L0–L23) |
| small-36 | 36 | Yes (L0–L35) | Yes (L0–L35) | Yes (L0–L35) | Yes (L0–L35) |
| medium-16 | 16 | Yes (L0–L15) | Yes (L0–L15) | Yes (L0–L15) | Yes (L0–L15) |
| large-16 | 16 | Yes (L0–L15) | Yes (L0–L15) | Yes (L0–L15) | Yes (L0–L15) |

- Test artifacts: `.pkl` per layer in `linear_probes/test_data/` exist for all above (trained & random).
- Logs: `logs/test_*` contain the per-layer printed test accuracies.

---

### Randomized baselines (summary)

Observed test accuracy ranges across layers (randomized-weight models):

- small-16: ~0.664–0.710
- small-24: ~0.656–0.696
- small-36: ~0.634–0.710
- medium-16: ~0.639 (L15 example); full range is ~0.64–0.66
- large-16: ~0.655 (L15 example); full range is ~0.65–0.67

Source logs: `logs/test_random_*.err`. Files saved under `linear_probes/test_data/tf_lens_*_RANDOM_chess_piece_probe_layer_*.pkl`.

---

### Per-layer test accuracies (trained models)

All values are test accuracies on the 10,000-game evaluation set.

#### small-16 (L0–L15)

| Layer | Test Acc |
|---:|---:|
| 0 | 0.7643 |
| 1 | 0.7690 |
| 2 | 0.7697 |
| 3 | 0.7715 |
| 4 | 0.7877 |
| 5 | 0.8064 |
| 6 | 0.8325 |
| 7 | 0.8652 |
| 8 | 0.9160 |
| 9 | 0.9415 |
| 10 | 0.9642 |
| 11 | 0.9782 |
| 12 | 0.9878 |
| 13 | 0.9832 |
| 14 | 0.9667 |
| 15 | 0.9529 |

Source: `logs/test_trained_models_4732587.err` (small-16 section).

#### small-24 (L0–L23)

| Layer | Test Acc |
|---:|---:|
| 0 | 0.7572 |
| 1 | 0.7634 |
| 2 | 0.7661 |
| 3 | 0.7647 |
| 4 | 0.7660 |
| 5 | 0.7803 |
| 6 | 0.8136 |
| 7 | 0.8376 |
| 8 | 0.8400 |
| 9 | 0.8650 |
| 10 | 0.8934 |
| 11 | 0.9147 |
| 12 | 0.9387 |
| 13 | 0.9554 |
| 14 | 0.9652 |
| 15 | 0.9797 |
| 16 | 0.9789 |
| 17 | 0.9825 |
| 18 | 0.9903 |
| 19 | 0.9867 |
| 20 | 0.9805 |
| 21 | 0.9691 |
| 22 | 0.9555 |
| 23 | 0.9442 |

Source: `test_results.csv` (`small24_v100`).

#### small-36 (L0–L35)

| Layer | Test Acc |
|---:|---:|
| 0 | 0.7478 |
| 1 | 0.7576 |
| 2 | 0.7646 |
| 3 | 0.7630 |
| 4 | 0.7625 |
| 5 | 0.7644 |
| 6 | 0.7661 |
| 7 | 0.7840 |
| 8 | 0.8111 |
| 9 | 0.8212 |
| 10 | 0.8302 |
| 11 | 0.8387 |
| 12 | 0.8476 |
| 13 | 0.8614 |
| 14 | 0.8668 |
| 15 | 0.8716 |
| 16 | 0.8771 |
| 17 | 0.8985 |
| 18 | 0.9106 |
| 19 | 0.9396 |
| 20 | 0.9589 |
| 21 | 0.9743 |
| 22 | 0.9777 |
| 23 | 0.9767 |
| 24 | 0.9868 |
| 25 | 0.9947 |
| 26 | 0.9943 |
| 27 | 0.9939 |
| 28 | 0.9940 |
| 29 | 0.9901 |
| 30 | 0.9836 |
| 31 | 0.9758 |
| 32 | 0.9642 |
| 33 | 0.9538 |
| 34 | 0.9474 |
| 35 | 0.9383 |

Sources: `test_results.csv` (`small36_v100` L0–L22) and `logs/test_small36_missing_4732581.err` (L23–L35).

#### medium-16 (L0–L15)

| Layer | Test Acc |
|---:|---:|
| 0 | 0.7633 |
| 1 | 0.7716 |
| 2 | 0.7693 |
| 3 | 0.7841 |
| 4 | 0.8181 |
| 5 | 0.8555 |
| 6 | 0.9010 |
| 7 | 0.9342 |
| 8 | 0.9617 |
| 9 | 0.9771 |
| 10 | 0.9892 |
| 11 | 0.9972 |
| 12 | 0.9950 |
| 13 | 0.9878 |
| 14 | 0.9777 |
| 15 | 0.9667 |

Sources: `test_results.csv` (`medium16_3967312` L0–L14) and `logs/test_medium16_missing_4732580.err` (L15).

#### large-16 (L0–L15)

| Layer | Test Acc |
|---:|---:|
| 0 | 0.7622 |
| 1 | 0.7739 |
| 2 | 0.7726 |
| 3 | 0.7963 |
| 4 | 0.8513 |
| 5 | 0.9021 |
| 6 | 0.9208 |
| 7 | 0.9487 |
| 8 | 0.9737 |
| 9 | 0.9967 |
| 10 | 0.9972 |
| 11 | 0.9959 |
| 12 | 0.9947 |
| 13 | 0.9885 |
| 14 | 0.9804 |
| 15 | 0.9713 |

Sources: `test_results.csv` (`large16_3967311` L0–L12) and `logs/test_large16_missing_4732579.err` (L13–L15).

---

### What might be missing or worth adding

- Training-set accuracies per layer were not consistently logged to a CSV for all models. Test accuracies are complete; training accuracies can be reconstructed from training logs if needed.
- Before upload, exclude/move unrelated Lichess probe files in `linear_probes/saved_probes/`.
- Optional aggregation to add here:
  - Per-model summary stats (mean/median/stdev across layers)
  - Plots of accuracy vs. layer for trained vs. random
  - One consolidated CSV for test results across all models/layers

---

### Notes on fixes and methodology

- Resolved data artifact: ensured PGN format uses `1.e4` (no space after move numbers), so `find_dots_indices` probes valid, non-trivial positions.
- Randomized baselines: robust re-initialization with Xavier-style scaling; verified difference from originals; baselines ~0.64–0.71.
- SLURM scripts fixed (partitions, GRES, env activation) and jobs re-run to completion on `gpuA`/`gpuV`.
