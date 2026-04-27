# Assignment: fourier-sigma-sweep
**Student**: charliepai2c3-nezuko
**Branch**: charliepai2c3-nezuko/fourier-sigma-sweep
**Advisor branch**: icml-appendix-charlie-pai2c-r3

## Hypothesis

Fourier positional encoding (PR #261) gave a 12.7% improvement in val_avg/mae_surf_p (131.71 → 115.02). The key finding was that even with default sigma=1.0, the spatial frequency encoding dramatically helped.

According to Tancik et al. 2020, **sigma (the standard deviation of the random Gaussian frequency matrix B)** is the most impactful hyperparameter of random Fourier features. Sigma controls the frequency scale of the features:
- Low sigma (0.1-0.5): encodes mostly low-frequency spatial structure (smooth global patterns)
- Medium sigma (1.0): default, captures mid-range spatial frequencies
- High sigma (3.0-10.0): captures fine-grained, high-frequency patterns (boundary layers, sharp pressure gradients near foil surface)

For CFD pressure fields, surface boundary layers and wake regions exhibit **high-frequency spatial structure** — pressure drops sharply at the foil surface and recovers. The current sigma=1.0 may be undershooting the optimal frequency range for these physically important patterns.

**Hypothesis**: A higher sigma (3.0 or 10.0) will improve surface pressure prediction by giving the model richer high-frequency spatial vocabulary for boundary layer and wake representation. We sweep sigma ∈ {0.5, 1.0, 3.0, 10.0} to find the optimal value on top of the winning Fourier PE implementation.

## Instructions

You are building on the **winning Fourier PE implementation** from PR #261 — the `train.py` on the advisor branch already has `FourierEmbedding` and the Transolver changes. Your only job is to sweep `fourier_sigma`.

In `target/train.py`:

**Step 1: Find the `fourier_sigma` parameter in the model instantiation.** It should look something like:
```python
fourier_sigma=1.0,  # or fourier_sigma: float = 1.0 in some form
```

**Step 2: Run 4 experiments sequentially, changing only `fourier_sigma`** for each run:

```bash
# sigma=0.5
cd target/ && python train.py \
  --agent charliepai2c3-nezuko \
  --experiment_name fourier-sigma-0.5

# sigma=1.0 (current best — reference point)
cd target/ && python train.py \
  --agent charliepai2c3-nezuko \
  --experiment_name fourier-sigma-1.0

# sigma=3.0 (main hypothesis — richer high-freq)
cd target/ && python train.py \
  --agent charliepai2c3-nezuko \
  --experiment_name fourier-sigma-3.0

# sigma=10.0 (maximum high-freq)
cd target/ && python train.py \
  --agent charliepai2c3-nezuko \
  --experiment_name fourier-sigma-10.0
```

**How to change sigma**: In `train.py`, find where `Transolver` is instantiated (in the `build_model` function or equivalent). It will have a parameter like `fourier_sigma=1.0`. Change it to the target value before each run. All other hyperparameters remain at their defaults:
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`, `epochs=50`
- `num_fourier_freqs=16` (keep unchanged)

**Note:** Each run is independent — you can hard-code the sigma value in the model instantiation, run the experiment, then change it for the next run.

If the 30-minute timeout limits you to fewer runs, prioritize: **sigma=3.0 first**, then sigma=10.0, then sigma=0.5. sigma=1.0 is already known from PR #261.

## What to Report

For each sigma value that completed:
- `val_avg/mae_surf_p` at best checkpoint (primary metric)
- `test_avg/mae_surf_p`  
- Per-split val breakdown table
- Best epoch number
- Training time, peak VRAM

Summary table format:
| sigma | val_avg/mae_surf_p | test_avg/mae_surf_p | best_epoch |
|---|---:|---:|---:|
| 0.5 | ... | ... | ... |
| 1.0 | ... (PR #261 reference) | ... | 14 |
| 3.0 | ... | ... | ... |
| 10.0 | ... | ... | ... |

