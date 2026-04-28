# BASELINE — icml-appendix-charlie-pai2d-r3

## Current measured baseline

PR #280 (charliepai2d3-alphonse) — **L1 surface loss** (volume MSE
unchanged), all other knobs at the unmodified Transolver defaults
(`bs=4`, `lr=5e-4`, `n_hidden=128`, `n_layers=5`, `n_head=4`,
`slice_num=64`, `mlp_ratio=2`, `surf_weight=10`, cosine T_max=50).

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14/14) | **102.64** |
| `test_avg/mae_surf_p` (NaN-safe re-eval, best-val checkpoint) | **97.73** |
| Per-epoch wallclock | ~132 s |
| Peak GPU memory (batch=4) | 42.13 GB |
| Epochs completed before 30-min timeout | 14 / 50 |

Per-split val (best epoch 14):

| split | mae_surf_p |
|-------|-----------|
| val_single_in_dist     | 121.18 |
| val_geom_camber_rc     | 125.01 |
| val_geom_camber_cruise |  73.22 |
| val_re_rand            |  91.14 |
| **val_avg**            | **102.64** |

Per-split test (NaN-safe, best-val checkpoint):

| split | mae_surf_p |
|-------|-----------|
| test_single_in_dist     | 109.80 |
| test_geom_camber_rc     | 114.60 |
| test_geom_camber_cruise |  79.92 |
| test_re_rand            |  86.58 |
| **test_avg**            | **97.73** |

Reproduce:

```bash
cd target/
python train.py --experiment_name baseline_ref
```

(L1 surface loss is now baked into `train.py`. The scoring fix is
already on the advisor branch — `test_avg/*_p` should land as a clean
number, not NaN.)

## Round 3 progress

| Round | Best val_avg/mae_surf_p | Best test_avg/mae_surf_p | Lever |
|-------|------------------------:|-------------------------:|-------|
| Pre-r3 | TBD (no measured pre-r3 baseline on this branch) | — | — |
| PR #306 (merged) | 135.20 | 123.15 | bs=8, sqrt LR |
| **PR #280 (merged, current)** | **102.64** | **97.73** | **L1 surface loss** |

Notes:

- PR #280 was run with `bs=4, lr=5e-4` (unmodified defaults except for
  the L1 surface loss). PR #306 was `bs=8, lr=7.07e-4` with MSE. So the
  −24% val win for PR #280 conflates **L1 vs MSE** with **bs=4 vs bs=8**.
  Per the head-to-head: PR #306 with MSE@bs=8 = 135.20; PR #280 with
  L1@bs=4 = 102.64. The L1 effect is much larger than the bs effect —
  L1 is the dominant lever.
- Only ~14 of 50 scheduled epochs ran before the 30-min wallclock cap on
  both runs. Cosine T_max is set to 50, so neither run reached the cosine
  tail. Round 4 should consider `--epochs 14` (or wallclock-aware T_max)
  to test whether full cosine decay adds further gains.

## Reference (unmodified Transolver) configuration

Defaults from `train.py` (now uses L1 surface loss after PR #280 merge,
volume MSE unchanged):

| Knob | Value |
|------|-------|
| Model | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| Optimizer | AdamW(lr=5e-4, weight_decay=1e-4) |
| Schedule | CosineAnnealingLR(T_max=epochs) |
| Loss | `vol_loss + 10.0 * surf_loss`, **MSE volume + L1 surface** (post-#280) |
| Sampler | `WeightedRandomSampler` (balanced over 3 train domains) |
| Batch size | 4 |
| Epochs | 50 |
| Wallclock cap | 30 minutes (`SENPAI_TIMEOUT_MINUTES`) |

## Primary ranking metric

Lower is better, equal-weight mean across the four splits:

- val: `val_avg/mae_surf_p`
- test: `test_avg/mae_surf_p` (paper-facing, computed on the best-val checkpoint)
