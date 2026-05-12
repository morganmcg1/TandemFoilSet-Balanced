# Baseline — icml-appendix-willow-pai2g-24h-r1

Baseline reference for the willow-pai2g-24h-r1 logging-ablation launch. This
track is a controlled Charlie-vs-Willow comparison; each per-run training is
capped at SENPAI_TIMEOUT_MINUTES=30, so hypotheses are atomic single-knob
variants of the default `train.py` configuration.

## Default config (the comparison point)

`python train.py --epochs 50` with:

| Hyperparameter | Default |
|----------------|---------|
| lr             | 5e-4    |
| weight_decay   | 1e-4    |
| batch_size     | 4       |
| surf_weight    | 10.0    |
| epochs         | 50      |
| optimizer      | AdamW   |
| scheduler      | CosineAnnealingLR (T_max=epochs) |
| n_hidden       | 128     |
| n_layers       | 5       |
| n_head         | 4       |
| slice_num      | 64      |
| mlp_ratio      | 2       |
| loss           | MSE on normalized space, surface-weighted (`vol_loss + surf_weight * surf_loss`) |
| sampler        | WeightedRandomSampler (balanced across 3 domains) |

## Primary ranking metric

- val: `val_avg/mae_surf_p` — equal-weight mean of surface pressure MAE across
  the four validation splits (`val_single_in_dist`, `val_geom_camber_rc`,
  `val_geom_camber_cruise`, `val_re_rand`).
- test (paper-facing): `test_avg/mae_surf_p` — same average, computed once at
  end-of-run from the best validation checkpoint.

Lower is better.

## Current frontier (round 2 partial — data-bug diagnosed)

No clean settled baseline yet. The fleet-wide NaN on `test_avg/mae_surf_p` is
now confirmed to be a **data bug**, not a model failure: nezuko (PR #1377)
found that `test_geom_camber_cruise/000020.pt` has `y` containing `+inf` in
the hidden pressure GT, and `data/scoring.py::accumulate_batch` propagates the
inf into `mae_surf` via `inf * 0 = NaN`. This is the only such sample across
all 8 val/test splits, but it makes `test_avg/mae_surf_p = NaN` for **every**
PR on this fleet regardless of model quality. Filed as issue #1567.

`data/` and `data/scoring.py` are read-only per `program.md`, so the fix is a
train.py-side filter that drops samples with non-finite `y` **before**
`accumulate_batch`. Frieren is implementing this in the #1515 rework.

Best validation reading observed so far (informational only — not a merged
baseline because test_avg is NaN until the train.py filter lands):

| Source | val_avg/mae_surf_p | partial test_avg (3 of 4) | Notes |
|---|---:|---:|---|
| **#1515 (grad-clip-1.0, sent back for filter)** | **115.78** | **114.96** | 14/50 epochs, 100% steps clipped, effective LR ≈ 1.1e-5 |
| #1377 (mlp_ratio=4, closed) | 146.34 | 146.32 | 13/50 epochs, found the data bug |
| #1382 (wd=3e-4, closed) | 149.40 | 153.20 | 10/50 epochs |
| #1372 (n_head=8, closed) | 153.84 | 141.53 | 11/50 epochs |
| #1378 (n_hidden=192, closed) | 155.16 | 159.62 | 10/50 epochs |

#1515's `val_avg=115.78` is the strongest reading on the fleet by a wide
margin. Frieren's grad-norm analysis showed median pre-clip grad norm = 44.68
and 100% of steps clipped — so `max_norm=1.0` is acting as a global ~45×
LR-shrink, not "occasional outlier damping". Effective LR ≈ 1.1e-5. Whether
the win is fundamentally LR-scale-driven or adaptive-damping-driven is the
next question to test.

Round 2 priorities:
- frieren #1515 (rework): grad-clip-1.0 + train.py-side bad-sample filter →
  unblock paper-facing `test_avg/mae_surf_p` and merge as the round-2 anchor.
- tanjiro #1516: bf16 autocast for ~1.3–1.8× throughput → more epochs/run.
- thorfinn #1538: Huber loss on volume term for robustness against high-Re
  pressure extremes.
- nezuko (incoming): direct lr=1e-5 test to disentangle LR-shrink vs adaptive
  damping in grad-clip's win.

## Reproduce command

```
cd target/ && python train.py --epochs 50
```
