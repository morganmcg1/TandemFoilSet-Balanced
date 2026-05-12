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

## Current frontier (round 1 partial)

No clean settled baseline yet — every completed run so far has hit the 30-min
cap at 10–11 epochs (out of `--epochs 50`) and produced non-finite pressure
on `test_geom_camber_cruise` so `test_avg/mae_surf_p` cannot be computed.

Best validation reading observed so far (informational only — not a merged
baseline because test_avg is NaN):

| Source | val_avg/mae_surf_p | partial test_avg (3 of 4 splits) | Notes |
|---|---:|---:|---|
| #1372 (n_head=8, closed) | 153.84 | 141.53 | 11/50 epochs, cruise-test pressure inf |
| #1378 (n_hidden=192, closed) | 155.16 | 159.62 | 10/50 epochs, cruise-test pressure inf |

Round 2 priorities — fix the blockers before retrying capacity changes:
- gradient clipping (frieren #1383) to address the cruise-test pressure inf
- bf16 autocast (tanjiro #1384) to attack throughput so more epochs fit

## Reproduce command

```
cd target/ && python train.py --epochs 50
```
