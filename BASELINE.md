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

Lower is better. No measured numbers are recorded for this track yet; the
first completed default-config run on this fleet will populate this baseline.

## Reproduce command

```
cd target/ && python train.py --epochs 50
```
