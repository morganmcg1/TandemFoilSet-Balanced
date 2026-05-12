# Baseline — icml-appendix-charlie-pai2g-24h-r5

Charlie no-W&B logging arm, round 5. Each training execution is capped at
`SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock cap). Local JSONL metrics only,
no W&B.

## Current best

No experiments merged yet on this advisor branch. The baseline number to beat
is the out-of-the-box `train.py` Transolver on this branch, evaluated with
`SENPAI_TIMEOUT_MINUTES=30`. Every student PR must report
`val_avg/mae_surf_p` (primary), `test_avg/mae_surf_p` (paper number), and
per-split MAE for surface pressure on all four val and test tracks.

## Baseline configuration (from `target/train.py`)

| Lever | Value |
|---|---|
| Model | Transolver, n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2 |
| Activation | GELU |
| Loss | MSE in normalized space: `vol_loss + 10 * surf_loss` |
| Surface weight | 10.0 |
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| LR schedule | CosineAnnealingLR, T_max=epochs |
| Batch size | 4 (variable mesh sizes, pad_collate to N_max) |
| Sampler | WeightedRandomSampler (balanced domain mix) |
| Max epochs | 50 (configurable via `--epochs`) |
| Timeout | 30 min wall-clock (hard cap) |
| Precision | FP32 |

## Primary metric

`val_avg/mae_surf_p` — equal-weight mean surface-pressure MAE across the four
validation splits:

- `val_single_in_dist` — single-foil random holdout (sanity check)
- `val_geom_camber_rc` — unseen front-foil camber M=6-8 (raceCar)
- `val_geom_camber_cruise` — unseen front-foil camber M=2-4 (cruise)
- `val_re_rand` — stratified Re holdout across tandem domains

Best checkpoint = lowest `val_avg/mae_surf_p`. Test eval at end of training
uses that checkpoint and reports `test_avg/mae_surf_p` for the paper.

## How to claim a win

A PR is a winner if its terminal `SENPAI-RESULT` marker reports a strictly
lower `val_avg/mae_surf_p` than this file's current best (or, before the
first merge, beats the out-of-the-box baseline of the same epoch/timeout
budget by a clearly statistically meaningful margin and reports the
matching test number).

## Update procedure

When a new winner merges, update this file with:

- The merged PR number
- New `val_avg/mae_surf_p`
- Matching `test_avg/mae_surf_p`
- One-line note on what changed
