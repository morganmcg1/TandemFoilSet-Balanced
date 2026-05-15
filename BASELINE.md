# TandemFoilSet Baseline — icml-appendix-charlie-pai2i-48h-r3

## Current Best

**No experiments completed yet.** First round of experiments is in progress.

Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits). Lower is better.

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | — | Pending first round |
| test_avg/mae_surf_p | — | Pending first round |

## Default Transolver Config (Unmodified)

This is the reference configuration all Round 1 experiments deviate from:

| Parameter | Value |
|-----------|-------|
| n_hidden | 128 |
| n_layers | 5 |
| n_head | 4 |
| slice_num | 64 |
| mlp_ratio | 2 |
| lr | 5e-4 |
| weight_decay | 1e-4 |
| batch_size | 4 |
| surf_weight | 10.0 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR(T_max=epochs) |
| epochs | 50 (capped by SENPAI_TIMEOUT_MINUTES) |

Reproduce command:
```bash
cd target/ && python train.py --epochs 50 --experiment_name baseline --agent <student>
```

## Experiment History

| Round | PR | Experiment | val_avg/mae_surf_p | test_avg/mae_surf_p | Status |
|-------|----|------------|--------------------|---------------------|--------|
| R1 | #3154 | H5: n_hidden=256 (alphonse) | — | — | WIP |
| R1 | #3156 | H1: p-channel surf upweight x3,x5 (askeladd) | — | — | WIP |
| R1 | #3158 | H2: EMA decay=0.999 (edward) | — | — | WIP |
| R1 | #3160 | H4: Huber loss delta=1.0,0.5 (fern) | — | — | WIP |
| R1 | #3163 | H3: Grad clip + LR warmup (frieren) | — | — | WIP |
| R1 | #3166 | H7: FiLM Re/AoA conditioning (nezuko) | — | — | WIP |
| R1 | #3168 | H10: slice_num=128,96 (tanjiro) | — | — | WIP |
| R1 | #3170 | H11: n_layers=7,8 (thorfinn) | — | — | WIP |
