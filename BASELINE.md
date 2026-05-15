# TandemFoilSet Baseline — icml-appendix-charlie-pai2i-48h-r3

## Current Best

**PR #3166 — H7: FiLM Re/AoA conditioning (nezuko)**  
Merged 2026-05-15. 14 epochs completed (30-min timeout cap; model not converged — still improving at cutoff).

Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits). Lower is better.

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | **114.6268** | PR #3166 / metrics.yaml |
| val_single_in_dist/mae_surf_p | 129.7991 | PR #3166 |
| val_geom_camber_rc/mae_surf_p | 129.0683 | PR #3166 |
| val_geom_camber_cruise/mae_surf_p | 94.8233 | PR #3166 |
| val_re_rand/mae_surf_p | 104.8163 | PR #3166 |
| test_avg/mae_surf_p | NaN (⚠ scoring bug) | PR #3166 |
| test_avg/mae_surf_p (3-split, excl. cruise) | 111.155 | PR #3166 |

**⚠ data/scoring.py NaN bug:** `test_geom_camber_cruise` sample index 20 has non-finite GT; `nan * 0 = nan` propagates through the masked sum in `accumulate_batch`. `test_avg/mae_surf_p` is NaN as a result. All val metrics are clean. This affects any PR's test numbers until the bug is resolved.

**Note on FiLM baseline validity:** No unmodified Transolver has been run on this branch yet. The 114.63 value includes FiLM conditioning (cond_dim=11). Future PRs beating 114.63 may be beating FiLM, not the raw baseline. A clean baseline run (cond_dim=0) is tracked as a Round 2 priority.

**Artifacts:** `models/model-charliepai2i48h3-nezuko-h7-film-re-aoa-cond-20260515-131301/`

**Reproduce:**
```bash
# FiLM-conditioned (current best)
cd target/ && python train.py --epochs 50 \
  --experiment_name h7-film-re-aoa-cond --agent <student>
# Note: model_config must include cond_dim=11 (see merged train.py)
```

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
| R1 | #3166 | H7: FiLM Re/AoA conditioning (nezuko) | **114.6268** | NaN (scoring bug) | **MERGED** |
| R1 | #3168 | H10: slice_num=128,96 (tanjiro) | — | — | WIP |
| R1 | #3170 | H11: n_layers=7,8 (thorfinn) | — | — | WIP |
