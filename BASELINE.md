# TandemFoilSet Baseline — icml-appendix-charlie-pai2i-48h-r3

## Current Best

**PR #3160 — H4: Huber loss δ=0.5 (fern)**  
Merged 2026-05-15. 14 epochs completed (30-min timeout cap; not converged).

Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits). Lower is better.

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | **112.8406** | PR #3160 / Huber δ=0.5 |
| val_single_in_dist/mae_surf_p | 144.92 | PR #3160 |
| val_geom_camber_rc/mae_surf_p | 125.53 | PR #3160 |
| val_geom_camber_cruise/mae_surf_p | 81.82 | PR #3160 |
| val_re_rand/mae_surf_p | 99.10 | PR #3160 |
| test_avg/mae_surf_p | NaN (⚠ scoring bug) | PR #3160 |
| test_avg/mae_surf_p (3-split, excl. cruise) | 113.44 | PR #3160 |

**Important:** fern's run used the pre-FiLM-merge train.py — i.e. **Huber δ=0.5 ALONE, no FiLM**. The current merged train.py contains both FiLM and Huber together, but the FiLM+Huber compound has not been tested yet. The 112.84 number is **Huber δ=0.5 only**.

**⚠ data/scoring.py NaN bug:** `test_geom_camber_cruise` sample 20 has non-finite GT; `nan * 0 = nan` propagates through the masked sum. `test_avg/mae_surf_p = NaN` for all PRs. File is read-only.

**Artifacts:** `models/model-h4-huber-delta-0.5-20260515-135951/`

**Reproduce:**
```bash
# Huber δ=0.5 (current best, no FiLM)
cd target/ && python train.py --epochs 50 \
  --experiment_name h4-huber-delta-0.5 --agent <student> --huber_delta 0.5
# Note: merged train.py has cond_dim=11 by default (FiLM on); fern's run was BEFORE FiLM merge
```

## Previous best (overridden)

| PR | Experiment | val_avg/mae_surf_p | Status |
|----|------------|--------------------|--------|
| #3166 | H7: FiLM Re/AoA conditioning (nezuko) | 114.6268 | Merged, outperformed by #3160 |

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
| R1 | #3160 | H4: Huber loss δ=0.5 (fern) | **112.8406** | NaN | **MERGED — NEW BEST** |
| R1 | #3163 | H3: Grad clip + LR warmup (frieren) | — | — | WIP |
| R1 | #3166 | H7: FiLM Re/AoA conditioning (nezuko) | **114.6268** | NaN (scoring bug) | **MERGED** |
| R1 | #3168 | H10: slice_num=128,96 (tanjiro) | 149.27 (no FiLM) | 137.35 (3-split) | Closed |
| R1 | #3170 | H11: n_layers=7,8 (thorfinn) | — | — | WIP |
