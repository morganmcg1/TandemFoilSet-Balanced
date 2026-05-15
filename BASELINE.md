# TandemFoilSet Baseline — icml-appendix-charlie-pai2i-48h-r3

## Current Best

**PR #3408 — H19: FiLM + Huber δ=0.5 + T_max=15 triple compound (nezuko)**
Merged 2026-05-15. 14 epochs completed (30-min timeout cap; LR fully annealed — best epoch = final epoch).

Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits). Lower is better.

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | **83.8136** | PR #3408 |
| val_single_in_dist/mae_surf_p | 96.4406 | PR #3408 |
| val_geom_camber_rc/mae_surf_p | 93.7378 | PR #3408 |
| val_geom_camber_cruise/mae_surf_p | 62.8339 | PR #3408 |
| val_re_rand/mae_surf_p | 82.2422 | PR #3408 |
| test_avg/mae_surf_p | NaN (⚠ scoring bug) | PR #3408 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **80.2415** | PR #3408 |
| test_single_in_dist/mae_surf_p | 83.0363 | PR #3408 |
| test_geom_camber_rc/mae_surf_p | 85.5143 | PR #3408 |
| test_re_rand/mae_surf_p | 72.1738 | PR #3408 |

**Configuration:** FiLM cond_dim=11 (default on) + Huber δ=0.5 + T_max=15 (default). FiLM was ON during this run.

**Why triple stacking works:** FiLM conditions the model on Re/AoA, reducing cross-regime variance. Huber δ=0.5 clips extreme gradient contributions from high-Re samples. T_max=15 ensures LR fully anneals within the ~14-epoch timeout window, enabling stable Huber-regime refinement. All three improvements compound multiplicatively.

**⚠ data/scoring.py NaN bug:** `test_geom_camber_cruise` sample 20 has non-finite GT; `test_avg/mae_surf_p = NaN` for all PRs. File is read-only.

**Artifacts:** `models/model-h19-film-huber-tmax15-triple-20260515-193153/`

**Reproduce:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h19-film-huber-tmax15-triple --agent <student> \
  --huber_delta 0.5
# FiLM cond_dim=11 and CosineAnnealingLR T_max=15 are now the merged defaults
```

## Previous Best (overridden by #3408)

**PR #3335 — H15: Huber δ=0.5 + T_max=15 compound (nezuko)**
Merged 2026-05-15. 14 epochs completed (30-min timeout cap; LR fully annealed — best epoch = final epoch).

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | **94.6764** | PR #3335 |
| val_single_in_dist/mae_surf_p | 112.4778 | PR #3335 |
| val_geom_camber_rc/mae_surf_p | 102.4805 | PR #3335 |
| val_geom_camber_cruise/mae_surf_p | 72.9612 | PR #3335 |
| val_re_rand/mae_surf_p | 90.7862 | PR #3335 |
| test_avg/mae_surf_p | NaN (⚠ scoring bug) | PR #3335 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **92.4234** | PR #3335 |
| test_single_in_dist/mae_surf_p | 100.6682 | PR #3335 |
| test_geom_camber_rc/mae_surf_p | 93.1860 | PR #3335 |
| test_re_rand/mae_surf_p | 83.4160 | PR #3335 |

**Configuration:** Huber δ=0.5 + T_max=15. **FiLM was OFF** (`--cond_dim 0`). Adding FiLM on top of this config was the H19 follow-up.

## Earlier Bests (overridden)

| PR | Experiment | val_avg/mae_surf_p | Status |
|----|------------|--------------------|--------|
| #3335 | H15: Huber δ=0.5 + T_max=15, no FiLM (nezuko) | 94.6764 | Merged 2026-05-15, overridden by #3408 |
| #3160 | H4: Huber loss δ=0.5, no FiLM (fern) | 112.8406 | Merged 2026-05-15, overridden |
| #3166 | H7: FiLM Re/AoA conditioning (nezuko) | 114.6268 | Merged 2026-05-15, overridden |

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
| scheduler | CosineAnnealingLR(T_max=15) ← updated from T_max=epochs |
| epochs | 50 (capped by SENPAI_TIMEOUT_MINUTES) |

Reproduce command:
```bash
cd target/ && python train.py --epochs 50 --experiment_name baseline --agent <student>
```

## Experiment History

| Round | PR | Experiment | val_avg/mae_surf_p | test_avg/mae_surf_p | Status |
|-------|----|------------|--------------------|---------------------|--------|
| R1 | #3154 | H5: n_hidden=256 (alphonse) | — | — | Closed |
| R1 | #3156 | H1: p-channel surf upweight x3,x5 (askeladd) | — | — | WIP |
| R1 | #3158 | H2: EMA decay=0.999 (edward) | — | — | WIP |
| R1 | #3160 | H4: Huber loss δ=0.5 (fern) | **112.8406** | NaN | **MERGED — prev best** |
| R1 | #3163 | H3: Grad clip + LR warmup (frieren) | 120.09 (clip=1.0) | — | Closed (dead end) |
| R1 | #3166 | H7: FiLM Re/AoA conditioning (nezuko) | **114.6268** | NaN | **MERGED — prev best** |
| R1 | #3168 | H10: slice_num=128,96 (tanjiro) | 149.27 (no FiLM) | 137.35 (3-split) | Closed |
| R1 | #3170 | H11: n_layers=7,8 (thorfinn) | — | — | Closed (budget-limited) |
| R2 | #3284 | H12: Clean baseline + T_max=15 ablation (nezuko) | 114.19 (T_max=15 arm) | — | Closed |
| R2 | #3297 | H13: Surface dual-head (tanjiro) | 130.54 | — | Closed (dead end, no FiLM) |
| R2 | #3311 | H14: FiLM + Huber compound (fern) | — | — | Closed |
| R2 | #3335 | H15: Huber δ=0.5 + T_max=15 compound (nezuko) | **94.6764** | **92.4234** (3-split) | **MERGED — NEW BEST** |
| R2 | #3338 | H16: FiLM + Surface Head (askeladd) | — | — | WIP |
| R2 | #3339 | H8: Per-sample norm (tanjiro) | — | — | WIP |
| R2 | #3340 | H9: WSD schedule + beta2=0.98 (thorfinn) | — | — | WIP |
| R2 | #3341 | H5b: Wider model matched-budget (alphonse) | — | — | WIP |
| R2 | #3342 | H2b: EMA decay=0.999 (edward) | — | — | WIP |
| R2 | #3343 | H17: Per-channel Huber (fern) | — | — | WIP |
| R2 | #3349 | H18: Grad clip=1.0, no warmup (frieren) | — | — | WIP |
| R3 | #3408 | H19: FiLM+Huber+T_max=15 triple compound (nezuko) | **83.8136** | **80.2415** (3-split) | **MERGED — NEW BEST** |
