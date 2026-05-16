# TandemFoilSet Baseline — icml-appendix-charlie-pai2i-48h-r3

## Current Best

**PR #3651 — H38: Weight decay reduction (wd=5e-5) at lr=1e-3 + clip=1.0 (frieren)**
Merged 2026-05-16. 13 epochs completed (30-min timeout cap; best epoch = final epoch).

Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits). Lower is better.

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | **68.1932** | PR #3651 Arm B |
| val_single_in_dist/mae_surf_p | 76.8452 | PR #3651 Arm B |
| val_geom_camber_rc/mae_surf_p | 84.3542 | PR #3651 Arm B |
| val_geom_camber_cruise/mae_surf_p | 44.4649 | PR #3651 Arm B |
| val_re_rand/mae_surf_p | 67.1084 | PR #3651 Arm B |
| test_avg/mae_surf_p | NaN (⚠ scoring bug) | PR #3651 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **65.4393** | PR #3651 Arm B |
| test_single_in_dist/mae_surf_p | 66.5542 | PR #3651 Arm B |
| test_geom_camber_rc/mae_surf_p | 72.9271 | PR #3651 Arm B |
| test_re_rand/mae_surf_p | 56.8366 | PR #3651 Arm B |

**Configuration:** FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 (merged defaults) + CosineAnnealingLR T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + **wd=5e-5**.

**Context:** AdamW applies `lr × wd × param` per step, so raising lr from 5e-4 → 1e-3 effectively doubled the per-step L2 penalty (wd=1e-4 was tuned at 5e-4). wd=5e-5 restores the original effective regularization strength per step. Arm A (wd=0) also beat baseline by -1.16 but less convincingly. Arm B improved **all 4 val splits** vs H32 (single_in_dist -2.83, cruise -2.80, re_rand -0.74, rc -0.11). Orthogonal to clip — weight decay (parameter-norm penalty) is a different mechanism from gradient clipping (gradient-norm bound).

**⚠ data/scoring.py NaN bug:** `test_geom_camber_cruise` sample 20 has non-finite GT. File is read-only.

**Artifacts:** `models/model-h38-wd5e5-lr1e3-clip1-20260516-052550/`

**Reproduce:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h38-wd5e5-lr1e3-clip1 --agent <student> \
  --weight_decay 5e-5 --lr 1e-3 --clip_grad_norm 1.0
# FiLM cond_dim=11, Huber δ_vel=0.5/δ_p=0.25, T_max=15 are merged defaults
```

## Previous Best (overridden by #3651)

**PR #3557 — H32: LR=1e-3 + clip=1.0 on H20 base (thorfinn)**
Merged 2026-05-16. 13 epochs completed (30-min timeout cap; best epoch = final epoch).

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | **69.4381** | PR #3557 Arm A |
| val_single_in_dist/mae_surf_p | 79.6711 | PR #3557 Arm A |
| val_geom_camber_rc/mae_surf_p | 84.4672 | PR #3557 Arm A |
| val_geom_camber_cruise/mae_surf_p | 47.2669 | PR #3557 Arm A |
| val_re_rand/mae_surf_p | 66.3473 | PR #3557 Arm A |
| test_avg/mae_surf_p (3-split, excl. cruise) | **69.1774** | PR #3557 Arm A |

**Configuration:** FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + wd=1e-4 (default).

**Artifacts:** `models/model-h32-lr1e3-clip1-20260516-012246/`

## Previous Best (overridden by #3557)

**PR #3452 — H27b: LR=1e-3 + clip=1.0 on H20 base (frieren)**
Merged 2026-05-16. 13 epochs completed (30-min timeout cap; best epoch = final epoch).

Primary metric: `val_avg/mae_surf_p` (equal-weight mean surface pressure MAE across 4 val splits). Lower is better.

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | **71.7713** | PR #3452 Arm B |
| val_single_in_dist/mae_surf_p | 83.7818 | PR #3452 Arm B |
| val_geom_camber_rc/mae_surf_p | 85.0398 | PR #3452 Arm B |
| val_geom_camber_cruise/mae_surf_p | 49.5211 | PR #3452 Arm B |
| val_re_rand/mae_surf_p | 68.7425 | PR #3452 Arm B |
| test_avg/mae_surf_p | NaN (⚠ scoring bug) | PR #3452 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **70.6226** | PR #3452 Arm B |
| test_single_in_dist/mae_surf_p | 72.9392 | PR #3452 Arm B |
| test_geom_camber_rc/mae_surf_p | 78.0408 | PR #3452 Arm B |
| test_re_rand/mae_surf_p | 60.8879 | PR #3452 Arm B |

**Configuration:** FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 (merged defaults) + CosineAnnealingLR T_max=15 + clip_grad_norm=1.0 + **lr=1e-3**.

**Why higher LR works:** With clip=1.0 bounding per-step gradient norms, a higher peak LR is stable. Under T_max=15 cosine with 13-epoch wall budget, a higher peak LR covers more loss-landscape area in the high-LR phase. Pre-clip grad norms decayed from 8.6→2.3 — clip was active every step, confirming stability. Training monotone on both arms. **Every split improved** vs H20 baseline.

**Note on Arm A (lr=7e-4):** val_avg=75.9937 — essentially tied with H20 (75.4955). The monotone trend 5e-4→7e-4→1e-3 is: 75.50 (tie) → 75.99 (tie) → **71.77 (clear win)**. The jump happens in the 1e-3 range, not 7e-4.

**Note on `--huber_delta 0.5`:** This flag is a no-op in the current train.py (loss always uses per-channel `huber_delta_vel`/`huber_delta_p`). The realized loss config is δ_vel=0.5, δ_p=0.25 from merged defaults.

**⚠ data/scoring.py NaN bug:** `test_geom_camber_cruise` sample 20 has non-finite GT; `test_avg/mae_surf_p = NaN` for all PRs. File is read-only.

**Artifacts:** `models/model-h27b-lr1e3-clip1-20260516-012724/`

**Reproduce:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h27b-lr1e3-clip1 --agent <student> \
  --clip_grad_norm 1.0 --lr 1e-3
# FiLM cond_dim=11, Huber δ_vel=0.5/δ_p=0.25, CosineAnnealingLR T_max=15 are merged defaults
```

## Previous Best (overridden by #3452)

**PR #3445 — H20: Gradient clip=1.0 on H19 triple compound (nezuko)**
Merged 2026-05-16. 14 epochs completed (30-min timeout cap; LR fully annealed — best epoch = final epoch).

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | **75.4955** | PR #3445 Arm A |
| val_single_in_dist/mae_surf_p | 85.7272 | PR #3445 Arm A |
| val_geom_camber_rc/mae_surf_p | 85.4700 | PR #3445 Arm A |
| val_geom_camber_cruise/mae_surf_p | 55.7886 | PR #3445 Arm A |
| val_re_rand/mae_surf_p | 74.9964 | PR #3445 Arm A |
| test_avg/mae_surf_p | NaN (⚠ scoring bug) | PR #3445 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **73.1556** | PR #3445 Arm A |
| test_single_in_dist/mae_surf_p | 77.4314 | PR #3445 Arm A |
| test_geom_camber_rc/mae_surf_p | 77.5658 | PR #3445 Arm A |
| test_re_rand/mae_surf_p | 64.4696 | PR #3445 Arm A |

**Configuration:** FiLM cond_dim=11 (default) + Huber δ=0.5 + CosineAnnealingLR T_max=15 (default) + clip_grad_norm=1.0.

**Why grad clipping works:** Pre-clip gradient norm was 5–17× throughout training, meaning clipping was active at every step. Clamping per-step update magnitude to max_norm=1.0 prevents the Huber-activated tail gradients (from high-Re samples with large pressure spikes) from taking disproportionately large optimizer steps. Combined with FiLM's regime conditioning and T_max=15's full annealing, clip=1.0 provides the final stabilization layer that lets the model refine more consistently across all splits.

**Note on Arm B (clip=0.5):** val_avg=77.0687 — also beats H19, but over-clips gradient (17–34× reduction vs Arm A's 5–17×). clip=1.0 is the optimum.

**⚠ data/scoring.py NaN bug:** `test_geom_camber_cruise` sample 20 has non-finite GT; `test_avg/mae_surf_p = NaN` for all PRs. File is read-only.

**Artifacts:** `models/model-h20-clip1-h19-20260515-212335/`

**Reproduce:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h20-clip1-h19 --agent <student> \
  --huber_delta 0.5 --clip_grad_norm 1.0
# FiLM cond_dim=11 and CosineAnnealingLR T_max=15 are now the merged defaults
```

## Previous Best (overridden by #3445)

**PR #3450 — H25: Per-channel Huber δ_vel=1.0, δ_p=0.25 on H19 stack (askeladd)**
Merged 2026-05-15. 14 epochs completed (30-min timeout cap; LR fully annealed — best epoch = final epoch).

| Metric | Value | Source |
|--------|-------|--------|
| val_avg/mae_surf_p | **75.7713** | PR #3450 Arm B |
| val_single_in_dist/mae_surf_p | 86.5482 | PR #3450 Arm B |
| val_geom_camber_rc/mae_surf_p | 87.4861 | PR #3450 Arm B |
| val_geom_camber_cruise/mae_surf_p | 55.2883 | PR #3450 Arm B |
| val_re_rand/mae_surf_p | 73.7625 | PR #3450 Arm B |
| test_avg/mae_surf_p | NaN (⚠ scoring bug) | PR #3450 |
| test_avg/mae_surf_p (3-split, excl. cruise) | **73.0704** | PR #3450 Arm B |
| test_single_in_dist/mae_surf_p | 74.5330 | PR #3450 Arm B |
| test_geom_camber_rc/mae_surf_p | 78.8537 | PR #3450 Arm B |
| test_re_rand/mae_surf_p | 65.8246 | PR #3450 Arm B |

**Configuration:** FiLM cond_dim=11 + per-channel Huber δ_vel=1.0, δ_p=0.25 + T_max=15.

## Previous Best (overridden by #3450)

**PR #3408 — H19: FiLM + Huber δ=0.5 + T_max=15 triple compound (nezuko)**
Merged 2026-05-15. 14 epochs completed (30-min timeout cap; LR fully annealed — best epoch = final epoch).

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

**Artifacts:** `models/model-h19-film-huber-tmax15-triple-20260515-193153/`

## Previous Bests (overridden by #3408)

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
| #3408 | H19: FiLM+Huber δ=0.5+T_max=15 triple compound (nezuko) | 83.8136 | Merged 2026-05-15, overridden by #3450 |
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
| R3 | #3408 | H19: FiLM+Huber+T_max=15 triple compound (nezuko) | **83.8136** | **80.2415** (3-split) | **MERGED — overridden by #3450** |
| R3 | #3450 | H25: Per-channel Huber δ_vel=1.0/δ_p=0.25 on H19 (askeladd) | **75.7713** | **73.0704** (3-split) | **MERGED — NEW BEST** |
| R3 | #3447 | H22: Uniform Huber δ=0.1 on H19 stack (fern) | 78.8321 | ~73.2 (3-split) | **MERGED** (beats H19; below H25) |
