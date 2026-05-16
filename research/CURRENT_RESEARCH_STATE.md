# SENPAI Research State

- **Date**: 2026-05-16 12:45
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active — GEGLU architecture breakthrough; mega-stacking phase
- **Most recent human research directive**: None received

## Current Best

**PR #3834 (H48: GEGLU gated FFN, askeladd) — val_avg/mae_surf_p = 58.6268** (MERGED)

Test 3-split avg (excl. cruise NaN bug): **56.6976** — largest OOD gains in the run.

| Reference | val_avg/mae_surf_p | Status |
|-----------|--------------------|--------|
| **H48 GEGLU (lr=1e-3 + n_head=2 + wd=5e-5 + clip=1.0 + ffn_act=geglu)** | **58.6268** | **CURRENT BEST (PR #3834)** |
| H48 SwiGLU (same stack + swiglu) | 61.4410 | Merged |
| H49 Lion Arm A (optimizer=lion + lr=1e-4 + wd=1e-3, H37b base) | 60.3008 | Closed PR #3859 |
| H39 Arm C (n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0) | 63.4385 | Merged PR #3683 (documentation) |
| H37b (n_head=2 + lr=1e-3 + clip=1.0, default wd) | 66.1060 | Overridden |

**Δ H48 GEGLU vs H37b: −7.48 pts val, −7.75 pts test.** Largest single-PR gain since the T_max mismatch fix.

## Key Confirmed Insights

1. **GEGLU gated FFN is a major win (H48)**: val=58.63 vs H37b 66.11. Spatial selectivity via multiplicative gating is a direct fit for CFD boundary-layer gradients. GEGLU outperforms SwiGLU by 2.8 pts.
2. **GEGLU generalizes better than it in-fits**: test gain > val gain. Gating mechanism reduces OOD sensitivity — critical for our cross-geometry/Reynolds evaluation.
3. **T_max=15 hardcoded mismatch was the first-order fix (R1)**: 11.7-pt gain.
4. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 — merged defaults.
5. **Grad clip=1.0 is effective (H20)**: clip=2/3 regress; lower clip (H56 in-flight) may help at lr=2e-3.
6. **lr monotone trend holds through 2e-3 (H39 Arm C)**: −2.67 pts vs H37b at n_head=2+wd=5e-5 stack.
7. **wd=5e-5 is LR-normalized regularization (H38)**: Both H48 and H39 used wd=5e-5.
8. **n_head=2 is the global optimum (H37b, H46)**: U-shape 8→4→2→1 confirmed.
9. **Architecture width fails (H33)**: n_hidden=192/256 regress.
10. **n_layers=3 isolated win does NOT stack with n_head=2 (H42 Arm C)**: Capacity reductions destroy each other — BUT this was pre-GEGLU. H60 revisits depth in GEGLU context.
11. **β₁=0.8 isolated win does NOT stack (H44 Arm C)**: β₁=0.9 is optimal.
12. **Schedule lever exhausted at 14-epoch budget**: H43/H41C/H47/H50 all failed. WSD (H50) also failed — cosine T_max=15 is the right schedule for our compute regime.
13. **Lion optimizer beats AdamW at H37b base (H49)**: val=60.30 (Arm A lr=1e-4), −5.80 vs H37b. Mechanism: sign-normalization removes gradient imbalance between high-Re and low-Re samples. Closed because H48 GEGLU baseline at 58.63 is better; Lion+GEGLU (H58) is the follow-up.
14. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.

## Active WIP Experiments

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#3918** | askeladd | **H57: GEGLU + lr=2e-3** (mega-stack) | **CRITICAL** | ~55-57 |
| **#3965** | edward | **H58: Lion + GEGLU** (mega-stack) | **CRITICAL** | ~52-55 |
| **#3966** | fern | **H59: RMSNorm in GEGLU Transolver** | HIGH | ~57-58 |
| **#3968** | thorfinn | **H60: GEGLU + n_layers (4 vs 6)** | HIGH | ~57-58 |
| #3899 | tanjiro | H55: Mixup (α=0.2, 0.4) | HIGH | Beats 58.63 if orthogonal |
| #3898 | nezuko | H54: surf_weight (5, 20) | MEDIUM | ~58-60 (pre-GEGLU design) |
| #3897 | frieren | H56: lower clip (0.5, 0.7) | MEDIUM | May compound with GEGLU |
| #3896 | alphonse | H51: LR ceiling (2.5e-3, 3e-3) | MEDIUM | ~60+ (pre-GEGLU design) |

All 8 students active. Zero idle.

**Note:** H51/H56/H54/H55 were designed for H39 Arm C config (val=63.44). With GEGLU now at 58.63, these need to beat 58.63 to matter. H51 and H54 are likely underperforming — still useful as data on LR ceiling and surf_weight. H55 (Mixup) and H56 (lower clip) have highest chance of beating GEGLU baseline because data augmentation and gradient control are orthogonal to the architecture.

## Cycle 8 PR Decisions

| PR | Decision | Reason |
|----|----------|--------|
| #3683 (thorfinn H39 Arm C) | **MERGED** | Documentation; BASELINE.md already updated; artifacts now on advisor branch |
| #3862 (fern H50 WSD) | **CLOSED** | Both arms regress vs H37b (−1.15 to −5.21 pts). Schedule angle exhausted — 4th schedule failure (H43/H41C/H47/H50) at this budget |
| #3859 (edward H49 Lion) | **CLOSED → H58** | Lion beats H37b by −5.80 but doesn't beat H48 GEGLU (1.67 pts gap). Forwarded as Lion+GEGLU mega-stack |

## Key Open Questions

1. **Does GEGLU + lr=2e-3 compound?** H57 askeladd (#3918). Predicted ≈ 55-57. HIGHEST PRIORITY.
2. **Does Lion + GEGLU compound?** H58 edward (#3965). Predicted ≈ 52-55. Would be largest single gain since T_max fix.
3. **Does RMSNorm help GEGLU's gate?** H59 fern (#3966). Mechanistically motivated, clean binary test.
4. **Does depth become a fresh lever in GEGLU?** H60 thorfinn (#3968). n_layers=4 vs 6.
5. **Does Mixup help OOD?** H55 tanjiro (#3899). Mixup + GEGLU may compound.
6. **What's next below 55?**:
   - GEGLU + lr=2e-3 + Lion (if H57 + H58 both win)
   - GEGLU + lr=2e-3 + Mixup (if H55 confirms)
   - GEGLU + RMSNorm + lr=2e-3 (if H59 wins)
   - GEGLU + n_layers=6 + lr=2e-3 (if H60 Arm A wins)
   - Architecture-level changes: attention mechanism variants, FiLM conditioning improvements, PDE-informed residuals

## Baseline Progression

| Val avg/mae_surf_p | Test 3-split | Event |
|---|---|---|
| 114.63 | — | R1 start (FiLM only, T_max=50 mismatch) |
| 83.81 | 80.24 | H19: T_max=15 + Huber + FiLM |
| 75.50 | 73.16 | H20: clip=1.0 |
| 71.77 | 70.62 | H27b/H32: lr=1e-3 |
| 68.19 | 65.44 | H38: wd=5e-5 |
| 66.11 | 64.45 | H37b: n_head=2 + lr=1e-3 |
| 63.44 | 61.39 | H39 Arm C: + lr=2e-3 (merged #3683, documentation) |
| **58.63** | **56.70** | **H48 GEGLU: + ffn_act=geglu** |

Total gain: **−56.0 pts val** (48.8% reduction from 114.63 to 58.63).

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `T_max=15` hardcoded in scheduler — students doing T_max sweeps must add CLI flag.
- H51/H56/H54/H55 were designed against H39 Arm C baseline (63.44). Compare against new baseline 58.63 when they land.
