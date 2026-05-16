# SENPAI Research State

- **Date**: 2026-05-16 12:00
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active — GEGLU architecture breakthrough; mega-stack now in progress
- **Most recent human research directive**: None received

## Current Best

**PR #3834 (H48: GEGLU gated FFN, askeladd) — val_avg/mae_surf_p = 58.6268** (MERGED)

Test 3-split avg (excl. cruise NaN bug): **56.6976** — largest OOD gains in the run.

| Reference | val_avg/mae_surf_p | Status |
|-----------|--------------------|--------|
| **H48 GEGLU (lr=1e-3 + n_head=2 + wd=5e-5 + clip=1.0 + ffn_act=geglu)** | **58.6268** | **CURRENT BEST (PR #3834)** |
| H48 SwiGLU (same stack + swiglu) | 61.4410 | Merged |
| H39 Arm C (n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0) | 63.4385 | Pending merge (rebase #3683) |
| H37b (n_head=2 + lr=1e-3 + clip=1.0, default wd) | 66.1060 | Overridden |

**Δ H48 GEGLU vs H37b: −7.48 pts val, −7.75 pts test.** Largest single-PR gain since the T_max mismatch fix.

## Key Confirmed Insights

1. **GEGLU gated FFN is a major win (H48)**: val=58.63 vs H37b 66.11. Spatial selectivity via multiplicative gating is a direct fit for CFD boundary-layer gradients. GEGLU outperforms SwiGLU by 2.8 pts.
2. **GEGLU generalizes better than it in-fits**: test gain > val gain. Gating mechanism reduces OOD sensitivity — critical for our cross-geometry/Reynolds evaluation.
3. **T_max=15 hardcoded mismatch was the first-order fix (R1)**: 11.7-pt gain.
4. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 — merged defaults.
5. **Grad clip=1.0 is effective (H20)**: clip=2/3 regress; lower clip (H56 in-flight) may help at lr=2e-3.
6. **lr monotone: trend holds through 2e-3 (H39 Arm C)**: −2.67 pts vs H37b. H51 testing ceiling.
7. **wd=5e-5 is LR-normalized regularization (H38)**: Both H48 and H39 used wd=5e-5.
8. **n_head=2 is the global optimum (H37b, H46)**: U-shape 8→4→2→1 confirmed.
9. **Architecture width fails (H33)**: n_hidden=192/256 regress.
10. **n_layers=3 isolated win does NOT stack with n_head=2 (H42 Arm C)**: Capacity reductions destroy each other.
11. **β₁=0.8 isolated win does NOT stack (H44 Arm C)**: β₁=0.9 is optimal.
12. **Schedule lever exhausted at 14-epoch budget**: H43/H41C/H47 all failed. WSD (H50) is the last open angle.
13. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Use 3-split excl. cruise.

## Active WIP Experiments

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#3918** | askeladd | **H57: GEGLU + lr=2e-3** (mega-stack) | **CRITICAL** | ~55-57 |
| **#3683** | thorfinn | H39 Arm C rebase | HIGH | pending rebase |
| #3896 | alphonse | H51: LR ceiling (2.5e-3, 3e-3) at H39 stack | MEDIUM | ~62-65 (pre-GEGLU) |
| #3897 | frieren | H56: lower clip (0.5, 0.7) at H39 stack | MEDIUM | ~62-65 (pre-GEGLU) |
| #3898 | nezuko | H54: surf_weight (5, 20) at H39 stack | MEDIUM | ~62-65 (pre-GEGLU) |
| #3899 | tanjiro | H55: Mixup (α=0.2, 0.4) | HIGH | May compound with GEGLU |
| #3859 | edward | H49: Lion optimizer | MEDIUM | ~55-66 |
| #3862 | fern | H50: WSD schedule | MEDIUM | ~56-60 |

**Note:** H51/H56/H54 were designed for the H39 Arm C config (val=63.44). With GEGLU now at 58.63, these need to beat 58.63 to matter — they may not. They still provide useful data on LR ceiling, clip, and surf_weight. If they land before H57, compare against 58.63.

All 8 students active. Zero idle.

## Key Open Questions

1. **Does GEGLU + lr=2e-3 compound?** H57 askeladd. Predicted ≈ 55-57. HIGHEST PRIORITY.
2. **Does LR ceiling continue past 2e-3?** H51 alphonse. On pre-GEGLU stack — still useful.
3. **Does WSD schedule help?** H50 fern.
4. **Does Lion beat AdamW?** H49 edward.
5. **Does Mixup help OOD generalization?** H55 tanjiro. May compound with GEGLU.
6. **What's next below 55?**:
   - GEGLU + lr=2e-3 + lr ceiling push (if H51/H57 show higher LR still helps)
   - GEGLU + lr=2e-3 + Mixup
   - GEGLU + lr=2e-3 + WSD (if H50 wins)
   - GEGLU + n_layers=3 (revisit depth in new architecture context)

## Baseline Progression

| Val avg/mae_surf_p | Test 3-split | Event |
|---|---|---|
| 114.63 | — | R1 start (FiLM only, T_max=50 mismatch) |
| 83.81 | 80.24 | H19: T_max=15 + Huber + FiLM |
| 75.50 | 73.16 | H20: clip=1.0 |
| 71.77 | 70.62 | H27b/H32: lr=1e-3 |
| 68.19 | 65.44 | H38: wd=5e-5 |
| 66.11 | 64.45 | H37b: n_head=2 + lr=1e-3 |
| 63.44 | 61.39 | H39 Arm C: + lr=2e-3 (pending merge) |
| **58.63** | **56.70** | **H48 GEGLU: + ffn_act=geglu** |

Total gain: **−56.0 pts val** (48.8% reduction from 114.63 to 58.63).

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `T_max=15` hardcoded in scheduler — students doing T_max sweeps must add CLI flag.
- PR #3683 has merge conflict (train.py conflict with H48 GEGLU) — thorfinn rebasing again.
- H51/H56/H54 were designed against H39 Arm C baseline (63.44). Compare against new baseline 58.63 when they land.
