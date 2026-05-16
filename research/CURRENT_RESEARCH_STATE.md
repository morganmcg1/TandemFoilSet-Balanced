# SENPAI Research State

- **Date**: 2026-05-16 09:30
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active — stacking wave in full swing
- **Most recent human research directive**: None received

## Current Best

**PR #3629 (H37b: n_head=2 + lr=1e-3 + clip=1.0, tanjiro) — val_avg/mae_surf_p = 66.1060** (merged 2026-05-16)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| **H37b (n_head=2 + lr=1e-3 + clip=1.0)** | **66.1060** | **Current best (PR #3629)** |
| H39 Arm B (lr=2e-3 + clip=1.0) | 66.3351 | Sent back for n_head=2 + wd=5e-5 stack |
| H38 Arm B (wd=5e-5 + lr=1e-3 + clip=1.0) | 68.1932 | Orthogonal to H37b |
| H41 Arm A (T_max=20 + lr=1e-3 + clip=1.0) | 66.9242 | Sent back for n_head=2 + wd=5e-5 stack |
| H42 Arm B (n_layers=3 + lr=1e-3 + clip=1.0) | 67.9740 | Sent back for n_head=2 + wd=5e-5 stack |
| H44 Arm A (β₁=0.8 + lr=1e-3 + wd=5e-5) | 66.6492 | Sent back for n_head=2 stack |
| H32 Arm A (lr=1e-3 + clip=1.0, default wd) | 69.4381 | Overridden |

**Test metrics (3-split avg, excl. cruise NaN bug):** 64.4522 (H37b)

## Key Confirmed Insights

1. **T_max mismatch was the dominant first-order bottleneck** (R1): T_max=15 fix gave 11.7-pt gain.
2. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 now merged defaults.
3. **Grad clip=1.0 is effective (H20)**: Active every step. Clip=2.0/3.0 regress (H40 confirmed).
4. **lr monotone: ceiling not visible at lr=2e-3 (H39)**: 1e-3→1.5e-3→2e-3 gives 69.44→68.12→66.34.
5. **wd=5e-5 is better at lr=1e-3 (H38)**: LR-normalized regularization — orthogonal.
6. **n_head=2 stacks super-additively (H37b)**: head_dim=64 > 32, super-additive with lr.
7. **T_max=20 wins (H41)**: Keeps final-epoch LR at 21% of peak vs 4.5%.
8. **Architecture width fails (H33)**: n_hidden=192/256 regress. Capacity is not the bottleneck.
9. **n_layers=3 wins via budget × structure efficiency (H42)**: 524K params, 21 epochs in 30min. Over-parameterized at n_layers=5.
10. **β₁=0.8 promising (H44)**: Faster moment decay, -1.54 vs H38. Below seed var threshold alone, needs stack.
11. **Warmup eats budget (H43)**: Budget-stealing at fixed wall cap. Cosine duration loss > gradient stability gain.
12. **β₂=0.999 confirmed (H36)**: β₂=0.95 hurts.
13. **EMA/SWA fail at 14-epoch budget**: No plateau to average over.
14. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Use 3-split excl. cruise.

## Active R5 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#3683** | thorfinn | **H39 Arm C: n_head=2 + lr=2e-3 + wd=5e-5** | SENT BACK (predicted ≈ 63–64) — HIGHEST PRIORITY |
| **#3688** | fern | **H41 Arm C: T_max=20 + n_head=2 + wd=5e-5** | SENT BACK (predicted ≈ 63–64) — HIGH |
| **#3689** | alphonse | **H42 Arm C: n_layers=3 + n_head=2 + wd=5e-5** | SENT BACK (predicted ≈ 63–65, 21 ep budget) — HIGH |
| **#3737** | frieren | **H44 Arm C: β₁=0.8 + n_head=2 + wd=5e-5** | SENT BACK (predicted ≈ 64.5–65.5) |
| **#3834** | askeladd | **H48: GEGLU/SwiGLU FFN gating** | NEW (predicted ≈ 64–65.5) |
| #3805 | tanjiro | H46: n_head=1 (head_dim=128) | WIP — limit of monotone trend |
| #3807 | nezuko | H47: cosine eta_min sweep (5e-5, 1e-4) | WIP — orthogonal LR floor lever |
| #3767 | edward | H45: DropPath (0.1, 0.2) on H38 base | WIP |

## Key Open Questions

1. **Can n_head=2 + lr=2e-3 + wd=5e-5 triple-stack?** H39 Arm C (thorfinn). Predicted ≈ 63–64. Highest priority.
2. **Can T_max=20 + n_head=2 + wd=5e-5 compound?** H41 Arm C (fern). Same priority.
3. **Can n_layers=3 + n_head=2 + wd=5e-5 stack?** H42 Arm C (alphonse). Predicted ≈ 63–65 — *extra headroom* from 21 epochs vs 14.
4. **Does β₁=0.8 + n_head=2 + wd=5e-5 stack?** H44 Arm C (frieren). Predicted ≈ 64.5–65.5.
5. **Does GEGLU FFN win independently?** H48 (askeladd). Architecture tier change.
6. **Does n_head=1 extrapolate the monotone trend?** H46 (tanjiro). Limit test.
7. **Does cosine eta_min > 0 help?** H47 (nezuko). LR floor lever orthogonal to T_max.
8. **Does DropPath regularize?** H45 (edward). Stochastic depth.

## Key Closed Dead Ends This Round

- **H43 (warmup 1/2 ep)**: Warmup eats cosine budget at fixed wall cap. Closure confirmed.
- **H40 (clip=2.0/3.0)**: Regress. clip=1.0 confirmed optimal.
- **H36 (β₂=0.95)**: Hurts. β₂=0.999 stays.
- **H35 (slice_num=96/128)**: Walltime-confounded.
- **H34 (element-wise clip)**: Too aggressive.
- **H33 (n_hidden=192/256)**: Width excess fails.

## Predicted Next Directions (after current WIPs land)

- **n_layers=3 + n_head=2 + lr=2e-3 + T_max=20 + wd=5e-5**: Super-compound if all 4 stack simultaneously. Could push to ≤61.
- **n_head=1 isolation**: If H46 confirms monotone, single-head is the floor.
- **H50 (WSD trapezoidal schedule)**: Holds lr at peak for more epochs, then sharp cooldown. Researcher-agent Priority 2.
- **H49 (Lion optimizer)**: Bold swing, researcher-agent Priority 3.
- **β₁=0.8 isolation confirmed**: If H44 Arm C stacks, this becomes a merged default.
- **n_layers=3 isolation**: If H42 Arm C wins, n_layers=3 becomes a new merged default with 21-epoch budget advantage.

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `T_max=15` hardcoded in scheduler — students doing T_max sweeps must add CLI flag (H41 confirmed pattern).
- PRs #3683/#3688/#3689/#3737 all need rebase against updated advisor branch (H37b merge). Students instructed to rebase before running Arm C.
