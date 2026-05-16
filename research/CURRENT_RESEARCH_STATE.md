# SENPAI Research State

- **Date**: 2026-05-16 10:45
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active — winner pending merge; 4 new stacking arms launched
- **Most recent human research directive**: None received

## Current Best

**PR #3683 (H39 Arm C: n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0, thorfinn) — val_avg/mae_surf_p = 63.4385** (pending rebase + merge)

Test 3-split avg (excl. cruise NaN bug): **61.3910**.

Last merged baseline: PR #3629 (H37b: n_head=2 + lr=1e-3 + clip=1.0) val_avg = 66.1060.

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| **H39 Arm C (n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0)** | **63.4385 (best) / 64.47 (mean of 2 seeds)** | **NEW BEST — pending merge after thorfinn rebase (PR #3683)** |
| H37b (n_head=2 + lr=1e-3 + clip=1.0, default wd) | 66.1060 | Last merged (PR #3629) |
| H39 Arm B (lr=2e-3 + clip=1.0) | 66.3351 | Pre-stack 2-way |
| H38 Arm B (wd=5e-5 + lr=1e-3 + clip=1.0) | 68.1932 | Pre-stack 2-way |
| H47 eta_min=5e-5 | 67.3452 | CLOSED — schedule lever exhausted |
| H44 Arm C (β₁=0.8 + n_head=2 + wd=5e-5) | 67.2459 | CLOSED — β₁ doesn't stack |
| H46 (n_head=1) | 69.1666 | CLOSED — monotone trend breaks |
| H42 Arm C (n_layers=3 + n_head=2 + wd=5e-5) | 69.16 | CLOSED — capacity reductions don't stack |

**Test metrics:** 61.391 (3-split avg, excl. cruise NaN; H39 Arm C best seed)

## Key Confirmed Insights

1. **Super-additive stacking confirmed at H39 Arm C**: n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0 = -6.00 pts vs H32 baseline, -2.67 pts vs H37b.
2. **T_max=15 hardcoded match is the dominant first-order bottleneck fix (R1)**: 11.7-pt gain.
3. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 — merged defaults.
4. **Grad clip=1.0 is effective (H20)**: Active every step; clip 2/3 regress (H40 confirmed).
5. **lr monotone NOT broken at 2e-3 (H39)**: 1e-3→1.5e-3→2e-3 gives 69.44→68.12→66.34→63.44 with full stack. H51 (lr=2.5e-3, 3e-3) now testing whether trend continues.
6. **wd=5e-5 is LR-normalized regularization (H38)**: Stacks with higher LR.
7. **n_head=2 is the global optimum (H37b, H46)**: U-shape 8→4→2→1 — head_dim=64 with 2-head ensembling.
8. **Architecture width fails (H33)**: n_hidden=192/256 regress.
9. **n_layers=3 wins isolated (budget × structure efficiency) but DOES NOT stack with n_head=2 (H42 Arm C)**: Capacity reductions interact destructively.
10. **β₁=0.8 isolated win does NOT stack (H44 Arm C)**: β₁=0.9 is the right Adam default for this config.
11. **Warmup eats budget (H43)**: Schedule deviations cost epochs at fixed wall cap.
12. **Schedule lever exhausted at 14-15 epoch budget (H43 warmup, H41 Arm C T_max stretch, H47 eta_min all failed)**: WSD (H50, in-flight) is the last open schedule angle — mechanistically distinct from those failures.
13. **β₂=0.999 confirmed (H36)**: β₂=0.95 hurts.
14. **EMA/SWA fail at 14-epoch budget**: No plateau to average over.
15. **DropPath fails at 14-epoch budget (H45)**: Pre-overfit regime.
16. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Use 3-split excl. cruise.

## Active R5 cycle 5 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| **#3683** | thorfinn | **H39 Arm C (val=63.44)** — needs rebase + merge | SENT BACK |
| **#3896** | alphonse | **H51: LR ceiling lr=2.5e-3, 3e-3** at H39 Arm C stack | WIP — pushes monotone trend |
| **#3897** | frieren | **H56: lower clip (0.5, 0.7)** at H39 Arm C stack | WIP — orthogonal step-size lever |
| **#3898** | nezuko | **H54: surf_weight sweep (5, 20)** at H39 Arm C stack | WIP — direct metric alignment |
| **#3899** | tanjiro | **H55: Mixup data augmentation (α=0.2, 0.4)** | WIP — data-side regularizer |
| #3834 | askeladd | H48: GEGLU/SwiGLU FFN | WIP — architectural lever |
| #3859 | edward | H49: Lion optimizer | WIP — optimizer-tier swap |
| #3862 | fern | H50: WSD trapezoidal schedule | WIP — schedule shape (last open angle) |

All 8 students active. Zero idle.

## Key Open Questions

1. **Does LR ceiling continue past 2e-3?** H51 alphonse. Predicted 62-65 if monotone trend holds.
2. **Does clip < 1.0 win at lr=2e-3?** H56 frieren. Tests step-size lever orthogonal to LR.
3. **Does direct metric alignment via surf_weight help?** H54 nezuko. Surf=20 doubles surface focus.
4. **Does Mixup help OOD splits at the new stack?** H55 tanjiro. Data-side regularizer for camber/Re generalization.
5. **Does GEGLU FFN win independently?** H48 askeladd. Architecture tier change.
6. **Does Lion optimizer beat AdamW?** H49 edward. Optimizer-tier swap.
7. **Does WSD trapezoidal schedule work?** H50 fern. Last open schedule shape.

## Key Closed Dead Ends This Round

- **H42 Arm C (n_layers=3 + n_head=2 + wd=5e-5)**: +5.72 vs H39 Arm C. Capacity reductions stack destructively.
- **H44 Arm C (β₁=0.8 + n_head=2 + wd=5e-5)**: +3.81 vs H39 Arm C. β₁ isolated win was config-sensitive.
- **H46 (n_head=1)**: +3.06 vs H37b. n_head=2 is the floor of the monotone trend.
- **H47 (eta_min sweep)**: +1.24 vs H37b. Schedule deviation hurts at this budget.
- **H45 (DropPath 0.1/0.2)**: Pre-overfit regime at 14-epoch budget.
- **H41 Arm C (T_max=20 + n_head=2 + wd=5e-5)**: Schedule extension strips fine-tune tail.
- **H43 (warmup 1/2 ep)**: Warmup eats cosine budget at fixed wall cap.
- **H40 (clip=2.0/3.0)**: clip=1.0 confirmed optimal.

## Predicted Next Directions

- **If H51 (LR ceiling) wins** at lr=2.5e-3 or 3e-3 → push higher, find peak.
- **If H56 (lower clip) wins** at clip=0.5/0.7 → test clip=0.3 + lr=2e-3 (tighten further).
- **If H54 surf_weight=20 wins** → push to surf_weight=40 (double again).
- **If H55 Mixup wins** at α=0.2 or 0.4 → test α=0.6 + CutMix variant for stronger augmentation.
- **If H50 WSD wins** → test WSD + lr=2e-3 + n_head=2 + wd=5e-5 mega-stack (5-way compound).
- **If H49 Lion wins** → test Lion + n_head=2 + wd=5e-5 stack at H37b base.
- **If H48 GEGLU wins** → test GEGLU + H39 Arm C 4-way mega-stack.

The 4-way H39 Arm C stack opens significant new search space. Any of the cycle-5 levers (LR ceiling, clip, surf_weight, Mixup, optimizer, schedule, architecture) that win can compound on top of the stack for a 5-way mega-experiment.

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `T_max=15` hardcoded in scheduler — students doing T_max sweeps must add CLI flag (H41 confirmed pattern).
- PR #3683 has merge conflict against advisor branch — thorfinn rebasing.
