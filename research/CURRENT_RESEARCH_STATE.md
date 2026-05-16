# SENPAI Research State

- **Date**: 2026-05-16 08:00
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active (H37b merged as new best → stacking round fully launched)
- **Most recent human research directive**: None received

## Current Best

**PR #3629 (H37b: n_head=2 + lr=1e-3 + clip=1.0, tanjiro) — val_avg/mae_surf_p = 66.1060** (merged 2026-05-16)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| **H37b (n_head=2 + lr=1e-3 + clip=1.0)** | **66.1060** | **Current best (PR #3629)** |
| H39 Arm B (lr=2e-3 + clip=1.0) | 66.3351 | (PR #3683) — 0.23 above baseline, sent back |
| H38 Arm B (wd=5e-5 + lr=1e-3 + clip=1.0) | 68.1932 | (PR #3651) — orthogonal to H37b |
| H41 Arm A (T_max=20 + lr=1e-3 + clip=1.0) | 66.9242 | (PR #3688) — 0.81 above baseline, sent back |
| H32 Arm A (lr=1e-3 + clip=1.0, default wd) | 69.4381 | (PR #3557) — overridden |

**Test metrics (3-split avg, excl. cruise NaN bug):** 64.4522 (H37b) — **vs 65.4393 (H38)**

**Key finding (H37b):** n_head=2 + lr=1e-3 stacking is **super-additive** — predicted 66.83 from independent gains (H37 isolated at -2.61 on H20), actual **66.11**. n_head: 8→4→2 monotone improving; head_dim progression (16→32→64) drives richer per-head slice projection.

## Key Confirmed Insights

1. **T_max mismatch was the dominant first-order bottleneck** (R1): T_max=15 fix gave 11.7-pt gain.
2. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 now merged defaults.
3. **Grad clip=1.0 is effective (H20)**: Active every step, prevents high-Re samples dominating.
4. **Clip absorbs gradient-magnitude-shaping interventions (H23b/H29/H30/H31/H26b)**: Confirmed 5×. Clip=2.0/3.0 regress at lr=1e-3 (H40) — 1.0 is confirmed optimum.
5. **lr=1e-3 is dramatically better (H27b/H32)**: Monotone 5e-4→8e-4→1e-3 gives 75.50→73.11→69.44.
6. **LR ceiling not visible at lr=2e-3 (H39)**: 1e-3→1.5e-3→2e-3 gives 69.44→68.12→66.34. Monotone continues.
7. **wd=5e-5 is better at lr=1e-3 (H38)**: LR-normalized regularization — orthogonal to clip.
8. **n_head=2 stacks super-additively (H37b)**: head_dim=64 vs 32 → current best 66.11.
9. **T_max=20 wins (H41)**: Keeps final-epoch LR at 21% of peak vs 4.5%. +0.82 epoch budget critical.
10. **Architecture width fails (H33)**: n_hidden=192/256 both regress +15-22%.
11. **β₂=0.999 confirmed (H36)**: β₂=0.95 hurts in cosine-decay tail.
12. **Element-wise clip too aggressive (H34)**: Dead end.
13. **δ_p=0.25 is the optimum (H31)**: Knob exhausted.
14. **Averaging (EMA/SWA) fails at 14-epoch budget (H20-era)**: No plateau to average over.
15. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Use 3-split excl. cruise.

## Active R5 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3683 | thorfinn | **H39 Arm C: n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0** | SENT BACK (predicted ≈ 63–64) |
| #3688 | fern | **H41 Arm C: T_max=20 + n_head=2 + wd=5e-5 + clip=1.0** | SENT BACK (predicted ≈ 63–64) |
| #3729 | askeladd | H43: Linear warmup (1, 2 ep) + lr=1e-3 + wd=5e-5 + clip=1.0 | WIP |
| #3737 | frieren | H44: AdamW β₁ sweep (0.8, 0.95) on H38 base | WIP |
| #3767 | edward | H45: DropPath (0.1, 0.2) on H38 base | WIP |
| #3689 | alphonse | H42: n_layers sweep (7, 3) on lr=1e-3 base | WIP (rate-limited, will resume) |
| **#3805** | tanjiro | **H46: n_head=1 (head_dim=128) — limit of monotone trend** | NEW |
| **#3807** | nezuko | **H47: cosine eta_min sweep (5e-5, 1e-4) on H37b base** | NEW |

## Key Open Questions

1. **Can lr=2e-3 + n_head=2 + wd=5e-5 stack?** H39 Arm C (thorfinn). Predicted ≈ 63–64. Potentially the highest-impact experiment running.
2. **Can T_max=20 + n_head=2 + wd=5e-5 stack?** H41 Arm C (fern). Parallel highest-impact test.
3. **Does n_head=1 extrapolate the monotone trend?** H46 (tanjiro). n_head 8→4→2 monotone, limit test.
4. **Does cosine eta_min > 0 help at n_head=2 + wd=5e-5?** H47 (nezuko). Orthogonal to T_max — controls LR floor vs schedule shape.
5. **Does warmup help at lr=1e-3?** H43 (askeladd). Early-gradient instability probe.
6. **Does β₁ matter?** H44 (frieren). Last clean AdamW knob.
7. **Does DropPath regularize effectively?** H45 (edward). Stochastic depth on H38 base.
8. **Does depth help where width fails?** H42 (alphonse). n_layers=7/3.

## Key Closed Dead Ends This Round

- **H40 (clip=2.0/3.0)**: Regress at lr=1e-3. clip=1.0 is confirmed optimum.
- **H36 (β₂ sweep)**: β₂=0.95 hurts. β₂=0.999 stays.
- **H35 (slice_num)**: Walltime-confounded — shelved.
- **H34 (element-wise clip)**: Too aggressive.
- **H31 (δ_p push)**: δ_p=0.25 optimal.
- **H26b (cond_dim reduction)**: FiLM path weakened by clip.

## Predicted Next Directions (when current WIPs land)

- **Triple stack (n_head=2 + lr=2e-3 + T_max=20 + wd=5e-5)**: If H39 Arm C and H41 Arm C both win, the full compound could push to ≤62.
- **n_head=1 (single head, head_dim=128)**: If monotone trend continues → H46.
- **Lion optimizer**: Different update mechanism (sign of grad) — compresses directional signal, orthogonal to clip magnitude control.
- **LLRD (per-layer LR decay)**: Lower LR for deeper Transolver blocks.
- **Focal Huber / adaptive loss weights**: Per-sample loss reweighting for high-Re outliers.
- **Schedule-Free AdamW**: Eliminates LR schedule entirely — may be simpler and equally effective.

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. File is read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `T_max=15` hardcoded in train.py scheduler — students doing T_max sweeps must add CLI flag (H41 confirmed working).
- H42 (alphonse) rate-limited — will auto-resume when GH API resets.
- GH REST API rate-limited until ~09:00 UTC; labels on H46/H47 PRs set correctly via GraphQL.
