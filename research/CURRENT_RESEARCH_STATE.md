# SENPAI Research State

- **Date**: 2026-05-16 05:45
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active (H32 new best → R5 experiments H37b–H43)
- **Most recent human research directive**: None received

## Current Best

**PR #3557 (H32: LR=1e-3 + clip=1.0, thorfinn) — val_avg/mae_surf_p = 69.4381** (merged 2026-05-16)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| **H32 Arm A (lr=1e-3 + clip=1.0)** | **69.4381** | **Current best (PR #3557)** |
| H27b Arm B (lr=1e-3 + clip=1.0) | 71.7713 | (PR #3452) — same config, seed variance |
| H37 Arm B (n_head=2 on H20 base) | 72.8859 | (PR #3629) — sent back for lr=1e-3 retest |
| H32 Arm B (lr=8e-4 + clip=1.0) | 73.1104 | (PR #3557) |
| Grad clip=1.0 + H19 (H20) | 75.4955 | (PR #3445) |
| Per-channel Huber δ_vel=1.0/δ_p=0.25 (H25) | 75.7713 | (PR #3450) |
| FiLM + Huber δ=0.5 + T_max=15 (H19) | 83.8136 | (PR #3408) |

**Test metrics (3-split avg, excl. cruise NaN bug):** 69.1774 (H32 Arm A)

**Seed-variance note:** H27b (71.77) and H32 (69.44) ran identical configs (lr=1e-3+clip=1.0) ~1h apart. 2.33 pt spread is the single-seed noise floor for this stack. Improvements > 2.3 pts are strong signal.

## Key Confirmed Insights

1. **T_max mismatch was the dominant first-order bottleneck** (R1): T_max=15 fix gave 11.7-pt gain.
2. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 now merged defaults.
3. **Grad clip=1.0 is effective (H20)**: Active every step, prevents high-Re samples dominating.
4. **Clip absorbs gradient-magnitude-shaping interventions**: Confirmed across H29/H23b/H30/H31/H26b. Interventions that change the *scale* of gradients don't compound with clip_grad_norm=1.0.
5. **lr=1e-3 is dramatically better (H27b/H32)**: Monotone trend 5e-4→8e-4→1e-3 gives 75.50→73.11→69.44. With clip=1.0 as safety rail, high LR is stable. Ceiling not yet visible.
6. **Architecture width fails (H33)**: n_hidden=192/256 both regress +15-22%. Budget-constrained, not capacity-constrained at the hidden-dim level.
7. **n_head=2 is promising (H37)**: head_dim=64 beats head_dim=32 by -2.61 pts on H20 base. Sent back for critical lr=1e-3 stacking test (H37b, tanjiro WIP).
8. **Element-wise clip too aggressive at clip_value=1.0 (H34)**: Dead end.
9. **δ_p floor is 0.25 (H31)**: Optimum confirmed in both directions. Knob exhausted.
10. **cond_dim reduction fails with clip (H26b)**: FiLM path weakened by global clip rescale. cond_dim=11 optimal.
11. **β₂=0.999 is correct for this regime (H36)**: β₂=0.95 hurts in cosine-decay tail (amplifies noise when LR is tiny). β₂=0.999 stays.
12. **Averaging (EMA/SWA) fails at 14-epoch budget**: No plateau to average over.
13. **WSD fails at this budget (H21)**: Decay phase never fires.
14. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Workaround: 3-split excl. cruise.
15. **n_head=2 memory bonus**: 39.6 GB vs 44.6 GB baseline — enables headroom for larger batches or longer schedules.

## Active R5 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3629 | tanjiro | **H37b: n_head=2 + lr=1e-3 + clip=1.0 stacking test** | WIP (HIGHEST PRIORITY — predicted ≈66.8 if additive) |
| #3729 | askeladd | **H43: Linear warmup (1, 2 ep) + lr=1e-3 + clip=1.0** | WIP (NEW) |
| #3683 | thorfinn | H39: LR ceiling (lr=1.5e-3, 2e-3) + clip=1.0 | WIP |
| #3685 | nezuko | H40: Clip sweep (2.0, 3.0) at lr=1e-3 | WIP |
| #3688 | fern | H41: T_max sweep (20, 18) on lr=1e-3 base | WIP |
| #3689 | alphonse | H42: n_layers sweep (7, 3) on lr=1e-3 base | WIP |
| #3651 | frieren | H38: Weight decay sweep (wd=0, 5e-5) on lr=1e-3 base | WIP (actively training) |
| #3623 | edward | H35: slice_num sweep (96, 128) on H20 base | WIP (actively training, on old lr=5e-4 base) |

**Note on H35/H38 bases:** H38 (frieren, wd sweep) is on H27b base = lr=1e-3 + clip=1.0 ≡ H32 config. Results directly comparable to baseline. H35 (edward, slice_num) is on H20 base (lr=5e-4) — result will need retest at lr=1e-3 if promising.

## Key Open Questions

1. **Can n_head=2 stack with lr=1e-3?** H37b (tanjiro). Predicted ~66.8 if additive. Highest priority.
2. **Does warmup help at lr=1e-3?** H43 (askeladd). Tests whether early gradient instability limits current best.
3. **Is lr>1e-3 stable?** H39 (thorfinn) tests lr=1.5e-3/2e-3. Monotone trend still rising.
4. **Does looser clip work at lr=1e-3?** H40 (nezuko). H30's null at lr=5e-4 doesn't extrapolate.
5. **Can T_max extension improve?** H41 (fern). At T_max=15, ep13 LR = 4.5% of peak. T_max=20 keeps ep13 at 27.5%.
6. **Does depth (n_layers) help where width fails?** H42 (alphonse). n_layers=7 adds passes; n_layers=3 speeds convergence.
7. **Does weight decay matter at lr=1e-3?** H38 (frieren). Default wd=1e-4 effectively doubled in per-step penalty when LR went from 5e-4 to 1e-3.
8. **Physical representation budget?** H35 (edward). slice_num=96/128 on H20 base — retest at lr=1e-3 if promising.

## Key Closed Dead Ends This Round

- **H36 (β₂ sweep)**: β₂=0.95 hurts (+3.93 vs H20). β₂=0.999 confirmed as correct for short-budget cosine regime.
- **H34 (element-wise clip)**: clip_grad_value=1.0 too aggressive; clips more than norm clip.
- **H31 (δ_p push)**: δ_p=0.25 is the optimum. Both directions confirmed worse.
- **H26b (cond_dim reduction with clip)**: FiLM path weakened. cond_dim=11 stays.

## Potential Next Research Directions

- **Compound n_head=2 + T_max=20 + lr=1e-3**: If H37b and H41 both win, this triple may push to ~65.
- **n_head=1 (single head, head_dim=128)**: Trend n_head=8→4→2 is monotone improving. Test the limit.
- **n_head=2 + slice_num scaling**: With head_dim=64, slice projection Linear(64,64) is square. More slices (96/128) may distribute richer per-head capacity across more physics tokens.
- **Lion optimizer**: sign(gradient) — purely directional. May compound differently with clip than AdamW.
- **AdamW β₁ sweep**: β₁=0.9 default never tested. Different mechanism from β₂. Medium priority given β₂=0.999 confirmed.
- **Per-layer LR decay (LLRD)**: Lower LR multiplier on deeper Transolver blocks — separates adaptation rates.
- **Per-FiLM decoupled LR**: Higher LR for FiLM scale/shift to counteract global clip downsizing.
- **Stochastic depth in Transolver blocks**: Randomly drop blocks (DropPath). Regularizes depth; may help at lr=1e-3.
- **Non-contiguous FiLM conditioning**: (Re, AoA1, AoA2) indices rather than contiguous slice.

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. File is read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op. Loss always uses `huber_delta_vel`/`huber_delta_p`.
- `T_max=15` hardcoded in train.py scheduler line — students doing T_max or warmup sweeps must add CLI flag.
