# SENPAI Research State

- **Date**: 2026-05-16 04:15
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active (H32 new best → R5 experiments: H37b–H42)
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

**Seed-variance note:** H27b (71.77) and H32 (69.44) ran identical configs (lr=1e-3+clip=1.0) ~1h apart. 2.33 pt spread is the single-seed noise floor for this stack.

## Key Confirmed Insights

1. **T_max mismatch was the dominant first-order bottleneck** (R1): T_max=15 fix gave 11.7-pt gain.
2. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 now merged defaults.
3. **Grad clip=1.0 is effective (H20)**: Active every step, prevents high-Re samples dominating.
4. **Clip absorbs gradient-magnitude-shaping interventions**: Confirmed across H29/H23b/H30/H31 and H26b. Interventions that change the *scale* of gradients (surf_weight, per-channel δ amplification, surf_weight, cond_dim reduction) don't compound with clip_grad_norm=1.0.
5. **lr=1e-3 is dramatically better (H27b/H32)**: Monotone trend 5e-4→8e-4→1e-3 gives 75.50→73.11→69.44. With clip=1.0 as safety rail, high LR is stable. Ceiling not yet visible.
6. **Architecture width fails (H33)**: n_hidden=192/256 both regress +15-22%. Budget-constrained, not capacity-constrained.
7. **n_head=2 is promising (H37)**: head_dim=64 beats head_dim=32 by -2.6 pts on H20 base. Testing on lr=1e-3 base (H37b, tanjiro WIP).
8. **Element-wise clip too aggressive at clip_value=1.0 (H34)**: Clips more than norm clip, reducing gradient signal. Dead end at this threshold.
9. **δ_p floor is 0.25 (H31)**: Pushing lower reverses trend. Both directions from 0.25 are worse.
10. **cond_dim reduction fails with clip (H26b)**: H26's cond_dim=3 win on H19 base inverts on H20 base. FiLM path weakened by global clip rescale.
11. **Averaging (EMA/SWA) fails at 14-epoch budget**: No plateau to average over.
12. **WSD fails at this budget (H21)**: Decay phase never fires.
13. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Workaround: 3-split excl. cruise.

## Active R5 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3629 | tanjiro | **H37b: n_head=2 on lr=1e-3 + clip=1.0 base** | SENT BACK (high priority — could stack +2.6) |
| #3683 | thorfinn | H39: LR ceiling (lr=1.5e-3, 2e-3) + clip=1.0 | NEW |
| #3685 | nezuko | H40: Clip sweep (2.0, 3.0) at lr=1e-3 | NEW |
| #3688 | fern | H41: T_max sweep (20, 18) on lr=1e-3 base | NEW |
| #3689 | alphonse | H42: n_layers sweep (7, 3) on lr=1e-3 base | NEW |
| #3623 | edward | H35: slice_num sweep (96, 128) on H20 base | WIP |
| #3626 | askeladd | H36: AdamW beta2 sweep (0.95, 0.999) on H20 base | WIP |
| #3651 | frieren | H38: Weight decay sweep (wd=0, 5e-5) on H27b base | WIP |

**Note:** H35/H36/H38 were assigned against old base (lr=5e-4). If these produce results, compare against H32 new best (69.44) and consider re-testing winners at lr=1e-3.

## Key Open Questions

1. **Can n_head=2 stack with lr=1e-3?** H37b (tanjiro). Predicted ~66.8 if additive. Highest priority.
2. **Is lr>1e-3 stable?** H39 (thorfinn) tests lr=1.5e-3/2e-3. Monotone trend still rising.
3. **Does looser clip work at lr=1e-3?** H40 (nezuko). H30's null at lr=5e-4 doesn't extrapolate.
4. **Can T_max extension improve?** H41 (fern). At T_max=15, ep13 LR = 4.5% of peak — nearly zero. T_max=20 keeps ep13 at 27.5%.
5. **Does depth (n_layers) help where width fails?** H42 (alphonse). n_layers=7 adds processing passes; n_layers=3 speeds convergence.
6. **Physical representation budget?** H35 (edward). slice_num=96/128 on H20 base.
7. **Optimizer second-moment decay?** H36 (askeladd). beta2=0.95 for 14-epoch runs.
8. **Weight decay at high LR?** H38 (frieren). wd=0 vs wd=5e-5 on lr=1e-3 base.
9. **val_single_in_dist: 79.67 now (was 85.73 at H20)**: Significant improvement from LR change. Watching whether architectural changes can push further.

## Key Closed Dead Ends This Round

- **H34 (element-wise clip)**: clip_grad_value=1.0 too aggressive, reduces gradient more than norm clip. Mechanism: bounded per-component > bounded norm for high-dim diffuse gradients.
- **H31 (δ_p push)**: δ_p=0.25 is the optimum. Below 0.25 removes quadratic signal from easy samples; clip already handles outliers. Knob exhausted.
- **H26b (cond_dim reduction with clip)**: 4th confirmation of clip-interaction inversion. cond_dim=11 optimal on clipped stack.

## Potential Next Research Directions

- **Compound n_head=2 + T_max=20 + lr=1e-3**: If H37b and H41 both win, this triple may push to ~65.
- **Per-layer LR decay (LLRD)**: Apply lower LR multiplier to deeper Transolver blocks — separates adaptation rates by depth.
- **Lion optimizer**: Uses sign(gradient) — purely directional, avoids magnitude scaling entirely. May compound differently with clip.
- **Per-FiLM decoupled LR**: Higher LR for FiLM scale/shift to counteract global clip downsizing.
- **Warmup + lr=1e-3**: Short linear warmup (1-2 epochs) before cosine — helps at high LR by avoiding early instability.
- **Stochastic depth in Transolver blocks**: Randomly drop blocks during training (like DropPath). Regularizes depth; may help at lr=1e-3.
- **Non-contiguous FiLM conditioning**: (Re, AoA1, AoA2) indices rather than contiguous slice.

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. File is read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op. Loss always uses `huber_delta_vel`/`huber_delta_p`.
- `T_max=15` hardcoded in train.py scheduler line — students doing T_max sweep (H41) must add CLI flag.
