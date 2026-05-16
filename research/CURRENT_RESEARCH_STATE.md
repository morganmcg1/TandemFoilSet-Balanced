# SENPAI Research State

- **Date**: 2026-05-16 03:10
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 4 active (H20 merged base → H27b new best → R4 experiments: H31–H38)
- **Most recent human research directive**: None received

## Current Best

**PR #3452 (H27b: LR=1e-3 + clip=1.0 on H20 base, frieren) — val_avg/mae_surf_p = 71.7713** (merged 2026-05-16)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| **lr=1e-3 + clip=1.0 (H27b Arm B)** | **71.7713** | **Current best (PR #3452)** |
| lr=7e-4 + clip=1.0 (H27b Arm A) | 75.9937 | (PR #3452) — tied with H20 |
| Grad clip=1.0 + H19 (H20 Arm A) | 75.4955 | (PR #3445) — prev best |
| Per-channel Huber δ_vel=1.0/δ_p=0.25 (H25 Arm B) | 75.7713 | (PR #3450) |
| FiLM + Huber δ=0.5 + T_max=15 (H19) | 83.8136 | (PR #3408) |

**Test metrics (3-split avg, excl. cruise NaN bug):** 70.6226 (H27b Arm B)

## Key Confirmed Insights

1. **T_max mismatch was the dominant first-order bottleneck**: T_max=15 fix gave 11.7-pt gain.
2. **Per-channel Huber wins over uniform (H25)**: δ_p=0.25, δ_vel=0.5 — now merged defaults.
3. **Grad clip=1.0 is independently effective (H20)**: Active every step, prevents high-Re samples dominating.
4. **Gradient-magnitude-shaping does NOT compound with clip=1.0 (H29/H23b/H30)**: Per-channel δ_vel amplification, surf_weight reduction, looser clip threshold — all absorbed by global norm clip. Principle: target directions, not magnitudes.
5. **LR=1e-3 is the new optimum (H27b)**: Monotone trend 5e-4→7e-4→1e-3: 75.50→75.99(tie)→**71.77(win)**. With clip=1.0 as safety rail, lr=1e-3 is stable (pre-clip norms 8.6→2.3, monotone training). Jump happens in 1e-3 range, not at 7e-4.
6. **Averaging (EMA H24, SWA H28) fails at 14-epoch budget**: No converged basin to average.
7. **WSD fails on this budget (H21)**: Decay phase never fires. CosineAnnealingLR T_max=15 is right.
8. **cond_dim=3 beats cond_dim=11 (H26)**: Geometry tail dims zero out for single-foil samples.
9. **surf_weight=10 optimal on H20+ base (H23b)**: clip=1.0 absorbs the H23 sw=5 win.
10. **Architecture capacity (n_hidden) NOT the bottleneck (H33)**: n_hidden=192/256 both regress +15-22%.
11. **Scoring NaN bug confirmed**: test_geom_camber_cruise sample 20. Workaround: 3-split excl. cruise.

## Active R4 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3587 | nezuko | **H34: Element-wise clip (clip_grad_value) + per-channel Huber** | WIP — highest priority |
| #3557 | thorfinn | H32: LR ceiling — lr=1.5e-3 vs lr=2e-3 + clip=1.0 | WIP — **redirected** (Arm A lr=1e-3 now covered by H27b) |
| #3553 | fern | H31: δ_p push (0.1, 0.05) + lr=1e-3 + clip=1.0 | WIP — **rebased** on new lr=1e-3 base |
| #3451 | alphonse | H26b: cond_dim=3/2 + clip=1.0 rebase | WIP (rebase in progress) |
| #3623 | edward | H35: slice_num sweep (96, 128) on H20 base | WIP |
| #3626 | askeladd | H36: AdamW beta2 sweep (0.95, 0.999) on H20 base | WIP |
| #3629 | tanjiro | H37: n_head sweep (8, 2) on H20 base | WIP |
| #3651 | frieren | H38: Weight decay sweep (wd=0, 5e-5) on H27b base | NEW |

**Note:** All new R4 PRs should target the **H27b base**: FiLM cond_dim=11, δ_vel=0.5/δ_p=0.25, T_max=15, clip_grad_norm=1.0, lr=1e-3. H35/H36/H37 were assigned at H20 base (lr=5e-4) — if any of these produce interesting results, a follow-up at lr=1e-3 may be warranted.

## Key Open Questions (Post-H27b)

1. **Does element-wise clip preserve per-channel benefit?** H34 (nezuko) — highest priority; tests H29 mechanism.
2. **LR ceiling: how high can we go?** H32 (thorfinn) redirected to lr=1.5e-3/2e-3. Training at 1e-3 was stable with clip active.
3. **δ_p push on new base?** H31 (fern) — δ_p=0.1/0.05 + lr=1e-3. Does further L1-ification of pressure loss help?
4. **Weight decay at new LR?** H38 (frieren) — wd=0 vs wd=5e-5. At lr=1e-3, effective wd is 2× what it was at lr=5e-4.
5. **cond_dim=3 with clip?** H26b (alphonse) — if cond_dim=3 compounds with clip+lr, additive win possible.
6. **Physical representation (slice_num)?** H35 (edward) — 96/128 on H20 base; directional architecture change.
7. **Optimizer momentum (beta2)?** H36 (askeladd) — 0.95 vs 0.999 on H20 base. 14-epoch budget may favor smaller beta2.
8. **Attention head granularity?** H37 (tanjiro) — n_head=8 vs n_head=2. Architectural direction, not magnitude.
9. **val_single_in_dist improving (83.78 now vs 85.72 at H20)**: H27b helped here (-1.95). Still the worst split.

## Guiding Principle (Post-H23b/H29/H30)

**Clip absorbs gradient-magnitude-shaping interventions.** Target:
- **Directional/structural**: attention head count (H37), physical representation budget (H35), conditioning (H26b)
- **Optimizer dynamics**: momentum decay (H36), LR schedule shape (H32), weight decay (H38)
- **Loss semantics**: δ_p push (H31), element-wise clip (H34)

## Potential Next Research Directions (After Active PRs Close)

- **Element-wise clip compound**: If H34 wins, compound δ_vel=1.0 + clip_grad_value as new default.
- **Compound lr=1e-3 + δ_p=0.1**: If H31 wins at lr=1e-3 base, both improvements compound naturally.
- **LR + cond_dim=3 compound**: If both H32 (higher LR) and H26b (cond_dim=3) win independently, compound.
- **FiLM head decoupled LR**: Separate LR group for FiLM parameters vs backbone.
- **Non-contiguous FiLM conditioning**: (Re, AoA1, AoA2) indices rather than contiguous slice.
- **Per-foil FiLM heads**: Single vs tandem mode routing.
- **n_layers sweep**: n_layers=3/7 — depth is a different axis than n_hidden width (which failed H33).
- **SWA with late start (start_epoch=11+) if budget allows**.

## Known Issues

- `data/scoring.py` NaN propagation: `test_geom_camber_cruise` sample 20 has non-finite GT. File is read-only. Report 3-split test_avg excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — loss always uses `huber_delta_vel`/`huber_delta_p`. Passing `--huber_delta` is a no-op.
