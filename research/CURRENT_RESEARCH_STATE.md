# SENPAI Research State

- **Date**: 2026-05-16 07:40
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active (H38 new best → continuing R5 + new H43/H44/H45)
- **Most recent human research directive**: None received

## Current Best

**PR #3651 (H38: wd=5e-5 + lr=1e-3 + clip=1.0, frieren) — val_avg/mae_surf_p = 68.1932** (merged 2026-05-16)

| Reference | val_avg/mae_surf_p | Notes |
|-----------|--------------------|-------|
| **H38 Arm B (wd=5e-5 + lr=1e-3 + clip=1.0)** | **68.1932** | **Current best (PR #3651)** |
| H38 Arm A (wd=0 + lr=1e-3 + clip=1.0) | 70.6064 | (PR #3651) — also beats H32 |
| H32 Arm A (lr=1e-3 + clip=1.0, default wd) | 69.4381 | (PR #3557) — overridden by H38 |
| H27b Arm B (lr=1e-3 + clip=1.0) | 71.7713 | (PR #3452) — same config as H32, seed variance |
| H37 Arm B (n_head=2 on H20 base) | 72.8859 | (PR #3629) — sent back for lr=1e-3 retest |
| Grad clip=1.0 + H19 (H20) | 75.4955 | (PR #3445) |

**Test metrics (3-split avg, excl. cruise NaN bug):** 65.4393 (H38 Arm B) — **vs 69.1774 (H32)**

**Key finding:** H38 validates that raising lr 5e-4→1e-3 doubled the effective AdamW L2 penalty (wd=1e-4 → net 2×). Halving wd to 5e-5 restores the original per-step regularization strength and improves all 4 val splits. Mechanism orthogonal to clip (weight norm vs gradient norm).

## Key Confirmed Insights

1. **T_max mismatch was the dominant first-order bottleneck** (R1): T_max=15 fix gave 11.7-pt gain.
2. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 now merged defaults.
3. **Grad clip=1.0 is effective (H20)**: Active every step, prevents high-Re samples dominating.
4. **Clip absorbs gradient-magnitude-shaping interventions (H23b/H29/H30/H31/H26b)**: Confirmed 5× — scale-of-gradient changes don't compound with clip_grad_norm=1.0.
5. **lr=1e-3 is dramatically better (H27b/H32)**: Monotone 5e-4→8e-4→1e-3 gives 75.50→73.11→69.44. LR ceiling not yet visible.
6. **wd=5e-5 is better at lr=1e-3 (H38)**: Default wd=1e-4 over-regularizes after LR doubling. wd=5e-5 restores balance. Orthogonal to clip.
7. **Architecture width fails (H33)**: n_hidden=192/256 both regress +15-22%. Capacity not the bottleneck.
8. **n_head=2 is promising (H37)**: head_dim=64 beats H20 by -2.61. Sent back for lr=1e-3 stacking test (H37b).
9. **β₂=0.999 confirmed (H36)**: β₂=0.95 hurts in cosine-decay tail. β₂=0.999 stays.
10. **Element-wise clip too aggressive at clip_value=1.0 (H34)**: Dead end.
11. **δ_p=0.25 is the optimum (H31)**: Knob exhausted.
12. **cond_dim reduction fails with clip (H26b)**: cond_dim=11 optimal on clipped stack.
13. **Averaging (EMA/SWA) fails at 14-epoch budget (H20-era)**: No plateau to average over.
14. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Use 3-split excl. cruise.

## Active R5 WIP Experiments

| PR | Student | Hypothesis | Status |
|----|---------|------------|--------|
| #3629 | tanjiro | **H37b: n_head=2 + lr=1e-3 + wd=5e-5 + clip=1.0 stacking test** | WIP (HIGHEST PRIORITY — predicted ≈66.8 if additive) |
| #3729 | askeladd | H43: Linear warmup (1, 2 ep) + lr=1e-3 + wd=5e-5 + clip=1.0 | WIP |
| #3737 | frieren | **H44: AdamW β₁ sweep (0.8, 0.95) on H38 base** | NEW |
| #3683 | thorfinn | H39: LR ceiling (lr=1.5e-3, 2e-3) + clip=1.0 | WIP |
| #3685 | nezuko | H40: Clip sweep (2.0, 3.0) at lr=1e-3 | WIP |
| #3688 | fern | H41: T_max sweep (20, 18) on lr=1e-3 base | WIP |
| #3689 | alphonse | H42: n_layers sweep (7, 3) on lr=1e-3 base | WIP |
| #3767 | edward | **H45: DropPath (0.1, 0.2) on H38 base** | NEW (stochastic depth regularizer) |

**Note on wd for in-flight experiments:** H37b, H43, and all H39-H42 were assigned before H38 merged. If any of these win, compare against 68.1932. If they only beat H32 (69.44) but not H38, request a wd=5e-5 retest before deciding merge/close.

## Key Open Questions

1. **Can n_head=2 stack with lr=1e-3 + wd=5e-5?** H37b (tanjiro). Predicted ≈66.8 from H37 data. Highest priority.
2. **Does warmup help at lr=1e-3?** H43 (askeladd). Tests early-gradient instability hypothesis.
3. **Does β₁ matter?** H44 (frieren). Last clean AdamW knob before optimizer-switch experiments.
4. **Is lr>1e-3 stable?** H39 (thorfinn). Monotone trend still rising at 1e-3.
5. **Does looser clip work at lr=1e-3?** H40 (nezuko). H30's null at lr=5e-4 doesn't extrapolate.
6. **Can T_max extension improve?** H41 (fern). T_max=15 → 20: ep13 LR from 4.5% → 27.5% of peak.
7. **Does depth help where width fails?** H42 (alphonse). n_layers=7 vs 3.
8. **Physical representation budget?** H35b (edward). slice_num=96/128 on new H38 base.

## Key Closed Dead Ends This Round

- **H36 (β₂ sweep)**: β₂=0.95 hurts (+3.93 vs H20). β₂=0.999 stays.
- **H35 (slice_num sweep)**: Walltime-confounded — slice96/128 lose 2-3 epochs of LR anneal at our 30-min cap. Per-epoch trajectory shows representation is fine, but budget is binding. Shelved.
- **H34 (element-wise clip)**: clip_grad_value=1.0 too aggressive.
- **H31 (δ_p push)**: δ_p=0.25 is optimal, both directions confirmed worse.
- **H26b (cond_dim reduction)**: FiLM path weakened by clip. cond_dim=11 stays.

## Potential Next Research Directions

- **Compound n_head=2 + T_max=20 + lr=1e-3 + wd=5e-5**: If H37b and H41 both win, triple stack may push to ~65.
- **n_head=1 (single head, head_dim=128)**: Trend n_head=8→4→2 is monotone improving. Test the limit.
- **n_head=2 + slice_num scaling**: With head_dim=64, slice projection Linear(64,64) is square. More slices (96/128) could distribute richer per-head capacity.
- **Lion optimizer**: sign(gradient) — purely directional. May compound differently with clip than AdamW.
- **wd fine-sweep near 5e-5**: {2.5e-5, 5e-5, 7.5e-5} to pin the inverted-U. 3 quick runs.
- **Per-layer LR decay (LLRD)**: Lower LR multiplier on deeper Transolver blocks.
- **Stochastic depth (DropPath)**: Randomly drop Transolver blocks during training.
- **Non-contiguous FiLM conditioning**: (Re, AoA1, AoA2) indices rather than contiguous slice.

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. File is read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op. Loss always uses `huber_delta_vel`/`huber_delta_p`.
- `T_max=15` hardcoded in train.py scheduler — students doing T_max or warmup sweeps must add CLI flag.
- In-flight experiments H39-H42 use old wd=1e-4. If they beat H32 but not H38, request wd=5e-5 retest.
