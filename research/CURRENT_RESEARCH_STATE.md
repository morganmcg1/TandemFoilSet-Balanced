# SENPAI Research State

- **Last updated:** 2026-05-12 ~22:40 (Smooth L1 PR #1414 merged −7.3%; new baseline 95.336; alphonse assigned full-stack validation #1663)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 95.336`** — PR #1414 (Smooth L1 β=0.1 + channel_weights=[1,1,3] + NaN-skip), epoch 13, 0.66M param Transolver.  

> ⚠️ The validated metric (95.336) was from Smooth L1 + CW on lr=5e-4 (pre-#1424 state). The merged advisor branch NOW has all 4 axes combined: (i) channel_weights=[1,1,3], (ii) lr=7e-4 + 2-epoch warmup, (iii) grad_clip=1.0, (iv) Smooth L1 β=0.1 + NaN-skip. The precise post-merge metric for the full stack is PENDING (alphonse's #1663 confirmation run).

**NaN-skip fix now merged**: `test_avg/mae_surf_p` is now a clean 4-split (best so far: 85.648).  
See `BASELINE.md` for full per-split details.

## In-flight PRs

| PR | Student | Slug | Axis | vs. Baseline |
|----|---------|------|------|---|
| #1421 | edward | `surf-only-channel-weight` | Decouple vol/surf channel weights | STALE WIP (notified of new baseline 95.336, needs rebase onto current HEAD) |
| #1432 | tanjiro | `wall-distance-rebased` | Wall-dist + full stack stacked | NEEDS REBASE — merge conflict (notified of new baseline 95.336) |
| #1435 | thorfinn | `unified-pos-ref16-nopad` | Unified pos encoding ref=16 | NEEDS REBASE (notified of new baseline 95.336) |
| #1597 | frieren | `depth-6-layers` | Depth n_layers 5→6 | WIP |
| #1657 | fern | `rff-pos-encoding` | Fourier RFF encoding (space_dim 2→64) | WIP — just assigned |
| #1658 | askeladd | `swa-ep10-14` | SWA averaging epochs 10–14 | WIP — just assigned |
| #1659 | nezuko | `slice-96-stable` | slice_num 64→96 intermediate | WIP — just assigned |
| #1663 | alphonse | `smooth-l1-full-stack` | Full-stack validation (no code change, just run) | WIP — just assigned |

### Closed as dead ends (this round)
- #1426 frieren hidden-192-head-6: +12.8% worse
- #1429 nezuko slice-128-mlp-4: +6.97% worse, overflow
- #1517 askeladd ema-0.99-adaptive: neutral (+0.40%)
- #1598 nezuko mlp-ratio-4-alone: +7.0% worse (under-trained)

## Current research focus

1. **🔥 Full-stack validation (alphonse #1663)** — most urgent. Confirms actual metric for merged config (Smooth L1 + CW + warmup + clip). Expected 85–93. If confirmed, all subsequent work is measured against this.
2. **Wall-distance rebase (tanjiro #1432)** — physics-informed input feature. Was −0.96% on old code. Needs rebase onto current HEAD (all 4 axes).
3. **Depth n_layers=6 (frieren #1597)** — WIP. Tests whether depth helps on new Smooth L1 baseline.
4. **Unified pos encoding ref=16 (thorfinn #1435)** — signal on cruise OOD at ref=8. Needs rebase.
5. **Fourier RFF encoding (fern #1657)** — novel input representation, high potential for OOD.
6. **SWA epochs 10–14 (askeladd #1658)** — weight averaging after convergence.
7. **slice_num=96 (nezuko #1659)** — intermediate token count.
8. **Decoupled channel weights (edward #1421)** — vol vs surf separate weighting, needs rebase.

## Key research insights so far

- **Loss shape wins big:** Smooth L1 (β=0.1) → −26% alone, −22% with CW. Mechanism: L1 regime minimizes median absolute error = MAE eval criterion.
- **Channel weighting:** [1,1,3] → −9.5% standalone. May mildly antagonize Smooth L1 (both operate on large pressure residuals). Full-stack validation pending.
- **LR warmup + grad clip:** +40% LR (7e-4), 2-epoch warmup, clip=1.0 → −16.1% on MSE. Effect on Smooth L1 pending.
- **NaN-skip fix now canonical:** `y_finite + nan_to_num` in evaluate_split merged. Clean 4-split test_avg from now on.
- **Architecture axes:** Width-scaling killed by 30-min budget. Depth (n_layers=6) in progress. slice_num intermediate probe in progress.
- **Positional encoding:** OOD signal at ref=8 (−1.8% cruise, +9.1% re_rand). Needs proper bandwidth fix at ref=16.
- **EMA/SWA:** EMA wrong for rapid-descent regime. SWA (late-epoch averaging) in progress.

## Next research directions (when new slots open)

1. **β sweep for Smooth L1** — β ∈ {0.05, 0.02} (narrower quadratic) and {0.3, 1.0} (wider). Do after full-stack confirmed.
2. **Pure L1 loss** — `F.l1_loss` directly. Cleaner, same regime hypothesis.
3. **T_max alignment** — set T_max=14 to match actual 30-min budget.
4. **Per-sample loss normalization** — normalize each sample's loss by per-sample std.
5. **Multi-resolution slice pooling** — vary slice_num per block.

## Epoch budget arithmetic

- Epoch time: ~131s (baseline 662K params), ~138–142s for larger models
- 30-min cap: ~14 epochs baseline-equivalent
- T_max=20 → LR at epoch 14 is ~45% through schedule; model still descending at cutoff
