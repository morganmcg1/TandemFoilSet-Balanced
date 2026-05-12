# SENPAI Research State

- **Last updated:** 2026-05-12 ~23:10 (closed #1432 tanjiro wall-dist +12.8% worse, closed #1597 frieren depth-6 +36% worse; assigned tanjiro #1682 pure-L1, frieren #1684 T_max-aligned-14)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 95.336`** — PR #1414 (Smooth L1 β=0.1 + channel_weights=[1,1,3] + NaN-skip), epoch 13, 0.66M param Transolver.

> ⚠️ The validated metric (95.336) was from Smooth L1 + CW on lr=5e-4 (pre-#1424 state). The merged advisor branch NOW has all 4 axes combined: (i) channel_weights=[1,1,3], (ii) lr=7e-4 + 2-epoch warmup, (iii) grad_clip=1.0, (iv) Smooth L1 β=0.1 + NaN-skip. The precise post-merge metric for the full stack is PENDING (alphonse's #1663 confirmation run).

**NaN-skip fix now merged**: `test_avg/mae_surf_p` is a clean 4-split (best so far: 85.648).  
See `BASELINE.md` for full per-split details.

## In-flight PRs

| PR | Student | Slug | Axis | vs. Baseline |
|----|---------|------|------|---|
| #1421 | edward | `surf-weight-25` | Surface loss weight 10→25 | STALE WIP (notified of new baseline 95.336) |
| #1435 | thorfinn | `unified-pos-ref8` | Unified pos encoding ref=8 | WIP (notified of new baseline 95.336) |
| #1657 | fern | `rff-pos-encoding` | Fourier RFF encoding (space_dim 2→64) | WIP |
| #1658 | askeladd | `swa-ep10-14` | SWA averaging epochs 10–14 | WIP |
| #1659 | nezuko | `slice-96-stable` | slice_num 64→96 intermediate | WIP |
| #1663 | alphonse | `smooth-l1-full-stack` | Full-stack validation (no code change) | WIP — most urgent |
| #1682 | tanjiro | `pure-l1-loss` | Pure L1 loss (remove Smooth L1 quadratic regime) | WIP — just assigned |
| #1684 | frieren | `tmax-aligned-14` | T_max=14 → cosine fully anneals in 30-min budget | WIP — just assigned |

### Closed as dead ends (this round)
- #1426 frieren hidden-192-head-6: +12.8% worse
- #1429 nezuko slice-128-mlp-4: +6.97% worse, overflow
- #1517 askeladd ema-0.99-adaptive: neutral (+0.40%)
- #1598 nezuko mlp-ratio-4-alone: +7.0% worse (under-trained)
- #1432 tanjiro wall-distance-rebased: +4.59% worse vs #1424 / +12.8% vs current (negative stacking with new loss/LR regime)
- #1597 frieren depth-6-layers: +5.91% worse vs #1418 / +36% vs current (capacity not bottleneck on 1500 samples)

## Current research focus

1. **🔥 Full-stack validation (alphonse #1663)** — most urgent. Confirms actual metric for merged config (Smooth L1 + CW + warmup + clip). Expected 85–93. If confirmed, all subsequent work is measured against this.
2. **Pure L1 loss (tanjiro #1682)** — test whether removing the Smooth L1 quadratic regime at β=0.1 further improves the MAE criterion alignment. One-line change, directionally motivated.
3. **T_max alignment (frieren #1684)** — cosine schedule was being truncated at ~50% peak LR. Aligning T_max=14 to the feasible epoch budget is a free optimization with −1% to −3% expected gain.
4. **Fourier RFF encoding (fern #1657)** — novel input representation, high OOD potential.
5. **SWA epochs 10–14 (askeladd #1658)** — weight averaging after convergence.
6. **slice_num=96 (nezuko #1659)** — intermediate token count, stable regime.
7. **Unified pos encoding ref=8 (thorfinn #1435)** — OOD signal at ref=8.
8. **Surf weight 10→25 (edward #1421)** — stale; may need reassignment.

## Key research insights so far

- **Loss shape wins big:** Smooth L1 (β=0.1) → −26% alone, −22% with CW. Mechanism: L1 regime minimizes median absolute error = MAE eval criterion.
- **Channel weighting:** [1,1,3] → −9.5% standalone. May mildly antagonize Smooth L1 (both operate on large pressure residuals). Full-stack validation pending.
- **LR warmup + grad clip:** +40% LR (7e-4), 2-epoch warmup, clip=1.0 → −16.1% on MSE. Effect on Smooth L1 pending.
- **NaN-skip fix now canonical:** `y_finite + nan_to_num` in evaluate_split merged. Clean 4-split test_avg from now on.
- **Architecture axes exhausted at budget:** Width-scaling killed by 30-min budget. Depth (n_layers=6) regressed +5.9% — capacity not bottleneck on 1500 samples. Depth axis closed.
- **T_max must match epoch budget:** Frieren's #1597 confirmed T_max=20 with 14 feasible epochs wastes the last ~50% of cosine schedule. Explicitly testing T_max=14 alignment in #1684.
- **Wall-distance negative stacking:** log-distance feature helped −0.96% on flat-loss baseline but hurt +4.6% on CW+warmup baseline — per-batch standardization noise amplified in steeper loss landscape.
- **EMA/SWA:** EMA wrong for rapid-descent regime. SWA (late-epoch averaging) in progress.

## Next research directions (when new slots open)

1. **β sweep for Smooth L1** — β ∈ {0.05, 0.02} (narrower quadratic) and {0.3, 1.0} (wider). After #1682 pure-L1 completes.
2. **Per-sample loss normalization** — normalize each sample's loss by per-sample std.
3. **Data augmentation** — left-right foil mirroring (physics: tandem foil has no L/R symmetry directly, but the mesh does), mild geometry perturbation.
4. **Gradient accumulation** — higher effective batch size within the 30-min wall-clock.
5. **Loss: log-cosh or relative pressure** — alternative to L1/Smooth-L1 that handles the heavy-tailed pressure distribution.

## Epoch budget arithmetic

- Epoch time: ~131s (baseline 662K params), ~138–142s for larger models
- 30-min cap: ~14 epochs baseline-equivalent, ~12 epochs for larger models
- T_max=20 → LR at epoch 14 is ~50% through schedule (still high!) — T_max alignment is immediately testable fix
