# SENPAI Research State
- 2026-04-29 updated (post-#817-close, alphonse reassigned #926); branch icml-appendix-charlie-pai2e-r5
- Most recent research direction from human researcher team: None received yet.

## Current Best Baseline

**PR #901** — `val_avg/mae_surf_p` = **71.2882** (epoch 13/~14, timeout-bound, still descending)
Branch: `charliepai2e5-askeladd/cosine-tmax-15` (merged)

Configuration: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, **Lion** optimizer, lr=3e-4, wd=1e-2, surf_weight=20, batch_size=4, **CosineAnnealingLR T_max=15** (aligned to ~14-epoch timeout budget), **L1 loss** (vol + surf_weight * surf), grad_clip=1.0, no EMA, no bf16.

Per-split surf p MAE: single=79.41, camber_rc=83.18, camber_cruise=54.18, re_rand=68.38.

Baseline improvement trajectory: 128.83 (MSE+AdamW) → 97.45 (L1+AdamW, PR #798, −24.4%) → 77.30 (L1+Lion+clip, PR #799, −20.7%) → **71.29 (Cosine T_max=15 aligned, PR #901, −7.78%)**

**All current experiments must rebase on the Lion+L1+clip+T_max=15 recipe.**

## Completed Experiments (this round)

### Merged (wins)
| PR | Hypothesis | Result | Δ |
|----|------------|--------|---|
| #798 | L1 loss (align objective with metric) | 97.45 | −24.4% vs 128.83 |
| #799 | Lion optimizer + L1 + clip=1.0 | 77.30 | −20.7% vs 97.45 |
| #901 | Cosine LR T_max budget alignment (T_max 50→15) | 71.29 | −7.78% vs 77.30 |

### Closed (dead ends)
| PR | Hypothesis | Result | Reason |
|----|------------|--------|--------|
| #803 | Surface feature noise | 142.25 | +46% regression |
| #804 | Cosine LR warmup (3-epoch) | 128.23 | +32% vs current baseline |
| #805 | Preprocess MLP depth +1 residual layer | 138.60 | +42% regression |
| #806 | FiLM domain conditioning | 106.49 | +9.3% regression |
| #802 | bf16 autocast + TF32 + batch_size=8 | 129.14 | +32.5% regression |
| #822 | SmoothL1/Huber loss (beta sweep) | 103.00 | +5.7% regression |
| #824 | Gradient clipping at 0.5/1.0/5.0 | 101.23 | +3.9% regression (wrong regime — natural grad norm 85–115, not 1–10) |
| #823 | asinh pressure target transform (scale 100/500/2000) | 99.26 | +28.4% vs current 71.29 — symmetric squash mismatched to asymmetric pressure tail |
| #817 | surf_weight sweep L1 AdamW (10/15/20/25/30) | 95.56 at sw=25 | Doesn't beat 71.29 baseline (run predated PR #901); directional finding: optimum shifts to sw=25 under L1. Superseded by #926 |
| #852 | Per-channel L1 loss weighting: amplify pressure channel | — | Closed — superseded/redirected |
| #879 | Wider hidden dim n_hidden 256 (AdamW, wrong recipe) | 121.34 | +57% regression vs current baseline; wrong optimizer |
| #857 | Drop-path stochastic depth regularization | — | Closed |

## Currently Running (status:wip)

| PR | Student | Hypothesis |
|----|---------|------------|
| #893 | charliepai2e5-frieren | Lion lr sweep: test lr=1e-4, 5e-4, 6e-4 vs baseline 3e-4 |
| #894 | charliepai2e5-nezuko | Lion+L1 surf_weight re-tune: sweep 5/10/30/40 vs baseline 20 |
| #908 | charliepai2e5-thorfinn | slice_num sweep 32/64/128: physics attention bottleneck tuning |
| #801 | charliepai2e5-edward | EMA model averaging (decay=0.995) for better generalization |
| #913 | charliepai2e5-tanjiro | n_layers depth sweep (4/5/6) on Lion+L1+clip |
| #922 | charliepai2e5-askeladd | Multi-step LR schedule for Lion (step decay ×0.3 at 50%/80% of budget vs cosine) |
| #926 | charliepai2e5-alphonse | surf_weight=25 validation on Lion+T_max=15 baseline (sw sweep 20/25/28) |
| #801 | charliepai2e5-edward | EMA model averaging (decay=0.995) for better generalization |

## Idle Students Needing Assignment

None — all students assigned.

## Current Research Focus

With the **Lion + L1 + clip + T_max=15** combination established as the new baseline (71.29), the remaining levers to investigate are:

1. **LR schedule shape for Lion**: Multi-step decay (step × 0.3 at 50%/80% budget) vs cosine — Lion paper shows step LR can outperform cosine (in-flight #922)
2. **Lion-specific LR tuning**: 3e-4 may not be optimal — sweep 1e-4/5e-4/6e-4 (in-flight #893)
3. **Loss surface refinements on Lion+L1 base**: surf_weight re-tuning for Lion — PR #817 showed sw=25 wins under AdamW+L1; #926 validates this with Lion+T_max=15 (in-flight #894, #926)
4. **Architecture capacity**: slice_num sweep, n_layers depth (in-flight #908, #913)
5. **Regularization**: EMA model averaging (in-flight #801)
6. **Data pipeline**: bf16 at bs=4 for ~1.3× more epochs (not yet assigned — bs=8 previously hurt)
7. **Ensemble / test-time**: train multiple seeds, average predictions

## Potential Next Research Directions

### High priority (not yet assigned)
1. **bf16 autocast at bs=4** (NOT bs=8): Pure throughput gain → ~1.3× more epochs within timeout. Previously tested at bs=8 which hurt convergence; bs=4 with bf16 is untested and should be safe. Expected: more effective epochs = lower val_avg/mae_surf_p.
2. **Focal L1 / hard-example mining**: Weight loss by node-wise rolling error to chase the long pressure tail — camber_rc split (83.18) is the worst and likely drives the average.
3. **Geometry-physics bottleneck token**: Compact geometry token (chord/camber/gap/stagger/Re → learnable embedding) injected via cross-attention key/value into Transolver blocks.

### Medium priority
4. **Temperature scaling on attention slices**: Learnable temperature τ per head for Transolver's slice attention (already has a temperature parameter — probe initialization and learning rate).
5. **Re-conditional normalization (volume only)**: Pass Re number through FiLM on volume nodes only — FiLM on all surface nodes hurt (#806), but Re conditioning only on flow domain may help.
6. **Position encoding on foil surface arc-length**: Explicit ordering of surface nodes by arc-length position as positional encoding.
7. **Test-time ensemble (k=5 seeds)**: Trivial parallelism win if variance across seeds is high — average predictions at test time.
8. **Weight decay sweep for Lion**: wd=1e-2 was carried from the initial Lion config; Lion may respond differently than AdamW to weight decay values (1e-3, 5e-3, 2e-2).

### Bigger swings (architecture/representation)
9. **Deeper preprocess MLP**: 2-layer preprocess with skip connection (previous attempt failed at 1 residual layer with wrong architecture — a clean 3-layer MLP may work).
10. **Separate surface/volume encoders**: Dedicated MLP towers for surface nodes vs. volume nodes before the Transolver blocks — respects the physical distinction in the data.

## Known Issues

- **`data/scoring.py:accumulate_batch` NaN bug**: `test_geom_camber_cruise/mae_surf_p` is NaN across all runs because the scoring code doesn't guard against non-finite model predictions (only non-finite ground truth). Affects test metrics, not val metrics. Three students have independently flagged this. Needs organizer fix in `data/scoring.py`.
