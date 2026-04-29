# SENPAI Research State

- 2026-04-29 10:30
- Most recent research direction from human researcher team: None (no open GitHub Issues)
- Current research focus: Beating the PSN+compound baseline (val_avg/mae_surf_p=61.5855, test_avg/mae_surf_p=54.3573). Round 5 experiments launched to probe orthogonal axes after the compound PSN stack was established.

## Current Research Focus and Themes

### The Problem
TandemFoilSet CFD surrogate: predict Ux, Uy, pressure at every mesh node given geometry + flow conditions. Primary metric: `val_avg/mae_surf_p` (surface pressure MAE, lower is better — equal-weight mean over 4 val splits). Literature target: `test_avg/mae_surf_p` ~ 40.93. Current best: 54.3573 test / 61.5855 val.

### Compound Baseline (PR #1050)
The current best configuration:
- Architecture: n_hidden=256, n_head=8, n_layers=3, slice_num=16, mlp_ratio=2
- Training: Huber loss (delta=1.0), epochs=30, grad_clip=1.0, ema_decay=0.999, per_sample_norm=True
- lr=5e-4, batch_size=4, surf_weight=10.0, bf16 mixed precision
- Timeout constraint at 30 min: actual run terminated at epoch 22/30 (val still falling ~2.8%/epoch at termination, LR=8.27e-5)
- Peak VRAM: 30.44 GB | 1,606,219 parameters

### Per-split Val Breakdown (PR #1050 best)
| Split | val/mae_surf_p |
|-------|---------------|
| single_in_dist | 68.3069 |
| geom_camber_rc | 72.6498 |
| geom_camber_cruise | 44.8940 |
| re_rand | 60.4914 |
| **avg** | **61.5855** |

`geom_camber_cruise` is consistently the easiest split (both val and test ~38-45). `geom_camber_rc` is hardest (72-79). Focus on improving `geom_camber_rc` and `single_in_dist` would yield the biggest aggregate gain.

### What We Know Works
- `--per_sample_norm`: normalizes each sample's Huber loss by per-sample std, equalizing the 15× Reynolds-number gradient-magnitude spread. Significant improvement independently validated (PR #795: -4.5% over epoch=12 baseline).
- EMA weight averaging (decay=0.999): marginally better than lower EMA decays (PR #882: -0.86%).
- Huber loss (delta=1.0): robust against extreme high-Re pressure values (PR #788: -8.85% over MSE).
- `n_hidden=256, n_head=8`: best architecture width/attention configuration found.
- `n_layers=3, slice_num=16`: deeper/more slices helps (PR #1005: -8.31%).
- `grad_clip=1.0`: stabilizes training with large pressure values.
- BF16 mixed precision: included in compound stack; VRAM efficient.
- surf_weight=30: better than lower values (PR #827: -5.26%); PR #1050 uses surf_weight=10.0.
- Longer training: val still falling at timeout — epochs=30 better than epochs=24 (-7.8%).

### Round 5 Experiments In-Flight (2026-04-29)
| PR | Student | Experiment | Hypothesis |
|----|---------|------------|------------|
| #1118 | edward | epochs=50 | More training time; val still falling at epoch 22 |
| #1119 | thorfinn | cosine eta_min=5e-5 | LR floor prevents over-decay |
| #1120 | nezuko | n_layers=2 | Shallower = faster = more epochs in budget |
| #1121 | fern | huber_delta=0.1 | Tighter Huber emphasizes surface pressure precision |
| #1122 | alphonse | lr=1e-3 | Higher LR with cosine — faster convergence |
| #1123 | tanjiro | n_hidden=320 | Wider model: more capacity |
| #1124 | askeladd | weight_decay=0 | No L2 regularization |
| #1125 | frieren | surf_weight=5 | Reduce surface emphasis (baseline=10) to test balance |

## Potential Next Research Directions

### High Priority
1. **Longer training with warmup restarts** — val curve still falling at ep22/30. SGDR (warm restarts) or cyclical schedules might help escape local optima. The 30-min hard timeout limits epoch count; every epoch matters.
2. **LR schedule optimization** — cosine annealing currently decays to near-zero. A nonzero `eta_min` floor or restart schedule may prevent premature LR death.
3. **surf_weight tuning** — baseline uses 10; prior experiments showed surf_weight=30 improved things. r5 is testing surf_weight=5. The optimal balance between surface and volume loss weighting is not yet established.

### Medium Priority
4. **Batch size + gradient accumulation** — effective batch scaling via gradient accumulation (e.g., accum_steps=4) could improve optimization stability within the VRAM budget.
5. **Surface-pressure-specific heads** — separate decoder heads for surface vs. volume predictions, optimized independently.
6. **Multi-scale attention** — different slice_num values at different layers (coarse→fine hierarchy).
7. **Feature engineering** — Re-embedding as Fourier features, or additional physics-motivated node features.
8. **Warm restart schedule (SGDR)** — cosine annealing with periodic restarts to escape shallow optima.

### Lower Priority / Bold Ideas
9. **Physics-informed regularization** — continuity equation or Bernoulli constraint as auxiliary loss.
10. **Ensemble methods** — average predictions from multiple seeds (free gain if inference budget allows).
11. **Curriculum learning** — easy (low-Re, single-foil) examples first, harder (high-Re, tandem) later.
12. **Separate Re-conditioned normalization** — instead of per_sample_norm, learn separate batch norm stats per Re range.
13. **Larger model exploration** — n_hidden=384 or 512 with gradient accumulation to fit VRAM; n_layers=4.
