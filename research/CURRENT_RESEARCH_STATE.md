# SENPAI Research State

- 2026-04-29 11:15
- Most recent research direction from human researcher team: None (no open GitHub Issues)
- Current research focus: Compounding wins on top of the new huber_delta=0.1 baseline (PR #1121, val=58.4790, test=51.3554). Round-r5 first review wave produced 1 winner and 3 closes; second wave probes orthogonal axes (loss tighter, MLP capacity, surf weighting up, attention slices).

## Current Research Focus and Themes

### The Problem
TandemFoilSet CFD surrogate: predict Ux, Uy, pressure at every mesh node given geometry + flow conditions. Primary metric: `val_avg/mae_surf_p` (surface pressure MAE, lower is better — equal-weight mean over 4 val splits). Compete target: `test_avg/mae_surf_p` ~ 40.93 (Transolver paper). Current best: 51.3554 test / 58.4790 val. Gap to close: ~20%.

### NEW Compound Baseline (PR #1121 — merged 2026-04-29)
- Architecture: n_hidden=256, n_head=8, n_layers=3, slice_num=16, mlp_ratio=2
- Training: Huber loss (**delta=0.1** ← new), epochs=30, grad_clip=1.0, ema_decay=0.999, per_sample_norm=True
- lr=5e-4, batch_size=4, surf_weight=10.0, bf16 mixed precision
- Timeout-bound at epoch 22/30 (val still falling ~3%/epoch, LR=8.27e-5 at termination)
- 1,606,219 params | Peak VRAM 30.45 GB

### Per-split Val Breakdown (PR #1121 best — equal-weight mean)
| Split | val/mae_surf_p | test/mae_surf_p |
|-------|---------------|-----------------|
| single_in_dist | 66.29 | 59.57 |
| geom_camber_rc | 71.21 | 64.96 |
| geom_camber_cruise | **39.32** | **32.35** |
| re_rand | 57.10 | 48.55 |
| **avg** | **58.4790** | **51.3554** |

`geom_camber_cruise` is much easier than the others. `geom_camber_rc` lags hardest (test 65) — gap to close. Surface velocity channels (Ux, Uy) all improved -12 to -15% with huber_delta=0.1.

### What We Know Works
- **`--huber_delta 0.1`** (NEW from PR #1121, -5.04% val, -5.52% test). Tighter Huber clamp on PSN-normalized residuals. Cruise (-14.4% test) and re_rand (-9.4% test) gained most.
- `--per_sample_norm`: equalizes 15× Re-driven gradient-magnitude spread across samples.
- EMA weight averaging (decay=0.999): marginal but consistent.
- Huber loss + surf_weight=10 baseline.
- `n_hidden=256, n_head=8, n_layers=3, slice_num=16, mlp_ratio=2`: best architecture found.
- `grad_clip=1.0`: stabilizes against extreme pressure values.
- BF16 mixed precision.
- Cosine annealing T_max=epochs (decays to 0 over the schedule).
- epochs=30 (val still falling at timeout — more is likely better).

### What Just Failed (closed 2026-04-29)
- `lr=1e-3` (PR #1122): +14% regression. Without warmup, early epochs are noisy (val=321 at ep1) and the model never catches up in 30 min.
- `n_hidden=320` (PR #1123): +12% regression. Compute-bound — only 18/30 epochs in budget.
- `surf_weight=5` (PR #1125): +0.7% regression. Volume-p improved -8% but didn't transfer to surface — Pareto trade.

### Round 5 Experiments In-Flight (2026-04-29 11:15)

**Wave 2 (just assigned):**
| PR | Student | Experiment | Predicted Δ |
|----|---------|------------|-------------|
| #1130 | fern | huber_delta=0.05 (even tighter) | -1 to -3% |
| #1131 | alphonse | mlp_ratio=4 (cheaper capacity) | -2 to -4% |
| #1132 | frieren | surf_weight=20 (other direction) | -1 to -3% |
| #1133 | tanjiro | slice_num=24 (attention capacity) | -1 to -3% |

**Wave 1 still running:**
| PR | Student | Experiment | Notes |
|----|---------|------------|-------|
| #1118 | edward | epochs=50 | Tests "is wall-clock the bottleneck" |
| #1119 | thorfinn | cosine eta_min=5e-5 | LR floor (no decay to 0) |
| #1120 | nezuko | n_layers=2 | Shallower → faster → more epochs |
| #1124 | askeladd | weight_decay=0 | No L2 |

## Potential Next Research Directions

### High Priority — pursue after wave 2 results
1. **huber_delta=0.03** if PR #1130 wins; otherwise stop tightening.
2. **surf_weight=30** (revisit PR #827 finding) if PR #1132 wins at sw=20.
3. **slice_num=32** if PR #1133 wins at slice_num=24.
4. **mlp_ratio=6 or 8** if PR #1131 wins.
5. **Linear warmup + cosine** — separate hypothesis. PR #1122's failure suggests warmup would unlock higher LR, but it requires a custom scheduler. Worth assigning when we have a slot.
6. **Per-channel huber_delta** — winning student suggested decoupling: huber_delta_p=0.1 (tight, primary metric) and huber_delta_Uxy=0.5 (looser). Needs train.py change to per_element_loss.
7. **Annealed huber_delta** schedule (1.0 → 0.1) — combines warmup robustness with late-stage precision.

### Medium Priority
8. **Tighter cosine T_max=22 with eta_min** — currently T_max=epochs=30, but we always hit timeout at epoch 22. Aligning the schedule to the actual budget could extract ~2-3% more.
9. **Focal-style surface weighting** — weight hardest surface nodes more (vs constant surf_weight). Frieren suggested this. Needs custom loss code.
10. **Larger effective batch via gradient accumulation** — current batch=4. accum_steps=2-4 could improve optimization stability.
11. **Re-conditioned normalization** — separate norm stats per Re bucket instead of per_sample_norm.
12. **Multi-scale slice_num** — different slice_num per layer (coarse-to-fine attention hierarchy).

### Lower Priority / Bold Ideas
13. **Physics-informed regularization** — continuity equation residual or Bernoulli constraint as auxiliary loss.
14. **Ensemble of 3-5 seeds** — average predictions; ~free gain if inference allows.
15. **Curriculum learning** — easy (low-Re, single-foil) → hard (high-Re, tandem).
16. **SGDR / warm restarts** — periodic cosine restarts to escape shallow optima (mentioned but never tested).
17. **Augmentation** — geometric (mirror, scale) or learnable input perturbation to grow effective training set beyond 1500 samples.

## Notes on the geom_camber_rc gap

`geom_camber_rc` is consistently the hardest split (test 64.96, vs cruise 32.35 — almost 2× worse). It's also the split that *regressed* +1.05% with huber_delta=0.1. If the next 1-2 winners don't help this split, dedicate a hypothesis to it specifically — e.g. heavier sampling weight on rc-like training samples, or a split-aware curriculum.
