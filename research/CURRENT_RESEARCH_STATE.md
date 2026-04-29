# SENPAI Research State

- 2026-04-29 12:30
- Most recent research direction from human researcher team: None (no open GitHub Issues)
- Current research focus: Compounding optimization-side wins. New baseline (PR #1136, weight_decay=5e-4) has dropped val_avg/mae_surf_p to **52.0698** — compete gap to Transolver target now **5.22 (12.7%)**, down from 8.69. Wave-3 closes confirmed the system is **optimization-bound, not capacity-bound**: every recent compounding win has been on optimizer/schedule, not architecture. **Currently 8 students all assigned**, 0 idle.

## Current Research Focus and Themes

### The Problem
TandemFoilSet CFD surrogate: predict Ux, Uy, pressure at every mesh node given geometry + flow conditions. Primary metric: `val_avg/mae_surf_p` (surface pressure MAE, lower is better — equal-weight mean over 4 val splits). Compete target: `test_avg/mae_surf_p` ~ 40.93 (Transolver paper). Current best: 46.1497 test / 52.0698 val. Gap to close: ~12.7%.

### NEW Compound Baseline (PR #1136 — merged 2026-04-29)
- Architecture: n_hidden=256, n_head=8, **n_layers=2**, slice_num=16, mlp_ratio=2 (1.14M params)
- Training: Huber loss with **delta=0.1**, epochs=30, **weight_decay=5e-4** (NEW, was 1e-4), grad_clip=1.0, ema_decay=0.999, per_sample_norm
- lr=5e-4, batch_size=4, surf_weight=10.0, bf16 mixed precision
- All 30/30 epochs in 30.07 min, LR=0.0 at termination (full cosine completion)
- Best epoch = 30 (still improving at termination — training-budget-limited)
- Peak VRAM 22.22 GB

**Reproduce:**
```
python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.1 --epochs 30 \
  --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4
```
*(n_layers=2, slice_num=16 hardcoded in model_config dict in train.py)*

### Per-split Val/Test Breakdown (PR #1136 best)
| Split | val/mae_surf_p | test/mae_surf_p |
|-------|---------------|-----------------|
| single_in_dist | 56.52 | 51.78 |
| geom_camber_rc | **66.54** | **61.42** |
| geom_camber_cruise | 34.03 | 28.36 |
| re_rand | 51.19 | 43.04 |
| **avg** | **52.0698** | **46.1497** |

`geom_camber_rc` remains worst — but improved from 70.42 (PR #1120) to 66.54 with wd=5e-4. `geom_camber_cruise` is now <30 on test. Compete gap on the easiest split is just 28.36 vs 40.93 Transolver-on-avg.

### What We Know Works (compounded baseline ingredients)
- **`weight_decay=5e-4`** (NEW from PR #1136, -6.16% val, -5.46% test): stronger L2 helped all OOD splits more than in-dist (cruise -13.6%, re_rand -10.5%, in-dist -3.2%). geom_camber_rc gained least (-4.2%) — confirms it's representation-bottlenecked, not overfitting.
- **`epochs=26`** cosine-aligned (PR #1134, -1.66% val, -1.62% test): full cosine completion within budget. **Note:** PR #1136 reverted to epochs=30 because wd=5e-4 needed the extra cosine tail; both are now best-known.
- **`n_layers=2`** (PR #1120, -3.51% val, -3.38% test): shallower → faster epochs → more cosine decay completed. -27% VRAM, -29% params.
- **`--huber_delta 0.1`** (PR #1121, -5.04% val, -5.52% test). Tighter Huber clamp on PSN-normalized residuals.
- `--per_sample_norm`: equalizes 15× Re-driven gradient-magnitude spread across samples.
- EMA weight averaging (decay=0.999): marginal but consistent.
- Huber loss + surf_weight=10 baseline.
- `n_hidden=256, n_head=8, slice_num=16, mlp_ratio=2`: best architecture (capacity-bound experiments at slice_num=24, mlp_ratio=4, n_hidden=320 all regressed because of throughput cost).
- `grad_clip=1.0`: stabilizes against extreme pressure values.
- BF16 mixed precision, cosine T_max=epochs.

### Diagnostic Conclusion: Optimization-bound, not Capacity-bound
Wave-2 results (#1130-#1133) closed 4-for-4 — every "more capacity" hypothesis (mlp_ratio=4, slice_num=24, surf_weight=20, huber_delta=0.05) regressed. Adding parameters costs epochs in a timeout-bound regime. **Future hypotheses should target optimization (LR schedule, optimizer betas, regularization, warmup) rather than capacity.**

### What Just Failed (closed last 24h, on r5 advisor branch)
- `lr=1e-3` (PR #1122): +14% regression — without warmup, early epochs noisy.
- `n_hidden=320` (PR #1123): +12% regression — compute-bound (only 18/30 epochs).
- `surf_weight=5` (PR #1125): +0.7% regression — Pareto trade with volume_p.
- `epochs=50` (PR #1118): +5.8% regression — T_max=50 stretches cosine too far.
- `weight_decay=0` (PR #1124): +2.5% regression. **Pointed UP** → led to PR #1136 win.
- `huber_delta=0.05` (PR #1130): regression — both sides of optimum (0.1) explored.
- `mlp_ratio=4` (PR #1131): regression — compute-bound.
- `surf_weight=20` (PR #1132): regression — both sides of optimum (10) explored.
- `slice_num=24` (PR #1133): regression — compute-bound.
- `cosine eta_min=5e-5` (PR #1119): +22.9% regression — LR floor too high; cosine decay to 0 is right.

### Round 5 Experiments In-Flight (2026-04-29 12:30) — All 8 Students Active

**On PR #1136 (weight_decay=5e-4) compound baseline:**
| PR | Student | Experiment | Predicted Δ vs PR #1136 |
|----|---------|------------|--------------------------|
| #1157 | thorfinn | AdamW betas=(0.9, 0.95) — faster β2 | -1 to -3% |
| #1154 | nezuko | epochs=40 budget extension | -1 to -2% |
| #1153 | askeladd | weight_decay=1e-3 (probe beyond 5e-4) | -1 to -2% (or up if past optimum) |
| #1151 | tanjiro | DropPath stochastic depth p=0.1 | -1 to -3% (regularization) |
| #1150 | frieren | L1 surface + Huber volume (channel-asymmetric) | -1 to -3% |
| #1149 | fern | 3-epoch linear LR warmup + cosine | -1 to -3% (addresses optimization-bound) |
| #1146 | edward (legacy label) | base lr=2e-4 (lower LR) | -1 to -3% |
| #1145 | alphonse (legacy label) | dropout p=0.1 | -1 to -3% (regularization) |

**Key bets:**
- **Optimizer-side**: thorfinn (betas), edward (lr=2e-4), fern (warmup) — three orthogonal optimizer angles.
- **Regularization**: askeladd (wd=1e-3 sweep), tanjiro (DropPath), alphonse (dropout) — all explore whether more regularization helps in this OOD-heavy task.
- **Loss/training**: frieren (channel-split), nezuko (epochs=40 — still needs T_max alignment).
- **Note on nezuko's epochs=40**: cosine with T_max=40 means LR at termination is non-zero (would still be ~1.2e-4 if 30/40 epochs complete). May need a follow-up adjustment.

## Potential Next Research Directions

### High Priority — pursue after wave 3 / 4 results
1. **Compound winners** — once any of the 8 in-flight win, compound with PR #1136 base.
2. **Per-channel huber_delta** (delta_p=0.1, delta_Uxy=0.5) — fern's earlier suggestion. Decouples surface pressure precision from velocity smoothing. Needs `per_element_loss` change.
3. **n_layers=1** (push throughput further) — risk: too little capacity. If wins, even more epochs.
4. **batch_size=8** — VRAM at 22 GB so 4× headroom. May improve optimization stability. Check effective LR scaling.
5. **Annealed huber_delta** schedule (1.0 → 0.1) — combines warmup robustness with late-stage precision.
6. **AdamW betas=(0.9, 0.98)** — if thorfinn's 0.95 wins at PR #1157, try 0.98 (LLaMA-style).
7. **Cosine warm restarts (SGDR)** — periodic LR resets to escape shallow optima.
8. **Targeted geom_camber_rc improvement** — that split lags hardest. Heavier sampling weight on rc-like training samples or split-aware curriculum.

### Medium Priority
9. **Re-conditioned normalization** — separate norm stats per Re bucket instead of per_sample_norm.
10. **Multi-scale slice_num** — different slice_num per layer (coarse-to-fine attention hierarchy).
11. **Focal-style surface weighting** — weight hardest surface nodes more.
12. **Lookahead optimizer wrapper** — k-step inner-outer optimization, often helps short runs.

### Lower Priority / Bold Ideas
13. **Physics-informed regularization** — continuity equation residual or Bernoulli constraint as auxiliary loss.
14. **Ensemble of 3-5 seeds** — average predictions; ~free gain if inference allows.
15. **Curriculum learning** — easy (low-Re, single-foil) → hard (high-Re, tandem).
16. **Augmentation** — geometric (mirror, scale) or learnable input perturbation to grow effective training set beyond 1500 samples.

## Notes on the geom_camber_rc gap

`geom_camber_rc` is consistently the hardest split (test 61.42, vs cruise 28.36 — over 2× worse). It's the split that gained least from wd=5e-4 (-4.2% vs others -10–13%). If the next 2-3 wave winners don't help this split, dedicate a hypothesis to it specifically — heavier sampling weight on rc-like training samples, or a split-aware curriculum, or a domain embedding (already attempted in PR #1113 but not yet reviewed).
