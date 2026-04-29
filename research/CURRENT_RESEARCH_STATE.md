# SENPAI Research State

- 2026-04-29 23:10 (updated — 8 WIP PRs, 0 review-ready, 0 idle students)
- **Most recent research direction from human researcher team:** None (no GitHub Issues)
- **Current best:** `val_avg/mae_surf_p` = **45.5945**, `test_avg/mae_surf_p` = **39.7038** (PR #1319, epochs=40 + huber_delta=0.12 + lr=7e-4)
- **Compete target:** `test_avg/mae_surf_p` = 40.93 (Transolver paper) — **BEATEN** (gap = **-1.2262**, 3.0% below target)

## Current Research Focus and Themes

The research is in **round r5** on branch `icml-appendix-charlie-pai2f-r5`. The current baseline compounds:
- n_layers=2 (hardcoded)
- slice_num=8 (hardcoded)
- huber_delta=0.12 (**current baseline** — beats 0.1)
- epochs=40 (SequentialLR: LinearLR warmup 3 epochs + CosineAnnealingLR over 37 epochs)
- warmup_epochs=3
- weight_decay=5e-4
- lr=7e-4 (**current baseline** — PR #1319 uses 7e-4 on the huber_delta=0.12 stack; lr=8e-4 win from PR #1275 was on huber_delta=0.1 stack)
- batch_size=4, grad_clip=1.0, ema_decay=0.999, per_sample_norm
- adamw_beta2=0.985

**Compete target BEATEN by -1.2262 (3.0%).** The test_avg is now 39.7038 vs target 40.93. The model remains training-budget-limited.

**Reproduce:**
```bash
python train.py --n_hidden 256 --n_head 8 --loss huber --huber_delta 0.12 --epochs 40 \
  --grad_clip 1.0 --ema_decay 0.999 --per_sample_norm --weight_decay 5e-4 --warmup_epochs 3 \
  --adamw_beta2 0.985 --lr 7e-4
```

*(Note: n_layers=2, slice_num=8 are hardcoded in model_config dict in train.py)*

### Per-split Breakdown (PR #1319 current best)
| Split | val/mae_surf_p | test/mae_surf_p |
|-------|---------------|-----------------|
| single_in_dist | — | 43.3242 |
| geom_camber_rc | — | 53.0699 |
| geom_camber_cruise | — | 24.5326 |
| re_rand | — | 37.8885 |
| **avg** | **45.5945** | **39.7038** |

`geom_camber_rc` remains the hardest OOD split (+13 MAE above average). Compete target beaten by -1.2262 — pushing further below.

### Active Experiments — 8 WIP PRs (0 idle students)

| PR | Student | Experiment | Goal |
|----|---------|------------|------|
| #1334 | charliepai2f5-askeladd | Gradient accumulation grad_accum=4 (effective batch=16) | Zero-VRAM-cost effective batch 4x: reduce gradient noise, improve OOD |
| #1331 | charliepai2f5-tanjiro | SWA: weight-average final epochs | Post-training SWA for OOD generalization |
| #1330 | charliepai2f5-fern | lr=9e-4 probe | Next step on LR sweep above 8e-4 |
| #1329 | charliepai2f5-frieren | lr=8e-4 + huber_delta=0.12 compound | Stack two recent wins not yet tested together |
| #1306 | charliepai2f5-alphonse | n_hidden=192 slim model OOD probe | Smaller model: faster epochs → more cosine steps → better OOD? |
| #1281 | charliepai2f5-edward | AdamW beta1 sweep (0.95, 0.85) | Probe beta1 axis on lr=7e-4+beta2=0.985 stack |
| #1276 | charliepai2f5-thorfinn | warmup_epochs=5 | Longer linear warmup on beta2=0.985 stack |
| #1268 | charliepai2f5-nezuko | OneCycleLR max_lr=3e-3 | Higher peak LR for faster one-cycle convergence |

## Consolidated History of What Works vs What Failed

### Confirmed Wins (compounded into baseline)
- **epochs=40** (PR #1319, -2.32% val, -0.58% test): budget extension still improving at epoch boundary — new baseline
- **huber_delta=0.12** (PR #1311, -2.73% val, -0.76% test vs 0.10): looser Huber boundary wins over 0.10
- **lr=8e-4** (PR #1275, -2.39% test): decisive win on the beta2=0.985+huber_delta=0.1 stack — COMPETE TARGET BEATEN by 2.43%
- **lr=7e-4** (PR #1242, -0.25% test): higher LR improves exploration in the cosine phase
- **adamw_beta2=0.985** (PR #1241, -0.75% val): fine-tuned between 0.98 and 0.999 default
- **adamw_beta2=0.98** (PR #1191, +win): LLaMA-style slower second moment decay
- **weight_decay=5e-4** (PR #1136, -6.16% val): stronger OOD L2 regularization
- **epochs=26→32 cosine-aligned** (PR #1134, #1154, -1.66% val): full cosine completion budget
- **warmup_epochs=3** (PR #1149, -1.75% val): smooth early-epoch gradient noise
- **slice_num=8** (PR #1194, -2.1% val): fewer attention slices → faster epochs → more cosine steps
- **huber_delta=0.1** (PR #1121, -5.04% val): tighter Huber clamp on PSN-normalized residuals
- **per_sample_norm** (baseline): equalizes 15× Re-driven gradient-magnitude spread

### Closed / Dead Ends
- **multi-step LR gamma=0.3** (PR #1322): +1.88% val regression vs baseline (PR #1319) — complex schedule did not improve on simple cosine
- **huber_delta=0.15** (PR #1316): +5.58% regression — bracket closed: optimal is 0.10–0.12
- **huber_delta=0.08** (PR #1293): too tight — between 0.1 (best then) and 0.05 (dead end)
- **huber_delta=0.05** (PR #1240): too tight — pressure gradients over-clamped
- **huber_delta per-channel** (PR #1188): dead end
- **huber_delta annealed** (PR #1192): dead end
- **surf_weight=20** (PR #1229): regression — surf_weight=10 is optimal, do NOT exceed
- **surf_weight=5** (PR #1171): regression
- **SWA** (PR #1233): no gain over EMA baseline
- **ema_decay=0.9995** (PR #1224): regression
- **ema_decay=0.9999**: regression — 0.999 confirmed optimal for ~11k-step budget
- **adamw betas=(0.9, 0.95)** (PR #1157): +4.19% regression
- **n_layers=1** (PR #1189): capacity floor — faster but weaker
- **n_hidden=384** (PR #1212): too slow per epoch
- **n_head=16**: no gain over 8 heads
- **slice_num=24** (PR #1133): compute-bound
- **batch_size=8** (PR #1190): no consistent gain
- **weight_decay=1e-3** (PR #1153): past optimum
- **weight_decay=2e-4 / 1e-3**: dead ends (5e-4 confirmed optimal)
- **droppath p=0.1** (PR #1151): regression
- **dropout p=0.1** (PR #1248): closed — implicit regularization did not help OOD
- **one-cycle LR (PR#1246 at peak 2e-3)**: merged but marginal

## Current Research Themes

### 1. LR Sweep — Upper Bound
- **In flight:** lr=9e-4 probe (PR #1330, fern) — natural next step on LR sweep above 8e-4
- **In flight:** lr=8e-4 + huber_delta=0.12 compound (PR #1329, frieren) — stack two wins not yet tested together
- **Key question:** Where is the actual LR optimum above 8e-4? Does it compound with huber_delta=0.12?

### 2. Budget Extension
- **Merged:** epochs=40 (PR #1319) — model still improving at epoch boundary; now the baseline
- **In flight:** SWA over final epochs (PR #1331, tanjiro) — post-training weight averaging for OOD

### 3. Effective Batch Size — Gradient Accumulation
- **Just assigned:** grad_accum=4, effective batch=16 (PR #1334, askeladd) — VRAM only 22% used; 4x effective batch at zero cost
- **Key hypothesis:** Larger effective batches reduce gradient noise → better OOD generalization on geom_camber_rc

### 4. LR Schedule Optimization
- **In flight:** warmup_epochs=5 (PR #1276, thorfinn), OneCycleLR max_lr=3e-3 (PR #1268, nezuko)
- **Closed:** multi-step LR (PR #1322) — regression vs baseline
- **Key question:** Can we extract more from the 40-epoch budget with a different schedule shape?

### 5. Optimizer Tuning
- **In flight:** AdamW beta1 sweep 0.95/0.85 (PR #1281, edward)
- **beta2 axis CLOSED:** beta2=0.985 is optimal.
- **weight_decay axis CLOSED:** 5e-4 is optimal.
- **EMA axis CLOSED:** 0.999 is optimal.

### 6. Architecture & Capacity
- **In flight:** n_hidden=192 slim model (PR #1306, alphonse) — can a smaller model exploit cosine schedule better due to faster epoch time?

## Potential Next Research Directions (after current wave resolves)

### High Priority
1. **Compound winners from current wave** — stack any WIP wins (lr=9e-4, lr=8e-4+delta=0.12, grad_accum, SWA) onto PR #1319 baseline
2. **Log-cosh loss** — smooth Huber alternative with natural gradient behavior, no delta hyperparameter
3. **lr=1e-3 probe** — if 9e-4 wins, continue sweeping up; the LR optimum hasn't plateaued yet
4. **grad_accum sweep** — if accum=4 helps, test accum=8 (effective batch=32); VRAM still has headroom

### Medium Priority
5. **Separate Re embedding** — explicitly embed Reynolds number as learnable conditioning (targets geom_camber_rc)
6. **Deferred EMA start** — decay=0 for warmup epochs, ramp to 0.999 over a few hundred steps
7. **slice_num=4** — even fewer slices → even faster epochs (probing the floor)
8. **Cosine restart schedule (SGDR)** — T_0=10, T_mult=1 for 3-4 warm restarts within 40 epochs

### Bold/Creative Directions
9. **Physics-informed loss** — pressure divergence (∇p) or Kutta condition consistency as aux loss
10. **Geometry encoder** — pre-encode foil shape with small CNN, inject as model conditioning
11. **Ensemble at inference** — run 3-5 models with different seeds, average predictions (free gains)
12. **FNO-style spectral layer** — add one spectral convolution layer before Transolver attention
13. **Test-time augmentation** — mirror geometry inputs and average predicted fields
14. **Focal loss variant** — upweight hard OOD samples (geom_camber_rc) in the loss

## Notes
- **COMPETE TARGET BEATEN** by -1.2262 (3.0%) as of PR #1319 (test=39.7038 vs target 40.93). Goal now is to push further below.
- **Key compound opportunity:** lr=8e-4 + huber_delta=0.12 (PR #1329, frieren) in flight — these two wins haven't been stacked on the current baseline.
- geom_camber_rc split is still the hardest OOD split — high Re + geometry shift. Focus on it.
- The model is consistently training-budget-limited. Best improvements come from: more throughput, better schedules, budget extension, or effective batch size increase (gradient accumulation).
- VRAM is heavily underutilized (~21 GB / 96 GB available). Gradient accumulation (PR #1334) assigned as top priority.
- EMA axis CLOSED — 0.999 is optimal for the current ~11k-step training budget.
- surf_weight axis CLOSED — 10 is optimal; exceeding it hurts OOD generalization.
- huber_delta axis CLOSED — 0.10–0.12 is the optimal range; 0.15 tested and failed.
- multi-step LR schedule CLOSED — regression vs cosine baseline (PR #1322).
- The `--adamw_beta1` flag was added in commit d386a5f (betas=(cfg.adamw_beta1, cfg.adamw_beta2) in AdamW constructor).
- The `--grad_accum` flag does NOT yet exist in train.py — student askeladd must implement it (PR #1334).
