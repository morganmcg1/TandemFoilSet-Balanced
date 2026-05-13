# SENPAI Research State

- **Date:** 2026-05-13 10:00
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — 11-compound winner; MASSIVE)

**PR #1953 — alphonse n_hidden=192 + epochs=50** (merged 2026-05-13 10:00):
- `val_avg/mae_surf_p = 55.7634` (↓ from 63.4801, **−12.17%**)
- `test_avg/mae_surf_p = 48.0960` (↓ from 54.9834, **−12.53%**)
- **All 4 test splits improve dramatically** (in_dist −9.56, camber_rc −6.59, camber_cruise −4.67, re_rand −6.73 vs PR #1930)
- Config: EMA (decay=0.999) + Huber β=0.5 + bf16 autocast + LR warmup 1ep + `torch.compile(dynamic=True)` + n_hidden=192, n_layers=3, slice_num=64, mlp_ratio=2, n_head=4, grad-clip max_norm=5.0, **`--epochs 50` (T_max=50)**
- **FIRST direct measurement of the full 10-compound stack + schedule fix.** All three mergers (n_hidden=192, grad-clip=5.0, T_max=50) compounded as predicted.
- **EMA−live gap flipped from +0.42 to −8.32** — at the new compound point, EMA shadow is doing real work absorbing per-step gradient variance, not just smoothing convergence noise.
- Best epoch 30/30 (every single epoch was a new EMA best). Val slope at termination **−0.84/ep — model is epoch-saturated, not capacity-saturated**.
- **W&B run:** `vnsqnuoy`
- **Reproduce:** `cd target/ && python train.py --agent <student> --wandb_name "<name>" --n_hidden 192 --n_layers 3 --epochs 50`

**IMPORTANT:** PR #1953 merge produced an EMPTY DIFF — student ran the winning config purely via CLI flags. The advisor branch default still has `n_hidden=128`, `n_layers=5`, and default epochs. All subsequent student PRs MUST specify `--n_hidden 192 --n_layers 3 --epochs 50` (or higher T_max) in reproduce commands to test against the actual 11-compound baseline.

**Cumulative compounding (11 merges):**

| Baseline | val | test | Key change |
|----------|-----|------|------------|
| Stock (MSE, fp32) | ~160+ | ~130+ | — |
| PR #1419 alphonse bf16 | 109.29 | 97.67 | bf16 autocast → +4 epochs in budget |
| PR #1436 fern Huber β=1.0 | 96.49 | 86.33 | Smooth L1 → loss-shape MAE alignment |
| PR #1606 fern EMA | 92.35 | 81.63 | Weight averaging → reduces noise ball at eval |
| PR #1689 fern Huber β=0.5 | 85.92 | 76.55 | Tighter MAE alignment in moderate-error band |
| PR #1672 nezuko warmup 1ep | 85.09 | 75.52 | LR warmup → AdamW 2nd-moment stabilization |
| PR #1763 edward torch.compile | 71.44 | 62.59 | 44% speedup → 29 vs 17 epochs in budget |
| PR #1875 frieren n_layers=3 | 69.45 | 61.19 | 35% further speedup + capacity-right-sizing |
| PR #1784 tanjiro grad-clip=10 | 65.98 | 57.07 | Soft-scaling regime damps gradient heavy tail |
| PR #1899 alphonse n_hidden=192 | 63.72 | 55.64 | Width reinvestment — compact+wide beats compact+narrow on depth-limited stack |
| PR #1930 tanjiro grad-clip=5 | 63.48 | 54.98 | Tighter threshold → 90% clip rate, 4.3× downscaling |
| **PR #1953 alphonse T_max=50 + full stack** | **55.76** | **48.10** | First full-stack measurement + schedule fix. All 4 splits −12% uniformly |

## Active experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #2000 | T_max=80 schedule extension (epochs 50→80) | Schedule | WIP | #1953 MERGED (11th winner, val=55.76, MASSIVE −12.17%). Val descending at −0.84/ep at termination. At T_max=80, LR at ep 30 ≈ 3.45e-4 (vs 1.73e-4 at T_max=50). Tests: schedule push compounds further OR T_max=50 was sweet spot |
| askeladd | #1841 | slice_num=48 — retest on full 8-merge stack (n_layers=3 + grad-clip=10) | Architecture / throughput | WIP-REBASE | First-pass val=70.76 beat OLD baseline (71.44) but not n_layers=3 (69.45) or new grad-clip baseline (65.98). Mechanism (3/4 splits improve, capacity-right-sizing) is clean. Expected retest val ≈ 65.35 if relative −0.95% holds |
| edward | #1833 | `--epochs 40` (T_max=40) — convert throughput headroom into more training | LR schedule / training duration | WIP | Running on older stack. Needs to beat new baseline (val < 65.98) to merge; will likely need rebase + retest if beats only intermediate baselines |
| fern | #1805 | Adaptive Huber β annealing — retest on n_layers=3 baseline | Loss shape / schedule | WIP-REBASE | v2 result (val=71.16) beat old compile baseline but not 69.45 or 65.98; mechanism confirmed sound. Retest on full 8-merge stack |
| frieren | #1898 | n_layers=3 + epochs=50 — cosine schedule T_max tuning | LR schedule / training duration | WIP | Critical follow-up: #1875 ran 30 epochs at T_max=30, but ~44 epochs fit in budget. Setting T_max=50 keeps LR positive through all 44 actual epochs |
| nezuko | #1994 | n_head=4→8 — attention head diversity | Architecture (attention) | WIP | #1878 CLOSED (+10.5% regression; mlp_ratio=1 already +0.99% on n_layers=3 stack). FFN capacity-down bracketed: mlp_ratio=2 is optimum. Testing attention diversity: 8 heads with head_dim=24 on n_hidden=192 |
| tanjiro | #1982 | grad-clip max_norm=2.5 — threshold scan step 3 | Gradient stability (threshold scan) | WIP | #1930 MERGED (10th winner, val=63.48). 3/4 splits improved; in_dist regressed +1.00. At 2.5: ~97% clip, ~8-9× downscaling — tests whether crossing into direction-normalization regime. Outcome: (A) val < 63.48 → keep going; (B) val > 63.48 → U-shape confirmed, bracket between 2.5–5.0 |
| thorfinn | #1960 | n_layers=2 + n_hidden=192 — depth floor test | Architecture (depth) | WIP | #1913 grad-accum=2 closed (+8.9% val, +19.7% vs current baseline; undertrained at fixed --epochs due to half opt-steps). Pivoting from trajectory-quality to architecture axis. Expected ~36s/ep, ~50 epochs in budget |

**Baseline alert**: New baseline is PR #1953 (**val=55.7634, test=48.0960**). All future merges must beat this. WIP PRs running against the older PR #1930 baseline (val=63.48) MUST be rebased and retested with `--epochs 50` (T_max=50). The schedule fix alone is the dominant lift — any experiment without it cannot beat the new baseline.

**Critical mandate**: ALL student reproduce commands MUST now specify `--n_hidden 192 --n_layers 3 --epochs 50` (or `--epochs 80` to match the in-flight schedule push). The PR #1953 merge produced an empty diff — these CLI flags are how students access the 11-compound stack.

## Critical diagnostic: schedule truncation pattern — CONFIRMED MASSIVE

**RESOLVED by PR #1953**: T_max=50 vs T_max=30 (same wall-clock budget) delivered the dominant single-merge improvement in 11 cycles (−12% across all 4 test splits). The schedule axis is now confirmed as the highest-leverage remaining dimension.

**Current status:** model is epoch-saturated, not capacity-saturated. Val slope at #1953 termination was **−0.84/ep at epoch 30/30**. PR #2000 (alphonse T_max=80) is the immediate follow-up to test whether schedule can extract further gains.

**Retroactive implications:** all previously-closed experiments measured under T_max=30 may have hidden viability under the new T_max=50 protocol. Specifically: the asymmetric OOD/IID trade-off observed at grad-clip=5 (in_dist +1.00 vs OOD wins) might reverse under longer effective training.

## Closed hypotheses (all rounds)

### Loss / feature engineering
- **per-channel surface weights (0.5, 0.5, 2.0)** (#1445 v2, nezuko) — +1.4% worse. p already dominates.
- **SiLU activation** (#1648, edward) — +5.0% worse. GELU/lr=5e-4 is well-tuned.
- **surf_weight=30** (#1427 v2, askeladd) — +3.6% worse. Huber β=0.5 already does the work.
- **surf_weight=5** (#1743, askeladd) — +2.05% worse. surf_weight=10 is optimum on primary metric. Sweep bracketed: 5/10/30.

### Regularization / noise
- **Dropout=0.1, 0.05** (#1629 v2/v3, thorfinn) — both +2% worse. β=0.5 sharpened landscape; per-step Bernoulli noise becomes gradient corruption.
- **Gradient clipping max_norm=1.0** (#1534 v2, tanjiro) — +1.6% worse. 100% clipping = normalized SGD with ~22× downscaling. Asymmetric OOD-helps/IID-hurts split. Direction-normalization regime.
- **Lookahead optimizer k=5, α=0.5** (#1783, thorfinn) — +1.39% worse. Competes with EMA for trajectory-smoothing budget; EMA-live gap collapses from −10.5 to −1.6.
- **SGDR cosine warm restarts (T_0=10, T_mult=2)** (#1858, thorfinn) — val=72.99 (+2.17% vs OLD 71.44 baseline; +5.09% vs NEW 69.45 baseline). LR restart mechanism worked exactly as designed. Cycle 1 wasted on converge-then-reset; EMA-tax of ~10 epochs to wash cycle-2 high-LR noise > marginal exploration benefit.
- **Gradient accumulation steps=2 (effective bs=8)** (#1913, thorfinn) — val=75.64 (+8.9% vs old baseline; +18.7% vs current). Mechanism diagnosis was sharp: per-epoch wall time unchanged but opt-steps halved → cosine starves → model undertrained at +0.5/ep at the cap. EMA-live gap DID close to +1.88 (variance reduction effect IS present) but step-count deficit dominates at fixed --epochs. Fixed wall-clock retest mechanically valid but baseline shifted; closed.
- **Refined pattern**: 5 trajectory-shape interventions (dropout, grad-clip 1.0, surf_weight=30, Lookahead, SGDR) failed on β=0.5+EMA stack. But **grad-clip=10 in the soft-scaling regime succeeded** (PR #1784, −7.65%). The failure pattern wasn't about clipping/perturbing per se — it was about direction-normalization (clip 100%, ~22× scaling) and exploration-vs-EMA conflict. Soft proportional damping of the heavy gradient tail (72% clip, ~2.1× typical scaling) preserves bulk direction signal while suppressing outliers. Future smoothing experiments should target this regime, not the high-intensity end.

### Gradient stability / heavy-tail damping (NEW direction — opens after #1784)
- **Gradient clipping max_norm=10** (#1784, tanjiro) — **MERGED, −7.65% val, −6.74% test**. All 4 splits improve. Soft-scaling regime: clip rate 72.4%, typical step downscaling ~2.1×, p99 downscaling 9.2×.
- **Gradient clipping max_norm=5.0** (#1930, tanjiro) — **MERGED, −0.38% val, −1.18% test**. 3/4 splits improve (in_dist regresses +1.00 — first sign of asymmetric OOD/IID trade-off). Clip rate 90.1%, downscaling 4.3×. Predictions matched exactly.
- **Threshold scan in progress**: tanjiro #1982 max_norm=2.5 — regime shift test. (~97% clip, ~8-9× downscaling predicted). Two outcomes: continued gain or U-shape confirmation → bracket at 2.5–5.0.

### LR warmup
- **Warmup 1 epoch** (#1672 v2, nezuko) — MERGED, −0.96%. Mechanism: AdamW 2nd-moment stabilization, not EMA catch-up.
- **Warmup 2 epochs** (#1806, nezuko) — +3.57% worse. Hypothesis prediction directly falsified. Bracketed at 1 epoch.

### Loss shape / Huber β
- **Huber β=0.5** (#1689, fern) — MERGED, −6.96%. Direct MAE alignment in moderate-error band.
- **Huber β=0.25** (#1705, fern) — +9.31% worse. β sweep bracketed: 0.25 < 0.5 (BEST) > 1.0.

### Architecture (capacity-up — all fail)
- **n_layers=8** (#1546, edward) — +24.9% worse. 155 s/epoch; underfitting at budget.
- **mlp_ratio=4** (#1544, alphonse) — +5.3% worse. Over-parameterized for 1500-sample dataset.
- **slice_num=96** (#1550, thorfinn) — +10.4% worse. Confirmed in two runs.
- **n_hidden=192** (#1442 v2, frieren) — +12.5% worse. **On n_layers=5 stack — now retesting on n_layers=3 (#1899).**
- **Pattern**: 4/4 capacity-up fails on n_layers=5 compile stack. Capacity-up direction is closed for n_layers=5; open inversion on n_layers=3.

### Architecture (FFN capacity-down — LOSS)
- **mlp_ratio=1** (#1878, nezuko) — +10.5% vs current 10-compound baseline; +0.99% vs n_layers=3 baseline. **CLOSED.** FFN capacity is not the bottleneck at n_layers=3 + n_hidden=192. mlp_ratio=2 is the optimum for this 1500-sample dataset. Per-axis capacity-down matrix: depth-down WIN, FFN-down LOSS, slice-down in-flight (#1841).

### Attention configuration (NEW direction — opens with nezuko #1994)
- **n_head=8** (#1994, nezuko) — in progress. Doubles attention heads 4→8, head_dim 48→24. Tests whether attention diversity improves generalization on geometry+physics+Re multi-aspect task.

### Architecture (capacity-down + width reinvestment — winning direction)
- **n_layers=3** (#1875 v2, frieren) — MERGED, −2.78% val. 35% throughput boost, param count 0.23× baseline. Val descending at final epoch.
- **n_hidden=192 × n_layers=3** (#1899, alphonse) — **MERGED, −8.25% val, −9.06% test. All 4 splits improve.** 0.93M params, 54 s/epoch. "Compact but wide" hypothesis confirmed: per-layer expressivity was the bottleneck at n_layers=3; wider layers compensate for reduced composition depth. Val still descending at ep 30 → schedule fix in progress.

### LR magnitude
- **lr=7e-4** (#1791, alphonse) — +0.42% worse vs pre-compile baseline. EMA half-life fixed; faster convergence + larger EMA step cancel. LR-magnitude sweep dead.

### Training efficiency
- **EMA without diagnostic pass** (#1626, fern) — within noise. Diagnostic overhead was small.
- **torch.compile(model, dynamic=True)** (#1763, edward) — MERGED, −16.06%. Dominant throughput lever.
- **EMA decay=0.9995** (#1669, edward) — catastrophic (+41 MAE). 3.7-epoch half-life can't settle in budget.

## Potential next directions

### High priority (compile + n_layers=3 + n_hidden=192 + grad-clip=10 stack)
1. **T_max=80 schedule push** — #2000 (alphonse): direct follow-up to #1953. Tests if schedule axis has more headroom.
2. **Grad-clip threshold scan step 3** — #1982 (tanjiro): max_norm=2.5 on PRE-T_max=50 stack — likely will need retest after #2000 result.
3. **n_head=8** — #1994 (nezuko): attention head diversity. PRE-T_max=50 stack — likely will need retest.
4. **n_layers=3 + epochs=50 retest (n_hidden=128)** — #1898 (frieren): STALE 2h+, nudged. May be DEAD given alphonse delivered the schedule-fix win on the wider stack.
5. **β annealing retest** — #1805 (fern): NEEDS REBASE; retest on full 11-compound stack (val < 55.76).
6. **slice_num=48 retest** — #1841 (askeladd): capacity-down on slice axis; needs retest on full stack.
7. **n_layers=2 + n_hidden=192** — #1960 (thorfinn): depth floor test on PRE-T_max=50 stack — may need retest.
8. **n_hidden=224 or 256 × n_layers=3 + epochs=50** — width scan extension. Wall-time constrained but well-motivated since #1899 showed compact+wide hypothesis.
9. **EMA decay tuning** — EMA−live gap is now −8.32 (big!). Try ema_decay=0.998 or 0.9995 to balance smoothing vs reactivity.
10. **batch_size=8 + epochs=50** — possible with smaller model footprint. Compounds with schedule fix.

### Medium priority
- **n_hidden=160 × n_layers=3** — bracket width from below; is 192 the sweet spot or above it?
- **n_layers=2 × n_hidden=192** — push "compact+wide" further along the depth axis.
- **Grad-clip max_norm=15-20** — bracket the upper side of the threshold.
- **n_layers=3 + n_hidden=192 + slice_num=48** — compound width + capacity-down on slice axis.
- **batch_size=8** — GPU memory freed by smaller model (~46-50 GB for n_hidden=192).

### Longer horizon
- **Re-aware embeddings** — explicit log-Re positional encoding; re_rand split (54.10 at new baseline) is the clearest OOD gap.
- **Surface-aware dual-head decoder** — separate volume/surface heads; surface MAE is the metric.
- **Spectral / Fourier neural operator hybrid** — if attention-based approach plateaus.
- **Systematic retest under new combined stack** — all previously-closed capacity/LR experiments should be reconsidered under n_layers=3 + n_hidden=192 + grad-clip=10 + schedule-fixed.
