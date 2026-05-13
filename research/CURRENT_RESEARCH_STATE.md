# SENPAI Research State

- **Date:** 2026-05-13 07:00
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — 8-compound winner)

**PR #1784 — tanjiro grad-clip max_norm=10** (merged 2026-05-13 07:00, stacked on compile+EMA+Huber+bf16+warmup+n_layers=3 stack):
- `val_avg/mae_surf_p = 65.9757` (epoch 29 of 30; vs n_layers=3 baseline 69.4518 → **−5.00%**, vs old compile baseline 71.44 → **−7.65%**)
- `test_avg/mae_surf_p = 57.0711` (vs 61.1887 → **−6.74%**)
- **All 4 test splits improve cleanly** (in_dist −3.28, camber_rc −3.64, camber_cruise −4.89, re_rand −4.65)
- Config: EMA (decay=0.999) + Huber β=0.5 + bf16 autocast + LR warmup 1ep (start_factor=0.2) + `torch.compile(model, dynamic=True, mode='default')` + **grad-clip max_norm=10**, n_hidden=128, n_layers=3 (advisor branch), slice_num=64, mlp_ratio=2, lr=5e-4, bs=4
- **Mechanism (soft-scaling regime)**: clip rate 72.4% on compile stack; typical clipped step (norm ~21) → 2.1× downscaling; heavy tail p99 ~92 → 9.2× downscaling. Heavy-tail damping without erasing bulk direction signal. Pre-compile lever (−0.95%) compounded with compile to deliver −7.65% on full stack.

**Caveat on current advisor state**: PR #1784 was *measured* at n_layers=5 + grad-clip=10. The squash-merge applied the grad-clip change on top of the existing n_layers=3 advisor branch — so the live advisor branch is now n_layers=3 + grad-clip=10, a config that has not been directly evaluated. The grad-clip mechanism operates after `loss.backward()` and is mechanistically orthogonal to architecture, so the combined stack is expected to be ≤ 65.98. The next round of n_layers=3-baseline tests (tanjiro #1930, askeladd #1841 retest) will directly measure this combined state.

**Cumulative compounding (8 merges):**

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
| **PR #1784 tanjiro grad-clip=10** | **65.98** | **57.07** | Soft-scaling regime damps gradient heavy tail without erasing direction |

## Active experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #1899 | n_layers=3 + n_hidden=192 — width reinvestment after depth reduction | Architecture (width) | WIP | Prior n_hidden=192 failed on n_layers=5 (+12.5%). On n_layers=3 (0.42M params), compact+wide hypothesis. Expected ~0.94M params, ~65s/epoch. Will run against grad-clip=10 baseline implicitly |
| askeladd | #1841 | slice_num=48 — retest on full 8-merge stack (n_layers=3 + grad-clip=10) | Architecture / throughput | WIP-REBASE | First-pass val=70.76 beat OLD baseline (71.44) but not n_layers=3 (69.45) or new grad-clip baseline (65.98). Mechanism (3/4 splits improve, capacity-right-sizing) is clean. Expected retest val ≈ 65.35 if relative −0.95% holds |
| edward | #1833 | `--epochs 40` (T_max=40) — convert throughput headroom into more training | LR schedule / training duration | WIP | Running on older stack. Needs to beat new baseline (val < 65.98) to merge; will likely need rebase + retest if beats only intermediate baselines |
| fern | #1805 | Adaptive Huber β annealing — retest on n_layers=3 baseline | Loss shape / schedule | WIP-REBASE | v2 result (val=71.16) beat old compile baseline but not 69.45 or 65.98; mechanism confirmed sound. Retest on full 8-merge stack |
| frieren | #1898 | n_layers=3 + epochs=50 — cosine schedule T_max tuning | LR schedule / training duration | WIP | Critical follow-up: #1875 ran 30 epochs at T_max=30, but ~44 epochs fit in budget. Setting T_max=50 keeps LR positive through all 44 actual epochs |
| nezuko | #1878 | mlp_ratio=1 — capacity-down on FFN axis | Architecture / throughput | WIP | Completes 3-axis capacity-down matrix (depth=frieren, slice=askeladd, MLP=nezuko). Running against older stack — may need rebase + retest |
| tanjiro | #1930 | grad-clip max_norm=5.0 — threshold scan on new 8-merge stack | Gradient stability (threshold scan) | WIP | Direct continuation of #1784 win. Tests if threshold-vs-quality relationship is monotonic (push lower) or U-shaped (settle at 10). At threshold 5: ~100% clip rate, ~4.2× typical downscaling vs 2.1× at 10 |
| thorfinn | #1913 | gradient accumulation steps=2 (effective bs=8) on n_layers=3 stack | Gradient quality | WIP | #1858 SGDR closed. Pivoting to gradient-quality axis: lower-variance updates via accumulation, no dataloader bottleneck. Will implicitly measure on 8-merge stack |

**Baseline alert**: New baseline is PR #1784 (**val=65.9757, test=57.0711**). All future merges must beat this. WIP PRs running against older baselines should be sent back for retest if they show clean mechanism but their delta wouldn't beat 65.98.

**Combined-stack measurement priority**: tanjiro #1930 (grad-clip=5) and the askeladd #1841 retest are the first runs to *directly* measure the n_layers=3 + grad-clip=10 combined advisor state. Both serve dual purpose: testing their nominal hypothesis AND confirming the merged baseline is at expected ~65.98.

## Critical diagnostic: schedule truncation pattern

n_layers=3 gives ~44 epochs in 30 min but `--epochs 30` sets T_max=30. Cosine LR hits zero at epoch 30; remaining ~14 epochs are at or near LR=0. The fact that PR #1875's best epoch was 30/30 (still descending at T_max) strongly suggests this is wasting capacity. PR #1898 (frieren epochs=50) directly tests whether fixing the schedule unlocks additional improvement.

**If #1898 wins:** T_max tuning will become the new default for n_layers=3 experiments. Likely need to retroactively retest other hypotheses on n_layers=3 with correct T_max.

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
- **Refined pattern**: 5 trajectory-shape interventions (dropout, grad-clip 1.0, surf_weight=30, Lookahead, SGDR) failed on β=0.5+EMA stack. But **grad-clip=10 in the soft-scaling regime succeeded** (PR #1784, −7.65%). The failure pattern wasn't about clipping/perturbing per se — it was about direction-normalization (clip 100%, ~22× scaling) and exploration-vs-EMA conflict. Soft proportional damping of the heavy gradient tail (72% clip, ~2.1× typical scaling) preserves bulk direction signal while suppressing outliers. Future smoothing experiments should target this regime, not the high-intensity end.

### Gradient stability / heavy-tail damping (NEW direction — opens after #1784)
- **Gradient clipping max_norm=10** (#1784, tanjiro) — **MERGED, −7.65% val, −6.74% test**. All 4 splits improve. Soft-scaling regime: clip rate 72.4%, typical step downscaling ~2.1×, p99 downscaling 9.2×. Compounded strongly with compile (pre-compile lever was only −0.95%).
- **Threshold scan in progress**: tanjiro #1930 max_norm=5.0 — tests monotonic vs U-shaped quality-threshold relationship.

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

### Architecture (capacity-down — opens winning direction)
- **n_layers=3** (#1875 v2, frieren) — MERGED, −2.78% val. 35% throughput boost, param count 0.23× baseline. Val descending at final epoch.

### LR magnitude
- **lr=7e-4** (#1791, alphonse) — +0.42% worse vs pre-compile baseline. EMA half-life fixed; faster convergence + larger EMA step cancel. LR-magnitude sweep dead.

### Training efficiency
- **EMA without diagnostic pass** (#1626, fern) — within noise. Diagnostic overhead was small.
- **torch.compile(model, dynamic=True)** (#1763, edward) — MERGED, −16.06%. Dominant throughput lever.
- **EMA decay=0.9995** (#1669, edward) — catastrophic (+41 MAE). 3.7-epoch half-life can't settle in budget.

## Potential next directions

### High priority (compile + n_layers=3 + grad-clip=10 stack)
1. **Grad-clip threshold scan** — #1930 (tanjiro): max_norm=5.0 on 8-merge stack. If wins → push lower (max_norm=2.5? 1.0 retest?). If loses → grad-clip=10 is the optimum, freeze.
2. **T_max schedule tuning** — #1898 (frieren): epochs=50. If unlocks the unused ~14-epoch budget, becomes new default. Independent of grad-clip.
3. **Width reinvestment** — #1899 (alphonse): n_hidden=192 on n_layers=3. Compact+wide; prior failure was deep+wide.
4. **β annealing retest** — #1805 (fern send-back): mechanism confirmed sound, needs retest on new stack.
5. **slice_num=48 retest** — askeladd #1841: capacity-down on slice axis, full 8-merge stack.
6. **mlp_ratio=1** — nezuko #1878: complete 3-axis capacity-down matrix.
7. **Grad-accumulation steps=2** — thorfinn #1913: gradient-quality axis, distinct from heavy-tail damping.

### Medium priority
- **Grad-clip max_norm=15-20** — bracket the upper side of the threshold; check whether the relationship is truly monotonic-decreasing or peaks at 10 from above too.
- **Per-layer / per-parameter-group grad-clip** — separate thresholds for attention vs MLP vs embedding layers, on top of #1930 result.
- **n_layers=3 + slice_num=48** — compound capacity-down if askeladd's slice test wins on full stack.
- **n_layers=3 + n_hidden=192 + epochs=50** — compound width + schedule fix. Only if #1898 and #1899 both win independently.
- **batch_size=8 on n_layers=3** — GPU memory freed by smaller model; ~15-18 GB vs 23.8 GB. Larger batches improve gradient quality. Complementary to grad-accum.
- **Selective gradient damping** — apply soft-clipping only to outlier-prone parameter groups (e.g., positional embeddings, FiLM layers).

### Longer horizon
- **Re-aware embeddings** — explicit log-Re positional encoding; re_rand split (55.22 at new baseline) is a clear OOD target.
- **Surface-aware dual-head decoder** — separate volume/surface heads; surface MAE is the metric.
- **Spectral / Fourier neural operator hybrid** — if attention-based approach plateaus.
- **Compounding n_layers=3 + grad-clip merges** — all previously-closed capacity/LR experiments should be reconsidered under the new throughput regime (44 epochs vs 29) AND the new gradient-stability regime.
