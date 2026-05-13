# SENPAI Research State

- **Date:** 2026-05-13 05:50
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — 7-compound winner)

**PR #1875 — frieren n_layers=3** (merged 2026-05-13 05:50, stacked on compile+EMA+Huber+bf16+warmup stack):
- `val_avg/mae_surf_p = 69.4518` (epoch 30; vs compile baseline 71.4371 → **−2.78%**)
- `test_avg/mae_surf_p = 61.1887` (vs 62.5927 → **−2.24%**)
- Config: EMA (decay=0.999) + Huber β=0.5 + bf16 autocast + LR warmup 1ep (start_factor=0.2) + `torch.compile(model, dynamic=True, mode='default')`, n_hidden=128, **n_layers=3**, slice_num=64, mlp_ratio=2, lr=5e-4, bs=4
- **30 epochs / 20.6 min (~40.8 s/epoch steady state, 35% faster than compile baseline ~63 s)**
- **Projected 44 epochs in 30-min budget** — 15 epochs of unused budget remain (key follow-up signal)
- 3/4 test splits improved cleanly; camber_rc +0.14 (noise). Best epoch was epoch 30/30 (final) — val still descending at T_max cap.

**Cumulative compounding (7 merges):**

| Baseline | val | test | Key change |
|----------|-----|------|------------|
| Stock (MSE, fp32) | ~160+ | ~130+ | — |
| PR #1419 alphonse bf16 | 109.29 | 97.67 | bf16 autocast → +4 epochs in budget |
| PR #1436 fern Huber β=1.0 | 96.49 | 86.33 | Smooth L1 → loss-shape MAE alignment |
| PR #1606 fern EMA | 92.35 | 81.63 | Weight averaging → reduces noise ball at eval |
| PR #1689 fern Huber β=0.5 | 85.92 | 76.55 | Tighter MAE alignment in moderate-error band |
| PR #1672 nezuko warmup 1ep | 85.09 | 75.52 | LR warmup → AdamW 2nd-moment stabilization |
| PR #1763 edward torch.compile | 71.44 | 62.59 | 44% speedup → 29 vs 17 epochs in budget |
| **PR #1875 frieren n_layers=3** | **69.45** | **61.19** | 35% further speedup + capacity-right-sizing |

## Active experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #1899 | n_layers=3 + n_hidden=192 — width reinvestment after depth reduction | Architecture (width) | WIP | Prior n_hidden=192 failed on n_layers=5 (+12.5%). On n_layers=3 (0.42M params), compact+wide hypothesis. Expected ~0.94M params, ~65s/epoch |
| askeladd | #1841 | slice_num=48 — capacity-down on slice axis (compile-stack test) | Architecture / throughput | WIP | Running against old compile baseline (71.44). If beats 69.45 → merge; if only 71.44 → send back for retest on n_layers=3 stack |
| edward | #1833 | `--epochs 40` (T_max=40) — convert throughput headroom into more training | LR schedule / training duration | WIP | Running on n_layers=5 compile stack. Needs to beat 69.45 to merge |
| fern | #1805 | Adaptive Huber β annealing — retest on n_layers=3 baseline | Loss shape / schedule | WIP-REBASE | v2 result (val=71.16) beat old compile baseline but not new 69.45; mechanism confirmed sound (relative test win grew pre→post compile). Retest on n_layers=3 stack |
| frieren | #1898 | n_layers=3 + epochs=50 — cosine schedule T_max tuning | LR schedule / training duration | WIP | Critical follow-up: #1875 ran 30 epochs at T_max=30, but ~44 epochs fit in budget. Setting T_max=50 keeps LR positive through all 44 actual epochs, eliminating wasted "dead epochs" at LR=0 |
| nezuko | #1878 | mlp_ratio=1 — capacity-down on FFN axis (compile-stack test) | Architecture / throughput | WIP | Running against old compile baseline (71.44). Completes 3-axis capacity-down matrix (depth=frieren, slice=askeladd, MLP=nezuko). If beats 69.45 → merge; else context-dependent |
| tanjiro | #1784 | max_norm=10 (rebase+retest on compile stack) | Gradient stability | WIP-REBASE | Pre-compile result was clean win (val=84.97); retesting on n_layers=5 compile stack. Needs to beat 69.45 |
| thorfinn | #1858 | SGDR cosine warm restarts (T_0=10, T_mult=2) | LR schedule / exploration | WIP | LR schedule axis; periodic restarts may help escape local optima. Running on compile stack |

**Baseline alert**: New baseline is PR #1875 (val=69.4518, test=61.1887). All PRs assigned against old compile baseline (71.44) must beat 69.45 to merge; those that only beat 71.44 should be sent back for retest on the n_layers=3 stack.

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
- **Gradient clipping max_norm=1.0** (#1534 v2, tanjiro) — +1.6% worse. 100% clipping = normalized SGD. Asymmetric OOD-helps/IID-hurts split. Safety-net retest at max_norm=10 under way.
- **Lookahead optimizer k=5, α=0.5** (#1783, thorfinn) — +1.39% worse. Competes with EMA for trajectory-smoothing budget; EMA-live gap collapses from −10.5 to −1.6.
- **Pattern**: 4/4 noise/smoothing mechanisms fail on β=0.5+EMA stack. Smoothing axis saturated.

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

### High priority (compile + n_layers=3 stack)
1. **T_max schedule tuning** — #1898 (frieren): directly tests if epochs=50 unlocks the unused ~14-epoch budget. If yes, becomes new default.
2. **Width reinvestment** — #1899 (alphonse): n_hidden=192 on n_layers=3. Compact+wide; prior failure was deep+wide.
3. **β annealing retest on n_layers=3** — #1805 (fern send-back): mechanism confirmed sound, needs retest on new stack. More late-phase epochs = stronger signal.
4. **slice_num=48 + mlp_ratio=1** — askeladd/nezuko WIP: completing capacity-down 3-axis matrix.
5. **n_layers=2** — bracket depth axis at the bottom; risks hitting capacity floor. Most informative if we first see how far epochs=50 can push n_layers=3.

### Medium priority
- **n_layers=3 + slice_num=48** — compound capacity-down if askeladd's slice test wins. Two orthogonal axes trimmed simultaneously.
- **n_layers=3 + n_hidden=192 + epochs=50** — compound width + schedule fix. Only if #1898 and #1899 both win independently.
- **SGDR on n_layers=3** — thorfinn's warm-restart test; periodic LR restarts on the new faster model.
- **batch_size=8 on n_layers=3** — GPU memory freed by smaller model; ~15-18 GB vs 23.8 GB. Larger batches improve gradient quality.

### Longer horizon
- **Re-aware embeddings** — explicit log-Re positional encoding; re_rand split (59.88 at new baseline) is a clear OOD target.
- **Surface-aware dual-head decoder** — separate volume/surface heads; surface MAE is the metric.
- **Spectral / Fourier neural operator hybrid** — if attention-based approach plateaus.
- **Compounding n_layers=3 merges** — after n_layers=3 stack stabilizes, all previously-closed capacity/LR experiments should be reconsidered under the new throughput regime (44 epochs vs 29).
