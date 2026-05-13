# SENPAI Research State

- **Date:** 2026-05-13 03:25
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — 6-compound winner)

**PR #1763 — edward torch.compile** (merged 2026-05-13 02:20, stacked on top of #1672):
- `val_avg/mae_surf_p = 71.4371` (epoch 29; vs warmup baseline 85.0926 → **−16.06%**)
- `test_avg/mae_surf_p = 62.5927` (vs 75.5171 → **−17.11%**)
- Config: EMA (decay=0.999) + Huber β=0.5 + bf16 autocast + LR warmup 1ep (start_factor=0.2) + `torch.compile(model, dynamic=True, mode='default')`, n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2, lr=5e-4, bs=4
- **29 epochs / 30 min (~63 s/epoch steady state, 44% faster than baseline)** ← throughput win is clean
- All 4 test splits improved (in_dist −16.67, cruise −10.99, camber_rc −10.49, re_rand −13.55)
- Val curve was still descending at epoch 29 (~0.4 MAE/epoch); T_max=30 confounder noted

**Cumulative compounding (6 merges so far):**

| Baseline | val | test | Key change |
|----------|-----|------|------------|
| Stock (MSE, fp32) | ~160+ | ~130+ | — |
| PR #1419 alphonse bf16 | 109.29 | 97.67 | bf16 autocast → +4 epochs in budget |
| PR #1436 fern Huber β=1.0 | 96.49 | 86.33 | Smooth L1 → loss-shape MAE alignment |
| PR #1606 fern EMA | 92.35 | 81.63 | Weight averaging → reduces noise ball at eval |
| PR #1689 fern Huber β=0.5 | 85.92 | 76.55 | Tighter MAE alignment in moderate-error band |
| PR #1672 nezuko warmup 1ep | 85.09 | 75.52 | LR warmup compresses EMA-lag phase |
| PR #1763 edward torch.compile | **71.44** | **62.59** | 44% speedup → 29 vs 17 epochs in budget |

## Active experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #1791 | lr=7e-4 (raise peak LR, keep T_max=30 hot-cosine shape) | LR magnitude | WIP | #1647 T_max=18 closed: aligned cosine starves LR at end; inverted angle = raise peak LR |
| askeladd | #1841 | slice_num=48 — capacity-down on slice axis (compile-stack test) | Architecture / throughput | WIP | #1743 surf=5 closed (sweep bracketed); pivoted to capacity-down direction (slice axis). Complements frieren's n_layers=3 |
| edward | #1833 | `--epochs 40` (T_max=40) — convert throughput headroom into more training | LR schedule / training duration | WIP | #1763 compile MERGED (new best val=71.44); val still descending at cap with T_max=30 starving LR |
| fern | #1805 | Adaptive Huber β annealing (rebase+retest on compile stack) | Loss shape / schedule | WIP-REBASE | Pre-compile result was small win (val=84.46 vs 85.09, 3/4 splits); sent back for compile-stack retest |
| frieren | #1875 | n_layers=3 v2 — fresh retry on compile-stack baseline | Architecture (depth, throughput angle) | WIP | #1792 closed without result; depth-axis capacity-down test on compile stack (companion to askeladd #1841 slice axis) |
| nezuko | #1806 | LR warmup 2 epochs (extend to test more cold-start EMA compression) | LR schedule | WIP | #1672 warmup 1ep MERGED; extend warmup to see if EMA catch-up gain scales |
| tanjiro | #1784 | max_norm=10 (rebase+retest on compile stack) | Gradient stability | WIP-REBASE | Pre-compile result was clean win on all 4 splits (val=84.97 vs 85.92); sent back to retest on compile stack |
| thorfinn | #1858 | SGDR cosine warm restarts (T_0=10, T_mult=2) | LR schedule / exploration | WIP | #1783 Lookahead closed (competes with EMA); pivot to LR schedule exploration via periodic restarts |

**Critical baseline note**: All PRs must now beat `val_avg/mae_surf_p < 71.4371` (PR #1763 torch.compile, test=62.5927, W&B o6k5dj4g). PRs that only beat the pre-compile baseline (85.09) are outdated — the hypothesis must be retested ON TOP OF the compile stack to be meaningful.

**Important caveat for in-flight WIPs**: PRs #1743, #1783, #1784, #1791, #1792, #1805, #1806 were all assigned BEFORE the torch.compile merge. They are running on the pre-compile stack and will complete ~17 epochs (not 29). Evaluation will compare them against the old 85.09 baseline for mechanism validation only. Any result that beats 71.44 is a clean MERGE; results between 71.44 and 85.09 show the hypothesis has merit but needs retesting on top of compile.

## Closed hypotheses (all rounds)

### Loss / feature engineering
- **per-channel surface weights (0.5, 0.5, 2.0)** (#1445 v2, nezuko) — val=93.60 (+1.4% worse). p already dominates gradient signal; re-weighting backfired, U down-weighting removed geometric regularization.
- **SiLU activation** (#1648, edward) — val=96.99 (+5.0% worse). "Smoother but slower" trajectory; GELU/lr=5e-4 is well-tuned for this regime.
- **surf_weight=30 on β=0.5 baseline** (#1427 v2, askeladd) — val=88.99 (+3.6% worse). Huber β=0.5 already does the MAE-alignment work surf_weight was reaching for; over-weighting amplifies gradient variance (EMA-vs-live gap widens from −10.5 to −22).
- **surf_weight=5 on β=0.5 baseline** (#1743, askeladd) — val=87.68 (+2.05% worse). All 4 surf-p splits regress, but ALL 4 vol-p splits IMPROVE 5-10%. surf_weight sweep bracketed: 5/10/30 → 10 is optimum on primary metric (surface). EMA-vs-live gap monotonic with surf_weight (5→−3, 10→−10.5, 30→−22) — surf_weight controls *useful gradient variance* the EMA averages over, not just routing.

### Regularization / noise on β=0.5 stack
- **Dropout=0.1 then 0.05 on β=0.5 baseline** (#1629 v2/v3, thorfinn) — val=87.61 (p=0.1) then 87.91 (p=0.05); both +2% worse. Monotonicity violation (p=0.05 worse than p=0.1) rules out tuning. β=0.5 sharpens loss curvature in small-residual regime; per-step Bernoulli noise becomes coordinate-wise gradient corruption, not regularization. EMA half-life 1.85 ep insufficient to wash out.
- **Gradient clipping max_norm=1.0 on β=0.5 baseline** (#1534 v2, tanjiro) — val=87.27 (+1.6% worse). 6375/6375 steps clipped (100%) at peak norm 140 → effectively normalized SGD (direction-only). Clean OOD-helps (camber_cruise/re_rand) / IID-hurts (in_dist/camber_rc) split — flatter loss-landscape traversal at IID cost. Different mechanism than originally hypothesized; new attempt with max_norm=10 (rare-spike safety net only).

- **Lookahead optimizer k=5, α=0.5** (#1783, thorfinn) — val=87.11 (+1.39% worse). All 4 splits regress. **Mechanism breakthrough**: Lookahead and EMA compete for the same trajectory-smoothing budget — they don't stack. Lookahead's *live* model at ep 17 was 7.7 MAE better than baseline live (smoothing works), but EMA−live gap collapsed from −10.5 to −1.6, so EMA-evaluated checkpoint regressed. EMA at 0.999 has saturated the trajectory-smoothing axis at this budget.

**Pattern**: 4 of 4 noise/smoothing mechanisms (surf_weight=30, dropout, grad-clip 1.0, Lookahead) regress on the β=0.5+EMA stack. The β=0.5-sharpened landscape + EMA-smoothed eval has saturated the trajectory-smoothing axis. **Future smoothing/regularization knobs likely won't help**; gains must come from structural changes (architecture, LR schedule shape, capacity, data) — not from another layer of noise control.

### LR warmup
- **Warmup 1 epoch** (#1672 v2, nezuko) — val=85.09 (−0.96% vs β=0.5 baseline). MERGED. All 4 splits improved; re_rand best (−1.41). Mechanism: post-warmup EMA catch-up phase compressed (epoch-4 EMA-live gap −26 MAE vs baseline). T_max confounder (~6% higher late LR) still present; 2-epoch warmup under test.

### Loss shape / Huber β
- **Huber β=0.5** (#1689, fern) — val=85.92 (−6.96% vs β=1.0 baseline). MERGED. All 4 splits improved; largest gains on hardest splits (in_dist −7.6%, camber_rc −7.0%). Mechanism: β=0.5 moves L1 gradient into the moderate-error bulk where loss density lives, directly aligning with MAE metric.
- **Huber β=0.25** (#1705, fern) — val=93.92 (+9.31% vs β=0.5 baseline). β sweep bracketed: β=0.25 (worse), β=0.5 (BEST), β=1.0 (worse). Mechanism: quadratic region |x| < 0.25 too small for moderate errors; constant L1 gradient is too slow. in_dist hurt most (+17.81%), cruise least (+4.14%); consistent with error-distribution explanation. Adaptive β schedule (1.0→0.5 anneal) under test in #1805.

### Training efficiency / throughput
- **EMA without diagnostic pass** (#1626, fern) — val=92.46 (+0.12 within noise). Diagnostic overhead was ~8 s/epoch not the predicted ~25 s; +1 epoch in budget (18 vs 17) insufficient to escape noise. Bottleneck is training step, not val. Useful intel: peak mem 32.9 GB / 96 GB.
- **torch.compile(model, dynamic=True)** (#1763, edward) — val=71.44 (−16.06% vs warmup baseline). MERGED. 44% per-epoch speedup, 29 epochs vs 17 in 30 min. All 4 splits dramatically improved. Val still descending at cap → follow-up epochs=40 under test (#1833).

### EMA variants
- **EMA decay=0.9995** (#1669, edward) — val=133.43 (+41 MAE, catastrophic). At 30-min cap, 3.7-epoch half-life can't reach steady state — shadow stays anchored to high-loss init iterates. Clean isolation: live model trajectory identical to baseline. Mechanism plausible at ≥20 epoch budget; falsified at ours.

### Architecture / capacity

- **n_layers=8** (#1546, edward) — val=136.50 (+24.9% vs bf16 baseline). 155 s/epoch → 12 epochs. All splits worse; OOD worst (+37 camber_rc). Classic underfitting: deeper model needs more steps to converge; budget prevents it.
- **mlp_ratio=4** (#1544, alphonse) — val=115.04 (+5.3% vs bf16 baseline). 108 s/epoch → 17 epochs. 3 of 4 splits worse. Conventional ratio is over-parameterized for 1500-sample dataset at 30-min cap.
- **slice_num=96** (#1550, thorfinn) — val=120.69, +10.4% worse. Two-run confirmation. +20% per-epoch cost (15 epochs vs 18) + slower per-step convergence. OOD splits worse, not better.
- **slice_num=128** (#1451, thorfinn) — confounded by bs=2 OOM. Superseded.
- **batch_size=8** (#1447, tanjiro) — dataloader bottleneck; no per-epoch speedup.
- **lr=1e-3 + warmup** (#1430, edward) — cosine T_max mismatch; schedule under-annealed. Reassigned to LR schedule alignment.
- **n_hidden=192 v2 (post-rebase)** (#1442 v2, frieren) — val=96.66 (+12.5% vs β=0.5 baseline). All 4 splits regress, in_dist worst (+16.93). 4/4 capacity-up pattern complete.

**Pattern**: 4 of 4 architecture capacity experiments (wider/deeper/more slices/more MLP/wider hidden) fail under our 30-min cap. Capacity is NOT the bottleneck at 1500 training samples. The bottleneck is training duration, schedule alignment, and optimization quality — that's where remaining experiments should focus. Open inversion: shallower (`n_layers=3`) under test in #1792 to convert capacity savings into throughput.

### LR schedule
- **Cosine T_max=18 aligned to actual epoch budget** (#1647, alphonse) — val=94.44 (+9.9% vs β=0.5 baseline). All 4 splits regress. LR-magnitude math: at T_max=30 epoch 17 LR ≈ 1.5e-4 (moderate); at T_max=18 same epoch LR ≈ 4e-6 (near-zero). The "mismatch" wasn't a bug — it was effectively a hot-LR plateau throughout training. Inverted angle: raise peak LR (lr=7e-4 under test in #1791) rather than decay faster.

## Key observations

1. **bf16 is the dominant lever** — 18 epochs/30 min vs 11-14 for fp32. Merged.
2. **Huber loss is the second lever** — loss-shape alignment with MAE metric; ~4 epochs of effective speedup vs MSE. Merged.
3. **EMA weight averaging is the third lever** — reduces the SGD noise ball at eval; EMA consistently outperforms live weights from epoch 9+ (epoch 17: −25 MAE). Merged.
4. **torch.compile is the fourth major lever** — 44% per-epoch speedup, 29 epochs in 30 min. This is transformative: the entire throughput ceiling has shifted. All previously-failed capacity experiments that were ~1.2× slower than baseline should be retested on compile stack.
5. **Val is still falling at the cap** — at epoch 29 (~0.4 MAE/epoch). We are not at convergence. Each new epoch gets cheaper; the ceiling keeps rising.
6. **Gradient norms are massive without clipping** — 5250/5250 steps clipped at max_norm=1.0, max norm 837. Acts as full gradient normalization. Tanjiro's retest on bf16 will show whether smoother trajectory compounds with the existing stack.
7. **Per-epoch throughput is king** — any lever that doesn't speed up wall-clock per-epoch or improve sample-efficiency struggles. Architecture levers (wider, deeper, more slices) face this headwind. But the compile win shows there was a **5× larger** throughput gain available than any architecture tweak could deliver.
8. **val_single_in_dist is hardest** (~112-175 MAE across runs, ~70 post-compile). OOD camber_cruise is easiest (~44 post-compile). In_dist being hardest is likely extreme-Re / extreme-p samples in the in-distribution set, not an overfitting artifact.

## Potential next directions (post current round)

### Immediate (compile stack, higher epoch budget)
- **T_max decoupling**: `--epochs 40` (edward, #1833) — allow cosine to run longer into training; val still falling at cap
- **Even longer schedule**: if 40-epoch cosine wins, try `T_max=60` or warm-restart cosine
- **LR at cap analysis**: the "hot LR" from T_max >> actual epochs was accidentally beneficial; design this explicitly
- **All pre-compile hypotheses retested on compile stack**: alphonse lr=7e-4 (#1791), askeladd surf_weight=5 (#1743), frieren n_layers=3 (#1792), etc. — should all be retested on compile stack regardless of pre-compile result

### Loss / optimization
- **Adaptive β schedule** (fern, #1805) — still under test; mechanism now has more epochs to express itself (29 vs 17)
- **Gradient clipping safety-net** (tanjiro, #1784) — max_norm=10 vs max_norm=1.0 (100% clip); spike suppression only
- **Lookahead optimizer** (thorfinn, #1783) — EMA operates at weight level; Lookahead at optimizer level — complementary hypothesis
- **Re-conditioning**: explicit Re-aware embeddings or log-Re positional encoding (re_rand split: 61.35 → ~55 target)
- **Cycle LR / warm restarts**: SGDR cosine restarts every ~5-8 epochs might prevent stagnation

### Architecture (on compile stack — higher epoch budget changes the equation)
- **n_layers=3** (frieren, #1792): shallower model may now NOT need throughput savings (29 epochs is ample); test if 3-layer model is actually expressive enough at 29 epochs
- **n_layers=7**: was untested; on compile stack, per-epoch cost for +2 layers is smaller fraction of budget
- **slice_num=80**: not yet tested; small step toward slice_num=64 minimum
- **Batch size = 8 on compile stack**: dataloader was the bottleneck before; compile may shift it to GPU compute, making bs=8 viable again

### Architecture — fresh directions (post-plateau, when current stack is exhausted)
- **Surface-aware decoder / dual-head** — separate volume and surface heads; surface MAE is the metric, optimize directly
- **Spectral / Fourier neural operator hybrids** — fresh architecture direction if attention-based plateau
- **Test-time augmentation** using physical symmetries (mirroring flow domain)
- **Graph neural network surrogate** — physics-aware topology instead of attention slicing
