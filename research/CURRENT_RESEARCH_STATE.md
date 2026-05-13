# SENPAI Research State

- **Date:** 2026-05-13 02:10
- **Track:** `willow-pai2g-48h-r5` on advisor branch `icml-appendix-willow-pai2g-48h-r5`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-48h-r5`
- **Students (8, each 1× 96GB GPU):** alphonse, askeladd, edward, fern, frieren, nezuko, tanjiro, thorfinn
- **Per-run training cap:** `SENPAI_TIMEOUT_MINUTES=30` (hard wall-clock per training execution)
- **Most recent direction from human team:** None. Controlled 24h/48h Charlie-vs-Willow logging ablation; experiments run in isolation from other branches.

## Research target

CFD surrogate for TandemFoilSet. Predict normalized `(Ux, Uy, p)` at every mesh node from 24-dim node features. Primary metric `val_avg/mae_surf_p` and paper-facing `test_avg/mae_surf_p` — both **lower is better**, averaged across 4 splits (in-distribution, unseen front-foil camber raceCar, unseen front-foil camber cruise, stratified Re holdout).

## Current baseline (MERGED — 5-compound winner)

**PR #1672 — nezuko LR warmup 1 epoch v2** (merged 2026-05-13 02:05, stacked on top of #1689):
- `val_avg/mae_surf_p = 85.0926` (epoch 17; vs 85.9197 β=0.5 baseline → −0.96%)
- `test_avg/mae_surf_p = 75.5171` (vs 76.5495 → −1.35%)
- Config: EMA (decay=0.999) + Huber β=0.5 + bf16 autocast + LR warmup 1ep (start_factor=0.2), n_hidden=128, n_layers=5, slice_num=64, mlp_ratio=2, lr=5e-4, bs=4
- ~17 epochs / 30 min (~110 s/epoch)
- All 4 test splits improved; re_rand best (−1.41 MAE)

**Cumulative compounding (5 merges so far):**

| Baseline | val | test | Key change |
|----------|-----|------|------------|
| Stock (MSE, fp32) | ~160+ | ~130+ | — |
| PR #1419 alphonse bf16 | 109.29 | 97.67 | bf16 autocast → +4 epochs in budget |
| PR #1436 fern Huber β=1.0 | 96.49 | 86.33 | Smooth L1 → loss-shape MAE alignment |
| PR #1606 fern EMA | 92.35 | 81.63 | Weight averaging → reduces noise ball at eval |
| PR #1689 fern Huber β=0.5 | 85.92 | 76.55 | Tighter MAE alignment in moderate-error band |
| PR #1672 nezuko warmup 1ep | **85.09** | **75.52** | LR warmup compresses EMA-lag phase |

## Active experiments

| Student | PR | Hypothesis | Lever | Status | Note |
|---------|----|-----------|-------|------|-----|
| alphonse | #1791 | lr=7e-4 (raise peak LR, keep T_max=30 hot-cosine shape) | LR magnitude | WIP | #1647 T_max=18 closed: aligned cosine starves LR at end; inverted angle = raise peak LR |
| askeladd | #1743 | `surf_weight=5` (opposite direction) | Loss weighting | WIP | surf=30 closed (+3.6% worse); test if Huber β=0.5 has shifted optimum below 10 |
| edward | #1763 | torch.compile (attack throughput bottleneck) | Throughput | WIP | EMA=0.9995 closed (+41 MAE — half-life too long for budget); pivot to throughput |
| fern | #1805 | Adaptive Huber β annealing (β=1.0 → β=0.5 over epochs 1-10) | Loss shape / schedule | WIP | β sweep bracketed (β=0.25 closed +9.3%, β=1.0 closed); anneal β for best of both regimes |
| frieren | #1792 | n_layers=3 (shallower) | Architecture (depth, throughput angle) | WIP | #1442 v2 n_hidden=192 closed: 4/4 capacity-up regress; testing capacity-down for throughput gain |
| nezuko | #1806 | LR warmup 2 epochs (extend to test more cold-start EMA compression) | LR schedule | WIP | #1672 warmup 1ep MERGED (new best −0.96%); extend warmup to see if EMA catch-up gain scales |
| tanjiro | #1784 | max_norm=10 (true safety-net threshold above 70–140 peak norms) | Gradient stability | WIP | grad-clip=1.0 v2 closed: 100% clip rate = direction normalization, OOD-helps/IID-hurts |
| thorfinn | #1783 | Lookahead optimizer (k=5 inner / α=0.5 outer) | Optimizer / trajectory averaging | WIP | dropout 0.1/0.05 both regress on β=0.5; monotonicity violation rules out tuning |

**Critical baseline note**: All PRs must now beat `val_avg/mae_surf_p < 85.0926` (PR #1672 warmup 1ep, test=75.5171, W&B 1hn6ur4l). PRs that only beat the prior β=0.5 baseline (85.92) but not the current baseline will be sent back for retest.

## Closed hypotheses (all rounds)

### Loss / feature engineering
- **per-channel surface weights (0.5, 0.5, 2.0)** (#1445 v2, nezuko) — val=93.60 (+1.4% worse). p already dominates gradient signal; re-weighting backfired, U down-weighting removed geometric regularization.
- **SiLU activation** (#1648, edward) — val=96.99 (+5.0% worse). "Smoother but slower" trajectory; GELU/lr=5e-4 is well-tuned for this regime.
- **surf_weight=30 on β=0.5 baseline** (#1427 v2, askeladd) — val=88.99 (+3.6% worse). Huber β=0.5 already does the MAE-alignment work surf_weight was reaching for; over-weighting amplifies gradient variance (EMA-vs-live gap widens from −10.5 to −22).

### Regularization / noise on β=0.5 stack
- **Dropout=0.1 then 0.05 on β=0.5 baseline** (#1629 v2/v3, thorfinn) — val=87.61 (p=0.1) then 87.91 (p=0.05); both +2% worse. Monotonicity violation (p=0.05 worse than p=0.1) rules out tuning. β=0.5 sharpens loss curvature in small-residual regime; per-step Bernoulli noise becomes coordinate-wise gradient corruption, not regularization. EMA half-life 1.85 ep insufficient to wash out.
- **Gradient clipping max_norm=1.0 on β=0.5 baseline** (#1534 v2, tanjiro) — val=87.27 (+1.6% worse). 6375/6375 steps clipped (100%) at peak norm 140 → effectively normalized SGD (direction-only). Clean OOD-helps (camber_cruise/re_rand) / IID-hurts (in_dist/camber_rc) split — flatter loss-landscape traversal at IID cost. Different mechanism than originally hypothesized; new attempt with max_norm=10 (rare-spike safety net only).

**Pattern**: 3 of 3 noise/regularization mechanisms (surf_weight=30, dropout, grad-clip 1.0) that helped or were neutral on the old MSE/Huber-β=1.0 stack now regress on the β=0.5 stack. Loss-shape sharpening from β=0.5 has tightened the optimization neighborhood; mechanisms that perturb gradient direction or per-step gradients interfere with the finer adjustment.

### LR warmup
- **Warmup 1 epoch** (#1672 v2, nezuko) — val=85.09 (−0.96% vs β=0.5 baseline). MERGED. All 4 splits improved; re_rand best (−1.41). Mechanism: post-warmup EMA catch-up phase compressed (epoch-4 EMA-live gap −26 MAE vs baseline). T_max confounder (~6% higher late LR) still present; 2-epoch warmup under test.

### Loss shape / Huber β
- **Huber β=0.5** (#1689, fern) — val=85.92 (−6.96% vs β=1.0 baseline). MERGED. All 4 splits improved; largest gains on hardest splits (in_dist −7.6%, camber_rc −7.0%). Mechanism: β=0.5 moves L1 gradient into the moderate-error bulk where loss density lives, directly aligning with MAE metric.
- **Huber β=0.25** (#1705, fern) — val=93.92 (+9.31% vs β=0.5 baseline). β sweep bracketed: β=0.25 (worse), β=0.5 (BEST), β=1.0 (worse). Mechanism: quadratic region |x| < 0.25 too small for moderate errors; constant L1 gradient is too slow. in_dist hurt most (+17.81%), cruise least (+4.14%); consistent with error-distribution explanation. Adaptive β schedule (1.0→0.5 anneal) under test in #1805.

### Training efficiency
- **EMA without diagnostic pass** (#1626, fern) — val=92.46 (+0.12 within noise). Diagnostic overhead was ~8 s/epoch not the predicted ~25 s; +1 epoch in budget (18 vs 17) insufficient to escape noise. Bottleneck is training step, not val. Useful intel: peak mem 32.9 GB / 96 GB.

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
4. **All three stack orthogonally** — compounding from val~160 to val=92.35 confirms each lever is mostly independent. The remaining headroom from these three stacked should be explored before declaring a local minimum.
5. **Gradient norms are massive without clipping** — 5250/5250 steps clipped at max_norm=1.0, max norm 837. Acts as full gradient normalization. Tanjiro's retest on bf16 will show whether smoother trajectory compounds with the existing stack.
6. **Per-epoch throughput is king** — any lever that doesn't speed up wall-clock per-epoch or improve sample-efficiency struggles. Architecture levers (wider, deeper, more slices) face this headwind.
7. **val_single_in_dist is hardest** (~112-175 MAE across runs). OOD camber_cruise is easiest (58-87 MAE). In_dist being hardest is likely extreme-Re / extreme-p samples in the in-distribution set, not an overfitting artifact.

## Potential next directions (post current round)

- **Huber β=0.25** (assigned to fern) — continue pushing toward pure L1; EMA should buffer kink noise
- **Annealing β over training** — start β=1.0 (stable early when errors are large) → decay to β=0.25 (MAE-aligned late)
- **torch.compile** — throughput angle; ~1.2–1.5× speedup if compiles cleanly with bf16
- **Re-conditioning** — explicit Re-aware embeddings or log-Re positional encoding (re_rand split still underperforms)
- **Surface-aware decoder / dual-head** — separate volume and surface heads
- **Spectral / Fourier neural operator hybrids** — fresh architecture direction if attention-based plateau
- **Test-time augmentation** using physical symmetries (mirroring flow domain)
- **Lookahead optimizer** — complementary to EMA at optimizer level
