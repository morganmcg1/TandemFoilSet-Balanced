# SENPAI Research State

- **As of:** 2026-05-13 ~05:30 UTC
- **Track:** `willow-pai2g-24h-r4` (round 20 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best (single-run):** val_avg/mae_surf_p = **66.1352**, test_avg/mae_surf_p = **56.8971** (seed=0, W&B `vfkbmgnp`). **Caution: this is −1.15σ lucky tail.**
- **Current best (3-seed mean, paper-publishable):** val = **68.88 ± 2.40**, test = **59.61 ± 2.36**. Stack: OneCycleLR(max_lr=1.5e-3, pct_start=0.05) + compile + bf16 train + fp32 eval + eval_every_n_epochs=3 + slice_num=128.

## Current research focus

### ⚠️ Major Baseline Reframing (Round 20 — PR #1874 merged)

Thorfinn's 2-seed confirmation (PR #1874) **refutes the headline framing of #1719**. The +5.72% test gain was from seed=0 sitting at the −1.15σ lucky tail. 3-seed reality:
- **Mean test gain vs #1716: only −1.23%** (below 2% refutation threshold)
- **Real OOD/in-dist trade-off:** OOD splits improve ~2-3.5% in mean; in-dist **regresses** +3.8% test / +4.5% val
- **val_geom_camber_cruise −8.2% claim was actually ~−3.3% in 3-seed mean**
- **Standing rule now established:** any single-seed win below 6% requires ≥2 seeds before paper framing

The pct_start=0.05 mechanism (deep-decay tail extension → OOD refinement) is real but smaller than the single-seed number suggested. The in-dist regression reveals that 0.05 may be too aggressive a warmup reduction. **The pct_start bracket {0.05, 0.10, 0.15} is now the critical paper figure to complete.**

### OOD-asymmetric regularization (from #1860)

Round 18's wd=5e-4 finding: OOD splits regress MONOTONICALLY MORE than in-dist (+11.7% cruise vs +3.8% in-dist). The standard "regularization → better OOD" prior is inverted on this workload. Nezuko #1916 (wd=2e-5, symmetric test) in progress.

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1785 | OneCycleLR max_lr=2e-3 (LR ceiling probe) | WIP |
| askeladd | #1379 | Smooth-L1 β=0.5 (gradient-fairness mechanism probe, rebased β=1.0 sent back) | WIP |
| edward | #1809 | surface-only p-weight=2 (targeted vs #1383) | WIP |
| fern | #1390 | surf_weight=25 (needs rebase) | WIP (stale) |
| frieren | #1768 | OneCycleLR pct_start=0.15 (bracket completion) | WIP |
| nezuko | #1916 | weight_decay=2e-5 (5× LOWER, tests OOD-asymmetry hypothesis) | WIP |
| tanjiro | #1861 | OneCycleLR final_div_factor=1e4 (deep-LR floor) | WIP |
| thorfinn | — | OneCycleLR pct_start=0.10 (3-seed, bracket midpoint + in-dist regression test) | NEW |

## Key learnings so far

1. **Compute is the bottleneck** — model still descending at epoch 29. Every extra epoch of compute pays off.
2. **torch.compile is the biggest infrastructure lever** — 1.58× throughput. Stack with everything.
3. **OneCycleLR per-batch beats warmup+cosine per-epoch by −6.5% val / −8.1% test** — largest single gain. Key: 10875 per-batch LR updates vs 29 per-epoch; full decay tail fires in budget.
4. **LR schedule shape matters more than LR magnitude** — Moving from warmup+cosine to OneCycleLR (same max_lr=1e-3) gave 6.5% improvement; moving from lr=5e-4 to lr=1e-3 gave only 0.76%.
5. **bf16 eval causes cruise overflow** — nan_to_num zeros it (biased low). fp32 eval needed for faithful paper test_avg. FIXED in PR #1556 (eval_every_n_epochs=3 gate recovers wall-clock).
6. **Hidden-192 bottlenecked by epoch count** — 1.47× slower per epoch costs 9 epochs → truncated OneCycleLR decay → all 4 splits regress. Capacity-via-width dominated by capacity-via-more-epochs under 30-min cap.
7. **Single-seed warmup+cosine(T_max=27) hot signal** — val 69.26 / test 59.61 beats OneCycleLR baseline by 3.6% test on one seed. Multi-seed confirmation in progress (thorfinn #1628).
8. **RNG variance ≈ ±5% on val; ±4% on test (CV)** — sub-6% single-seed wins need multi-seed confirmation; 2%+ test MEAN changes are reliable.
9. **Python dispatch is NOT the throughput bottleneck** (tanjiro #1764 closed) — `reduce-overhead` CUDAGraphs cost more than they save under `dynamic=True` mesh.
10. **Channel-weight loss must be surface-only** (edward #1383 closed) — flat multipliers on (Ux, Uy, p) reshape gradient balance but the headline metric is in physical space and surface-only.
11. **max_lr and pct_start address orthogonal failure modes** (round 10) — max_lr accelerates in-dist basin descent; pct_start extends deep-decay tail. But the pct_start gain is SMALLER than single-seed suggested (see #15).
12. **Compile overhead must be amortized against achievable epoch count** (tanjiro #1807 closed) — `max-autotune-no-cudagraphs` delivers +5.5% per-epoch but 180s compile cost eats 2 epochs; if SCHEDULER_EPOCHS is hardcoded, the schedule tail never fires.
13. **Per-batch LR scheduling is structurally superior to per-epoch on this workload** (thorfinn #1628 closed) — SequentialLR(T_max=27) per-epoch is +3-4% WORSE vs NEW pct_start=0.05 baseline.
14. **The standard "regularization → better OOD" prior is INVERTED (paper-worthy)** (nezuko #1860 closed) — wd=5e-4 regresses ALL splits, regression monotonic with OOD distance: in-dist +3.78%, camber-rc +7.28%, camber-cruise +11.68%, re-rand +10.37%. Mechanism: OOD extrapolation needs richer features; uniform shrinkage destroys OOD features first.
15. **NEW (paper-critical): Single-seed wins below ~6% need multi-seed confirmation** (thorfinn #1874 merged) — pct_start=0.05 seed=0 was at −1.15σ lucky tail. 3-seed mean gain vs #1716 is only −1.23% (below noise threshold). Real effect: OOD splits improve ~2-3% in mean; in-dist REGRESSES ~4%. The pct_start=0.05 vs 0.10 trade-off is now the paper's key schedule question.
16. **Smooth-L1 β=1.0 ≈ MSE at convergence on this stack** (askeladd #1379 sent back) — bulk normalized residuals at convergence satisfy |err| < 1.0, keeping the loss in the quadratic regime. β=0.5 will be the first real test of gradient-fairness.

## Potential next research directions

### Immediate (waiting on current PRs)
- **pct_start=0.10 multi-seed** (thorfinn NEW) — fills bracket midpoint; tests whether 0.10 avoids the in-dist regression seen at 0.05 while retaining some OOD gain.
- **pct_start=0.15** (frieren #1768) — bracket completion.
- **max_lr=2e-3** (alphonse #1785) — LR ceiling probe.
- **final_div_factor=1e4** (tanjiro #1861) — deeper LR floor for OOD refinement.
- **wd=2e-5** (nezuko #1916) — symmetric test of OOD-asymmetry.
- **Smooth-L1 β=0.5** (askeladd #1379) — first real gradient-fairness probe.
- **surface-only p-weight=2** (edward #1809).

### Short-term (round 21+)
- **batch_size=8 sweep** — current peak GPU ~49GB, 96GB available. More gradient updates per epoch may compound with OOD-tail mechanism.
- **SDPA flash backend** — explicit `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)` context manager.
- **anneal_strategy=linear** — alternative to cosine for deep-decay tail extension.
- **pct_start bracket paper figure** — once 0.05/0.10/0.15 are all 3-seed confirmed, this is a clean mechanistic figure: OOD gain vs in-dist cost as a function of warmup fraction.

### Architecture and signal (longer term)
- SwiGLU MLP — swap GELU FF layers.
- Per-domain LR — higher LR for OOD splits (re_rand/cruise) vs in-dist.
- Explicit signed-distance-to-surface as positional feature.
- Loss: relative-MAE for pressure (handles dynamic range across Re).
- Test-time augmentation: average predictions over mirrored geometry.
- Geometric data augmentation (rotation/scaling/flipping) for OOD-camber generalization.
