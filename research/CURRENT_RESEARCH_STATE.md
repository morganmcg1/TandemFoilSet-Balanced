# SENPAI Research State

- **As of:** 2026-05-13 ~09:30 UTC
- **Track:** `willow-pai2g-24h-r4` (round 30 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best (3-seed mean, paper-publishable):** val = **65.35 ± 3.37**, test = **56.68 ± 2.66** (PR #1379 — Smooth-L1 β=0.5, 3-seed). Best seed: val=62.3972, test=54.4758 (seed=1, W&B `mqf224bq`).

## Current research focus

### pct_start bracket {0.05, 0.10, 0.15} — critical paper figure

The most important pending figure is the 3-seed multi-seed sweep of the OneCycleLR warmup fraction. The mechanism is confirmed (pct_start=0.05 extends the decay tail, improving OOD while regressing in-dist), but the magnitude was overstated by a lucky single seed.

**Important caveat**: thorfinn #1944's pct_start=0.10 3-seed result was on the MSE stack (not the current Smooth-L1 β=0.5 baseline). The bracket figure now has two layers:

**MSE-stack bracket (legacy figure):**
| pct_start | Seeds | val_avg | test_avg | Source |
|---:|---:|---:|---:|---:|
| 0.05 | 3 | 68.88 ± 2.40 | 59.61 ± 2.36 | #1874 |
| **0.10** | **3** | **67.339 ± 1.98** | **58.314 ± 1.76** | **#1944 (sent back, but data valid)** |
| 0.15 | 1 | 67.35 | 58.07 | #1768 (closed) |
| 0.15 | 3 | TBD | TBD | WIP — nezuko #2046 |

**MSE-stack finding (paper-relevant):** pct_start=0.10 is a clean Pareto improvement over 0.05 — ALL 4 val and 4 test splits improve, std SHRINKS, in-dist regression eliminated, OOD gain preserved.

**Smooth-L1 β=0.5 stack bracket (current baseline figure):**
| pct_start | Seeds | val_avg | test_avg | Source |
|---:|---:|---:|---:|---:|
| 0.05 (current baseline) | 3 | 65.35 ± 3.37 | 56.68 ± 2.66 | #1379 |
| 0.10 | 3 | TBD | TBD | thorfinn re-running #1944 on new stack |
| 0.15 | 3 | TBD | TBD | nezuko #2046 (forked before β=0.5 — may also need re-run) |

The high-value compose-able question: does β=0.5 + pct_start=0.10 stack additively? β=0.5 = gradient fairness across residual magnitudes; pct_start=0.10 = LR-schedule Pareto warmup/decay balance — orthogonal mechanisms, expected ~1-2 mae units of additional gain over β=0.5 + pct=0.05.

### OOD-asymmetric regularization — CLOSED (weight decay axis)

The weight_decay axis is exhausted. Both wd=5e-4 (#1860) and wd=2e-5 (#1916) regressed vs wd=1e-4. The basin is confirmed: wd=1e-4 is the optimum for this workload. **Key paper finding:** standard "regularization → better OOD" prior is INVERTED here — wd=5e-4 regressed OOD splits monotonically more than in-dist, suggesting richer features are needed for OOD extrapolation.

**New regularization angle (model stochastic):** fern #2126 now testing dropout=0.1 in Transolver attention and MLP layers. Stochastic regularization is mechanistically different from weight decay — it spreads predictive load across redundant paths rather than uniformly shrinking parameters. May be the right lever for OOD specifically.

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1785 | OneCycleLR max_lr=2e-3 (LR ceiling probe) | WIP |
| askeladd | #2037 | Smooth-L1 β=0.25 — downward bracket from β=0.5, 3-seed | WIP |
| edward | #2039 | AdamW β2=0.95 — faster 2nd-moment adaptation, 3-seed | WIP |
| fern | #2126 | Transolver dropout=0.1 — stochastic OOD regularization, 3-seed | WIP |
| frieren | #2002 | OneCycleLR anneal_strategy=cos → linear — cosine-tail shape probe, 3-seed | WIP |
| nezuko | #2046 | OneCycleLR pct_start=0.15 — 3-seed bracket completion (paper figure) | WIP |
| tanjiro | #1965 | batch_size=4 → 8 (gradient quality vs step-count probe, 3-seed) | WIP |
| thorfinn | #1944 | OneCycleLR pct_start=0.10 (sent back, re-running on Smooth-L1 β=0.5 stack — Pareto win on MSE confirmed) | WIP |

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
17. **Weight decay basin confirmed at wd=1e-4** (nezuko #1916 closed) — wd=2e-5 (5× lower) also regresses vs baseline. Both wd directions explored (5e-4 = too much regularization; 2e-5 = too little). wd=1e-4 is the optimum; weight decay axis is exhausted.
18. **pct_start=0.10 is the Pareto point on MSE stack** (thorfinn #1944 sent back, 3-seed) — 67.34 ± 1.98 val / 58.31 ± 1.76 test vs 0.05's 68.88 ± 2.40 / 59.61 ± 2.36. **ALL 4 val splits AND all 4 test splits improve simultaneously**, std SHRINKS in both metrics, in-dist regression at 0.05 is fully eliminated, OOD gain preserved. Result needs re-run on Smooth-L1 β=0.5 stack to validate as new baseline — β=0.5 + pct_start=0.10 is the highest-value compose-able experiment available.
19. **surf_weight=10 is a confirmed basin** (fern #2003 closed) — +50% step (surf_weight=15) is within noise of MSE baseline, mildly worse on all splits. Combined with #1390 (surf_weight=25, +20.7% catastrophic regression), the surf_weight attack class is exhausted. Both directions explored — 10 is the joint optimum for this shared-backbone + max_lr=1.5e-3 stack. Changing surf_weight changes effective surface gradient LR; any deviation from 10 either destabilizes (too high) or under-emphasizes (too low).

## Potential next research directions

### Immediate (waiting on current PRs)
- **pct_start=0.10 on Smooth-L1 β=0.5 stack** (thorfinn #1944 re-running) — Pareto win on MSE stack confirmed; rerun on current baseline tests compose-ability of pct_start=0.10 + β=0.5.
- **pct_start=0.15 multi-seed** (nezuko #2046) — bracket completion; 3-seed confirmation of single-seed value.
- **max_lr=2e-3** (alphonse #1785) — LR ceiling probe.
- **batch_size=4 → 8** (tanjiro #1965, 3-seed) — gradient quality vs per-batch-step-count probe; uses unused GPU (~46GB / 96GB).
- **Smooth-L1 β=0.25** (askeladd #2037) — downward bracket from β=0.5.
- **AdamW β2=0.95** (edward #2039) — faster 2nd-moment adaptation.
- **Transolver dropout=0.1** (fern #2126) — stochastic OOD regularization, first model-level regularization probe.
- **anneal_strategy=linear** (frieren #2002) — alternative cosine-tail shape probe.

### Short-term (round 31+)
- **Smooth-L1 β bracket** — once β=0.25 returns, decide whether to go lower (β=0.1) or higher (β=0.75). The goal is finding the optimal gradient-fairness point.
- **SDPA flash backend** — explicit `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)` context manager.
- **pct_start bracket paper figure** — once 0.05/0.10/0.15 are all 3-seed confirmed, this is a clean mechanistic figure: OOD gain vs in-dist cost as a function of warmup fraction.

### Architecture and signal (longer term)
- SwiGLU MLP — swap GELU FF layers.
- Per-domain LR — higher LR for OOD splits (re_rand/cruise) vs in-dist.
- Explicit signed-distance-to-surface as positional feature.
- Loss: relative-MAE for pressure (handles dynamic range across Re).
- Test-time augmentation: average predictions over mirrored geometry.
- Geometric data augmentation (rotation/scaling/flipping) for OOD-camber generalization.
