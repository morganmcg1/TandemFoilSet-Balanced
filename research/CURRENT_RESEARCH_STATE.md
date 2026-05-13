# SENPAI Research State

- **As of:** 2026-05-13 ~03:25 UTC
- **Track:** `willow-pai2g-24h-r4` (round 11 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best:** val_avg/mae_surf_p = **66.1352**, test_avg/mae_surf_p = **56.8971** (fp32 eval, paper-faithful). Stack: OneCycleLR(max_lr=1.5e-3, **pct_start=0.05**) + compile + bf16 train + fp32 eval + eval_every_n_epochs=3 + slice_num=128.

## Current research focus

Round 9 merged nezuko #1719 (pct_start=0.05): **−3.57% val / −5.72% test** over the alphonse #1716 baseline. The test gain exceeds the val gain (5.72% vs 3.57%) — strong generalization signal. The per-split decomposition revealed a clean mechanistic split:

- `max_lr=1.5e-3` (alphonse #1716) saturated `val_single_in_dist` (−8.8%); pct_start adds only −0.6% there.
- `pct_start=0.05` (nezuko #1719) unlocked **val_geom_camber_cruise −8.2%** — the biggest single-split move on the most-stuck OOD split, all from pct_start.

**Mechanism:** the two knobs are orthogonal. max_lr accelerates the in-dist basin descent; pct_start extends the deep-decay tail (~8 extra epochs at LR < 1e-4) which is where OOD-camber refinement happens. This refutes round-9's noise-floor read on pct_start alone — single-knob result against OLD baseline was noise because the in-dist basin wasn't reached.

**Round 10 closures:**
- **tanjiro #1807** (max-autotune-no-cudagraphs) CLOSED — split outcome. Kernel autotuning delivered +5.5% per-epoch as predicted, 36 Triton picks, no CUDAGraph warnings. But 180s compile cost ate 2 epochs (27/29) → SCHEDULER_EPOCHS=29 hardcoded → final LR ~8.5e-5 instead of 1.5e-6 → model under-trained → val +3.0% worse. Compile-mode attack class is now closed under the 30-min cap.

**Round 11 closures:**
- **thorfinn #1628** (SequentialLR T_max=27, 2-seed) CLOSED — refuted under multi-seed scrutiny. 2-seed mean within ±2% of OLD baseline (noise), but +3.49% val / +4.39% test WORSE vs NEW baseline #1719. Per-epoch scheduling has a structural granularity ceiling that per-batch OneCycleLR doesn't — can't reach the deep-decay refinement density that pct_start=0.05 exploits.

**Round 11 sends-back:**
- **frieren #1768** (pct_start=0.15) — student asked for direction. Approved option 1: rebase onto new baseline, complete the {0.05 ✓, 0.10, 0.15} bracket. Awaiting rebased run.

**Round 10 merges:**
- **nezuko #1719** (pct_start=0.05 composition) MERGED — new baseline.

**Active research questions:**
1. **LR ceiling beyond 1.5e-3** — alphonse #1785 testing max_lr=2e-3 vs the new baseline.
2. **OOD-tail refinement** — nezuko's win opens up the OOD lever for the first time. If pct_start=0.05 helps OOD via deep-LR refinement, deeper LR floor (final_div_factor 1e3 → 1e4) should extend the mechanism further. tanjiro #1861.
3. **OOD regularization** — wd=1e-4 may be undertuned given the larger schedule-shape gains. nezuko #1860 sweeps weight_decay=5e-4.
4. **pct_start=0.15 bracket completion** — frieren #1768 rebased onto new baseline; runs 0.15 vs the 0.05 winner for the paper-figure 3-point sweep.
5. **NEW baseline confirmation (2-seed)** — thorfinn #1874 runs seed=1 and seed=2 on pct_start=0.05 baseline. Single-seed +5.72% test gain is at the high end of RNG variance and must be confirmed before stacking further compositions.
6. **Loss formulation** — surface-only p-weighting (edward #1809) and smooth-L1 (askeladd #1379) still pending.
7. **Throughput attacks under cap closed** — Python dispatch (#1764) and static-kernel autotuning (#1807) both refuted. Remaining angles: SDPA flash backend, mesh-layout caching, batch_size sweep, hidden_dim sweep at constant compute.

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1785 | OneCycleLR max_lr=2e-3 (LR ceiling probe) | WIP |
| askeladd | #1379 | smooth-L1 loss | WIP (long-running) |
| edward | #1809 | surface-only p-weight=2 (targeted vs #1383) | WIP |
| fern | #1390 | surf_weight=25 (needs rebase) | WIP (stale) |
| frieren | #1768 | OneCycleLR pct_start=0.15 (bracket completion) | WIP (rebasing onto new baseline) |
| nezuko | #1860 | weight_decay=5e-4 (OOD regularization) | WIP |
| tanjiro | #1861 | OneCycleLR final_div_factor=1e4 (deep-LR floor) | WIP |
| thorfinn | #1874 | 2-seed confirmation of pct_start=0.05 baseline (seed=1, seed=2) | WIP (just assigned) |

## Key learnings so far

1. **Compute is the bottleneck** — model still descending at epoch 29. Every extra epoch of compute pays off.
2. **torch.compile is the biggest infrastructure lever** — 1.58× throughput. Stack with everything.
3. **OneCycleLR per-batch beats warmup+cosine per-epoch by −6.5% val / −8.1% test** — largest single gain. Key: 10875 per-batch LR updates vs 29 per-epoch; full decay tail fires in budget.
4. **LR schedule shape matters more than LR magnitude** — Moving from warmup+cosine to OneCycleLR (same max_lr=1e-3) gave 6.5% improvement; moving from lr=5e-4 to lr=1e-3 gave only 0.76%.
5. **bf16 eval causes cruise overflow** — nan_to_num zeros it (biased low). fp32 eval needed for faithful paper test_avg. FIXED in PR #1556 (eval_every_n_epochs=3 gate recovers wall-clock).
6. **Hidden-192 bottlenecked by epoch count** — 1.47× slower per epoch costs 9 epochs → truncated OneCycleLR decay → all 4 splits regress. Capacity-via-width dominated by capacity-via-more-epochs under 30-min cap.
7. **Single-seed warmup+cosine(T_max=27) hot signal** — val 69.26 / test 59.61 beats OneCycleLR baseline by 3.6% test on one seed. Multi-seed confirmation in progress (thorfinn #1628).
8. **RNG variance ≈ ±5%** — sub-1% val deltas need multi-seed confirmation; 2%+ test changes are reliable.
9. **Python dispatch is NOT the throughput bottleneck** (tanjiro #1764 closed) — `reduce-overhead` CUDAGraphs cost more than they save under `dynamic=True` mesh.
10. **Channel-weight loss must be surface-only** (edward #1383 closed) — flat multipliers on (Ux, Uy, p) reshape gradient balance but the headline metric is in physical space and surface-only.
11. **NEW: max_lr and pct_start compose because they hit different failure modes** (round 10) — max_lr accelerates in-dist basin descent (saturated), pct_start extends deep-decay tail (unlocks OOD). Schedule-shape changes can stack mechanistically; the same knob is noise vs the wrong baseline and gold vs the right one.
12. **NEW: compile overhead must be amortized against achievable epoch count, not nominal schedule length** (tanjiro #1807 closed) — `max-autotune-no-cudagraphs` delivers +5.5% per-epoch but 180s compile cost eats 2 epochs; if SCHEDULER_EPOCHS is hardcoded, the schedule tail never fires and metrics regress despite faster steps. Compile-mode attack class is now exhausted under the 30-min cap.
13. **NEW: per-batch LR scheduling is structurally superior to per-epoch on this workload** (thorfinn #1628 closed) — SequentialLR(T_max=27) per-epoch ties OneCycleLR per-batch on the OLD baseline (within ±2% noise) but is +3-4% WORSE vs the NEW pct_start=0.05 baseline. ~362× more LR updates is the mechanism (10875 per-batch vs 30 per-epoch). The original round-1 win from warmup+cosine → OneCycleLR was about update granularity, not schedule shape. **Implication for paper:** OneCycleLR's win is a robust architectural claim, not a single-baseline coincidence.

## Potential next research directions

### Immediate (waiting on current PRs)
- max_lr=2e-3 against new baseline (alphonse #1785) — does the LR ceiling extend past 1.5e-3? Note: the new pct_start=0.05 schedule now reaches peak LR at epoch 1.5; combining with even higher max_lr may either stack or saturate.
- Deep-LR floor: final_div_factor 1e3 → 1e4 (tanjiro #1810) — extends the OOD-refinement mechanism nezuko's win opened up.
- weight_decay=5e-4 (nezuko #1811) — OOD regularization on new baseline.
- pct_start=0.15 ablation (frieren #1768) — third point on the pct_start curve {0.05 ✓, 0.10, 0.15} for paper figure.
- Surface-only p-weight=2 (edward #1809) — targeted version of channel weighting.
- SequentialLR(T_max=27) 2-seed shootout (thorfinn #1628) — paper-tier baseline question.
- Smooth-L1 results (askeladd #1379).

### Short-term (round 11+)
- **batch_size=8 sweep** — current peak GPU ~49GB, 96GB available. More gradient updates per epoch may stack with the OOD-tail mechanism.
- **SDPA flash backend** — explicit `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)` context manager; throughput attack that doesn't go through compile.
- **Per-channel p-weight sweep** — if edward's surface-only p_weight=2 shows directional signal, sweep {1.5, 2, 2.5, 3}.
- **surf_weight=25 on new baseline** — fern still pending proper rebase.
- **anneal_strategy=linear** — alternative to cosine on the new schedule; could extend the deep-LR tail differently than pct_start.
- **OneCycleLR cycle_momentum** — current default is True; toggling may change momentum at deep-LR phase, potentially relevant to OOD refinement.

### Architecture and signal (longer term)
- SwiGLU MLP — swap GELU FF layers.
- Per-domain LR — higher LR for OOD splits (re_rand/cruise) vs in-dist.
- Explicit signed-distance-to-surface as positional feature.
- Loss: relative-MAE for pressure (handles dynamic range across Re).
- Test-time augmentation: average predictions over mirrored geometry.
- Geometric data augmentation (rotation/scaling/flipping) for OOD-camber generalization.
