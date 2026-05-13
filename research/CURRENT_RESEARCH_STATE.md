# SENPAI Research State

- **As of:** 2026-05-13 ~01:30 UTC
- **Track:** `willow-pai2g-24h-r4` (round 8 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best:** val_avg/mae_surf_p = **68.5843**, test_avg/mae_surf_p = **60.3521** (fp32 eval, paper-faithful). Stack: OneCycleLR(max_lr=1.5e-3) + compile + bf16 train + fp32 eval + eval_every_n_epochs=3 + slice_num=128.

## Current research focus

Round-7 merged alphonse #1716 (max_lr=1.5e-3): −3.3% val / −2.4% test over the prior 1e-3 baseline. The in-distribution split dominates the gain (−8.8%), while OOD splits (geom_camber_rc, geom_camber_cruise) are essentially unmoved. This tells us:

1. **LR tuning is approaching ceiling on in-dist** — best epoch moved from 29/29 to 27/29, last 3 epochs within 0.03 of each other. Need to test max_lr=2e-3 to know if there's more headroom.
2. **OOD splits need a different lever** — camber-rc and camber-cruise don't respond to LR at all. They're generalization-bottlenecked, not optimization-bottlenecked. Hypothesis: geometric feature augmentation, domain-adaptive loss weighting, or explicit angle-of-attack normalization.
3. **pct_start sweep** — nezuko (0.05) and frieren (0.15) will bracket the default 0.10. The model's best epoch shifted from last to 27th — more or less warmup time may shift the balance.
4. **Compute efficiency** — tanjiro's reduce-overhead compile mode (10-20% throughput gain) would yield 3-5 extra epochs per 30 min.
5. **Scheduler shootout** — thorfinn's 2-seed SequentialLR(T_max=27) vs OneCycleLR comparison is still in progress.

Active threads:
- **OneCycleLR max_lr=2e-3** — does the LR ceiling extend beyond 1.5e-3? (alphonse, next assignment)
- **OneCycleLR pct_start=0.05** — shorter warmup, more decay time (nezuko #1719)
- **OneCycleLR pct_start=0.15** — longer warmup bracket (frieren #1768)
- **reduce-overhead compile** — throughput → more epochs per budget (tanjiro #1764)
- **Thorfinn #1628** — 2-seed SequentialLR(T_max=27) vs OneCycleLR (hot signal: 3.6% test single-seed)
- **Askeladd #1379** — smooth-L1 loss on rebased OneCycleLR baseline
- **Edward #1383** — channel-weighted loss (p:3) on rebased OneCycleLR baseline
- **Fern #1390** — surf_weight=25 (needs rebase to new baseline)

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1785 | OneCycleLR max_lr=2e-3 (LR ceiling probe) | WIP (just assigned) |
| askeladd | #1379 | smooth-L1 loss | WIP |
| edward | #1383 | p-channel-weight | WIP |
| fern | #1390 | surf_weight=25 (needs rebase) | WIP (stale) |
| frieren | #1768 | OneCycleLR pct_start=0.15 | WIP |
| nezuko | #1719 | OneCycleLR pct_start=0.05 | WIP |
| tanjiro | #1764 | reduce-overhead compile mode (throughput) | WIP |
| thorfinn | #1628 | SequentialLR(T_max=27) vs OneCycleLR, 2-seed | WIP (rebasing) |

## Key learnings so far

1. **Compute is the bottleneck** — model still descending at epoch 29. Every extra epoch of compute pays off.
2. **torch.compile is the biggest infrastructure lever** — 1.58× throughput. Stack with everything.
3. **OneCycleLR per-batch beats warmup+cosine per-epoch by −6.5% val / −8.1% test** — largest single gain. Key: 10875 per-batch LR updates vs 29 per-epoch; full decay tail fires in budget.
4. **LR schedule shape matters more than LR magnitude** — Moving from warmup+cosine to OneCycleLR (same max_lr=1e-3) gave 6.5% improvement; moving from lr=5e-4 to lr=1e-3 gave only 0.76%.
5. **bf16 eval causes cruise overflow** — nan_to_num zeros it (biased low). fp32 eval needed for faithful paper test_avg. FIXED in PR #1556 (eval_every_n_epochs=3 gate recovers wall-clock).
6. **Hidden-192 bottlenecked by epoch count** — 1.47× slower per epoch costs 9 epochs → truncated OneCycleLR decay → all 4 splits regress. Capacity-via-width dominated by capacity-via-more-epochs under 30-min cap.
7. **Single-seed warmup+cosine(T_max=27) hot signal** — val 69.26 / test 59.61 beats OneCycleLR baseline by 3.6% test on one seed. Multi-seed confirmation in progress (thorfinn #1628). Could indicate per-epoch schedule is competitive at aligned T_max.
8. **RNG variance ≈ ±5%** — sub-1% val deltas need multi-seed confirmation; 2%+ test changes are reliable.

## Potential next research directions

### Immediate (waiting on current PRs)
- OneCycleLR max_lr sweep: 1.5e-3 (alphonse #1716) — is 1e-3 the ceiling?
- OneCycleLR pct_start={0.05, 0.10, 0.15} bracket (nezuko #1719, current, frieren #1768) — map warmup trade-off
- reduce-overhead compile (tanjiro #1764) — throughput gain → more epochs per 30 min
- Scheduler shootout 2-seed: SequentialLR(T_max=27) vs OneCycleLR (thorfinn #1628) — hot signal needs confirmation
- Results from askeladd/edward (loss formulation on rebased OneCycleLR baseline)

### Short-term (round 8)
- **OneCycleLR max_lr=2e-3** — if alphonse shows 1.5e-3 still improving, push further
- **pct_start winner → additional tuning** — 3-point bracket informs best warmup fraction
- **Smooth-L1 on OneCycleLR baseline** — if askeladd shows directional signal
- **Channel-weighted loss on OneCycleLR** — if edward shows directional signal
- **surf_weight=25 on OneCycleLR** — fern still pending proper rebase
- **SequentialLR(T_max=27) as baseline candidate** — if thorfinn's multi-seed confirms 3.6% test gain

### Architecture and signal (longer term)
- SwiGLU MLP — swap GELU FF layers
- Per-domain LR — higher LR for OOD splits (re_rand/cruise) vs in-dist
- Explicit signed-distance-to-surface as positional feature
- Loss: relative-MAE for pressure (handles dynamic range across Re)
- Test-time augmentation: average predictions over mirrored geometry
