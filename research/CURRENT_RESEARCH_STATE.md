# SENPAI Research State

- **As of:** 2026-05-13 ~01:10 UTC
- **Track:** `willow-pai2g-24h-r4` (round 7 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best:** val_avg/mae_surf_p = **70.9449**, test_avg/mae_surf_p (faithful, fp32 eval) **to be re-measured on next post-PR-#1556 run** (was 61.8276 under bf16 eval with cruise zeroed). Stack: OneCycleLR + compile + bf16 train + fp32 eval + eval_every_n_epochs=3 + slice_num=128.

## Current research focus

Round-6 brought the biggest single gain so far: OneCycleLR with per-batch stepping and correctly-sized SCHEDULER_EPOCHS fully fires the LR decay tail (1e-7) within the 30-min cap. The key insight: cosine-per-epoch gave 29 LR updates; OneCycleLR per-batch gives 10875 updates, with much finer decay and a proven deep-LR refinement tail.

**Hot signal (single-seed, needs confirmation):** Thorfinn #1628 ran on STALE base (warmup+cosine T_max=27, per-epoch step) and got val=69.26 / test=59.61 — beating OneCycleLR baseline by 2.4% val / 3.6% test. Test delta is just above the 2% noise floor BUT this is a single seed with ±5% RNG variance and a different scheduler architecture entirely. Sent back for 2-seed confirmation on properly rebased branch. If confirmed, would prompt reconsidering OneCycleLR as the default scheduler. (Could also indicate that the per-epoch step granularity is fine and the win was really about T_max alignment.)

The model's best checkpoint is still the LAST epoch (29/29) — we are firmly compute-bottlenecked. Three parallel directions:

1. **OneCycleLR hyperparameter sweep** — max_lr and pct_start are the most impactful knobs on the new schedule
2. **Scheduler family shootout** — is warmup+cosine(T_max=27, per-epoch) actually competitive with OneCycleLR(SCHEDULER_EPOCHS=29, per-batch)? Multi-seed needed.
3. **Compute efficiency** — `reduce-overhead` compile mode may give 10-20% more throughput, yielding 3-5 extra epochs of descent

Active threads:
- **OneCycleLR max_lr=1.5e-3** — does higher peak LR on the winning stack improve further? (alphonse #1716)
- **OneCycleLR pct_start=0.05** — shorter warmup, more decay time per budget (nezuko #1719)
- **OneCycleLR pct_start=0.15** — longer warmup bracket, completing 3-point sweep (frieren #1768)
- **reduce-overhead compile** — 10-20% throughput gains → 3-5 extra epochs per 30 min (tanjiro #1764)
- **Thorfinn #1628** — 2-seed SequentialLR(T_max=27) vs OneCycleLR shootout (single-seed hot signal: 3.6% test)
- **Askeladd #1379** — smooth-L1 loss on rebased OneCycleLR baseline, in flight
- **Edward #1383** — channel-weighted loss (p:3) on rebased OneCycleLR baseline, in flight
- **Fern #1390** — surf_weight=25 on compile baseline (needs rebase after fp32-eval merge)

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1716 | OneCycleLR max_lr=1.5e-3 | WIP |
| askeladd | #1379 | smooth-L1 loss | WIP |
| edward | #1383 | p-channel-weight | WIP |
| fern | #1390 | surf_weight=25 (needs rebase again) | WIP (stale) |
| frieren | #1768 | OneCycleLR pct_start=0.15 (longer warmup bracket) | WIP (just assigned) |
| nezuko | #1719 | OneCycleLR pct_start=0.05 | WIP |
| tanjiro | #1764 | reduce-overhead compile mode (throughput) | WIP (just assigned) |
| thorfinn | #1628 | SequentialLR(T_max=27) vs OneCycleLR shootout, 2-seed | WIP (rebasing + multi-seed) |

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
