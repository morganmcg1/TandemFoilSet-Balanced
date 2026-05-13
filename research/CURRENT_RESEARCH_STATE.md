# SENPAI Research State

- **As of:** 2026-05-13 ~00:30 UTC
- **Track:** `willow-pai2g-24h-r4` (round 6 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best:** val_avg/mae_surf_p = **70.9449**, test_avg/mae_surf_p = **61.8276** (PR #1404, OneCycleLR max_lr=1e-3, SCHEDULER_EPOCHS=29, per-batch stepping, on compile+bf16+slice128)

## Current research focus

Round-6 brought the biggest single gain so far: OneCycleLR with per-batch stepping and correctly-sized SCHEDULER_EPOCHS fully fires the LR decay tail (1e-7) within the 30-min cap. The key insight: cosine-per-epoch gave 29 LR updates; OneCycleLR per-batch gives 10875 updates, with much finer decay and a proven deep-LR refinement tail.

The model's best checkpoint is still the LAST epoch (29/29) — we are firmly compute-bottlenecked. Two parallel directions:

1. **OneCycleLR hyperparameter sweep** — max_lr and pct_start are the most impactful knobs on the new schedule
2. **Compute efficiency** — `reduce-overhead` compile mode may give 10-20% more throughput, yielding 3-5 extra epochs of descent

Active threads:
- **OneCycleLR max_lr=1.5e-3** — does higher peak LR on the winning stack improve further? (alphonse #1716)
- **OneCycleLR pct_start=0.05** — shorter warmup, more decay time per budget (nezuko #1719)
- **Thorfinn #1628** — T_max=30 run in flight (cosine stack, superseded by OneCycleLR, but run already started — closing and redirecting after results posted)
- **Askeladd #1379** — smooth-L1 loss on warmup+cosine baseline, in flight. If it beats 75.85, directional signal; won't beat 70.94.
- **Edward #1383** — channel-weighted loss (p:3) on warmup+cosine, in flight. Same.
- **Tanjiro #1522** — hidden-192 on compile+bf16, in flight. Same.
- **Frieren #1556** — fp32-eval every 3 epochs. Not metric-improving; establishing faithful test_avg.
- **Fern #1390** — surf_weight=25 on compile baseline (needs rebase again after OneCycleLR merge).

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1716 | OneCycleLR max_lr=1.5e-3 | WIP (just assigned) |
| askeladd | #1379 | smooth-L1 loss (warmup+cosine baseline) | WIP, run in flight |
| edward | #1383 | p-channel-weight (warmup+cosine baseline) | WIP, run in flight |
| fern | #1390 | surf_weight=25 (needs rebase again) | WIP (stale) |
| frieren | #1556 | fp32-eval every 3 epochs | WIP, run in flight |
| nezuko | #1719 | OneCycleLR pct_start=0.05 | WIP (just assigned) |
| tanjiro | #1522 | hidden-192 on compile+bf16 (warmup+cosine) | WIP, run in flight |
| thorfinn | #1628 | T_max=30 (cosine — superseded, run in flight) | WIP, will close after result |

## Key learnings so far

1. **Compute is the bottleneck** — model still descending at epoch 29. Every extra epoch of compute pays off.
2. **torch.compile is the biggest infrastructure lever** — 1.58× throughput. Stack with everything.
3. **OneCycleLR per-batch beats warmup+cosine per-epoch by −6.5% val / −8.1% test** — largest single gain. Key: 10875 per-batch LR updates vs 29 per-epoch; full decay tail fires in budget.
4. **LR schedule shape matters more than LR magnitude** — Moving from warmup+cosine to OneCycleLR (same max_lr=1e-3) gave 6.5% improvement; moving from lr=5e-4 to lr=1e-3 gave only 0.76%.
5. **bf16 eval causes cruise overflow** — nan_to_num zeros it (biased low). fp32 eval needed for faithful paper test_avg.
6. **Hidden-192 directional signal** — pending on compile+bf16 stack with full epoch budget.
7. **RNG variance ≈ ±5%** — sub-1% val deltas need multi-seed confirmation; 2%+ test changes are reliable.

## Potential next research directions

### Immediate (waiting on current PRs)
- OneCycleLR max_lr sweep: 1.5e-3 (alphonse #1716) — is 1e-3 the ceiling?
- OneCycleLR pct_start=0.05 (nezuko #1719) — shorter warmup, more decay time
- Results from askeladd/edward/tanjiro — directional signal on loss/architecture on old stack
- fp32 eval (frieren #1556) — paper-faithful test_avg

### Short-term (round 7)
- **reduce-overhead compile** — CUDA graph fused kernels may give 10-20% more throughput → 3-5 extra epochs
- **OneCycleLR max_lr=2e-3** — if 1.5e-3 is still improving, push further
- **OneCycleLR pct_start=0.15** — complement to pct_start=0.05 bracket
- **Smooth-L1 on OneCycleLR baseline** — if askeladd shows directional signal on old stack
- **Hidden-192 on OneCycleLR** — if tanjiro shows width helps, re-test on new default
- **Channel-weighted loss on OneCycleLR** — if edward shows directional signal
- **surf_weight=25 on OneCycleLR** — fern still pending proper rebase

### Architecture and signal (longer term)
- SwiGLU MLP — swap GELU FF layers
- Per-domain LR — higher LR for OOD splits (re_rand/cruise) vs in-dist
- Explicit signed-distance-to-surface as positional feature
- Loss: relative-MAE for pressure (handles dynamic range across Re)
- Test-time augmentation: average predictions over mirrored geometry
