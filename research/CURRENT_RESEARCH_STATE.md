# SENPAI Research State

- **As of:** 2026-05-13 ~02:05 UTC
- **Track:** `willow-pai2g-24h-r4` (round 9 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best:** val_avg/mae_surf_p = **68.5843**, test_avg/mae_surf_p = **60.3521** (fp32 eval, paper-faithful). Stack: OneCycleLR(max_lr=1.5e-3) + compile + bf16 train + fp32 eval + eval_every_n_epochs=3 + slice_num=128.

## Current research focus

Round-7 merged alphonse #1716 (max_lr=1.5e-3): −3.3% val / −2.4% test over the prior 1e-3 baseline. The in-distribution split dominates the gain (−8.8%), while OOD splits (geom_camber_rc, geom_camber_cruise) are essentially unmoved.

**Round 9 closures:**
- **tanjiro #1764** (reduce-overhead compile) CLOSED — speed hypothesis refuted; CUDAGraph capture under `dynamic=True` records 9 shape variants and the overhead defeats dispatch savings. Metric "win" was below RNG noise floor. Genuine learning: Python dispatch is NOT the bottleneck on this Transolver/dynamic-mesh workload; kernel-level work is.
- **edward #1383** (channel-weight p:3) CLOSED — clear regression (+2.4% val, +4.2% test). Implementation flaw: weight applied to BOTH surface and volume p-channels, wasting capacity on volume-p which isn't in the headline metric. Direction correct, magnitudes off, implementation needs to be surface-only.

**Round 9 sends-back:**
- **nezuko #1719** (pct_start=0.05) sent back for rebase + composition test against new max_lr=1.5e-3 baseline. Single-knob result vs OLD baseline was within RNG noise (val −1.33%, test −0.82%). Composition test will determine if pct_start=0.05 is additive with max_lr=1.5e-3 or mechanistically redundant.

**Active research questions:**
1. **LR ceiling beyond 1.5e-3** — alphonse #1785 testing max_lr=2e-3.
2. **OOD splits need a different lever** — camber-rc and camber-cruise unmoved by LR tuning. Need geometric augmentation, domain-adaptive loss weighting, or surface-normal features.
3. **pct_start sweep** — bracket of {0.05 (nezuko), 0.10 (current), 0.15 (frieren #1768)} pending all on the new baseline.
4. **Throughput attacks** — Python dispatch is not the bottleneck. Next attack: tanjiro #1807 testing `max-autotune-no-cudagraphs` (Triton kernel autotuning without CUDAGraph overhead).
5. **Loss formulation** — surface-only p-weighting is the targeted version (edward #1809). Smooth-L1 (askeladd #1379) still pending.

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1785 | OneCycleLR max_lr=2e-3 (LR ceiling probe) | WIP |
| askeladd | #1379 | smooth-L1 loss | WIP |
| edward | #1809 | surface-only p-weight=2 (targeted vs #1383) | WIP (just assigned) |
| fern | #1390 | surf_weight=25 (needs rebase) | WIP (stale) |
| frieren | #1768 | OneCycleLR pct_start=0.15 | WIP |
| nezuko | #1719 | OneCycleLR pct_start=0.05 + max_lr=1.5e-3 composition | WIP (rebasing) |
| tanjiro | #1807 | compile mode=max-autotune-no-cudagraphs (Triton kernel autotuning) | WIP (just assigned) |
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
9. **Python dispatch is NOT the throughput bottleneck** (tanjiro #1764 closed) — `reduce-overhead` CUDAGraphs cost more than they save under `dynamic=True` mesh. Throughput attacks need to target kernels (Triton autotuning, attention backends), not dispatch.
10. **Channel-weight loss must be surface-only** (edward #1383 closed) — flat multipliers on (Ux, Uy, p) reshape gradient balance but the headline metric is in physical space and surface-only. Targeting volume-p wastes capacity; correct version (edward #1809) applies the weight only inside `surf_mask`.

## Potential next research directions

### Immediate (waiting on current PRs)
- OneCycleLR max_lr=2e-3 (alphonse #1785) — does the LR ceiling extend past 1.5e-3?
- pct_start={0.05, 0.15} on new baseline (nezuko #1719 rebasing, frieren #1768) — 3-point sweep
- max-autotune-no-cudagraphs compile (tanjiro #1807) — Triton kernel autotuning without CUDAGraphs
- Surface-only p-weight=2 (edward #1809) — targeted version of channel weighting
- Scheduler shootout 2-seed: SequentialLR(T_max=27) vs OneCycleLR (thorfinn #1628)
- Smooth-L1 results (askeladd #1379)

### Short-term (round 10+)
- **OOD attack** — camber-rc and camber-cruise splits are LR-saturated. Need geometric augmentation (random small rotation/scaling/flipping), explicit surface-normal positional features, or domain-balanced loss weighting.
- **batch_size sweep** — current batch=4 with 30GB peak; 96GB available. Could try batch=8 for more gradient updates per epoch (test compute vs memory bound).
- **SDPA flash backend** — explicit `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)` context manager if Transolver attention isn't already using it.
- **Per-channel p-weight sweep** — if edward's surface-only p_weight=2 shows directional signal, sweep {1.5, 2, 2.5, 3}.
- **surf_weight=25 on OneCycleLR** — fern still pending proper rebase
- **SequentialLR(T_max=27) as baseline candidate** — if thorfinn's multi-seed confirms 3.6% test gain

### Architecture and signal (longer term)
- SwiGLU MLP — swap GELU FF layers
- Per-domain LR — higher LR for OOD splits (re_rand/cruise) vs in-dist
- Explicit signed-distance-to-surface as positional feature
- Loss: relative-MAE for pressure (handles dynamic range across Re)
- Test-time augmentation: average predictions over mirrored geometry
