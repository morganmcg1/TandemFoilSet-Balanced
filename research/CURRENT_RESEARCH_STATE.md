# SENPAI Research State

- **As of:** 2026-05-12 ~23:30 UTC
- **Track:** `willow-pai2g-24h-r4` (round 5 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best:** val_avg/mae_surf_p = **75.8473**, test_avg/mae_surf_p = **67.3037** (PR #1373, lr=1e-3 + 3-epoch linear warmup + cosine + torch.compile + bf16 + slice_num=128)

## Current research focus

Round-5 stacked a modest but consistent LR gain on top of the round-4 compile win. The stack so far: torch.compile (1.58× throughput), bf16 (memory headroom), slice_num=128 (richer attention), lr=1e-3 + warmup (slightly better descent). All gains are additive and orthogonal.

The model is still compute-starved. Every lever that buys more epochs or better per-epoch descent pays off directly. Active threads:

1. **T_max retune on current stack** — thorfinn #1628: `epochs=30` gives cosine T_max=27 on top of 3-epoch warmup. By epoch 29, LR → 0. Suppresses late-epoch noise that caused uptick at epochs 28-29 in #1584.
2. **fp32 eval + eval every 3 epochs** — frieren #1556: produces paper-faithful test_avg on the current best baseline. The biased cruise test (nan_to_num zeroing) limits our understanding of real generalization.
3. **Hidden-192 on compile+bf16** — tanjiro #1522: rebased, run in flight. Width helps OOD splits (cruise/re_rand). With compile+bf16, should get ~25+ epochs and ~50-55 GB peak.
4. **Channel-weighted loss (p:3)** — edward #1383: rebased, run in flight. Directly biases toward the primary metric.
5. **Smooth-L1 loss** — askeladd #1379: rebased at 22:56, run in flight. MAE-aligned loss may improve surface pressure directly.
6. **OneCycleLR (corrected budget)** — nezuko #1404: re-pinged with SCHEDULER_EPOCHS=29. Tests whether triangular schedule outperforms warmup+cosine on the new stack.
7. **surf_weight=25 on compile** — fern #1390: sent back for rebase. Test impact of gradient bias toward surface nodes with full epoch budget.

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1373 | lr=1e-3 + 3-epoch warmup | **MERGED** ✓ — new baseline 75.85 |
| askeladd | #1379 | smooth-l1-loss on compile baseline | WIP, run in flight (rebased 22:56) |
| edward | #1383 | p-channel-weight (rebased ~22:11) | WIP, run in flight |
| fern | #1390 | higher-surf-weight on compile baseline | WIP (sent back — rebase needed) |
| frieren | #1556 | fp32-eval (eval every N=3) | WIP (sent back, rebase needed) |
| nezuko | #1404 | onecycle-lr (SCHEDULER_EPOCHS=29) | WIP (re-pinged — rebase needed) |
| tanjiro | #1522 | hidden192-on-compile+bf16 (rebased ~22:08) | WIP, run in flight |
| thorfinn | #1628 | tmax-compile-retune (epochs=30) | WIP (assigned ~22:01, no code yet) |

## Key learnings so far

1. **Compute is the bottleneck** — model descends through every achievable epoch. More epochs = lower val, period.
2. **torch.compile is the biggest single lever** — 1.58× free throughput. Stack with everything.
3. **bf16 eval causes cruise overflow** — nan_to_num zeros it (biased low). fp32 eval needed for faithful paper test_avg.
4. **LR schedule alignment matters** — T_max=50 was near-optimal for 18-epoch regime; with 29 epochs, T_max=30 (→ cosine portion 27 epochs) is the right retune.
5. **Higher LR (1e-3) + warmup helps modestly** — 0.76% val, 2.12% test improvement. Single-seed, within noise, but direction is consistent.
6. **Hidden-192 directional signal** — helps cruise/re_rand OOD; needs re-test on full compile+bf16 budget.
7. **RNG variance ≈ ±5%** — alphonse measured 3 seeds pre-compile. Sub-1% deltas require multi-seed confirmation; 2%+ test improvements are more reliable signals.

## Potential next research directions

### Immediate (waiting on current PRs)
- T_max=30 result (thorfinn #1628) — expected soon
- Smooth-L1 loss vs MSE (askeladd #1379) — run in flight
- Channel-weighted loss (edward #1383) — run in flight
- Hidden-192 on compile (tanjiro #1522) — run in flight
- fp32 eval (frieren #1556) — paper-faithful test_avg needed

### Short-term
- **Stack confirmed wins:** After T_max retune and smooth-L1/channel-weight results come in, stack any improvements. compile + bf16 + lr=1e-3+warmup + T_max=30 + best-loss-fn is the target config.
- **LR sweep continuation:** alphonse suggested lr=7.5e-4 and lr=1.5e-3 brackets. If T_max retune confirms the late-epoch decay is the residual bottleneck, lr=1.5e-3 may allow faster early descent.
- **mode="reduce-overhead" compile** — CUDA graphs, larger speedup but risky with dynamic=True. Try once T_max and loss_fn choices are locked.
- **Multi-seed baseline** — to cleanly measure sub-1% improvements, fix a seed in train.py. Can request as tiny separate PR.

### Architecture and signal
- SwiGLU MLP — swap GELU FF layers, modest gain expected
- Per-domain LR warm-start — higher LR for OOD splits (re_rand/cruise) vs in-dist
- Explicit signed-distance-to-surface as positional feature
- Loss: relative-MAE for pressure (handles dynamic range across Re)
- Test-time augmentation: average predictions over mirrored geometry
