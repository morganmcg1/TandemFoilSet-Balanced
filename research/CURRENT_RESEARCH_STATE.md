# SENPAI Research State

- **As of:** 2026-05-12 ~22:00 UTC
- **Track:** `willow-pai2g-24h-r4` (round 4 of the Willow 24h ablation)
- **Most recent human directive:** Operator-defined isolation rules — 30-min hard cap.
- **Primary metric:** `test_avg/mae_surf_p` (val analogue: `val_avg/mae_surf_p`). Lower is better.
- **Current best:** val_avg/mae_surf_p = **76.4310** (PR #1584, torch.compile + bf16 + slice_num=128)
- **First faithful test_avg established:** 101.99 (frieren #1556 fp32-eval run, 13-epoch model — undertrained relative to best)

## Current research focus

Round-4 brought a massive win from torch.compile (1.58× throughput, 29 vs 18 epochs, val 76.43 vs 98.77). The model is still compute-starved — every lever that gives more epochs in the 30-min cap pays off directly.

Active threads:
1. **T_max retune now makes sense** — with 29 achievable epochs, T_max=30 aligns the cosine decay to the budget. Val curve showed uptick at epochs 28-29 consistent with high LR; T_max=30 should suppress this.
2. **fp32 eval still needed** — frieren's #1556 sent back to add `eval_every_n_epochs=3` to avoid the 45s/epoch overhead penalty. When merged, gives unbiased paper-facing test_avg.
3. **Hidden-192 still pending** — tanjiro needs to rebase on the compile baseline and re-run. Width helps OOD splits; with compile+bf16, should get 25+ epochs with ~50-55 GB peak.
4. **Stale students** — alphonse/askeladd/fern/nezuko have WIP PRs but no code activity in 4+ hours. Pods are running (confirmed); likely stuck in poll loops. Harvest workflow handles.

## Active PRs

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1373 | lr-warmup-1e-3 | stale WIP (4h+) |
| askeladd | #1379 | smooth-l1-loss | stale WIP (4h+) |
| edward | #1383 | p-channel-weight (rebased) | WIP, run pending |
| fern | #1390 | higher-surf-weight | stale WIP (4h+) |
| frieren | #1556 | fp32-eval (eval every N=3) | WIP (sent back) |
| nezuko | #1404 | onecycle-lr (corrected total_steps) | stale WIP (3h+) |
| tanjiro | #1522 | hidden192-on-bf16+compile baseline | WIP (sent back, rebase needed) |
| thorfinn | #1628 | **tmax-compile-retune** (T_max=30) | WIP (new) |

## Key learnings so far

1. **Compute is the bottleneck** — model descends through every achievable epoch. More epochs = lower val, period.
2. **torch.compile is the biggest single lever** — free throughput with no model changes. Stack with everything.
3. **bf16 eval causes cruise overflow** — nan_to_num zeros it (biased low). fp32 eval gives faithful numbers but needs eval-frequency gating to be affordable.
4. **T_max=50 was near-optimal for 18-epoch regime** — now with 29 epochs, T_max=30 is the right retune.
5. **Hidden-192 directional signal** — helps cruise/re_rand OOD; needs re-test on full compile+bf16 budget.

## Potential next research directions

### Immediate
- T_max=30 (thorfinn #1628) — complete the cosine for 29-epoch budget
- fp32 eval + eval_every_3 (frieren #1556 revised) — paper-faithful test_avg
- Hidden-192 on compile+bf16 (tanjiro #1522 rebase) — width gains with full epoch budget

### Short-term (next round)
- Stack torch.compile + T_max=30 + higher LR (1e-3) if both confirm independently
- mode="reduce-overhead" compile (CUDA graphs) — risky with dynamic=True, but thorfinn's data makes it worth trying
- Hidden-192 + compile — confirmed composition if tanjiro's retry succeeds
- Smooth-L1 / Huber loss (askeladd #1379) on compile baseline — needs pod to wake up

### Architecture and signal
- SwiGLU MLP — swap GELU FF layers, modest gain expected
- Per-domain LR warm-start — higher LR for OOD splits (re_rand/cruise) vs in-dist
- Explicit signed-distance-to-surface as positional feature
- Loss: relative-MAE for pressure (handles dynamic range across Re)
- Test-time augmentation: average predictions over mirrored geometry
