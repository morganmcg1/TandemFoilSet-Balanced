# SENPAI Research State

- **Date:** 2026-05-13 ~00:10
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none yet — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** ACTIVE — 5+ consecutive failed experiments. Researcher-agent dispatched for bolder hypotheses (mixed precision, Lion optimizer, loss reformulations, modern transformer ingredients).

## Current baseline

**`val_avg/mae_surf_p` = 101.810** (L1 loss + n_layers=6 + mlp_ratio=4, PR #1358)
**`test_avg/mae_surf_p` = 91.708** (first reliable test numbers — NaN-fix merged)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 124.150 | 110.726 |
| geom_camber_rc | 112.699 | 99.692 |
| geom_camber_cruise | 76.570 | 66.879 |
| re_rand | 93.820 | 89.536 |
| **avg** | **101.810** | **91.708** |

## What we've learned

### Big wins (merged)
1. **mlp_ratio=4**: −5% (PR #1408)
2. **n_layers=6**: −9.4% (PR #1392)
3. **L1 loss**: −20.5% (PR #1358) ← dominant factor

### Dead ends
- Width (n_hidden=192): too slow/epoch
- Channel-weighted loss p×3 in normalized space: counterproductive
- n_head=8: +43% per-epoch cost, +15.7% worse
- slice_num=128: +12% per-epoch cost, +17.8% worse
- Warmup+lr=1e-3: too hot (even T_max=15 fails at ~175 s/epoch)
- EMA (decay=0.999): cold-start drag (+41% worse); requires 100+ epochs or non-random init
- lr=3e-4: undertraining at 30-min cap; cosine barely decays with T_max=50 (sent back to try lower LR + short T_max later)
- n_layers=7: +51% worse, 9 epochs/30min, reproducible NaN on test_geom_camber_cruise
- grad clip max_norm=1.0: too aggressive for L1 constant-magnitude gradients; retry at max_norm=10 in progress
- Huber loss β=1.0: +15.7% worse; β=1.0 is "mostly-MSE" for normalized targets (std≈1). The constant-magnitude gradient advantage of L1 is what matters.

### Key insights
1. **L1 loss is the dominant lever** (−20.5% alone)
2. **Budget is the constraint**: 30 min → ~12-13 epochs at n_layers=6, ~9 at n_layers=7. Any deeper/wider arch needs a batching change.
3. **n_layers=6 + mlp_ratio=4 is the sweet spot** for the 30-min budget
4. **LR/schedule coupling**: can't tune LR without also fixing T_max; lower LR needs shorter T_max to decay properly
5. **Gradient clip threshold**: with L1 loss and 1.18M params, max_norm=1.0 clips virtually all updates; threshold must be >>1.0 to be useful

## Active experiments (Round 6 — Plateau Protocol, bolder hypotheses)

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1724 | bf16 mixed precision: 2x epoch throughput (~18-22 vs 12 epochs) | NEW |
| nezuko | #1678 | LR 5e-4 → 7e-4 (upper-stable range) | WIP |
| tanjiro | #1728 | GeGLU activation: gated MLP for flow-regime routing (mlp_ratio=3) | NEW |
| fern | #1726 | SWA late-start epoch 8: flat basin for OOD generalization | NEW |
| edward | #1725 | Lion optimizer lr=1e-4: sign-based updates for L1 gradients | NEW |
| askeladd | #1673 | AdamW eps 1e-8→1e-4 (adaptive scaling floor) | WIP |
| frieren | #1729 | RMSNorm replaces LayerNorm: faster norm, ~13-14 epochs | NEW |
| thorfinn | #1670 | weight_decay 1e-4 → 5e-4 (stronger L2) | WIP |

**Round 5 closed (all worse than baseline 101.810):** #1592 alphonse T_max=14 (+0.5%, seed noise across 3 seeds), #1661 fern warm restarts (+2.7%), #1671 edward β1=0.85 (+5.9%), #1634 tanjiro batch=8 (+23.6%), #1384 frieren surf_weight=25 (stale, never rebased).

**Round 6 bets:** Priority is #1724 bf16 (structural throughput gain) and #1725 Lion (optimizer paradigm shift). GeGLU, RMSNorm, and SWA are quality/efficiency improvements. If any win, compound them.

## Round 3/4 themes and open questions

1. **Schedule completion** (alphonse): T_max=14 = does aligning decay to actual epoch budget help?
2. **Gradient stability at scale** (nezuko retry): max_norm=10 — right clip threshold for L1 grad magnitudes?
3. **Batch size effect** (tanjiro): batch=4→8 — fewer, cleaner gradient steps; does this trade-off favor accuracy at our budget?
4. **OOD regularization** (edward): dropout=0.1 — improves hardest OOD splits?
5. **Multi-cycle schedule** (fern): CosineAnnealingWarmRestarts T_0=4 T_mult=2 — periodic LR restarts to escape L1 loss landscape traps?
6. **AdamW betas for L1** (askeladd): (0.95, 0.99) vs default — better second-moment tracking for constant-magnitude gradients?
7. **surf_weight with L1** (frieren): 10 → 25 on new baseline, after rebase
8. **weight_decay=0** (thorfinn): L1 as implicit regularizer may not need AdamW WD; removing it frees capacity

## Round 6 queued ideas (if current batch fails)

- **Physical-space L1**: Compute loss in denormalized units for direct metric alignment
- **surf_weight=5**: Reduce from 10 toward volume balance (not tested on L1 baseline)
- **mlp_ratio=8**: Bigger feedforward; note we're at width limit (n_hidden=128 × 8 = 1024-dim MLP). Per-epoch time TBD.
- **Data augmentation**: Mesh-coarsening, AoA jitter for OOD robustness
- **bf16 + Lion compound**: If both individually win, combine for max throughput + per-step quality
- **SwiGLU activation** (alternative to GeGLU): same gating idea, SiLU gate instead of GELU gate

## Key constraints

- 30 min / run cap: n_layers=6 → ~12-13 epochs (~175 s/epoch with L1)
- Per-epoch time budget eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~205 s/epoch)
- EMA eliminates: decay=0.999 with random init (cold-start drag, half-life 693 steps)
- Fourier L=4 does NOT compound with L1 (+5.6% worse); L1+n_layers=6 absorbs the same high-freq OOD content
- test_avg/mae_surf_p is RELIABLE since PR #1358 NaN-fix: 91.708 is the first accurate test baseline
- Gradient clip max_norm=1.0 AND max_norm=10.0: both worse. Clipping reduces oscillations but costs convergence speed. Oscillations are useful optimization search at 30-min budget.
- Dropout=0.1: +11.8% worse. Model is underfitting; regularizing an underfit model is counterproductive.
- AdamW betas (0.95, 0.99): +15.4% worse. β2=0.99 amplifies sign-flip noise for L1. Standard (0.9, 0.999) is correct.
- weight_decay=0: +3.2% worse (closest miss of round 4). WD provides useful regularization for high-magnitude splits.
- AdamW β1=0.85: +5.9% worse. Combined with β1=0.95 from round 4 (+15.4%), default β1=0.9 is correct in both directions. STOP tuning betas.
- CosineAnnealingWarmRestarts (T_0=4, T_mult=2): +2.7% worse. Restart at epoch 5 disrupts progress; needs longer budgets.
- Cosine T_max=14 (aligned to actual budget): essentially baseline (3 seeds: 102.30/104.11/107.09 = pure noise). Schedule decay is NOT the bottleneck.
- Batch=8 via accum_steps=2: +23.6% worse. Direct evidence we are step-count-limited; doubling effective batch halves optimizer steps in the budget. Stop increasing batch.
- **Round 4 meta-insight: CONVERGENCE-LIMITED. Anything slowing per-epoch progress loses at 30-min budget.**
- **Round 5 meta-insight: optimizer/schedule/batch hyperparameter space is FULLY EXHAUSTED. Plateau Protocol active — next round must be structural (loss reformulation, mixed precision, optimizer paradigm shift, architecture changes).**
