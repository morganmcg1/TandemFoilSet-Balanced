# SENPAI Research State

- **Date**: 2026-05-15 23:38
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 cleanup done; round 4 actively iterating. Round-1 winners merged:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → current best

System fix merged: **#3378 thorfinn (NaN-safe scoring)** — paper-wide finite 4-split test_avg.

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0.

## Closed this loop (Loop 5)

- **#3500 alphonse slice_num=32 (CLOSED +8.1%)**: throughput win confirmed (+15.4% per-epoch, 11 epochs realized) but capacity loss too steep — val_geom_camber_rc absorbed all the regression (+18.9% vs ≤+1.3% on cruise/re_rand). **Key permanent finding**: slice mechanism is **capacity-bottlenecked, not memory-bottlenecked** on the wider trunk; the slice projection's `B × N×slice × heads × 4` dominates VRAM, not the slice-attention matrix.
- **#3437 thorfinn domain curriculum (CLOSED +15.96%)**: predictions inverted (predicted rc/re_rand wins, both regressed worst). Three failure modes: hypothesis prediction inversion, +50% wall-time overhead from DataLoader rebuild, budget-elastic mechanism. Future curriculum requires infrastructure fix first (in-place sampler weight mutation).

## PRs in-flight (all 8 students active, zero idle GPUs)

| PR | Student | Axis | One-line summary | Round | Status |
|---|---|---|---|---|---|
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) rebased rerun | 1→rerun | wip (nudged earlier) |
| #3478 | edward | Width / throughput | Narrow trunk (n_h=128) + bf16: stop/pivot test | 2 | wip (nudged this loop — 6h stale) |
| #3496 | nezuko | Architecture (normalization) | RMSNorm: replace all LayerNorm, drop-in, -2112 params, +5-10% throughput | 3 | dispatched |
| #3498 | tanjiro | Architecture (MLP) | SwiGLU MLP iso-param d_ff=256 (PaLM/LLaMA gated FFN) | 3 | dispatched |
| #3525 | fern | Optimizer | Lion (sign-momentum, lr=1.5e-4, wd=1e-3) — tests camber_rc grad-norm mechanism | 4 | dispatched |
| #3526 | frieren | Throughput (graph compilation) | torch.compile(dynamic=True) — operator fusion, 10-20% per-epoch predicted | 4 | dispatched |
| #3555 | alphonse | Data augmentation | Coord jitter σ=0.01 on (x,z) input — denoising regularizer, fresh axis | 5 | dispatched |
| #3556 | thorfinn | Loss formulation | Mixed MSE+L1 (0.7/0.3) on surface — heavy-tail attack, distinct from SmoothL1 | 5 | dispatched |

## Recent decisions

- **#3500 alphonse slice_num=32 CLOSED**: capacity-bottleneck finding; rc most slice-sensitive split.
- **#3437 thorfinn curriculum CLOSED**: predictions inverted, infrastructure overhead, budget-elastic.
- **#3555 alphonse coord-jitter DISPATCHED**: fresh data-axis regularizer, direct attack on camber_rc overfitting from input side.
- **#3556 thorfinn MSE+L1 mix DISPATCHED**: loss formulation distinct from askeladd's SmoothL1; targets heavy-tailed pressure + camber_rc gradient-norm asymmetry.
- **#3478 edward narrow+bf16 NUDGED** (6h stale): stop/pivot test for wider-trunk merge revert.

## Systemic findings (load-bearing context)

1. **Schedule misalignment** — at `n_hidden=192`, fp32 30-min cap gives 8-9 realized epochs; bf16 gives 12; slice_num=32 gives 11 (but capacity loss). 7-way confirmed.
2. **camber_rc is the discriminator split**: most slice-sensitive (+18.9% at slice_num=32), structurally high gradient norms (per fern's grad-clip), curriculum predictions inverted. Three independent mechanism classes point to camber_rc as the discriminator for generalization quality.
3. **Slice mechanism is capacity-bottlenecked, not memory-bottlenecked**: slice projection dominates VRAM, not slice-attention matrix.
4. **CaIT cold-start ruled out** (via #3404 LayerScale init=1e-4).
5. **Depth co-adaptation hypothesis disconfirmed** (via #3455 DropPath).
6. **Aux task at saturation-prone targets ruled out** (via #3435).
7. **Domain curriculum has +50% wall-time overhead** (DataLoader rebuild structural — fixable but not in current scope).
8. **Low-dimensional per-sample conditioners ruled out** (via #3287 FiLM).
9. **#3478 stale_wip** is gating: stop/pivot determination delays.

## Stop / pivot criteria

- **#3478 result determines the next major direction**:
  - narrow+bf16 < 126.32 → revert wider trunk merge; re-baseline all round-3/4/5 PRs.
  - narrow+bf16 ≈ 126.32 ± 2% → width is a wash; keep merged config.
  - narrow+bf16 > 130 → wider thesis confirmed; throughput unlocks become priority.
- **Round-4 closures**:
  - Lion (#3525) regresses >3% → close optimizer-swap axis.
  - torch.compile (#3526) compile crashes → close engineering blocker; if no speedup → model not dispatch-bound.
- **Round-5 closures**:
  - Coord jitter (#3555) regresses >5% → close input-augmentation axis; if neutral but rc improves → win as diagnostic.
  - MSE+L1 mix (#3556) regresses >3% → close mixed-loss axis (and SmoothL1 if it also closes — both linear and smooth blends fail).

## Priority candidates if students free up next

1. **Per-domain LR / loss weighting** — give camber_rc its own LR or upweighting via WeightedRandomSampler. Direct attack on the camber_rc-is-the-discriminator finding.
2. **SWA tail (Stochastic Weight Averaging)** — uniform averaging in cosine annealing tail; complements EMA decay-based averaging.
3. **slice_num=48** — capacity/throughput tradeoff midpoint between 64 and 32.
4. **In-place WeightedRandomSampler weight mutation** — infrastructure unlock for any future curriculum.
5. **Mix MSE/L1 weight sweep** — if #3556 wins, sweep 0.5/0.5 and 0.9/0.1.
6. **gradient checkpointing** — trade compute for memory; revisits the bs=8 question with VRAM headroom (#3332 closed via memory-bandwidth-bound, but checkpointing changes the activation profile).
7. **DropToken (random subset of input nodes during training)** — different input-side regularizer if coord jitter wins.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
