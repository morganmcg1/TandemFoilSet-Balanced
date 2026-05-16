# SENPAI Research State

- **Date**: 2026-05-16 (Loop 6)
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 cleanup done; rounds 4-6 actively iterating. Round-1 winners merged:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → **current best**

System fix merged: **#3378 thorfinn (NaN-safe scoring)** — paper-wide finite 4-split test_avg.

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0.

## MAJOR SYSTEMIC FINDING: camber_rc-as-discriminator (5-way confirmed)

Every intervention that flattens or replaces gradient-magnitude information regresses `val_geom_camber_rc` by 11-19% while other splits are neutral or improved:

| Mechanism | camber_rc regression | What is removed from gradient |
|---|---|---|
| slice_num=32 (#3500, closed) | +18.9% | attention capacity |
| grad-clip max_norm=1.0 (#3336, closed) | +17.4% | gradient L2 magnitude |
| MSE+L1 mix 0.7/0.3 (#3556, closed) | +17.4% | sign(r) replaces magnitude |
| grad-clip max_norm=100 (#3336, closed) | +11.9% | gradient L2 magnitude (loose) |
| Lion sign-momentum (#3525, closed) | +10.84% | AdamW v adaptation |

**Interpretation**: camber_rc-loading parameters need disproportionately large effective updates carrying full gradient-magnitude information. This drives the current hypothesis design: attack from the data-distribution side (per-domain weighting, loop 6) instead of the gradient-flow side (all prior attempts).

## Closed this loop (Loop 6)

- **#3525 fern Lion (+0.33%, CLOSED)**: camber_rc +10.84% — 3rd mechanism class confirming the discriminator finding. Throughput improvement did not materialize (~205 s/ep). Closing optimizer-swap axis.
- **#3498 tanjiro SwiGLU (+4.79%, CLOSED)**: gated-FFN gives no throughput win on n_hidden=192 trunk (MLP is small FLOPs fraction vs PhysicsAttention). Camber_rc still worst split. Gated-FFN axis exhausted.
- **#3556 thorfinn MSE+L1 mix (+8.7%, CLOSED)**: per-split signature inverted (rc predicted biggest win, became biggest regression at +17.4%). Fifth mechanism class confirming camber_rc-as-discriminator. Linear-combination-loss axis closed.

## PRs in-flight (all 8 students active, zero idle GPUs)

| PR | Student | Axis | One-line summary | Round | Status |
|---|---|---|---|---|---|
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) rebased rerun — smooth blend vs MSE | 1→rerun | wip |
| #3478 | edward | Width / throughput | Narrow trunk (n_h=128) + bf16: stop/pivot test | 2 | wip (stale — key blocker) |
| #3496 | nezuko | Architecture (normalization) | RMSNorm: replace all LayerNorm, -2112 params, +5-10% throughput | 3 | wip |
| #3526 | frieren | Throughput (graph compilation) | torch.compile(dynamic=True) — operator fusion | 4 | wip |
| #3555 | alphonse | Data augmentation | Coord jitter σ=0.01 on (x,z) — denoising regularizer | 5 | wip |
| #3585 | fern | Loss / data distribution | Per-domain loss weight: racecar_tandem 2.0× — 1st data-side attack on camber_rc | 6 | dispatched |
| #3588 | tanjiro | Optimizer (meta) | Lookahead(AdamW, k=5, α=0.5) — slow-weight smoothing preserving v adaptation | 6 | dispatched |
| #3589 | thorfinn | Weight averaging | SWA tail (last 3 epochs): uniform average complementing EMA | 6 | dispatched |

## Recent decisions

- **#3556 thorfinn MSE+L1 mix CLOSED**: 5th mechanism class confirming camber_rc-as-discriminator; linear-combination-loss axis closed.
- **#3525 fern Lion CLOSED**: 3rd-mechanism-class confirmation (sign-momentum → camber_rc +10.84%); optimizer-swap axis closed.
- **#3498 tanjiro SwiGLU CLOSED**: no throughput win at n_hidden=192; gated-FFN axis closed.
- **#3585 fern per-domain-weight DISPATCHED**: racecar_tandem 2.0× — first data-distribution-side attack on the 5-way camber_rc finding.
- **#3588 tanjiro Lookahead DISPATCHED**: meta-optimizer wrapper around AdamW preserving v adaptation, fresh optimizer-meta axis.
- **#3589 thorfinn SWA tail DISPATCHED**: uniform tail averaging complementing EMA, fresh weight-averaging axis.

## Systemic findings (load-bearing context)

1. **camber_rc-as-discriminator (5-way confirmed)**: see table above. This is the defining finding of rounds 3-6. Every gradient-magnitude-flattening intervention regresses camber_rc by 11-19%.
2. **Schedule misalignment** — at `n_hidden=192`, fp32 30-min cap gives 8-9 realized epochs; bf16 gives 12; slice_num=32 gives 11 (but capacity loss). 7-way confirmed.
3. **Slice mechanism is capacity-bottlenecked, not memory-bottlenecked**: slice projection dominates VRAM, not slice-attention matrix (from #3500).
4. **Gated-FFN axis exhausted at n_hidden=192**: SwiGLU gives no throughput win because MLP is small FLOPs fraction on wider-trunk mesh inputs.
5. **CaIT cold-start ruled out** (via #3404 LayerScale init=1e-4).
6. **Depth co-adaptation hypothesis disconfirmed** (via #3455 DropPath).
7. **Aux task at saturation-prone targets ruled out** (via #3435).
8. **Domain curriculum has +50% wall-time overhead** (DataLoader rebuild structural — fixable but not in current scope).
9. **Low-dimensional per-sample conditioners ruled out** (via #3287 FiLM).
10. **#3478 edward stale_wip** is gating: stop/pivot determination delays.

## Stop / pivot criteria

- **#3478 result determines the next major direction**:
  - narrow+bf16 < 126.32 → revert wider trunk merge; re-baseline all round-3/4/5/6 PRs.
  - narrow+bf16 ≈ 126.32 ± 2% → width is a wash; keep merged config.
  - narrow+bf16 > 130 → wider thesis confirmed; throughput unlocks become priority.
- **Round-6 closures**:
  - Per-domain weight (#3585) regresses >5% → close data-distribution-side axis.
  - Lookahead (#3588) regresses >5% → close optimizer-meta axis.
  - SWA tail (#3589) regresses vs EMA and raw → close uniform-tail-averaging axis; EMA already covers it.

## Priority candidates if students free up next

1. **Per-channel pressure upweighting** — give `surf_p` channel its own multiplier within surface loss (e.g., 4× vs Ux/Uy). Direct attack on primary metric with no architectural risk.
2. **SWA + EMA ensemble** — if SWA wins, evaluate mean(SWA, EMA) as a 2-checkpoint ensemble.
3. **slice_num=48** — capacity/throughput tradeoff midpoint between 64 and 32.
4. **In-place WeightedRandomSampler weight mutation** — infrastructure unlock for any future curriculum.
5. **Mix MSE/L1 weight sweep** — only if SmoothL1 (#3127) wins; explore 0.9/0.1 if that direction proves valid.
6. **gradient checkpointing** — trade compute for memory; revisits bs=8 question.
7. **DropToken (random subset of input nodes during training)** — input-side regularizer distinct from coord jitter (#3555).
8. **Per-param effective-LR analysis** — if #3585 or #3588 reveals the camber_rc-loading parameter cluster, test asymmetric optimizer per parameter group.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
