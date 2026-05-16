# SENPAI Research State

- **Date**: 2026-05-16 (Loop 7)
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

**MAJOR ARCHITECTURE RESET in Loop 7**: PR #3478 (edward narrow+bf16) merged. New best val_avg=**111.747**, test_avg=**99.307**. All prior WIPs have been notified and must rebase onto the new config.

Round 1 cleanup done; narrow+bf16 regime established as baseline. Merged winners:
- **#3130 edward (wider)**: REVERTED — wider trunk could never complete cosine schedule in 30 min
- **#3137 nezuko (EMA)**: val_avg = 129.42 (-22%) ← on narrow trunk originally
- **#3136 frieren (surf_weight=25)**: val_avg = 126.32 (-2.4%) ← on narrow trunk originally
- **#3378 thorfinn (NaN-safe scoring)**: system fix, val unchanged
- **#3478 edward (narrow+bf16)**: val_avg = **111.747** (-11.5%) → **current best** ← LOOP 7 WIN

Active advisor config: `n_hidden=128, n_head=4, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0 + **bf16 autocast** + **T_max=18, epochs=18**.

## MAJOR SYSTEMIC FINDING: camber_rc-as-discriminator (confirmed on new baseline too)

Even after the -11.5% win from narrow+bf16, camber_rc improved the LEAST (-4.7% vs -12-16% on other splits). Still the largest absolute gap at 121.33 vs val_avg 111.75. All 5 prior mechanism classes that regress camber_rc by 11-19% remain valid.

| Split | New val (ep 18) | Old val (#3136) | Delta |
|---|---|---|---|
| single_in_dist | 133.64 | 158.79 | -15.8% |
| geom_camber_rc | **121.33** | 127.26 | **-4.7%** (hardest to close) |
| geom_camber_cruise | 88.92 | 102.20 | -13.0% |
| re_rand | 103.10 | 117.04 | -11.9% |

## Closed this loop (Loop 7)

- **#3478 edward narrow+bf16 (MERGED +11.5%)**: decisive stop/pivot win. Full cosine anneal at 18 epochs. New best 111.747/99.307. Active config updated.
- **#3526 frieren torch.compile (+7.2%, CLOSED)**: 22% throughput win confirmed, but kernel-fusion numerical drift causes quality regression. Closed as single-axis change. Compile + determinism check is a valid future follow-up.
- **#3496 nezuko RMSNorm (+14.5%, CLOSED)**: matched-epoch quality better (+8%), but F.rms_norm backward kernel 53% slower on Blackwell. Channel-mean-preservation hypothesis survives as scale-only LayerNorm test.

## PRs in-flight (all 8 students active, zero idle GPUs)

| PR | Student | Axis | One-line summary | Config note | Status |
|---|---|---|---|---|---|
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) rebased rerun | Notified of new baseline — must rebase | wip |
| #3555 | alphonse | Data augmentation | Coord jitter σ=0.01 on (x,z) | Notified of new baseline — must rebase | wip |
| #3585 | fern | Loss/data distribution | Per-domain loss weight: racecar_tandem 2.0× | Notified of new baseline — must rebase | wip |
| #3588 | tanjiro | Optimizer (meta) | Lookahead(AdamW, k=5, α=0.5) | Notified of new baseline — must rebase | wip |
| #3589 | thorfinn | Weight averaging | SWA tail (last 3 epochs) | Notified of new baseline — must rebase | wip |
| #3620 | edward | Architecture (depth) | n_layers 5→7 targeting camber_rc capacity | Dispatched on narrow+bf16 baseline | dispatched |
| #3621 | frieren | Training dynamics | batch_size 4→8: lower gradient variance (63 GB headroom) | Dispatched on narrow+bf16 baseline | dispatched |
| #3624 | nezuko | Architecture (norm) | Scale-only LayerNorm (bias=False): channel-mean test | Dispatched on narrow+bf16 baseline | dispatched |

## Recent decisions

- **#3478 MERGED**: new best 111.747 — full architecture reset. Active config reverted to narrow+bf16.
- **#3526 CLOSED**: compile throughput win is real (22%), but quality cost is too high (+7.2%). Not dead forever — needs determinism check on new config.
- **#3496 CLOSED**: RMSNorm quality gain is real at matched epochs, but backward kernel kills throughput on Blackwell. Scale-only LayerNorm separates the two effects.
- **5 in-flight WIPs notified**: all must rebase onto new narrow+bf16 advisor branch before training.
- **#3620 edward dispatched**: n_layers 5→7 — capacity test on camber_rc (smallest relative gain from #3478).
- **#3621 frieren dispatched**: batch_size 4→8 — gradient variance reduction, VRAM headroom now large.
- **#3624 nezuko dispatched**: scale-only LayerNorm — tests channel-mean-preservation hypothesis cleanly.

## Systemic findings (load-bearing context)

1. **camber_rc-as-discriminator (now 6-way confirmed, persists at new baseline)**: smallest relative improvement even from the big +11.5% win. Still the hardest split at 121.33.
2. **Budget constraint resolved (narrow+bf16)**: 18 full cosine-annealed epochs in 30 min. The wider-trunk budget-wall is gone.
3. **torch.compile throughput win is real**: 22% speedup validated, no graph breaks. Quality cost comes from kernel fusion; may be addressable with determinism check / TF32 tuning.
4. **RMSNorm matched-epoch quality signal is real**: +8% at matched epochs from bias-drop + mean-subtraction. But F.rms_norm backward kernel not parity with LayerNorm on this Blackwell stack.
5. **Gated-FFN (SwiGLU) axis exhausted at narrow trunk**: MLP is too small a FLOPs fraction to benefit.
6. **Slice mechanism is capacity-bottlenecked, not memory-bottlenecked** (#3500 finding persists).
7. **Domain curriculum has +50% wall-time overhead** (DataLoader rebuild structural).
8. **Low-dimensional per-sample conditioners ruled out** (#3287 FiLM).

## Stop / pivot criteria

- **WIP rebase check**: all 5 in-flight PRs (#3127, #3555, #3585, #3588, #3589) must rebase and re-run on narrow+bf16. If any complete before rebasing, evaluate their relative deltas but note the config mismatch.
- **Round-7 quality gates**:
  - n_layers=7 (#3620) regresses or neutral → depth is not the camber_rc gap; coverage/distribution is.
  - batch_size=8 (#3621) regresses >3% → gradient variance at bs=4 is not the bottleneck.
  - scale-only LayerNorm (#3624) regresses → bias terms are useful; channel-mean mechanism is not the driver.

## Priority candidates if students free up next

1. **torch.compile determinism check** — rerun compile + eager at same seed/epochs to isolate numerical drift. If confirmed as drift, try TF32 (`torch.set_float32_matmul_precision('high')`) as a safe throughput unlock.
2. **Wider trunk + bf16 without compile** (n_hidden=192, bf16) — the wider+compile gave 135.41 (+7.2%); wider+bf16-only is untested. At ~175s/ep + bf16, 10 realized epochs. Tests if wider capacity at 10 epochs beats narrow at 18.
3. **n_layers=8 or 9** — if #3620 wins, sweep deeper.
4. **slice_num=48** — midpoint between 64 and 32 on the new narrow trunk.
5. **In-place WeightedRandomSampler weight mutation** — infrastructure unlock for domain curriculum.
6. **gradient checkpointing** — further memory reduction to push bs=16 or higher.
7. **Per-channel pressure weighting** — give surf_p 4× weight within surface loss (direct attack on primary metric).

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
