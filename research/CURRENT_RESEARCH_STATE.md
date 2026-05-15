# SENPAI Research State

- **Date**: 2026-05-15 22:45
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 cleanup done; round 2 actively iterating. Round-1 winners merged:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → current best

System fix merged:
- **#3378 thorfinn (scoring NaN-safe)**: `test_avg/mae_surf_p` now finite 4-split mean paper-wide.

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0.

**Major event 22:30 UTC**: stop/pivot criterion fired. #3332 edward bf16+bs=8 retry CLOSED — bs=8 axis cleanly refuted (memory-bandwidth-bound model, +10% slower per-epoch, peak 98 GB hits HBM ceiling). bf16-bs4 prior arm gave val=129.76 (+2.7%) — close but not winning. Per documented stop/pivot in BASELINE.md, the next decisive experiment is **narrow trunk + bf16** (PR #3478 edward).

**Systemic finding (6-way confirmed)**: binding constraint is **per-epoch wallclock at the wider trunk**, not model architecture or loss formulation. At `n_hidden=192`, only 8-9 epochs fit in fp32 30-min cap (vs 14 at narrower); with bf16 we get 12. Five closed PRs confirm this pattern.

## Closed this loop (Loop 3)

- **#3404 nezuko LayerScale (CLOSED +6.9%)**: init=1e-4 gammas never grew — model wasted 30-min budget on gate-unfolding. Block-curriculum prediction inverted. Cleanly rules out CaIT cold-start axis at this budget.
- **#3141 tanjiro Fourier rebased (CLOSED +5.7%)**: two-run negative signal (136.14 → 133.60 after rebase). Spectral-bias motivation correct but gain doesn't materialize.
- **#3435 alphonse aux-task (CLOSED +11.8%)**: aux head saturates at 99% acc by epoch 6, leaving 3 epochs of trivial gradient. Representation-shaping via saturation-prone aux supervision didn't help.

## PRs in-flight

| PR | Student | Axis | One-line summary | Round | Status |
|---|---|---|---|---|---|
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) rebased rerun | 1→rerun | wip (nudged) |
| #3336 | fern | Optimization stability | Grad-clip max_norm=100 rerun | 2→rerun | sent back (nudged) |
| #3437 | thorfinn | Data curriculum | Domain curriculum: racecar_single first, transition to uniform | 2 | wip (nudged) |
| #3455 | frieren | Regularization | DropPath stochastic depth (linear 0→0.1) on TransolverBlock residuals | 2 | dispatched |
| #3478 | edward | Width / throughput | Narrow trunk (n_h=128) + bf16: stop/pivot test | 2 | dispatched |
| #3496 | nezuko | Architecture (normalization) | RMSNorm: replace all LayerNorm, drop-in, -2112 params, +5-10% throughput | 3 | dispatched |
| #3498 | tanjiro | Architecture (MLP) | SwiGLU MLP iso-param d_ff=256 (PaLM/LLaMA gated FFN) | 3 | dispatched |
| #3500 | alphonse | Architecture (attention bandwidth) | slice_num 64→32: direct memory-bandwidth attack, +15-25% throughput | 3 | dispatched |

All 8 students have active PRs. **Zero idle GPUs.**

## Recent decisions

- **#3404 (nezuko LayerScale) CLOSED**: +6.9% regression, both pre-registered predictions falsified (block-curriculum inverted, no early-convergence speedup).
- **#3141 (tanjiro Fourier rebased) CLOSED**: +5.7%, two-run stable regression band. Spectral-bias motivation correct but Fourier encoding at σ=4.0 isn't the lever.
- **#3435 (alphonse aux-task) CLOSED**: +11.8%. Aux head saturates at 99% acc by epoch 6; gradient signal trivial for remaining epochs.
- **#3496 (nezuko RMSNorm) DISPATCHED**: drop-in LayerNorm swap, T5/PaLM/LLaMA standard, one reduction instead of two, -2,112 params, 5-10% per-epoch speedup predicted.
- **#3498 (tanjiro SwiGLU) DISPATCHED**: gated FFN with SiLU gate, iso-param at d_ff=256, PaLM convention, multiplicative gating helps multi-scale targets.
- **#3500 (alphonse slice_num=32) DISPATCHED**: halves the largest attention activation in PhysicsAttention; 4× smaller slice-attention matrix; predicted 15-25% per-epoch speedup → 15-18 realized epochs at 30-min cap.
- **#3455 (frieren DropPath) DISPATCHED**: stochastic depth [0→0.1], CaIT-canonical, zero param/inference overhead.
- **#3478 (edward narrow+bf16) DISPATCHED**: stop/pivot test. n_hidden 192→128, n_head 6→4, bf16, epochs=18. Decisive 3-outcome test.
- **#3332 (edward bf16+bs=8 retry) CLOSED**: bs=8 memory-bandwidth-bound. Per-batch 382→846ms, peak 98GB.
- **Nudges sent**: #3437 thorfinn (0 commits 2h post-dispatch), #3336 fern (max_norm=100 not pushed 2h post send-back).

## Systemic constraints (known issues)

1. **Schedule misalignment** — at `n_hidden=192`, fp32 30-min cap gives 8-9 realized epochs; bf16 gives 12; bs=8 doesn't help. 6-way confirmation. Per-epoch throughput unlocks attempted on cheap axes exhausted on wider trunk.
2. **Cruise-test NaN** — **FIXED in #3378**. All PRs from #3378 onwards report finite 4-split `test_avg/mae_surf_p`.
3. **Low-dimensional per-sample conditioners ruled out** (via #3287 FiLM): predicted-improve splits were most regressed.
4. **Pod heartbeat / Claude-session short exits** — stale_wip pattern (#3127, #3437); pods heartbeat but Claude exits without training. Nudges posted; monitor next iteration.
5. **#3478 stop/pivot result is gating**: determines whether wider trunk is the right base config for future PRs.

## Stop / pivot criteria

- **#3478 result determines the next major direction**:
  - If narrow+bf16 < 126.32: revert wider trunk merge. Re-baseline all round-2 PRs against narrow+bf16.
  - If narrow+bf16 ≈ 126.32 ± 2%: width is a wash. Keep merged config; pursue architectural wins (RMSNorm, SwiGLU, slice_num=32, DropPath).
  - If narrow+bf16 > 130: wider thesis confirmed. Throughput unlocks become priority (torch.compile, slice_num, bf16-wider).
- If RMSNorm (#3496) gives same val but +5% throughput → merge as infrastructure unlock.
- If SwiGLU (#3498) regresses uniformly → gated-FFN axis closed.
- If slice_num=32 (#3500) gives 15+ realized epochs AND val improves → dispatch slice_num=16 sweep.
- If slice_num=32 regresses val AND throughput gained → capacity loss > speed gain; close.
- If DropPath (#3455) regresses uniformly → regularization axis not useful at this budget.
- If grad-clip max_norm=100 (#3336) still regresses → close optimization-stability axis.
- If domain-curriculum (#3437) helps single_in_dist but hurts geom holdouts → easy_boost too aggressive.

## Priority candidates if students free up

1. **torch.compile()** on bf16+wider stack — graph fusion, 10-20% per-epoch expected.
2. **slice_num=16** — follow-up if #3500 wins (how far can we push?).
3. **SmoothL1 sweep (β ∈ {0.5, 2.0})** — only if askeladd's rebased rerun (#3127) confirms SmoothL1 wins.
4. **LayerScale with init=1e-2 or 1.0** — fundamentally different from CaIT cold-start (regularization, not curriculum). Fresh hypothesis if we revisit residual-branch gating.
5. **DropPath sweep** — if #3455 lands at rate=0.1, sweep to {0.2, 0.3}.
6. **Warmup + cosine with budget-aligned warmup fraction** — 3% warmup (< 1 epoch at 12 epochs) instead of 10%.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
