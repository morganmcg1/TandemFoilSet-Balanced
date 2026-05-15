# SENPAI Research State

- **Date**: 2026-05-15 23:05
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 cleanup done; round 2 actively iterating. Round-1 winners merged:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → current best

System fix merged: **#3378 thorfinn (NaN-safe scoring)** — paper-wide finite 4-split test_avg.

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0.

## Closed this loop (Loop 4)

- **#3455 frieren DropPath (CLOSED +16.1%)**: directional disconfirmation — split-spread max/min got worse (1.70 vs ~1.55 baseline). OOD splits didn't outperform in-dist relatively. Lower rates wouldn't fix the directional pattern.
- **#3336 fern grad-clip max_norm=100 (CLOSED +0.81% on committed; ~0% on 3-seed mean)**: multi-seed evidence shows the effect is noise-equivalent. **Key permanent finding**: camber_rc systematically regresses across BOTH clip levels (+17.4% at max_norm=1.0, +11.9% at max_norm=100), indicating that split has gradient norms structurally different from re_rand/cruise. Whatever mechanism distinguishes them is a real research target.

## PRs in-flight (all 8 students active, zero idle GPUs)

| PR | Student | Axis | One-line summary | Round | Status |
|---|---|---|---|---|---|
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) rebased rerun | 1→rerun | wip (nudged) |
| #3437 | thorfinn | Data curriculum | Domain curriculum: racecar_single first, transition to uniform | 2 | wip (nudged) |
| #3478 | edward | Width / throughput | Narrow trunk (n_h=128) + bf16: stop/pivot test | 2 | dispatched |
| #3496 | nezuko | Architecture (normalization) | RMSNorm: replace all LayerNorm, drop-in, -2112 params, +5-10% throughput | 3 | dispatched |
| #3498 | tanjiro | Architecture (MLP) | SwiGLU MLP iso-param d_ff=256 (PaLM/LLaMA gated FFN) | 3 | dispatched |
| #3500 | alphonse | Architecture (attention bandwidth) | slice_num 64→32: direct memory-bandwidth attack | 3 | dispatched |
| #3525 | fern | Optimizer | Lion (sign-momentum, lr=1.5e-4, wd=1e-3) — tests camber_rc grad-norm mechanism | 4 | dispatched |
| #3526 | frieren | Throughput (graph compilation) | torch.compile(dynamic=True) — operator fusion, 10-20% per-epoch predicted | 4 | dispatched |

## Recent decisions

- **#3455 (frieren DropPath) CLOSED**: directional disconfirmation, split-spread got worse not better.
- **#3336 (fern grad-clip max_norm=100) CLOSED**: 3-seed mean neutral; camber_rc systematic regression is the permanent finding.
- **#3525 (fern Lion) DISPATCHED**: sign-momentum optimizer, different mechanism from AdamW. Tests whether camber_rc asymmetry is AdamW-`v`-mediated.
- **#3526 (frieren torch.compile) DISPATCHED**: operator fusion via inductor, dynamic shapes enabled. Priority #1 from prior research state — direct attack on dispatch-bound bottleneck different from slice_num=32 or bf16.
- Earlier this loop: dispatched #3496 nezuko RMSNorm, #3498 tanjiro SwiGLU, #3500 alphonse slice_num=32. Closed #3404 nezuko LayerScale, #3141 tanjiro Fourier rebased, #3435 alphonse aux-task.

## Systemic findings (load-bearing context)

1. **Schedule misalignment** — at `n_hidden=192`, fp32 30-min cap gives 8-9 realized epochs; bf16 gives 12. 6-way confirmed across closed PRs. The Round 4 dispatches (Lion, torch.compile) target throughput from new mechanism classes (optimizer-state reduction, kernel fusion) since the slice/bf16/bs axes are exhausted on the wider trunk.
2. **camber_rc grad-norm asymmetry**: split-systematic regression under grad-clip across both clip levels suggests camber_rc samples need large optimizer steps that re_rand/cruise don't. Lion's sign-only update tests this mechanism class directly.
3. **Low-dimensional per-sample conditioners ruled out** (via #3287 FiLM): predicted-improve splits were most regressed.
4. **CaIT cold-start ruled out** (via #3404 LayerScale, init=1e-4): 5-layer, 12-epoch budget can't grow gates to functional magnitude.
5. **Depth co-adaptation hypothesis disconfirmed** (via #3455 DropPath): split-spread got worse, in-dist regressed more than OOD.
6. **Aux task ruled out at saturation-prone targets** (via #3435 is_surface): aux head saturates at 99% acc by epoch 6.
7. **Cruise-test NaN FIXED in #3378**: 4-split finite test_avg paper-wide.
8. **Pod/Claude-session short-exit pattern**: #3127 askeladd and #3437 thorfinn still stale_wip post-nudge. Monitor next iteration; escalate if no training output emerges.

## Stop / pivot criteria

- **#3478 result determines the next major direction**:
  - narrow+bf16 < 126.32 → revert wider trunk merge; re-baseline all round-2/3/4 PRs.
  - narrow+bf16 ≈ 126.32 ± 2% → width is a wash; keep merged config, pursue architectural wins.
  - narrow+bf16 > 130 → wider thesis confirmed; throughput unlocks become priority.
- **#3526 torch.compile result is high-leverage**: if it gives 12+ realized epochs AND val < 124, it becomes a baseline-stack component for all future PRs.
- **Closure thresholds for round-4 dispatches**:
  - Lion (#3525): regresses by >3% → close optimizer-swap axis. If approximately neutral but camber_rc behavior changes → publishable mechanism finding.
  - torch.compile (#3526): compile crashes outright → close (engineering blocker). If compiles but no per-epoch speedup → model isn't dispatch-bound.

## Priority candidates if students free up next

1. **Per-domain sampling rate sweep** — if #3437 curriculum shows directional signal, sweep transition_frac ∈ {0.3, 0.5, 0.7}.
2. **SwiGLU + RMSNorm composition** — if both #3496 and #3498 win, the joint PR tests independence.
3. **slice_num=16 follow-up** — if #3500 wins at 32, how far can we push?
4. **gradient checkpointing + bs=8** — trade compute for memory; revisits the #3332 bs=8 attempt with VRAM headroom.
5. **Mix MSE/L1 loss** — direct attack on pressure tail; orthogonal to SmoothL1 single-beta test.
6. **Per-domain learning rate** — give camber_rc its own LR if that's the gradient-norm-driven split (follows from fern's grad-clip findings).
7. **LayerScale with init=1.0** — regularization (not cold-start) hypothesis. Fresh PR if we revisit residual gating.
8. **Per-head temperature annealing** in PhysicsAttention.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
