# SENPAI Research State

- **Date**: 2026-05-15 16:55
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 nearly closed out (one PR — fern's slice_num=128, #3134 — closed as a clear cost/benefit loss; round-1 send-backs are running). Round 2 actively dispatching. Three round-1 winners merged so far:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → current best

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0. Three-axis stack — never measured end-to-end on any single run. First round-2 PR rebased onto this config will produce the first true measurement.

**Most exciting in-flight signal**: askeladd's #3127 (SmoothL1) returned val_avg=**114.14** / test_avg=**102.32** on the OLD pre-merge config (n_hidden=128, n_head=4, surf_weight=10, no EMA) — a -31% improvement vs the comparable old-config MSE measurement. Hypothesis strongly validated; sent back for rebase + budget-align to lock in a clean drop-in measurement on the merged stack.

## PRs in-flight (status:wip after rebase send-back, or new round-2 dispatch)

| PR | Student | Axis | One-line summary | Round |
|---|---|---|---|---|
| #3121 | alphonse | LR schedule | Linear warmup (5%) + cosine annealing + epochs 50→100 (rebase + budget-align) | 1→rerun |
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) — strong-but-stale result 114.14, rebase + budget-align rerun | 1→rerun |
| #3141 | tanjiro | Input encoding | Random Fourier features on (x, z) position (rebase + budget-align rerun) | 1→rerun |
| #3144 | thorfinn | Depth | n_layers 5→8 (rebase + budget-align rerun) | 1→rerun |
| #3273 | edward | Re-conditioned scaling | MLP(log_Re) → per-sample output scaler (rebase + tighter bound + budget-align) | 2→rerun |
| #3277 | nezuko | Output decoupling | Separate Linear→GELU→Linear head per channel (Ux, Uy, p) | 2 |
| #3287 | frieren | Domain conditioning | Per-sample FiLM (scale, shift) on LayerNorm from gap+AoA features (Idea 13) | 2 |
| #3298 | fern | Per-channel loss weighting | `p_surf_weight=3.0` multiplier inside surface MSE (Idea 2 refinement) | 2 |

All 8 students have active PRs. **Zero idle GPUs.**

## Recent decisions

- **#3127 (askeladd SmoothL1) sent back**: val=114.14 / test=102.32 is the lowest result we've seen on this track, but measured on the OLD config (4 axes different from current baseline). Hypothesis validated, rebased rerun requested for a clean measurement.
- **#3134 (fern slice_num=128) CLOSED**: 191.65 val_avg = +52% regression. Per-epoch cost ~2× baseline means only 7 epochs realized; the capacity gain is more than offset by epoch-budget loss. Reassigned fern to `p_surf_weight=3.0` (#3298).
- **#3121 (alphonse warmup) and #3127 (askeladd Huber/SmoothL1)** both commented with rebase + budget-align recipe so the rerun produces a clean stacked measurement.
- **#3298 (fern p_surf_weight=3.0) dispatched**: round-2, single attributable knob, identity-at-`p_surf_weight=1.0`, budget-aligned epochs=12.

## Systemic constraints (known issues)

1. **Schedule misalignment**: 30-min wall-clock allows 9-15 epochs at current scale; cosine T_max=50 never anneals fully. All rerun send-backs now require `epochs ≈ realized_budget` with `T_max=epochs`. Flagged in BASELINE.md.
2. **Cruise-test NaN**: cruise test sample 20 has corrupt GT (761 Inf in `p` channel); `data/scoring.py::accumulate_batch` propagates NaN through `Inf * 0 = NaN`. Diagnosed independently by **four** students now (nezuko #3137, tanjiro #3141, frieren #3136, fern #3134). Two NaN-safe re-eval patterns established: tanjiro's `eval_test_clean.py` (mask-before-sum, per-element) and askeladd's `evaluate_split` per-sample `y_finite` filter (#3127). Both work; the dedicated `data/scoring.py` waived bug-fix PR is overdue.

## Round 2 stacking plan

Dispatched (round 2):
- edward #3273: ReScaler (Idea 4)
- nezuko #3277: Separate per-channel heads (Idea 12)
- frieren #3287: Domain-conditional FiLM (Idea 13)
- fern #3298: per-channel p_surf_weight (Idea 2 refinement)

Priority candidates as students free up:

1. **bf16 + batch_size=8 (Idea 3)** — throughput unlock (~2× epochs/30min); addresses schedule misalignment systemically. EMA must stay fp32.
2. **OneCycleLR (Idea 18)** — alternative scheduler if alphonse's warmup+cosine disappoints.
3. **Curriculum: low-Re first (Idea 16)** — upweight low-Re in WeightedRandomSampler for first 30% of epochs.
4. **Stochastic depth (Idea 14)** — linearly-increasing block-drop probability; pairs with thorfinn's deeper model.
5. **Temperature annealing (Idea 10)** — anneal `PhysicsAttention.temperature` from 2.0→0.1 across training.
6. **Larger model (Idea 7)** — push to n_hidden=256, n_layers=8, slice_num=128, n_head=8 if all three round-1 capacity axes (wider, deeper, more-slices) showed positive signal. (Slice axis didn't — proceed only on wider+deeper signal.)
7. **Dedicated scoring-fix PR with advisor waiver** — patches `data/scoring.py` to NaN-safe accumulation. Will fix test_avg/mae_surf_p across the board.
8. **SmoothL1 sweep (β ∈ {0.5, 2.0})** — only if askeladd's rebased rerun confirms SmoothL1 wins. β=1.0 is conservative; smaller β shifts more of the loss into the linear regime.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`

## Stop / pivot criteria

- If best round-2 PR shows <2% improvement over 126.32 → consider architectural pivot: a new backbone family before more hyperparameter tuning. FiLM is the most architectural of the round-2 set; if it fails, that's the strongest signal that we need to think bigger.
- If round-2 PRs reveal the orthogonality assumption is failing (e.g. ReScaler+EMA combined underperforms either alone by >3%) → run a clean combined-baseline measurement.
- If cruise-test NaN persists across 5+ more PRs → escalate `data/scoring.py` bug-fix to high-priority advisor-waived PR. The two NaN-safe re-eval patterns (tanjiro's element-mask, askeladd's sample-filter) are usable workarounds for now.
