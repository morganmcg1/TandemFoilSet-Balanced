# SENPAI Research State

- **Date**: 2026-05-15 17:18
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 cleanup ongoing; round 2 actively dispatching. Three round-1 winners merged:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → current best

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0. Three-axis stack — never measured end-to-end on any single run.

**Systemic finding (just established by edward's #3273)**: the binding constraint on this track is now **per-epoch wallclock at the wider trunk**, not model architecture or loss formulation. At `n_hidden=192`, only 8 epochs fit in the 30-min cap (vs 14 at narrower); the wider trunk's capacity advantage can't manifest. Edward reassigned to **bf16 mixed-precision (#3332)** to directly address this — expected to deliver ~30-40% per-epoch speedup, unlocking ~12-13 epochs at wider.

## PRs in-flight

| PR | Student | Axis | One-line summary | Round |
|---|---|---|---|---|
| #3121 | alphonse | LR schedule | Linear warmup (5%) + cosine annealing + epochs 50→100 (rebase + budget-align) | 1→rerun |
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) — strong-but-stale result 114.14, rebase + budget-align rerun | 1→rerun |
| #3141 | tanjiro | Input encoding | Random Fourier features on (x, z) position (rebase + budget-align rerun) | 1→rerun |
| #3144 | thorfinn | Depth | n_layers 5→8 (rebase + budget-align rerun) | 1→rerun |
| #3277 | nezuko | Output decoupling | Separate Linear→GELU→Linear head per channel (Ux, Uy, p) | 2 |
| #3287 | frieren | Domain conditioning | Per-sample FiLM (scale, shift) on LayerNorm from gap+AoA features (Idea 13) | 2 |
| #3332 | edward | Throughput | bf16 mixed-precision autocast on forward + loss (Idea 3) | 2 |
| #3336 | fern | Optimization stability | Global gradient norm clipping (`max_norm=1.0`) before `optimizer.step()` (NEW) | 2 |

All 8 students have active PRs. **Zero idle GPUs.**

## Recent decisions

- **#3273 (edward ReScaler rebased) CLOSED**: val=152.79 vs baseline 126.32 = +21% regression. ReScaler genuinely works at matched-epoch (+3% vs no-ReScaler matched-epoch) but the wider trunk only fits 8 epochs vs the narrower trunk's 14. Schedule/budget tradeoff dominates.
- **#3298 (fern p_surf_weight) CLOSED**: val=158.54 vs baseline 126.32 = +25% regression. Same wider-trunk-budget pattern as #3273 — matched-epoch comparison (158.54 vs 158.96) shows the hypothesis is approximately **neutral** at this scale. Per-channel reweighting cleanly tested and tested out.
- **#3332 (edward bf16) DISPATCHED**: systemic throughput unlock; targets the per-epoch wallclock that constrained #3273 and #3298. If successful, every future PR can use the wider trunk effectively.
- **#3336 (fern gradient clipping) DISPATCHED**: orthogonal per-step stabilization; doesn't depend on schedule annealing to validate, so it's robust to the current budget constraint.
- **#3277 (nezuko separate-heads)**: still status:wip 2.5h after assignment, but pod logs show iteration 35 picked up the assignment at 16:20 UTC and Claude exited normally at 16:27. Most likely currently running training. No action needed.

## Systemic constraints (known issues)

1. **Schedule misalignment** — at `n_hidden=192`, 30-min wall-clock gives 8 realized epochs; cosine `T_max=50` never anneals. Two paths to address: (a) bf16 (edward #3332 — assigned just now), (b) `epochs ≈ realized_budget` with `T_max=epochs` on every rerun. If bf16 lands as expected, the schedule alignment story simplifies.
2. **Cruise-test NaN** — cruise test sample 20 has corrupt GT (761 Inf in `p` channel); `data/scoring.py::accumulate_batch` propagates NaN through `Inf * 0 = NaN`. Diagnosed independently by **five** students now (nezuko #3137, tanjiro #3141, frieren #3136, fern #3134, edward #3273). Three NaN-safe re-eval patterns established and confirmed functional. Dedicated `data/scoring.py` bug-fix PR with advisor waiver is overdue.

## Round 2 stacking plan

Dispatched (round 2):
- nezuko #3277: Separate per-channel heads (Idea 12)
- frieren #3287: Domain-conditional FiLM (Idea 13)
- fern #3298: per-channel p_surf_weight (Idea 2 refinement)
- **edward #3332: bf16 mixed-precision (Idea 3) — systemic throughput unlock**

Priority candidates as students free up:

1. **OneCycleLR (Idea 18)** — alternative scheduler if alphonse's warmup+cosine disappoints.
2. **Curriculum: low-Re first (Idea 16)** — upweight low-Re in WeightedRandomSampler for first 30% of epochs.
3. **Stochastic depth (Idea 14)** — linearly-increasing block-drop probability; pairs with thorfinn's deeper model.
4. **Temperature annealing (Idea 10)** — anneal `PhysicsAttention.temperature` from 2.0→0.1 across training.
5. **Larger model (Idea 7)** — push to n_hidden=256, n_layers=8, slice_num=128, n_head=8 if all three round-1 capacity axes (wider, deeper, more-slices) showed positive signal. Slice axis already failed (#3134 closed); wider showed signal but is currently budget-constrained until bf16 lands.
6. **Dedicated scoring-fix PR with advisor waiver** — patches `data/scoring.py` to NaN-safe accumulation. Will fix test_avg/mae_surf_p across the board.
7. **SmoothL1 sweep (β ∈ {0.5, 2.0})** — only if askeladd's rebased rerun confirms SmoothL1 wins.
8. **batch_size=8 if bf16 lands** — next throughput stack on top of bf16 once edward proves the mixed-precision baseline is stable.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`

## Stop / pivot criteria

- If best round-2 PR shows <2% improvement over 126.32 → consider architectural pivot: a new backbone family before more hyperparameter tuning. FiLM is the most architectural of the round-2 set; if it fails, that's the strongest signal that we need to think bigger.
- If round-2 PRs reveal the orthogonality assumption is failing (e.g. multiple round-2 PRs combined with current stack underperform a clean baseline by >3%) → run a clean combined-baseline measurement.
- **If bf16 (#3332) fails to deliver meaningful per-epoch speedup**, escalate to either (a) `batch_size=8` (memory-bound, not compute-bound — bf16 isn't the right axis), or (b) profiling the model to find the actual bottleneck (PhysicsAttention slice softmax? the preprocess MLP?).
- If cruise-test NaN persists across 5+ more PRs (we're already at 5 independent diagnoses) → escalate `data/scoring.py` bug-fix to high-priority advisor-waived PR. The three NaN-safe re-eval patterns (tanjiro element-mask, askeladd sample-filter, edward sample-skip-via-y_finite) are usable workarounds for now.
