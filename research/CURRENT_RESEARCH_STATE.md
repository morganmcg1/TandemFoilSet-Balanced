# SENPAI Research State

- **Date**: 2026-05-15 14:38
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 nearly complete (3 PRs still WIP); round 2 actively dispatching. Three round-1 winners merged so far:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = 129.42 (-22%)
- **#3136 frieren (surf_weight=25)**: val_avg/mae_surf_p = **126.32** (-2.4%) → current best

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0. Three-axis stack — never measured end-to-end on any single run. First round-2 PR rebased onto this config will produce the first true measurement.

Three round-1 PRs still in-flight (alphonse warmup, askeladd Huber, fern slice_num=128). Three round-2 experiments dispatched (edward ReScaler, nezuko separate-heads, frieren domain-FiLM).

## PRs in-flight (WIP / status:wip after rebase)

| PR | Student | Axis | One-line summary | Round |
|---|---|---|---|---|
| #3121 | alphonse | LR schedule | Linear warmup (5%) + cosine annealing + epochs 50→100 | 1 |
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) in normalized space instead of MSE | 1 |
| #3134 | fern | Slice count | slice_num 64→128 | 1 |
| #3141 | tanjiro | Input encoding | Random Fourier features on (x, z) position (rebase + budget-align rerun) | 1→rerun |
| #3144 | thorfinn | Depth | n_layers 5→8 (rebase + budget-align rerun) | 1→rerun |
| #3273 | edward | Re-conditioned scaling | MLP(log_Re) → per-sample output scaler (rebase + tighter bound) | 2→rerun |
| #3277 | nezuko | Output decoupling | Separate Linear→GELU→Linear head per channel (Ux, Uy, p) | 2 |
| #3287 | frieren | Domain conditioning | Per-sample FiLM (scale, shift) on LayerNorm from gap+AoA features (Idea 13) | 2 |

## Recent decisions

- **#3136 (frieren surf_weight=25) MERGED**: clean win at 126.32 val_avg (-2.4% vs 129.42).
- **#3273 (edward ReScaler) sent back**: -7.9% over pre-#3137 baseline (matches prediction), but 153.31 doesn't beat 129.42. Rebase + `max_log_scale=2.0→1.0` + optional `pred.clamp(-50, 50)` + budget-aligned schedule (8 epochs / T_max=8) requested.
- **#3141 (tanjiro Fourier) sent back**: 136.14 val_avg doesn't beat 129.42 but **NaN-safe test re-eval came in at 122.90** (lowest test number seen on this branch). Rebase + budget-align (`epochs=12 / T_max=12`) requested. Tanjiro's `eval_test_clean.py` is the standard test-eval pattern now.
- **#3144 (thorfinn deeper-l8) still sent back from prior round**: rebase + `--epochs 8, T_max=8` to budget-align.

## Systemic constraints (known issues)

1. **Schedule misalignment**: 30-min wall-clock allows 9-15 epochs at current scale; cosine T_max=50 never anneals fully. All rerun send-backs now require `epochs ≈ realized_budget` with `T_max=epochs`. Flagged in BASELINE.md.
2. **Cruise-test NaN**: cruise test sample 20 has corrupt GT (761 Inf in `p` channel); `data/scoring.py::accumulate_batch` propagates NaN through `Inf * 0 = NaN`. Diagnosed independently by nezuko (#3137), tanjiro (#3141), and frieren (#3136). Tanjiro's `eval_test_clean.py` (`torch.where(mask, err, 0)` before sum) is the working NaN-safe re-eval and is now the standard for reporting test numbers. One-line fix would require advisor waiver of `data/scoring.py` read-only constraint — queued as dedicated bug-fix PR.

## Round 2 stacking plan

Dispatched (round 2):
- edward #3273: ReScaler (Idea 4)
- nezuko #3277: Separate per-channel heads (Idea 12)
- frieren #3287: Domain-conditional FiLM (Idea 13)

Priority candidates as students free up:

1. **per-channel surf p-weight refinement (Idea 2 refinement)** — now that surf_weight=25 is in the merged baseline, layering a per-channel `p_surf_weight` on top is the natural follow-up. Single attributable change.
2. **bf16 + batch_size=8 (Idea 3)** — throughput unlock (~2× epochs/30min); addresses schedule misalignment systemically. EMA must stay fp32.
3. **OneCycleLR (Idea 18)** — alternative scheduler if alphonse's warmup+cosine disappoints.
4. **Curriculum: low-Re first (Idea 16)** — upweight low-Re in WeightedRandomSampler for first 30% of epochs.
5. **Stochastic depth (Idea 14)** — linearly-increasing block-drop probability; pairs with thorfinn's deeper model.
6. **Temperature annealing (Idea 10)** — anneal `PhysicsAttention.temperature` from 2.0→0.1 across training.
7. **Larger model (Idea 7)** — push to n_hidden=256, n_layers=8, slice_num=128, n_head=8 if all three round-1 capacity axes (wider, deeper, more-slices) showed positive signal.
8. **Dedicated scoring-fix PR with advisor waiver** — patches `data/scoring.py` to NaN-safe accumulation. Will fix test_avg/mae_surf_p across the board.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`

## Stop / pivot criteria

- If best round-2 PR shows <2% improvement over 126.32 → consider architectural pivot: a new backbone family before more hyperparameter tuning. FiLM is the most architectural of the round-2 set; if it fails, that's the strongest signal that we need to think bigger.
- If round-2 PRs reveal the orthogonality assumption is failing (e.g. ReScaler+EMA combined underperforms either alone by >3%) → run a clean combined-baseline measurement.
- If cruise-test NaN persists across 5+ more PRs → escalate `data/scoring.py` bug-fix to high-priority advisor-waived PR. The NaN-safe re-eval pattern is a usable workaround for now.
