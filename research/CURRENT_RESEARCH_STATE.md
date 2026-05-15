# SENPAI Research State

- **Date**: 2026-05-15 14:24
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

Round 1 mostly in-flight, round 2 dispatching. Two round-1 winners merged:
- **#3130 edward (wider)**: val_avg/mae_surf_p = 166.50 → first reference
- **#3137 nezuko (EMA)**: val_avg/mae_surf_p = **129.42** → current best (-22% relative)

Active advisor config: `n_hidden=192, n_head=6, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999.

Five round-1 PRs are still in-flight (WIP). Two round-2 experiments dispatched to edward and nezuko.

## PRs in-flight (WIP)

| PR | Student | Axis | One-line summary | Round |
|---|---|---|---|---|
| #3121 | alphonse | LR schedule | Linear warmup (5%) + cosine annealing + epochs 50→100 | 1 |
| #3127 | askeladd | Loss formulation | SmoothL1 (Huber) in normalized space instead of MSE | 1 |
| #3134 | fern | Slice count | slice_num 64→128 | 1 |
| #3136 | frieren | Surface weighting | surf_weight 10→25 | 1 |
| #3141 | tanjiro | Input encoding | Random Fourier features on (x, z) position | 1 |
| #3273 | edward | Re-conditioned scaling | MLP(log_Re) → per-sample output scaler; addresses per-sample p magnitude variation and cruise NaN (Idea 4) | 2 |
| #3277 | nezuko | Output decoupling | Separate Linear→GELU→Linear head per channel (Ux, Uy, p); addresses channel-scale mismatch (Idea 12) | 2 |

## PRs sent back for revision

| PR | Student | Reason |
|---|---|---|
| #3144 | thorfinn | n_layers=5→8 gave 143.82 but only 9/50 epochs realised (~3.4 min/epoch); sent back for rebase onto merged config + `--epochs 8, T_max=8` to budget-align cosine |

## Systemic constraints (known issues)

1. **Schedule misalignment**: 30-min wall-clock allows 9-14 epochs at current scale; cosine T_max=50 never anneals fully. Thorfinn's send-back includes the fix. Round-2 PRs should use `--epochs ≈ realized_budget` with `T_max=epochs` — flagged as recommended in BASELINE.md.
2. **Cruise-test NaN**: cruise test sample 20 has corrupt GT (761 Inf in `p` channel); `data/scoring.py::accumulate_batch` propagates NaN through `Inf * 0 = NaN`. Diagnosed precisely by nezuko (#3137). One-line fix but requires advisor waiver of `data/scoring.py` read-only constraint. Queued — val_avg metrics unaffected, only test_avg impacted.

## Round 2 stacking plan

Dispatched so far (round 2):
- edward: ReScaler — Idea 4, Re-conditioned per-sample output scaler
- nezuko: Separate heads — Idea 12, per-channel output decoders

Priority candidates when further students free up:

1. **Thorfinn rerun (deeper-l8 rebased)** — clean test of depth vs merged config; budget-aligned to ~8 epochs.
2. **Domain-conditional FiLM (Idea 13)** — fully independent; derive domain from gap+AoA features; inject as FiLM shift+scale into LayerNorm.
3. **bf16 + batch_size=8 (Idea 3)** — throughput unlock (~2× epochs/30min); addresses schedule misalignment systemically. Must keep EMA in fp32.
4. **per-channel surf p-weight (Idea 2 refinement)** — if frieren's surf_weight=25 wins, sweep `p_surf_weight ∈ {2, 3, 5}`.
5. **OneCycleLR (Idea 18)** — alternative if alphonse's warmup+cosine disappoints.
6. **Curriculum: low-Re first (Idea 16)** — upweight low-Re in WeightedRandomSampler for first 30% of epochs.
7. **Stochastic depth (Idea 14)** — linearly-increasing block-drop probability; pairs with thorfinn's deeper model.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`

## Stop / pivot criteria

- If best round-2 PR shows <2% improvement over 129.42 → consider architectural pivot: domain FiLM (Idea 13) or a new backbone before more hyperparameter tuning.
- If multiple PRs crash under merged config (n_hidden=192 + EMA) → the stacking orthogonality assumption may fail; run a clean combined-baseline to confirm.
- If cruise-test NaN persists across 3+ more PRs → escalate `data/scoring.py` bug-fix to a high-priority advisor-waived PR — it's obscuring all paper-facing test metrics.
