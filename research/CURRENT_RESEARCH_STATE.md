# SENPAI Research State

- **Date:** 2026-05-16 00:42 UTC (Cycle 8)
- **Advisor branch:** `icml-appendix-willow-pai2i-24h-r2`
- **Target base branch:** `icml-appendix-willow`
- **W&B project:** `wandb-applied-ai-team/senpai-v1`
- **Hard limits:** `SENPAI_MAX_EPOCHS=50`, `SENPAI_TIMEOUT_MINUTES=30.0`, 1 GPU / 96GB per student

## Most recent research direction from human researcher team

None — no human directives on this launch.

## Current baseline (merged into advisor branch)

**PR #3215 (tanjiro) — SmoothL1 (Huber) loss β=0.05** — merged 2026-05-15 23:20

- `val_avg/mae_surf_p` = **90.6039**
- `test_avg/mae_surf_p` = **83.0029**
- W&B run: `iofja54s`

Per-split (val | test): single=112.03|101.95, camber_rc=104.42|97.84, camber_cruise=62.07|55.10, re_rand=83.89|77.11.

**Current merge bar: val_avg < 90.60 AND test_avg < 83.00.**

## Cycle 8 findings (00:42 UTC)

Two experiments have **already beaten the baseline** per W&B — waiting for terminal SENPAI-RESULT comments to confirm before merge:

| Run | PR | Config | W&B val | W&B test | vs baseline |
|---|---|---|---|---|---|
| `pykk0x44` | #3516 (tanjiro) | SmoothL1 β=0.02 | **88.11** | **77.91** | −2.75% val / −6.1% test |
| `a42b4ca9` | #3356 (thorfinn) | div-free (div_weight=0.01) + SmoothL1 | **87.87** | **78.83** | −3.0% val / −5.0% test |

Both show physics-informed and loss-hyperparameter approaches compounding. If confirmed, the new baseline will be ~val=87–88, test=77–78.

## Round 3 — Active WIP (00:42 UTC)

### New Round 3 assignments (all on SmoothL1 baseline):

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3516 | tanjiro | SmoothL1 β sweep: β={0.02, 0.03, 0.075} | β=0.02 finished (val=88.11, test=77.91 **WINNER**), β=0.03 running (~01:00 done), β=0.075 pending |
| #3520 | frieren | Pure L1 surface loss (directly aligns with MAE metric) | Running (~01:00 done) |
| #3523 | edward | Domain one-hot embedding (single vs tandem indicator) | Running (~01:00 done) |
| #3568 | fern | mlp_ratio=4 (widen FFN inside Transolver blocks) | Just assigned — pivot from failed depth experiment |

### Rebasing onto SmoothL1 baseline:

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3356 | thorfinn | Divergence-free aux loss div_weight=0.01 | Rebased, first arm finished (val=87.87, test=78.83 **WINNER**), second replication running (~00:52 done) |
| #3350 | alphonse | FiLM-Re conditioning (per-block Re conditioning) | Sent back 23:27, not yet started — student likely rebasing |
| #3194 | askeladd | LR warmup=3 cosine (vs warmup=0 baseline) | Sent back 23:29, not yet started |
| #3207 | nezuko | Geom-conditioned slice assignment | Sent back 23:31, not yet started |

### Closed this cycle

- **PR #3413 (fern n_layers=8 + bfloat16 AMP):** CLOSED. val=134.88/test=119.34 — worse than learnable-Fourier baseline (116.34/107.33). Depth scaling requires more training budget than 30-min wall-clock allows. bf16 AMP recipe validated and reusable.
- **PR #3198 (edward per-channel weights):** CLOSED (cycle 7). All 3 arms worse than old baseline.
- **PR #3441 (frieren slice_num=80):** CLOSED (cycle 7). val=130.17 (+12% worse).

## GitHub API rate limit note

Rate limit exhausted at 00:41 UTC (0/5000 remaining). Resets at **01:19 UTC**. Wakeup scheduled for 01:20 UTC to harvest results.

## Potential next research directions

**SmoothL1 is the foundation; we're now compounding.** Two experiments already beat val=90.60:
- β=0.02 (better β for SmoothL1) — val=88.11, test=77.91
- div-free + SmoothL1 — val=87.87, test=78.83

**Open question: can these two compound further?** SmoothL1 β=0.02 + div-free aux loss could push below val=85.

### Round 3 active bets:

1. **β sweep (tanjiro #3516):** Maps the optimum around β=0.05; β=0.02 already wins
2. **Pure L1 surface loss (frieren #3520):** L1 is the β→0 limit of SmoothL1; if the optimum is β<0.02, L1 should win
3. **Domain one-hot embedding (edward #3523):** Pure input feature — single vs tandem indicator
4. **mlp_ratio=4 (fern #3568):** Width expansion in Transolver FFN; orthogonal to all above
5. **div-free + SmoothL1 (thorfinn #3356):** Compounding physics regularization with SmoothL1
6. **FiLM-Re + SmoothL1 (alphonse #3350):** Compounding Re conditioning with SmoothL1
7. **LR warmup=3 + SmoothL1 (askeladd #3194):** warmup=3 was 14% better on old baseline; test compound
8. **Geom-slice + SmoothL1 (nezuko #3207):** Geometry OOD conditioning + SmoothL1

### Unexplored Round 3/4 backlog:

1. **β=0.02 + div_weight=0.01 compound** — stack the two proven winners
2. **OneCycleLR scheduler** — fast convergence in short compute budgets
3. **Per-channel β** (β_p ≠ β_Ux ≠ β_Uy) — different regularization per field
4. **Larger batch_size (8 or 12) + linear LR scaling** — exploits unused 53 GB VRAM
5. **N_FOURIER_BANDS sweep** (12/16 bands with SmoothL1)
6. **div_weight sweep (0.005, 0.01, 0.02, 0.05)** — map weight-vs-improvement curve around 0.01
7. **Wider geometry projection for geom-slice** (small MLP instead of linear)
8. **Stochastic Depth / DropPath** — regularization orthogonal to physics

### Architecture tier (if current levers saturate):

- GNN over mesh (graph-native geometry encoding)
- Galerkin transformer (spectral-basis attention)
- Spectral-conv hybrid (FNO layers + Transolver)
- Per-sample normalization with clipping (dynamic range attack)

**Plateau response:** 5+ consecutive failures → shift to architecture tier above.
