# SENPAI Research State

- **Date:** 2026-05-16 01:35 UTC (Cycle 9)
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

## Cycle 9 W&B Snapshot (01:25 UTC) — Confirmed Winners Awaiting SENPAI-RESULT

Four experiments beat the current baseline and are awaiting terminal student comments before merge:

| Run | PR | Student | Config | Best val | Best test | W&B State |
|---|---|---|---|---|---|---|
| `anr2xaul` | #3350 | alphonse | FiLM-Re + SmoothL1 β=0.05 | **86.53** | 80.47 | finished |
| `a42b4ca9` | #3356 | thorfinn | div-free w=0.01 + SmoothL1 β=0.05 | 87.87 | **78.83** | finished |
| `es15998q` | #3350 | alphonse | FiLM-Re + SmoothL1 (rep 1) | 87.51 | 81.36 | finished |
| `wju9cic5` | #3516 | tanjiro | β=0.03 | 88.83 | 80.02 | finished |

All confirmed on SmoothL1 baseline (smooth_l1_beta=0.05 in W&B config).

**Not yet with SENPAI-RESULT**: Students haven't flipped to `status:review` yet. They may still be completing additional sweep arms.

## Round 3 — Active WIP (01:35 UTC)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3516 | tanjiro | SmoothL1 β sweep: β={0.02, 0.03, 0.075} | β=0.02 done (val=88.11), β=0.03 done (val=88.83), β=0.075 running (~01:52 done) |
| #3520 | frieren | Pure L1 surface loss | 1st arm done (val=93.98, FAILED), 2nd arm just started (~01:55 done) |
| #3597 | edward | batch_size=8 + lr=1e-3 (linear LR scaling) | Just assigned |
| #3568 | fern | mlp_ratio=4 (FFN width expansion) | Running since ~01:21, finishing ~01:51 |
| #3356 | thorfinn | div-free + SmoothL1 β=0.05 | 1st run won (val=87.87, test=78.83), 2nd run finished (val=92.10 — worse), student posting results |
| #3350 | alphonse | FiLM-Re + SmoothL1 β=0.05 | 2 finished runs (val=86.53, 87.51), 3rd arm running (~01:52 done) |
| #3194 | askeladd | LR warmup=3 + SmoothL1 | Rebased 00:29, arm running (~01:52 done) |
| #3207 | nezuko | Geom-conditioned slice + SmoothL1 | Not yet rebased — pod may be slow |

## Closed this cycle

- **PR #3523 (edward domain one-hot):** CLOSED. val=96.26/test=86.25 — +6.25%/+3.91% worse. Binary indicator hurts (shortcut pathology).
- **PR #3413 (fern n_layers=8 + bf16):** CLOSED (cycle 8). val=134.88 — depth scaling fails at 30-min budget.

## Potential next research directions

**Multiple winners confirmed compounding with SmoothL1:**
- FiLM-Re (alphonse): val=86.53 — physics-informed Re conditioning works!
- div-free (thorfinn): val=87.87, test=78.83 — velocity divergence penalty helps OOD!
- β=0.02-0.03 (tanjiro): val=88.11-88.83 — tighter β threshold still helps!

**Key next step: compound the winners.** The strongest single-experiment win is alphonse's FiLM-Re (val=86.53, best val). Thorfinn's div-free wins on test (78.83, best test). If these can be stacked: FiLM-Re + div-free + SmoothL1 could push below val=85.

### After next merge(s):

1. **FiLM-Re + div-free compound** — stack alphonse and thorfinn's mechanisms
2. **β=0.02 + FiLM-Re compound** — if β=0.02 is the sweet spot
3. **warmup=3 + FiLM-Re compound** — if askeladd shows warmup helps on SmoothL1
4. **Per-channel β** (β_p=0.05, β_Ux=β_Uy=0.10) — different curvature per field
5. **N_FOURIER_BANDS=12** — more Fourier capacity on SmoothL1 baseline

### Unexplored but queued:

- Larger batch_size (edward #3597 — running)
- mlp_ratio=4 (fern #3568 — running)
- Pure L1 loss (frieren #3520 — running)
- Geom-slice (nezuko #3207 — not yet rebased, concerning)

### Architecture tier (if plateau):

- GNN over mesh
- Galerkin transformer
- Spectral-conv hybrid
- Per-sample normalization with clipping
