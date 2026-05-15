# SENPAI Research State

- **Date:** 2026-05-15 23:35
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

**New merge bar: val_avg < 90.60 AND test_avg < 83.00.**

Key insight: SmoothL1 (β=0.05) caps gradient contribution of large normalized residuals (|err| > 0.05 → linear instead of quadratic). Composes additively with learnable Fourier. Best epoch was the LAST (14/50, wall-clock limited) — val still declining, suggesting headroom.

## Merge history summary

| PR | val_avg | test_avg | Δ val |
|---|---|---|---|
| #3200 (fern) Fourier 8-band | 121.4956 | 112.4884 | first baseline |
| #3352 (fern) Learnable Fourier | 116.3411 | 107.3254 | −4.24% |
| **#3215 (tanjiro) SmoothL1 β=0.05** | **90.6039** | **83.0029** | **−22.13%** |

## Round 3 — Active WIP (23:35 UTC)

### New Round 3 assignments (all on SmoothL1 baseline):

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3516 | tanjiro | SmoothL1 β sweep: β={0.02, 0.03, 0.075} | Just assigned, 3 sequential arms |
| #3520 | frieren | Pure L1 surface loss (directly aligns with MAE metric) | Just assigned, 1 arm |
| #3523 | edward | Domain one-hot embedding (single vs tandem indicator) | Just assigned, 1 arm |

### Rebasing onto SmoothL1 baseline:

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3356 | thorfinn | Divergence-free aux loss div_weight=0.01 | Sent back 23:21 for rebase on SmoothL1 |
| #3350 | alphonse | FiLM-Re conditioning (per-block Re conditioning) | Sent back 23:27 for rebase on SmoothL1 |
| #3194 | askeladd | LR warmup=3 cosine (vs warmup=0 baseline) | Sent back 23:31 for rebase on SmoothL1 |
| #3207 | nezuko | Geom-conditioned slice assignment | Sent back 23:31 for rebase on SmoothL1 |

### Still in-flight (non-SmoothL1):

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3413 | fern | n_layers=8 + bfloat16 AMP (fp16 bug fixed) | Running since ~23:25; will finish ~23:55 |

### Closed this cycle

- **PR #3353 (frieren slice_num=96+ckpt):** CLOSED. Memory unused (16/96 GB peak); +50% epoch time → only 10 epochs.
- **PR #3441 (frieren slice_num=80 no ckpt):** CLOSED. val=130.17 (+12% worse); actual VRAM 90.6 GB (not 50-55 GB predicted). Slice_num scaling not the bottleneck.
- **PR #3198 (edward per-channel weights):** CLOSED. All 3 arms (p=2,3,5) worse than old baseline. Per-channel loss weights redundant when SmoothL1 already caps outlier gradients.

## Round 1/2 outcomes (all evaluated against OLD baseline 116.34/107.33):

- **PR #3215 (tanjiro SmoothL1 β=0.05):** MERGED ✓ — new baseline 90.60/83.00
- **PR #3356 (thorfinn div-free):** beaten old baseline (−2.5%/−4.2%), NOT new. Sent back for rebase.
- **PR #3350 (alphonse FiLM v3):** val=118.53/test=110.57, doesn't beat new baseline. Sent back for rebase.
- **PR #3194 (askeladd warmup=3):** val=108.10/test=96.40, doesn't beat new baseline. Sent back for rebase.
- **PR #3207 (nezuko geom-slice):** val=116.82/test=105.72, doesn't beat new baseline. Sent back for rebase.
- **PR #3198 (edward p-channel weights):** all arms worse than OLD baseline. CLOSED.
- **PR #3441 (frieren slice_num=80):** val=130.17 (+12% worse). CLOSED.
- **PR #3413 (fern n_layers=8+AMP):** fp16 NaN bug fixed with bfloat16. Re-running now.

## Potential next research directions

**SmoothL1 is now the dominant lever.** Round 3 builds on SmoothL1 as the new foundation (val=90.60, test=83.00). Key research questions now shift to: (a) what compounds WITH SmoothL1? and (b) can we push below val=80 / test=70?

### Round 3 active experiments (on SmoothL1 baseline):

1. **β sweep (tanjiro #3516):** β={0.02, 0.03, 0.075} — maps the optimum around β=0.05
2. **Pure L1 surface loss (frieren #3520):** directly aligned with MAE metric
3. **Domain one-hot embedding (edward #3523):** pure input augmentation, single vs tandem

### Round 3 rebase queue (waiting for compound test on SmoothL1 baseline):
- **thorfinn #3356:** div-free aux loss div_weight=0.01 — beat old baseline by 6%, could compound
- **alphonse #3350:** FiLM Re-conditioning — orthogonal to loss, could compound
- **askeladd #3194:** LR warmup=3 — warmup=3 beat warmup=0 by 14%, could add on SmoothL1
- **nezuko #3207:** geom-slice assignment — geometry OOD conditioning, orthogonal to loss
- **fern #3413:** n_layers=8 + bfloat16 AMP — currently running; if beats 116.34, rebase on SmoothL1

### Unexplored Round 3/4 backlog:
1. OneCycleLR + gradient clipping (scheduler improvement)
2. Per-channel β for surface loss (β_p ≠ β_Ux ≠ β_Uy)
3. SmoothL1 + per-sample normalization (stacked dynamic-range attack)
4. N_FOURIER_BANDS sweep (12/16 bands with SmoothL1)
5. Per-foil-pair conditioning (follow-up to nezuko's camber_rc underperformance)
6. Wider geometry projection for geom-slice (small MLP instead of linear)

**Plateau response:** 5+ consecutive failures → shift to architecture tier (GNN over mesh, Galerkin transformer, spectral-conv hybrid) or data representation (per-sample normalization with clipping).
