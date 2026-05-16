# SENPAI Research State

- **Date:** 2026-05-16 01:30 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

**Five axes confirmed and merged.** The compound stack is now: Huber loss + bf16 AMP + cosine T_max=15 + EMA decay=0.999 + FiLM conditioning.

Current best: **92.606 val_avg/mae_surf_p** (frieren #3122, FiLM on full EMA+bf16+T_max=15 stack, 2026-05-16 01:28)

Two additional promising architectural wins pending composition confirmation:
- Fourier features scale=2 (fern #3117): **−9.10% intra-PR on bf16-only**; sent back to rerun on full EMA+T_max=15+FiLM stack
- Two-shot FiLM per block (frieren #3584): just assigned, expected to compound on FiLM baseline

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 / −1.4% vs bf16 | 100.059 (fp32) |
| #3126 | EMA decay=0.999 Karras-ramp (nezuko) | −1.06% vs bf16+T_max=15 paired arm | 96.464 |
| #3122 | FiLM conditioning — log Re, AoA, NACA, gap, stagger (frieren) | −4.00% vs EMA baseline | **92.606** |

## Falsified hypotheses (closed, informative)

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting (alphonse) | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 (alphonse) | +8.3% regression | bf16 noise amplifies higher LR |
| #3321 | lr=1e-3 / 1.5e-3 + warmup on fp32 + bf16 (tanjiro) | +2.2-12% regression (6 arms) | Two-seed confirmation of higher-LR direction dead end |

## Active experiments

| Student | PR | Hypothesis | Stack | Status |
|---------|----|-----------|----|----|
| thorfinn | #3390 | bf16+T_max=15/20 compose verify | bf16+T_max | Stale_wip — pod active, likely finishing arms |
| alphonse | #3443 | lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15 | bf16+T_max=15 | Stale_wip — pod active, likely analyzing |
| askeladd | #3365 | batch_size=6/8 on bf16 | bf16 | Stale_wip — pod active |
| tanjiro | #3511 | grad_clip ∈ {0.5, 1.0, ∞} | bf16+T_max=15+EMA | Training (36 GB VRAM) |
| nezuko | #3492 | n_hidden=192 vs 128 | bf16+T_max=15+EMA | Training (85 GB VRAM) |
| edward | #3113 | slice_num=96 vs 64 | bf16+Huber | Rebased, training (has train signals) |
| frieren | #3584 | Two-shot FiLM (attn + MLP per block) | bf16+T_max=15+EMA+FiLM | Just assigned |
| fern | #3117 | Fourier scale=2 + concat raw (recompose R3) | bf16+T_max=15+EMA+FiLM | Sent back to rebase + rerun on full stack |

## Key research questions

1. **Fourier composition:** fern #3117 showed −9.10% on bf16-only. Does scale=2 concat-Fourier still win when stacked on EMA+T_max=15+FiLM? Expected ~85-89 if composition holds.
2. **Two-shot FiLM (frieren #3584):** incremental benefit of conditioning both attention and MLP per block. Expected −2-6% vs single-shot FiLM baseline (92.606).
3. **LR axis:** alphonse #3443 — does 2.5e-4 or 3.5e-4 help on the EMA+FiLM stack? EMA smoothing may widen the optimal LR range.
4. **Batch size:** askeladd #3365 — larger batch on bf16; if wins, compose with full stack.
5. **Model capacity:** nezuko #3492 — n_hidden=192 on full EMA stack, before FiLM merge.
6. **Grad clipping:** tanjiro #3511 — does clipping (0.5, 1.0) reduce bf16 noise outliers?
7. **Slice_num=96:** edward #3113 — more spatial tokens; additive with capacity changes.

## LR axis summary

| LR | dtype | T_max | val_avg/mae_surf_p | source |
|---|---|---|---:|---|
| 1e-3+warmup | bf16 | 50 | 107.457 | alphonse #3364 Arm B |
| 1e-3+warmup | bf16 | 50 | 100.272 | tanjiro #3321 Arm B |
| **5e-4 (default)** | bf16 | 15 | 97.492 | nezuko #3126 Arm A |
| **5e-4 (default)** | bf16 | 15+EMA | 96.464 | nezuko #3126 Arm B (merged) |
| **5e-4 (default)** | bf16 | 15+EMA+FiLM | **92.606** | frieren #3122 Arm B (merged) |
| 2.5e-4 | bf16 | 15 | TBD | alphonse #3443 Arm B |
| 3.5e-4 | bf16 | 15 | TBD | alphonse #3443 Arm C |

## Potential next hypotheses

1. **Compose Fourier+FiLM** — fern #3117 recompose on full stack likely to give ~85-88; very high priority.
2. **Two-shot FiLM composition** — frieren #3584, in-flight.
3. **Schedule-Free AdamW** — eliminates LR tuning. Strong candidate after LR axis closes.
4. **Sobolev loss** — gradient supervision near surface (pairs with Huber). Physically motivated, untested.
5. **SDF input features** — signed distance to surface as explicit geometry signal. Untested.
6. **n_layers=6/7** — after n_hidden=192 characterized by nezuko.
7. **FiLM on preprocess MLP** — frieren's suggestion #2; small additional param overhead.
8. **Lower LR below 2.5e-4** — if alphonse #3443 shows monotone improvement at lower LR.

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond`
- **GH API rate limit:** recurring ~40-min windows. Student pods auto-recover when limit clears.
- **test_geom_camber_cruise NaN:** known cruise-sample overflow in read-only `data/scoring.py`; use 3-split partial mean for test comparisons.
- **Fourier features (fern #3117):** −9.10% intra-PR on bf16-only confirmed; recompose run on full stack in progress.
- **edward #3113:** pre-EMA stack (Huber+bf16 only); result will be informative but not on full best stack.
