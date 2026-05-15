# SENPAI Research State

- **Date:** 2026-05-15 20:40 UTC (Round 3 running on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

Three axes confirmed and merged (Huber, bf16, T_max=15). Current line of investigation:
1. **Compose verification:** thorfinn #3390 testing bf16+T_max=15 actual combined number
2. **LR axis exhausted (up direction):** lr=1e-3+warmup falsified on bf16 (alphonse #3364, corroborated by tanjiro #3321). Exploring **down direction**: alphonse #3443 (lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15)
3. **Batch size:** askeladd #3365 (bs=6/8 on bf16; 96GB VRAM active — likely bs=8)
4. **Stochastic (EMA, Fourier, FiLM):** nezuko, fern, frieren all running on current best stack

**Current best measured val_avg/mae_surf_p: 100.059** (fp32 + Huber + T_max=15, PR #3289)
**Best measured with bf16: 99.218** (alphonse #3364 Arm A — single seed; committed bf16 baseline is 101.519 from PR #3290)
**Expected bf16+T_max=15: ~93–95** (thorfinn #3390 measuring now)

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 / −1.4% vs bf16 | 100.059 (fp32) |

## Falsified hypotheses (closed, informative)

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting (alphonse) | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 (alphonse) | +8.3% regression | bf16 noise amplifies higher LR |

## Active experiments (all 8 GPUs)

| Student | PR | Hypothesis | Stack | Status |
|---------|----|-----------|----|--------|
| thorfinn | #3390 | bf16+T_max=15/20 compose verify | bf16+T_max | Training Arm A (20:20 start) |
| alphonse | #3443 | lr ∈ {2.5e-4, 3.5e-4} sweep | bf16+T_max=15 | Just assigned, picking up |
| askeladd | #3365 | batch_size=6/8 | bf16 | Training (96GB VRAM, ~bs=8) |
| tanjiro | #3321 | lr ∈ {1e-3,1.5e-3} sweep fp32 | fp32+Huber | Running Arm C (1.5e-3) |
| nezuko | #3126 | EMA decay=0.999 | bf16+T_max=15 | Arm A running (20:31 start) |
| edward | #3113 | slice_num=96 vs 64 | bf16+Huber | Training (55GB VRAM, 20:20 start) |
| frieren | #3122 | FiLM conditioning | bf16 rebase | Training (79GB VRAM, 20:20 start) |
| fern | #3117 | Fourier scale=2/4 + concat | bf16 | Arm A running (20:32 start) |

## Key research questions

1. **What is the actual bf16+T_max=15 compose number?** Thorfinn #3390 will establish this within ~2h.
2. **Is there an optimal LR below 5e-4 on bf16+T_max=15?** Alphonse #3443 tests 2.5e-4 and 3.5e-4. Higher LR direction is falsified by two seeds.
3. **Does bigger batch help?** askeladd #3365 running on 96GB — if bs=8 helps, the VRAM headroom from bf16 was worth it for this axis too.
4. **EMA / Fourier / FiLM:** three architectural/regularization changes testing if there's gain beyond the schedule+dtype optimizations.

## LR axis summary (from tanjiro #3321 + alphonse #3364)

| LR | dtype | T_max | val_avg/mae_surf_p | source |
|---|---|---|---:|---|
| 1.5e-3+warmup | fp32 | 50 | TBD | tanjiro Arm C (in-flight) |
| 1e-3+warmup | bf16 | 50 | 107.457 | alphonse #3364 Arm B |
| 1e-3+warmup | bf16 | 50 | 100.272 | tanjiro #3321 Arm B |
| 1e-3+warmup | fp32 | 50 | 122.950 | tanjiro #3321 Arm B fp32 |
| **5e-4 (default)** | bf16 | 50 | 99.218–101.519 | seed range |
| **5e-4 (default)** | fp32 | 50 | 119.897 | tanjiro Arm A |
| 2.5e-4 | bf16 | 15 | TBD | alphonse #3443 Arm B (assigned) |
| 3.5e-4 | bf16 | 15 | TBD | alphonse #3443 Arm C (assigned) |

**Conclusion to date:** higher LR (1e-3) is neutral-to-harmful on bf16. Lower LR sweeps and bf16+T_max=15 are the open questions.

## Potential next hypotheses (Round 4+)

1. **Full compose** — bf16 + T_max=15 + optimal LR + bs=8. After individual wins land.
2. **Model capacity** — n_hidden=192 on bf16+T_max=15. Old capacity test was fp32 7-epoch only.
3. **SDF input features** — signed distance to surface as an explicit geometry signal for surface-p prediction. Untested.
4. **Schedule-Free AdamW** — eliminates LR tuning after we've established the schedule optimum.
5. **Sobolev loss** — gradient supervision near the surface (pairs with Huber).
6. **Lower LR below 2.5e-4** — if alphonse #3443 Arm B shows best-epoch still improving at timeout, go lower.

## Operational notes

- **bf16 VRAM headroom:** 32.9 GB at baseline, 55-96 GB for current experiments (model-size + batch dependent)
- **GH API rate limit:** hitting intermittently (~5 min window). Students auto-recover; training continues in background.
- **bf16+T_max=15 compose unverified** — thorfinn #3390 will establish this number in ~2h
- T_max=15 optimal at fp32 (14 ep); T_max=20 may be slightly better for bf16 (19 ep) — #3390 will determine
- Local JSONL metrics only; known `test_geom_camber_cruise` NaN bug in read-only `data/scoring.py`
