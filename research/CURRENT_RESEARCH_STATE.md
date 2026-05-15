# SENPAI Research State

- **Date:** 2026-05-15 18:35 UTC (Round 2 → Round 3 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

Two Round 2 axes confirmed and merged. The primary thesis — "wall-clock-truncated runs mean throughput and schedule-fit are first-class levers" — is now validated:

- **bf16 AMP (PR #3290):** 1.345× throughput, 14→19 epochs, val_avg = 101.519 (−8.98% vs Huber)
- **Cosine T_max=15 (PR #3289):** full LR schedule decay inside the wall-clock budget, val_avg = 100.059 on fp32 (−10.3% vs Huber)
- **Compose (pending verification):** bf16+T_max=15 expected ~93–95; thorfinn assigned to verify (#3390)

**Current best measured val_avg/mae_surf_p: 100.059** (fp32 + Huber + T_max=15, PR #3289)
**Best measured with bf16: 101.519** (bf16 + Huber + T_max=50, PR #3290)
**Expected after compose: ~93–95** (bf16 + Huber + T_max=15, to be measured by thorfinn #3390)

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 / −1.4% vs bf16 | 100.059 (fp32) |

## Round 3 assignments (active)

| Student | PR | Hypothesis | Expected val_avg |
|---------|----|-----------|--------------------|
| thorfinn | #3390 | bf16+T_max={15,20} composition verify | ~93–95 |
| alphonse | #3364 | LR=1e-3 + 3-ep warmup on bf16 | ~93–96 |
| askeladd | #3365 | batch_size=6/8 on bf16 | ~95–99 |

**All 8 GPUs occupied — zero idle students.**

## Round 2 still in-flight

| PR | Student | Hypothesis | Status |
|----|---------|-----------|--------|
| #3321 | tanjiro | LR=1e-3+warmup (fp32) | WIP, running |
| #3126 | nezuko | EMA weights | WIP, running |
| #3113 | edward | slice_num=96 (fp32) | WIP, student just resumed |
| #3122 | frieren | FiLM conditioning | WIP, needs Huber+bf16 rebase |
| #3117 | fern | Fourier lower scale + concat | WIP (sent back ×2, needs rebase+bf16) |

## Key research questions for Round 3+

1. **Does bf16+T_max=15 compose additively?** If both give ~-10% independently, does the compose give ~-18%? (thorfinn #3390)
2. **What's the LR optimum on bf16?** alphonse's #3364 tests lr=1e-3+warmup directly on bf16 — if tanjiro's fp32 LR result also lands, it will tell us whether the LR axis is additive with T_max.
3. **Does bigger batch help?** askeladd's #3365 tests bs=6/8 — the bf16 VRAM headroom finally enables this.
4. **Three-way compose:** if all three Round 3 experiments win, Round 4 targets bf16+T_max=15+lr=1e-3+bs=8 in a single config.

## Potential Round 4+ directions

1. **Full compose** — bf16 + T_max=15 + lr=1e-3 + bs=8. If each axis wins independently, this is the high-value compounding experiment.
2. **Model capacity** — n_hidden=192 or n_layers=6 with bf16+T_max=15. Round 1's capacity failure was fp32 only 7 epochs; 19 bf16 epochs changes the picture.
3. **Fourier features** (lower scale) — fern's in-flight send-back. scale=2/4 + concat raw, on bf16.
4. **FiLM conditioning** — frieren's in-flight rebase. If −1.55% on MSE holds on Huber+bf16, it's a merge.
5. **EMA weights** — nezuko's in-flight. Should be a positive or neutral result.
6. **SDF input features** — signed distance to surface as geometry signal. Untested; high-value for surface-pressure prediction.
7. **Schedule-Free AdamW** — eliminates T_max tuning entirely. After nailing down the schedule optimum, test removing it.
8. **Sobolev loss** — gradient supervision near the surface. Pairs with Huber and may help `single_in_dist`.

## Operational notes

- **Current BASELINE.md entries:** PR #3094 (Huber fp32, 111.531), PR #3290 (bf16, 101.519), PR #3289 (T_max=15 fp32, 100.059)
- **bf16+T_max=15 compose unverified** — thorfinn #3390 will establish this number
- All new experiments must include `--amp_dtype bf16`
- T_max=15 is the current best schedule; for bf16 (19 epochs), T_max=20 may be slightly better — #3390 will determine
- 30-min wall-clock cap × 50-epoch cap; ~19 epochs with bf16
- Local JSONL metrics only; known `test_geom_camber_cruise` NaN bug in read-only `data/scoring.py`
