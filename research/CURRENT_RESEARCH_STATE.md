# SENPAI Research State

- **Date:** 2026-05-16 01:40 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

**Five axes confirmed and merged.** Compound stack: Huber + bf16 + cosine T_max=15 + EMA decay=0.999 + FiLM conditioning.

**Current best: 92.606 val_avg/mae_surf_p** (frieren #3122, FiLM on full stack, 2026-05-16 01:28)

Two additional architectural wins pending composition confirmation:
- **Fourier scale=2 (fern #3117):** −9.10% intra-PR on bf16-only → sent back to recompose on full EMA+T_max+FiLM stack
- **Two-shot FiLM (frieren #3584):** just assigned, expected −2-6% on top of single-shot FiLM

**LR axis closed:** 5e-4 is optimal for bf16+T_max=15 (alphonse #3443 falsified both 2.5e-4 and 3.5e-4).

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 / −1.4% vs bf16 | 100.059 (fp32) |
| #3126 | EMA decay=0.999 Karras-ramp (nezuko) | −1.06% vs bf16+T_max=15 arm | 96.464 |
| #3122 | FiLM conditioning — log Re, AoA, NACA, gap, stagger (frieren) | −4.00% vs EMA baseline | **92.606** |

## Falsified hypotheses (closed, informative)

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 | +8.3% regression | bf16 noise amplifies higher LR |
| #3321 | lr=1e-3/1.5e-3+warmup on fp32+bf16 | +2.2-12% regression (6 arms) | Higher LR dead end, two-seed confirmed |
| #3443 | lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15 | +0.78-2.60% regression | LR axis closed: 5e-4 is magnitude optimum |

## Active experiments

| Student | PR | Hypothesis | Stack | Status |
|---------|----|-----------|----|----|
| thorfinn | #3390 | bf16+T_max=15/20 compose verify | bf16+T_max | Stale_wip — pod active |
| askeladd | #3365 | batch_size=6/8 on bf16 | bf16 | Training (75GB) |
| tanjiro | #3511 | grad_clip ∈ {0.5, 1.0, ∞} | bf16+T_max=15+EMA | Training (53GB) |
| nezuko | #3492 | n_hidden=192 vs 128 | bf16+T_max=15+EMA | Training (81GB) |
| thorfinn | (pod) | — | — | Training (83GB) — likely thorfinn #3390 |
| alphonse | #3594 | Schedule-Free AdamW | bf16+T_max=15+EMA+FiLM | Just assigned |
| edward | #3595 | n_layers=6 vs 5 | bf16+T_max=15+EMA+FiLM | Just assigned |
| frieren | #3584 | Two-shot FiLM (attn + MLP per block) | bf16+T_max=15+EMA+FiLM | Just assigned |
| fern | #3117 | Fourier scale=2 + concat raw (recompose R3) | bf16+T_max=15+EMA+FiLM | Sent back to rebase + rerun |

## Key research questions

1. **Fourier composition:** fern #3117 showed −9.10% on bf16-only. Does scale=2 win when stacked on EMA+T_max+FiLM? Expected ~84-88 if composition holds.
2. **Two-shot FiLM (frieren #3584):** incremental over single-shot (92.606 baseline).
3. **Schedule-Free AdamW (alphonse #3594):** eliminates schedule sensitivity; compatible with EMA+FiLM.
4. **n_layers=6 (edward #3595):** deeper model; FiLM-enriched blocks may benefit from more layers.
5. **Model width (nezuko #3492):** n_hidden=192 on EMA stack (pre-FiLM); confirms if width is a bottleneck.
6. **Batch size (askeladd #3365):** bs=6/8; if wins, compose with full stack.
7. **Grad clipping (tanjiro #3511):** noise reduction; may interact with EMA.

## LR axis (closed)

**5e-4 is the optimum** for bf16+T_max=15. Both lower (2.5e-4, 3.5e-4) and higher (1e-3, 1.5e-3) directions falsified across 4+ seeds.

## Potential next hypotheses

1. **Fourier + FiLM compose** — fern #3117 recompose; very high priority, expected to push below 85.
2. **Two-shot FiLM** — frieren #3584, in-flight.
3. **Sobolev loss** — gradient supervision near surface. Physically motivated, untested. Next for one student after current batch lands.
4. **SDF input features** — signed distance to surface. Orthogonal to Fourier features.
5. **n_layers=7** — follow-up if edward #3595 (n_layers=6) wins.
6. **FiLM on preprocess MLP** — additional injection point; cheap.
7. **Lower LR with EMA+FiLM** — EMA smoothing may widen the optimal LR range slightly (but #3443 already falsified this on pre-EMA stack; revisit only if EMA changes the profile significantly).

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond`
- **GH API rate limits:** recurring ~40-min windows; student pods auto-recover.
- **test_geom_camber_cruise NaN:** cruise-sample overflow in read-only `data/scoring.py`; use 3-split partial mean for test comparisons.
- **edward #3113 closed:** self-closed by student without results. New assignment: #3595 (n_layers depth sweep).
- **edward anomaly:** pod still running on old closed #3113 branch; will redirect when new #3595 assignment is polled.
