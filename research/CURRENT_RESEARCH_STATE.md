# SENPAI Research State

- **Date:** 2026-05-16 01:25 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

**Four axes confirmed and merged.** The compound stack is now: Huber loss + bf16 AMP + cosine T_max=15 + EMA decay=0.999.

Current best: **96.464 val_avg/mae_surf_p** (nezuko #3126, EMA + bf16 + T_max=15, 2026-05-15 22:32)

1. **Compose verification still pending:** thorfinn #3390 measuring bf16+T_max=15 (result: nezuko Arm A confirmed 97.492, close to the predicted 93–95 floor; thorfinn in-flight for 2nd seed)
2. **LR axis (down direction):** alphonse #3443 testing lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15. Higher LR falsified by two seeds.
3. **Batch size:** askeladd #3365 (bs=6/8 on bf16; ~77 GB VRAM, actively training)
4. **Architectural changes:** **fern #3117 (Fourier scale=2 concat raw) showed −9.10% intra-PR win (Arm B 93.967 vs Arm A 103.370 on bf16, no EMA/T_max) — sent back for rebase + recompose on full EMA+T_max=15 stack to confirm composition before merge.** frieren #3122 (FiLM, rebased 23:23, training)
5. **Model capacity:** nezuko just assigned #3492 (n_hidden=192 on full best stack)
6. **Slice num:** edward #3113 (slice_num=96 vs 64, rebased, training)
7. **LR sweep (fp32 higher):** tanjiro #3321 (just completed arms, GPU idle — likely posting results)

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 / −1.4% vs bf16 | 100.059 (fp32) |
| #3126 | EMA decay=0.999 Karras-ramp (nezuko) | −1.06% vs bf16+T_max=15 paired arm | **96.464** |

## Falsified hypotheses (closed, informative)

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting (alphonse) | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 (alphonse) | +8.3% regression | bf16 noise amplifies higher LR |
| #3321 | lr=1e-3 / 1.5e-3 + warmup on fp32 + bf16 (tanjiro) | +2.2-12% regression (6 arms) | Two-seed confirmation of higher-LR direction dead end |

## Active experiments

| Student | PR | Hypothesis | Stack | Status |
|---------|----|-----------|----|--------|
| thorfinn | #3390 | bf16+T_max=15/20 compose verify | bf16+T_max | 1 arm complete 22:58, post-training idle |
| alphonse | #3443 | lr ∈ {2.5e-4, 3.5e-4} sweep | bf16+T_max=15 | 2+ arms complete by 22:55, post-training idle |
| askeladd | #3365 | batch_size=6/8 | bf16 | Training (77 GB previously, now 0; analysis phase) |
| tanjiro | #3511 | grad_clip ∈ {0.5, 1.0, ∞} | bf16+T_max=15+EMA | Just assigned (after #3321 closed-falsified) |
| nezuko | #3492 | n_hidden=192 vs 128 | bf16+T_max=15+EMA | Just assigned |
| edward | #3113 | slice_num=96 vs 64 | bf16+Huber | Pod restarted iter 9 — picking up |
| frieren | #3122 | FiLM conditioning | bf16+T_max=15+EMA | Rebased 23:23, launching arms |
| fern | #3117 | Fourier scale=2 + concat raw (R3) | bf16+T_max=15+EMA | **R2 strong winner sent back for rebase + recompose: Arm B 93.967 on bf16-only (−9.10% intra-PR); rerun on full stack to confirm composition** |

## Key research questions

1. **EMA + compose:** What does bf16+T_max=15+EMA achieve on tanjiro/thorfinn seeds? Nezuko's paired Arm A (97.492) is first measurement; thorfinn will confirm.
2. **Is there an optimal LR below 5e-4?** Alphonse #3443 tests 2.5e-4 and 3.5e-4. Lower LR may interact with EMA smoothing.
3. **Does bigger batch help?** askeladd #3365 — if bs=8 wins, compose it with EMA stack.
4. **Model capacity:** nezuko #3492 — n_hidden=192 on full stack tests whether representation width is a bottleneck.
5. **Fourier / FiLM:** both blocked by merge conflicts. Priority to resolve.
6. **Slice_num=96:** edward #3113 — additive to n_hidden test if both win.

## LR axis summary

| LR | dtype | T_max | val_avg/mae_surf_p | source |
|---|---|---|---:|---|
| 1.5e-3+warmup | fp32 | 50 | TBD | tanjiro Arm C (GPU idle, likely done) |
| 1e-3+warmup | bf16 | 50 | 107.457 | alphonse #3364 Arm B |
| 1e-3+warmup | bf16 | 50 | 100.272 | tanjiro #3321 Arm B |
| **5e-4 (default)** | bf16 | 15 | 97.492 | nezuko #3126 Arm A (first compose measurement) |
| **5e-4 (default)** | bf16 | 15+EMA | **96.464** | nezuko #3126 Arm B (merged) |
| 2.5e-4 | bf16 | 15 | TBD | alphonse #3443 Arm B |
| 3.5e-4 | bf16 | 15 | TBD | alphonse #3443 Arm C |

## Potential next hypotheses (Round 4)

1. **Compose full stack** — bf16 + T_max=15 + EMA + optimal LR (after alphonse #3443 lands) + bs (after askeladd #3365 lands).
2. **EMA with current best LR** — once lower LR is characterized, re-test EMA at optimal LR.
3. **SDF input features** — signed distance to surface as an explicit geometry signal. Untested.
4. **Schedule-Free AdamW** — eliminates LR tuning. Strong candidate once we've characterized the LR axis.
5. **Sobolev loss** — gradient supervision near the surface (pairs with Huber). Physically motivated.
6. **Lower LR below 2.5e-4** — if alphonse #3443 Arm B best-epoch still improving at budget end.
7. **n_layers=6 or n_layers=7** — after n_hidden=192 is characterized.
8. **FiLM / Fourier** — once merge conflicts resolved; promising prior results from earlier rounds.

## Operational notes

- **EMA merged (PR #3126):** decay=0.999 Karras-ramp, built into train.py as `--use_ema --ema_decay 0.999`
- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999`
- **GH API rate limit:** recurring ~40-min windows. Student pods auto-recover when limit clears.
- **bf16+T_max=15 compose measured:** 97.492 (nezuko Arm A single seed); thorfinn #3390 verifying second seed
- Local JSONL metrics only; known `test_geom_camber_cruise` NaN bug in read-only `data/scoring.py`
- **Frieren/Fern merge conflicts:** both need rebase onto current advisor branch; pods re-invoked 22:23 and should be resolving
