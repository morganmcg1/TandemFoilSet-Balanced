# SENPAI Research State

- **Date:** 2026-05-15 17:25
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 122.81`, `test_avg/mae_surf_p = 111.16` (PR #3217 frieren H5 RFF, merged 2026-05-15 16:25)

## Merged improvements so far (baseline stack)

| PR | Hypothesis | val_avg delta | Cumulative val_avg |
|---|---|---|---|
| #3226 thorfinn H10 | Re-strat sampler (Re>1e6 ×2) | — (1st merge) | 127.84 |
| #3217 frieren H5 | RFF coord encoding (n_freq=32) + NaN fix | -5.03 (-3.9%) | **122.81** |

**Baseline now includes:** Re-strat sampler + RFF coord encoding + evaluate_split NaN workaround.

## Per-split current best

| Split | val | test |
|---|---|---|
| `single_in_dist` | 144.70 | 123.91 |
| `geom_camber_rc` | 125.95 | 114.82 |
| `geom_camber_cruise` | 101.61 | 88.14 |
| `re_rand` | 119.00 | 117.78 |
| **avg** | **122.81** | **111.16** |

## Current active WIP PRs (round 2 + new)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #3345 | thorfinn | H11 signed-log1p target transform | WIP (just assigned) |
| #3326 | fern | H12 MLP dropout (dropout=0.1 in FFN) | WIP (training) |
| #3318 | frieren | H6 grad clip + SGDR warm restarts | WIP (training) |
| #3224 | tanjiro | H13 geom-cond v2 (rebase + T_max=15) | WIP (rebasing, needs rebase) |
| #3222 | nezuko | H9 Cautious AdamW | WIP (training) |
| #3201 | edward | H3 channel-weighted surface loss (p×3) | WIP (starting) |
| #3197 | askeladd | H8 EMA (rerun on merged baseline) | WIP (training) |
| #3184 | alphonse | H1 LinearNO ablation | WIP (starting) |

## Closed/Failed this round

| PR | Student | Hypothesis | Outcome |
|---|---|---|---|
| #3291 | thorfinn | H7 two-branch head | Closed — 135.74 (+10.5% worse) |
| #3210 | fern | H2 scale 4M params | Closed — cap-bound (6 epochs) |
| #3224 v1 | tanjiro | H13 geom-cond v1 | Sent back — 134.31 on stale baseline, rerunning v2 |

## Research insights so far

1. **Spectral bias matters**: RFF gave -3.9% val_avg, acting on boundary-layer gradients that raw (x,z) coordinates can't represent efficiently.
2. **High-Re upweighting works**: Re-strat sampler gave -5.03% val_avg in its first pass.
3. **val_single_in_dist is the remaining bottleneck**: at 144.70 it's 20% above the best other split. Split shows wake-interaction complexity not captured by OOD-camber or OOD-Re generalization.
4. **Shared decoder beats specialized decoders at this param budget**: H7 two-branch head regressed every split. Cross-channel feature sharing provides implicit regularization.
5. **Test metrics now reliable**: NaN fix in baseline — test_avg = 111.16. test-val gap is 9.5%; test is consistently lower (better) than val.
6. **30-min cap limits full-cosine runs**: the decisive ~30-pt drop for frieren's RFF came at epoch 11 where cosine finally bit. Multiple restarts (H6 SGDR) or T_max reduction (tanjiro v2) are the countermeasures in flight.

## Round-2 high-value bets

- **H11 log1p targets (thorfinn #3345)**: compresses 13× y-range to ~3×; addresses per-sample magnitude domination of gradients. High leverage if high-Re batches are the bottleneck.
- **H6 SGDR (frieren #3318)**: multiple LR restarts in 50-epoch window; each is a chance to escape plateau. Grad clipping addresses noisy high-Re updates.
- **H13 geom-cond v2 (tanjiro #3224)**: gates learned (0.05→0.17) in v1 confirming mechanism works. v2 rebases on current baseline and applies T_max=15 so cosine fully anneals.
- **H9 Cautious AdamW (nezuko #3222)**: mask-disagreeing updates; expected to improve OOD generalization on top of stable RFF gradients.

## Open questions

- Does H11 log1p address the 144.70 single_in_dist bottleneck more than OOD splits?
- Will EMA + RFF + Re-strat compose to sub-120 val_avg?
- Does Cautious AdamW's gradient masking benefit increase with stable RFF feature space?
- Will tanjiro's geom-cond v2 beat 122.81 on the full cosine cycle?
