# SENPAI Research State

- **Date:** 2026-05-15 14:45
- **Branch:** `icml-appendix-charlie-pai2i-24h-r4`
- **Round:** charlie-pai2i-24h-r4 (24h, 8 students × 1 GPU, local JSONL metrics only)
- **Most recent human research directive:** _none — issue queue empty_
- **Primary metric:** `val_avg/mae_surf_p` (lower is better)
- **Current baseline:** `val_avg/mae_surf_p = 127.84` (PR #3226 thorfinn H10 Re-strat, merged 2026-05-15)

## Round-1 results summary

Four PRs landed for review. Ranking by `val_avg/mae_surf_p`:

| # | Student | Hypothesis | val_avg | Outcome |
|---|---------|------------|---------|---------|
| 3226 | thorfinn | H10 Re-strat sampler | **127.84** | **MERGED → new baseline** |
| 3197 | askeladd | H8 EMA (decay=0.999) | 132.17 | Send back: re-run on merged baseline |
| 3224 | tanjiro  | H13 gated geom-cond | 134.31 | Send back: re-run on merged baseline + fix cap |
| 3210 | fern     | H2 scale to 4M params | 158.40 | Send back: add grad clip + lower lr + smaller variant |

Four PRs still WIP from round 1: alphonse (H1 LinearNO), edward (H3 channel-weighted loss), frieren (H5 RFF), nezuko (H9 Cautious AdamW). They will likely come in for review during the next iteration.

## Current research focus

Round 2 begins with the Re-strat sampler in the baseline. The dominant signal so far:

- **High-Re upweighting works** — best val_re_rand (111.08) and val_geom_camber_cruise (91.50) among submitted PRs.
- **OOD vs in-dist asymmetry persists** — `val_single_in_dist` is the hardest split at ~160 mae_surf_p for the merged baseline. This is the obvious next target.
- **All three completed-but-not-merged ideas (EMA, geom-cond, scale-up)** are orthogonal mechanisms that should compose with Re-strat — re-running them on the merged baseline closes that experiment cleanly.

## Known branch-wide quirk

`test_avg/mae_surf_p` is currently **NaN** for every PR on this branch. Root cause: `data/scoring.py` (read-only, can't modify) accumulates `(pred - y).abs() * surf_mask`. Sample 20 in `test_geom_camber_cruise` has 761 `inf` values in `y[..., 2]` (p channel). Because `NaN * 0 = NaN` (IEEE 754), the infinity propagates through the mask multiplication into the accumulator, contaminating `test_avg`. Three of four students independently spotted this. **Workaround:** rank on `val_avg/mae_surf_p`, report the 3 finite test splits separately.

## Round-2 priorities

1. **Get round-1 reruns on the new baseline** — askeladd (EMA), tanjiro (geom-cond fixed cap), fern (smaller variant + grad clip). These are essentially free information: known mechanisms tested over a known better baseline.

2. **New mechanism for newly-idle thorfinn** — H7 two-branch output head (surface vs volume decoder), targeting `val_single_in_dist` and `val_geom_camber_rc` which are still the two hardest splits at 160 / 149. Dedicated surface decoder capacity should help the metric we're actually scored on.

3. **Watchlist for incoming round-1 PRs** — alphonse, edward, frieren, nezuko. If any beat 127.84, merge then chain.

## Potential next research directions (round 3+)

- **Compounding round:** stack the round-1+2 winners into a single bundled PR.
- **Asymmetric Q/K projections (H4):** orthogonal attention modification.
- **Gradient clipping + SGDR (H6):** stability + warm restarts.
- **MLP dropout (H12) or log1p target normalization (H11):** lightweight regularization.
- **GeoTransolver GALE (full):** multi-scale ball queries + full cross-attention conditioning if simple H13 shows OOD gains.
- **Loss reformulation:** Huber on surface pressure, per-domain loss normalization.
- **Curriculum/mining:** hard-sample mining by per-sample MAE during training.
- **Spectral targeting:** SIREN-style learned coordinate encoding, multi-band RFF sweep on σ if frieren H5 shows traction.

## Open questions

- Does EMA's gain on top of Re-strat reach val_avg < 125?
- Is the two-branch head the right level of capacity allocation, or is the surface task too small to need a dedicated decoder?
- The `val_single_in_dist` split is the hardest now (160.10) — what's different about it from val_re_rand (111.08)? Worth a data audit before round 3.

## Living document

Update this file each round with the latest research focus, themes, and
open questions. Prune stale entries; merge winners into the baseline.
