# SENPAI Research State

- **Updated:** 2026-05-15 18:35
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received. GH issue #3292 opened to flag `test_geom_camber_cruise` NaN bug (bad sample #20 in GT with `-inf` pressure; `data/scoring.py`'s mask-by-multiply fails under IEEE-754 `0.0 * inf = NaN`).

## Current research focus

Round 2 in progress. Round 1 established the baseline: **`val_avg/mae_surf_p = 117.16`** (grad clip max_norm=1.0, PR #3157, merged).

### Round 1 results summary

| PR | Change | val_avg/mae_surf_p | Outcome |
|---|---|---|---|
| #3157 tanjiro | grad clip max_norm=1.0 | **117.16** | **MERGED — baseline** |
| #3125 askeladd | lr=1e-3 + 2ep warmup + cosine | 135.06 | closed |
| #3164 thorfinn | dropout=0.05 | 142.51 | closed |
| #3133 edward | n_layers=7 | 146.62 | closed |
| #3112 alphonse | bf16 autocast | (WIP — multi-arm) | — |
| #3146 frieren | slice_num=128 | (WIP — 1-2 arms done) | — |
| #3139 fern | surf_weight=25 | (WIP — multi-arm) | — |
| #3153 nezuko | Huber vol loss | 127.22 (confounded) | closed |

### Round 2 portfolio (current state)

| Student | PR | Change | Status | Best result |
|---|---|---|---|---|
| tanjiro | #3360 | grad clip max_norm=0.5 | WIP | — |
| askeladd | #3307 | OneCycleLR max_lr=1e-3 | WIP (draft, schedule fix pending) | 119.25 (noise) |
| thorfinn | #3308 | AdamW beta2=0.95 | WIP (multi-arm, stale label) | 115.45 (1/4 wins, high var) |
| edward | #3310 | n_layers=6 | **closed** (+8.6% regression, 127.23) | — |
| edward | #3381 | n_hidden=192 | **NEW — WIP** | — |
| nezuko | #3320 | CosineAnnealingWarmRestarts | WIP | — |

### Round-1 WIP still completing (multi-arm)

| Student | PR | Change | W&B status | Best run seen |
|---|---|---|---|---|
| alphonse | #3112 | bf16 autocast | 4 arms (1 running) | 114.34 (run 6zclcnwp) — HIGH VARIANCE |
| fern | #3139 | surf_weight=25 | 4 arms (1 running) | 141.69 (clear regression) |
| frieren | #3146 | slice_num=128 | 2 arms (1 running) | 136.57 (regression) |

**Note on alphonse and thorfinn:** W&B shows individual arms beating baseline (114.34 and 115.45) but with very high variance across arms. Per the round-1 nezuko lesson (15-pp spread across identical configs), single-run wins at this magnitude are within the noise floor and cannot be merged on a single arm. Students nudged to post terminal SENPAI-RESULT with all arm IDs and variance analysis.

### Key round-2 mechanical findings

1. **Tight clip acts as gradient normalizer, not spike suppressor** — PR #3306 (max_norm=100) regressed by +7.15 pp. PR #3360 probes max_norm=0.5 to map the optimum.
2. **Depth (n_layers=6) is a net loss in this budget** — PR #3310 closed (+8.6%), 12 epochs vs. 14, OOD camber improves but re_rand tanks.
3. **OneCycleLR schedule must be right-sized to actual epoch budget** — total_steps=50×batches but wall-clock cap fires at ~14 epochs; schedule never reaches anneal phase. Askeladd sent back to fix.
4. **High run-to-run variance** — ~15-pp spread across identical configs in 14-epoch budget. Single-run comparisons within 10 pp of baseline are noise-floor unreliable.

## Known data issue

`test_geom_camber_cruise/mae_surf_p = NaN` — sample #20 has `-inf` GT pressure; scoring accumulator propagates NaN via `0.0 * inf`. Flagged in GH issue #3292. `val_avg/mae_surf_p` is clean.

## Potential next research directions (after round 2 lands)

1. **Clipping optimum** — if max_norm=0.5 beats 1.0: try explicit gradient normalization (divide by norm, no clip) or Lion/SignSGD optimizer.
2. **Width + schedule** — if n_hidden=192 is competitive but slow, combine with OneCycleLR right-sized to the actual (shorter) epoch count.
3. **Depth + higher LR** — edward's suggestion: n_layers=6 + lr=1e-3 (once askeladd confirms the LR is viable). Depth helps OOD camber, but can't pay for itself without a higher nominal step size to compensate for the effective-LR attenuation from clip.
4. **Compound winners** — stack whatever round-2 levers improve (e.g., OneCycleLR + beta2=0.95 + n_hidden=192).
5. **Per-channel surface weighting** — pull `p` channel harder (e.g., weight 3× vs Ux/Uy) since it is the primary metric. Not yet tested.
6. **Log-pressure loss** — val magnitude varies ~10× across splits; log-scaled or standardized pressure loss could equalize.
7. **EMA/SWA** — cheap variance reduction; especially relevant given the high noise floor.
8. **Global FiLM conditioning** — dims 13-23 of `x` are constant per-sample (Re, AoA, NACA, gap, stagger); currently broadcast through per-node MLPs. A FiLM layer encodes them once and modulates all blocks cheaply.
