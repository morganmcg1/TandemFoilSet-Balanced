# SENPAI Research State

- **Updated:** 2026-05-15 18:05
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received. GH issue #3292 opened to flag `test_geom_camber_cruise` NaN bug (bad sample #20 in GT with `-inf` pressure; `data/scoring.py`'s mask-by-multiply fails under IEEE-754 `0.0 * inf = NaN`).

## Current research focus

Round 2 in progress. Round 1 established the first baseline: **`val_avg/mae_surf_p = 117.16`** (grad clip max_norm=1.0, PR #3157, merged).

### Round 1 results summary

| PR | Change | val_avg/mae_surf_p | Outcome |
|---|---|---|---|
| #3157 tanjiro | grad clip max_norm=1.0 | **117.16** | **MERGED — new baseline** |
| #3125 askeladd | lr=1e-3 + 2ep warmup + cosine | 135.06 | closed (+15%) |
| #3164 thorfinn | dropout=0.05 | 142.51 | closed (+22%) |
| #3133 edward | n_layers=7 | 146.62 | closed (+25%, instability) |
| #3112 alphonse | bf16 autocast | (WIP) | — |
| #3146 frieren | slice_num=128 | (WIP) | — |
| #3139 fern | surf_weight=25 | (WIP) | — |

**Key insight from round 1:** max_norm=1.0 fires on 100% of steps (median pre-clip norm 45.7), giving effective LR ≈ 1.1e-5. The model is running normalized gradient descent at ~45× lower effective LR than nominal.

### Round 2 portfolio (current state)

| Student | PR | Change | Status | Result |
|---|---|---|---|---|
| tanjiro | #3360 | grad clip max_norm=0.5 | WIP | — |
| tanjiro | #3306 | grad clip max_norm=100.0 | **closed** | 124.31 (+7.15 regression) |
| askeladd | #3307 | OneCycleLR max_lr=1e-3 | **sent back (draft)** | 119.25 (within noise, schedule bug) |
| thorfinn | #3308 | AdamW beta2=0.999 → 0.95 | WIP | — |
| edward | #3310 | n_layers=5 → 6 | WIP | — |
| nezuko | #3320 | CosineAnnealingWarmRestarts T_0=5 T_mult=2 | WIP | — |

Still awaiting round-1 results from: alphonse (#3112 bf16), frieren (#3146 slice_num=128), fern (#3139 surf_weight=25).

### Key round-2 finding: gradient clip mechanism confirmed

PR #3306 (max_norm=100, clip fires on 20.88% of steps) regressed by +7.15 pp. This confirms: **the tight clip is acting as a gradient normalizer, not just spike suppression.** The beneficial mechanism is the uniform step scaling, not gradient sanitization. PR #3360 probes max_norm=0.5 (even tighter) to find the direction optimum.

**Implication for round 3:** if max_norm=0.5 improves further → consider explicit gradient normalization (divide by norm, no bound) or Lion/SignSGD. If max_norm=0.5 regresses → max_norm=1.0 is near-optimal in this direction, headroom is elsewhere.

### askeladd #3307 OneCycleLR schedule-sizing bug

At max_norm=100 analogy: total_steps was set to `len(train_loader) * MAX_EPOCHS` (50 epochs), but only 14 epochs ran under wall-clock cap. Schedule reached only ~28% of its arc — anneal phase never fired. Sent back with fix: `total_steps = len(train_loader) * 14` + guard `if global_step < scheduler.total_steps: scheduler.step()`.

## Critical finding: high run-to-run variance

Nezuko's round-1 experiment ran 3 identical configs and observed **15-point spread** in val_avg/mae_surf_p (127.22 / 141.16 / 141.93). Single-run comparisons in the 14-epoch budget regime are at or near the noise floor. For results within ~10 pp of the baseline, require 2-3 runs to confirm. For clear winners (>15 pp improvement), single runs are likely reliable.

## Known data issue

`test_geom_camber_cruise/mae_surf_p = NaN` on all runs — sample #20 in `test_geom_camber_cruise` GT has `-inf` pressure; the scoring accumulator propagates NaN via `0.0 * inf`. Flagged in GH issue #3292. The validation splits are unaffected; `val_avg/mae_surf_p` is clean.

## Potential next research directions (after round 2 lands)

1. **Understand clipping sweet spot** — if max_norm=0.5 improves: try explicit gradient normalization (divide by norm, no clip). If it regresses: max_norm=1.0 is the optimum; move to other levers.
2. **Explicit gradient normalization / Lion / SignSGD** — if confirmed that gradient direction (not magnitude) is all that matters, use an optimizer built for this (Lion uses sign of gradient + EMA).
3. **OneCycleLR (right-sized)** — askeladd's re-run should reveal whether per-batch scheduling helps once properly matched to the actual 14-epoch budget.
4. **AdamW beta2=0.95** (thorfinn #3308 in flight) — faster second-moment adaptation may help given large gradient scales.
5. **n_layers=6 on clipped baseline** (edward #3310 in flight) — round-1 showed 7 layers unstable; 6 is safer.
6. **Warm restarts** (nezuko #3320 in flight) — CosineAnnealingWarmRestarts T_0=5 T_mult=2.
7. **Compound winners** — stack top round-2 levers if multiple improve (e.g., OneCycleLR + beta2=0.95 + n_layers=6).
8. **surf_weight tuning** — fern's round-1 result (surf_weight=25) still pending. If not a win, try intermediate 15 or per-channel weights.
9. **Wider model** — n_hidden=128 → 192, conditional on bf16 (alphonse result) unlocking VRAM headroom.
10. **Log-pressure loss** — val magnitude varies ~10× across splits; log-scaled or standardized pressure loss could reduce imbalance.
11. **EMA/SWA** — cheap variance reduction for best-checkpoint selection.
12. **Global conditioning (FiLM)** — dims 13-23 of `x` are constant per-sample (Re, AoA, NACA params, gap, stagger); FiLM conditioning would encode them once and modulate all blocks cheaply.
13. **Per-channel surface weighting** — pull `p` channel harder (e.g., weight 3×) since it is the primary metric.
