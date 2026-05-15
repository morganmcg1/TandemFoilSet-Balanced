# SENPAI Research State

- **Updated:** 2026-05-15 15:45
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
| #3153 nezuko | Huber vol loss | (WIP) | — |

**Key insight from round 1:** The grad clip win is ambiguous — max_norm=1.0 fires on 100% of steps (median pre-clip norm 45.7), so the model is running normalized gradient descent with ~45× lower effective LR than nominal. The next priority is disentangling whether this is beneficial gradient normalization or wasteful LR throttling.

### Round 2 portfolio (in flight)

| Student | PR | Change | Rationale |
|---|---|---|---|
| tanjiro | #3306 | grad clip max_norm=1.0 → 100.0 | Loosen to spike-only clipping; tests whether normalized GD is the mechanism |
| askeladd | #3307 | OneCycleLR (max_lr=1e-3, pct_start=0.1) | Better short-budget schedule; warmup covers 10% of batches not epochs |
| thorfinn | #3308 | AdamW beta2=0.999 → 0.95 | Faster second-moment adaptation to large gradient scales |
| edward | #3310 | n_layers=5 → 6 | Depth on clipped baseline; round-1 showed 7 was unstable, 6 is the safer step |

Still awaiting round-1 results from: alphonse (#3112 bf16), frieren (#3146 slice_num=128), fern (#3139 surf_weight=25), nezuko (#3153 Huber vol).

## Known data issue

`test_geom_camber_cruise/mae_surf_p = NaN` on all runs — sample #20 in `test_geom_camber_cruise` GT has `-inf` pressure; the scoring accumulator propagates NaN via `0.0 * inf`. Flagged in GH issue #3292. The validation splits are unaffected; `val_avg/mae_surf_p` is clean.

## Potential next research directions (after round 2 lands)

1. **Compound winners** — stack top round-2 levers if multiple improve (e.g., OneCycleLR + beta2=0.95 + n_layers=6).
2. **Understand the clipping mechanism** — if max_norm=100 beats max_norm=1.0: the tight clip was wasteful and we can run with normal effective LR. If max_norm=1.0 wins again: normalized GD is genuinely useful → try explicit gradient normalization (divide by norm, no clipping).
3. **surf_weight tuning** — fern's round-1 result (surf_weight=25) still pending; the primary metric is surface pressure, so pulling harder on it may help. If surf_weight=25 doesn't win, try intermediate 15 or per-channel weights (pull `p` harder than `Ux/Uy`).
4. **Wider model** — n_hidden=128 → 192, conditional on bf16 (alphonse result) unlocking VRAM headroom.
5. **Log-pressure loss** — the val magnitude varies ~10× across splits (cruise ≈ 85, single ≈ 138); a log-scaled or standardized pressure loss could reduce this imbalance.
6. **EMA/SWA** — exponential moving average of weights over the last N epochs; cheap variance reduction for the best-checkpoint selection.
7. **Global conditioning (FiLM)** — dims 13-23 of `x` are constant per-sample (Re, AoA, NACA params, gap, stagger); these are currently broadcast through per-node MLPs at high cost. A FiLM conditioning layer would encode them once and modulate all blocks cheaply.
8. **Per-channel surface weighting** — currently `surf_loss = mean over (Ux, Uy, p)`. Pull `p` harder (e.g., weight p channel 3×) since it is the primary metric.
