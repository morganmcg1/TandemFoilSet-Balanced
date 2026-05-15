# SENPAI Research State

- **Updated:** 2026-05-15 18:50
- **Track:** `willow-pai2i-24h-r5` (advisor branch `icml-appendix-willow-pai2i-24h-r5`, base `icml-appendix-willow`)
- **Per-run budget:** 30 min wall clock, ≤50 epochs, 1 GPU @ 96 GB VRAM

## Most recent direction from human researcher team

No directives received. GH issue #3292 open for `test_geom_camber_cruise` NaN bug.

## ★ MAJOR FINDING: nezuko warm-restarts likely new baseline (~98.88)

PR #3320 (CosineAnnealingWarmRestarts T_0=5 T_mult=2) has 3 confirmed W&B arms:
- `oeo67jf2`: **98.88** (-15.6% vs 117.16)
- `79m50be7`: 100.90 (-13.9%)
- `iyhrbvuq`: 102.22 (-12.8%)

Mean ≈ 100.67, variance only ~3 pp — the first **replicated** improvement, robustly outside the noise floor. Advisor commented on PR asking nezuko to post terminal SENPAI-RESULT for immediate merge. Once merged, baseline drops to ~98.88.

**Why warm restarts works here:** The 30-min/14-epoch budget is short relative to the 50-epoch cosine design horizon. With T_0=5 and T_mult=2, restarts occur at epochs 5, 10, 20 — giving multiple "fresh start" escape-from-local-minima opportunities within the 14-epoch wall. This likely explains both the improvement AND the lower inter-run variance compared to round-1.

## Current baseline

**`val_avg/mae_surf_p = 117.16`** — grad clip max_norm=1.0, PR #3157 (warm-restarts merge imminent, pending terminal SENPAI-RESULT from nezuko)

## Round 2 portfolio (current state)

| Student | PR | Change | Status | Best result |
|---|---|---|---|---|
| nezuko | #3320 | CosineAnnealingWarmRestarts T_0=5 | **PENDING MERGE** | 98.88 (3 replicated arms!) |
| tanjiro | #3360 | grad clip max_norm=0.5 | WIP | — |
| askeladd | #3307 | OneCycleLR max_lr=1e-3 (right-sized) | WIP draft | — |
| edward | #3381 | n_hidden=192 wider model | WIP | — |
| thorfinn | #3416 | per-channel surf loss p×3 | **NEW WIP** | — |
| thorfinn | #3308 | AdamW beta2=0.95 | **closed** +15.1% regression | — |
| edward | #3310 | n_layers=6 | **closed** +8.6% regression | — |

### Round-1 WIP still completing (multi-arm, stale labels)

| Student | PR | Change | Best run seen | Status |
|---|---|---|---|---|
| alphonse | #3112 | bf16 autocast | 114.34 (1 run, high variance) | WIP multi-arm |
| fern | #3139 | surf_weight=25 | 141.69 (clear regression) | WIP |
| frieren | #3146 | slice_num=128 | 136.57 (regression) | WIP |

Advisor nudged all three. Fern and frieren results point toward close.

## Round 2 closed results

| PR | Change | val_avg | Δ | Outcome |
|---|---|---|---|---|
| #3306 tanjiro | grad clip 1.0→100 | 124.31 | +7.15 | closed — confirms tight clip is normalizer |
| #3308 thorfinn | AdamW beta2=0.95 | 134.89 | +17.7 | closed — mechanistically falsified |
| #3310 edward | n_layers=6 | 127.23 | +10.07 | closed — depth costs epochs |
| #3307 askeladd | OneCycleLR (bugged) | 119.25 | +2.09 | sent back — schedule-sizing fix needed |

## Key round-2 findings

1. **Warm restarts is the winner** (~98.88, -15.6%, replicated 3× with low variance)
2. **Tight grad clip is gradient normalizer** — loosening (max_norm=100) regressed; tightening (max_norm=0.5 in #3360) still testing
3. **Depth (n_layers=6) costs more epochs than it gains in this budget** — 31% slower, 2 fewer epochs, net regression
4. **AdamW beta2=0.95 increases gradient noise** — opposite of hypothesis; slow EMA acts as variance reduction in large-gradient regimes
5. **OneCycleLR must be sized to actual epoch budget** — right-sized fix in progress

## Potential next research directions

1. **Warm-restarts + per-channel p-weighting** — nezuko's win + thorfinn's new assignment (#3416) could stack
2. **Warm-restarts + OneCycleLR** — askeladd's right-sized fix now runs on top of warm-restarts baseline
3. **Warm-restarts + max_norm=0.5** — once #3360 finishes, determine whether tighter clip compounds
4. **Wider model on warm-restarts baseline** — edward's #3381 n_hidden=192 will land on warm-restarts if merge happens first
5. **Log-pressure loss** — val magnitude varies ~10× across splits; log scaling could equalize
6. **EMA/SWA** — low-variance alternative to best-checkpoint selection; cheap on top of warm-restarts
7. **FiLM conditioning on global features** — Re, AoA, NACA params are currently broadcast through all per-node MLPs; conditioning once per sample is cheaper and richer
8. **Lion optimizer** — if max_norm=0.5 confirms "tighter = better", Lion's sign-of-gradient update is the extreme case
