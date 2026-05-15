# SENPAI Research State

- **Last updated:** 2026-05-15 16:55 (round-1 fully closed; round-2 assigned to all 8 students)
- **Most recent research direction from human researcher team:** none (no open issues).
- **Current best:** `val_avg/mae_surf_p` = **109.681** (PR #3276 grad-clip + AdamW selective decay)
- **Current focus:** round-2 — all 8 students assigned; key theme is budget-aware schedule matching + orthogonal axis isolation.

## Branch context
`icml-appendix-charlie-pai2i-24h-r2`. Local JSONL metrics only.

## Established baseline stack (merged to HEAD)
1. **PR #3208** (Huber loss) — `val_avg/mae_surf_p` 116.61
2. **PR #3276** (grad-clip + AdamW selective decay + NaN guard) — `val_avg/mae_surf_p` **109.68** (current best)

Key config: SmoothL1 (Huber, β=1.0) + clip_grad_norm(1.0) + AdamW selective decay (LN/bias/1D no-decay) + NaN sample guard in evaluate_split.

## Active PRs (round-2, all WIP)

| PR | Student | Hypothesis | Previous result |
|----|---------|-----------|-----------------|
| #3276 | fern | Grad-clip + selective decay + NaN guard | **MERGED** — new baseline |
| #3314 | fern | weight_decay 1e-4→3e-4 on decay group (round-3, single-axis) | follow-up to #3276 |
| #3294 | tanjiro | Warmup+cosine over 14ep (budget-matched) | PR #3220: 148.20 (100ep, never annealed) |
| #3295 | edward | Slice_num=128 (single-axis) | PR #3205: 164.38 (5ep, OOM workarounds) |
| #3301 | alphonse | Width-192, epochs=10 (budget-matched) | PR #3179: 154.98 (10ep, cosine never annealed) |
| #3302 | askeladd | Depth-8, epochs=9 (budget-matched) | PR #3183: 154.95 (9ep, cosine never annealed) |
| #3304 | frieren | surf_weight=20 single-axis | PR #3214: 138.44 (surf_weight=30 + 2×p, too aggressive) |
| #3223 | thorfinn | BF16 autocast + batch_size=8 | (round-1, still running) |
| #3344 | nezuko | 32-freq Random Fourier Features (Tancik 2020 RFF) | PR #3216: 137.94 (prescription bug — collapsed to x+z) |

## Round-2 design rationale

### Budget-matching insight (critical learning from round-1)
Round-1 showed that training under a 30-min cap with a 50-epoch cosine schedule means the LR **never reaches its annealed floor** — we're running flat-high-LR training the whole time. Matching `epochs` to actual completable epochs so the cosine fully decays is the primary fix in round-2 for architecture PRs:
- alphonse: epochs=10 (~185 s/epoch)
- askeladd: epochs=9 (~206 s/epoch)
- tanjiro: epochs=14 with 2-ep warmup (the schedule Tmax=14 now cools fully)

### Optimizer leverage confirmed
PR #3276 showed that grad-clip + selective decay gave 5.94% improvement (nearly all val splits improving). This is now baked into baseline. The per-split pattern:
- cruise is easiest (78.85 val, 68.48 test)
- single is hardest (148.09 val, 123.24 test)
- geom_rc lags improvement the most (-2.3% vs -8% for others)

The geom_rc underperformance is an open question — could be capacity, domain coverage, or loss balance.

## Open questions for round-2
- Does width-192 actually help when the schedule fits the budget? (alphonse #3301)
- Does depth-8 help when the schedule fits? (askeladd #3302)
- Does a moderate surf_weight=20 improve on 109.68? (frieren #3304)
- Does a budget-matched warmup+cosine beat plain cosine? (tanjiro #3294)
- Does slice_num=128 beat 64 in a fair single-axis test? (edward #3295)
- Does BF16+batch8 give competitive results with faster throughput? (thorfinn #3223)
- Does corrected Random Fourier Features (true 2D directions) improve geometry-split generalization? (nezuko #3344)
- Does weight_decay=3e-4 outperform 1e-4 on the decay group? (fern #3314)

## Potential round-3 directions (from RESEARCH_IDEAS_2026-05-15_initial.md)
- H2: Per-sample output scale normalization (y-std variability is 40× across dataset)
- H1: FiLM conditioning on global Re/geometry params
- H3: Separate surface/volume decoder heads
- H13: Log-scale pressure loss in train only
- Stronger weight decay on decay group only (fern's suggestion: try weight_decay=3e-4 or 5e-4)
- Diagnose geom_rc lagging split — possibly domain-specific capacity issue

## Plateau watch
Not yet — only 2 merged results, and both improved. Reassess after round-2 results land.
