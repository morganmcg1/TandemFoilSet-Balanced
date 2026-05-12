<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State — `icml-appendix-willow-pai2g-24h-r2`

- **Date / time:** 2026-05-12 21:00 UTC
- **Advisor branch:** `icml-appendix-willow-pai2g-24h-r2`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`
- **Most recent human direction:** none.

## Current research focus

Round 2 of the 24h Willow logging ablation on TandemFoilSet. Single-run hypothesis tests under a hard 30-min wall-clock cap. Primary decision metric: `val_avg/mae_surf_p` (lower is better). Paper-facing `test_avg/mae_surf_p` is structurally NaN on this branch (cruise test split overflows in nearly every run including baseline).

## Cycle-2 update — noise floor is much bigger than first thought

Three alphonse baseline runs span **119.64 → 132.73 → 131.79** — a 13-point range (~10%) under identical config. The single-run noise floor on val_avg/mae_surf_p is therefore ~10%, not 0.5–1% as initially recorded. **Most hypotheses to date are inside this noise band.** This recalibrates the merge bar substantially.

## Live W&B observations (3h into launch)

| Student | Best val_avg | Δ vs baseline-best (119.64) | Δ vs baseline-median (~131) | Code committed? | Notes |
|---|---|---|---|---|---|
| frieren (p_weight) | **116.34** | -2.8% | -11.0% | partial (only p_weight=3 arm, not the re-run code) | Best in cohort; re-run with p_weight=2 + clip succeeded |
| fern (lr=2e-3+clip) | 118.77 | -0.7% | -9.3% | no | 2 new arms launched ~20:51 UTC, still running |
| alphonse (baseline) | 119.64 | reference | reference | no | 3 baseline runs span 13 points |
| thorfinn (bf16+accum) | 124.60 | +4.1% | -4.9% | no | More epochs/30min as designed; first arm was 118.17 |
| askeladd (surf_w=30) | 127.53 | +6.6% | -2.6% | no | Improving across retries |
| tanjiro (per-field heads) | 137.21 | +14.7% | +4.7% | no | 3 of 7 runs crashed — stability issues |
| nezuko (wider 256/8h) | 176.37 | +47.4% | +34.6% | no | 1 currently running; capacity hurt under 30-min cap |
| edward (Huber per-sample) | 275.04 | +130.0% | +110.0% | no | Direction is broken; close on submission |

**Key:**
- Δ vs baseline-best uses alphonse's best baseline run (119.64) — strict.
- Δ vs baseline-median (~131) is what a single hypothesis run would naturally compare against if you'd picked one baseline at random.
- "Code committed?" measured by branch having any commit beyond the empty `assign` commit. Only frieren's first iteration is committed (`acf88af`).

## Workflow observation — stale_wip PRs

All 7 non-frieren PRs are flagged `stale_wip`. Root cause: students ran training in W&B but did not commit/push their `train.py` modifications. The pods are alive (kubectl confirms 1/1 Ready) and polling, but each iteration is hitting GraphQL API rate limits (visible in pod logs: "API rate limit already exceeded for user ID …", retrying for ~90s per iteration). The host-side harvest workflow is expected to drive completion. No advisor nudges this cycle — they would compete for the same rate-limited tokens.

## Themes

**Working (under noise but consistent):**
- Higher LR + grad clip (fern) — multiple arms consistently below baseline median.
- Throughput (thorfinn) — second arm regressed; the bf16+accum benefit may be epoch-count-dependent and saturate.
- Region prioritization (askeladd) — moderate signal.
- Intra-channel weighting (frieren, p_weight=2 with clip) — current best, but only one stable arm.

**Not working:**
- Per-sample Huber rescaling (edward) — catastrophic 100%+ regression. Will close on submission.
- Wider model (nezuko) — 30-min cap turns the wider model into an under-trained model.
- Per-field heads (tanjiro) — wash + stability issues.

## Plateau status

Not in plateau. Most hypotheses sit inside the noise band but multiple directions are showing consistent (if small) movement. Need formal submissions to adjudicate.

## Next directions (r3 candidates)

Same as previous cycle, with new priorities given cycle-2 observations:

1. **Repeat-runs for variance estimation** — give the top 3 hypotheses (frieren-p_weight=2+clip, fern-lr=2e-3+clip, thorfinn-bf16+accum) a 2nd or 3rd seed to separate signal from noise. The ~10% noise band on baseline means single-run wins are unsafe.
2. **Stack winners** — once the top 3 hypotheses are independently confirmed, build a single stacked-arm PR combining bf16+accum + lr=2e-3+clip + surf_weight=30 (+ optional p_weight=2). Test compound improvement.
3. **Diagnose cruise-test NaN** — fix the overflow in `accumulate_batch` so `test_avg/mae_surf_p` becomes reportable. Either: clamp predictions before accumulation, or replace the inf-prone arithmetic. This unlocks paper-facing numbers.
4. **OneCycleLR + high peak** as a schedule alternative.
5. **Surface-aware sampling**, **EMA weights**, **slice_num sweep** — held back from r2.

## Operational notes

- Cycle 2 had no PRs marked ready for review and no idle students. No merges/closes possible this cycle.
- Frieren PR #1471 was sent back ~20:02 UTC; student completed the re-run in W&B (116.34) but hasn't committed the updated code or marked ready.
- The host-side harvest/kill is expected to manage final fleet shutdown after pods are Ready.
