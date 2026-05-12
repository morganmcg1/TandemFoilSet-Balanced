<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State — `icml-appendix-willow-pai2g-24h-r2`

- **Date / time:** 2026-05-12 21:45 UTC
- **Advisor branch:** `icml-appendix-willow-pai2g-24h-r2`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`
- **Most recent human direction:** none.

## Current research focus

Round 2 of the 24h Willow logging ablation on TandemFoilSet. Single-run hypothesis tests under a hard 30-min wall-clock cap. Primary decision metric: `val_avg/mae_surf_p` (lower is better). Paper-facing `test_avg/mae_surf_p` is structurally NaN on this branch (cruise test split overflows in nearly every run including baseline).

## Cycle-2 update — noise floor is much bigger than first thought

Three alphonse baseline runs span **119.64 → 132.73 → 131.79** — a 13-point range (~10%) under identical config. The single-run noise floor on val_avg/mae_surf_p is therefore ~10%, not 0.5–1% as initially recorded. **Most hypotheses to date are inside this noise band.** This recalibrates the merge bar substantially.

## Cycle-4 update — third independent cruise-NaN diagnosis + advisor decision

Alphonse (baseline PR #1461) independently diagnosed the same cruise-NaN root cause at 21:15 UTC. New facts:
- Scanned all 8 val/test splits (1000 files): `test_geom_camber_cruise/000020.pt` is the **only** file with non-finite `y` (`y[:, 2]` is `-Inf` on 761 nodes).
- The bad sample, and only it, poisons `test_geom_camber_cruise/{mae_surf_p, mae_vol_p}` whenever it lands in a mixed batch — which it does with default `batch_size=4`.

Advisor decision posted on #1461 at 21:45 UTC:
1. Data fix and `scoring.py` fix are out of scope (isolated launch + read-only file rule).
2. The `train.py:evaluate_split` sanitize-and-gate workaround already on #1466 / #1480 is the in-scope path; whichever lands first is the bug-fix vehicle.
3. Authorized alphonse's `test_avg/mae_surf_p_excluding_bad_sample` recompute from the best existing checkpoint (`hqj9bt84`) — becomes the canonical baseline test number for the round, available *before* the eval-time workaround lands.

## Cycle-3 update — cruise-test NaN root cause + workaround discovered

Two students independently nailed the systemic `test_geom_camber_cruise/mae_surf_p = NaN` issue in PR comments at ~21:00 UTC:

- **#1466 (edward)** and **#1480 (thorfinn)**: `data/scoring.py:accumulate_batch` propagates `0 * Inf = NaN` when a batch contains a sample with non-finite `y` values. Specifically, `test_geom_camber_cruise` sample 20 has 761 nodes with `y_p = -Inf`. The per-sample skip path in `accumulate_batch` is defeated by the trailing masked-multiply.
- Both implemented identical workarounds in `train.py:evaluate_split` (sanitize `y` and gate `mask` per-sample before calling `accumulate_batch`).
- **`data/scoring.py` is read-only per `program.md`** — neither student modified it. Both fixes live in `train.py`.
- Edward verified on run `wxpj1e4u`: `test_avg/mae_surf_p = 257.22` (was NaN), `test_geom_camber_cruise/mae_surf_p = 156.58` (was NaN).

**Implication:** when these fixes land, every future run on this branch should produce a finite `test_avg`. This unlocks the paper-facing metric. The fix is hypothesis-agnostic and should be merged as a baseline-hardening change even if the surrounding hypothesis (edward's Huber, thorfinn's bf16+accum) doesn't win on val. Plan to cherry-pick the workaround once a student actually commits/pushes it; right now both PRs are still draft with no code on the branch beyond the empty `assign` commit.

## Live W&B observations (3.5h into launch)

| Student | Best val_avg | Δ vs baseline-best (119.64) | Δ vs baseline-median (~131) | Code committed? | Currently running | Notes |
|---|---|---|---|---|---|---|
| frieren (p_weight) | **116.34** | -2.8% | -11.0% | partial (only p_weight=3 arm, not the re-run code) | `fshtpt6z` (new arm) | Best in cohort; re-run with p_weight=2 + clip succeeded |
| fern (lr=2e-3+clip) | 118.77 | -0.7% | -9.3% | no | `2ny5alj3` | Crash on `j6ugv3ik` then follow-up launched |
| alphonse (baseline) | 119.64 | reference | reference | no | `7uv601md` | Crash on `ytujykqu` then follow-up launched |
| thorfinn (bf16+accum) | 124.60 | +4.1% | -4.9% | no | `5wvm7na2` | First arm was 118.17; cruise-NaN workaround diagnosed |
| askeladd (surf_w=30) | 127.53 | +6.6% | -2.6% | no | `3cv4bxtr` + `qhnzquax` | Improving across retries |
| tanjiro (per-field heads) | 137.21 | +14.7% | +4.7% | no | `2wlx399x` | 3 of 7 runs crashed earlier — stability issues |
| nezuko (wider 256/8h) | 176.37 | +47.4% | +34.6% | no | `wp67vqws` | Capacity hurt under 30-min cap |
| edward (Huber per-sample) | 275.04 | +130.0% | +110.0% | no | `4bplylk3` | Direction broken; cruise-NaN workaround diagnosed; close on submission |

**Key:**
- Δ vs baseline-best uses alphonse's best baseline run (119.64) — strict.
- Δ vs baseline-median (~131) is what a single hypothesis run would naturally compare against if you'd picked one baseline at random.
- "Code committed?" measured by branch having any commit beyond the empty `assign` commit. Only frieren's first iteration is committed (`acf88af`).

## Workflow observation — stale_wip PRs

All 7 non-frieren PRs remain `stale_wip` through cycle 3. Root cause: students ran training in W&B but did not commit/push their `train.py` modifications. The pods are alive (kubectl confirms 1/1 Ready) and polling, but each iteration is hitting GraphQL API rate limits (visible in pod logs: "API rate limit already exceeded for user ID …", retrying for ~90s per iteration). The host-side harvest workflow is expected to drive completion. No advisor nudges this or last cycle — they would compete for the same rate-limited tokens.

Notably, edward and thorfinn both posted **detailed diagnostic comments** in cycle 3 identifying the cruise-test NaN root cause, but neither has committed the workaround code yet. The commenting traffic seems to use a different rate-limit lane than the harvest workflow — students can still push prose updates even while the harvest is throttled.

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

Updated for cycle 3:

1. **Cherry-pick the cruise-test NaN fix** — `train.py:evaluate_split` sanitize-and-gate workaround (independently posted on #1466 and #1480). This is hypothesis-agnostic baseline hardening that unlocks the paper-facing `test_avg` for every future run. Highest-leverage cherry-pick on the table once a student actually commits it.
2. **Repeat-runs for variance estimation** — give the top 3 hypotheses (frieren-p_weight=2+clip, fern-lr=2e-3+clip, thorfinn-bf16+accum) a 2nd or 3rd seed to separate signal from noise. The ~10% noise band on baseline means single-run wins are unsafe.
3. **Stack winners** — once the top 3 hypotheses are independently confirmed, build a single stacked-arm PR combining bf16+accum + lr=2e-3+clip + surf_weight=30 (+ optional p_weight=2). Test compound improvement.
4. **OneCycleLR + high peak** as a schedule alternative.
5. **Surface-aware sampling**, **EMA weights**, **slice_num sweep** — held back from r2.

## Operational notes

- Cycles 3 and 4 had no PRs marked ready for review and no idle students. No merges/closes possible.
- Frieren PR #1471 was sent back ~20:02 UTC; student completed the re-run in W&B (116.34) but hasn't committed the updated code or marked ready.
- Alphonse, edward, and thorfinn have all posted full root-cause diagnostics for the cruise-test NaN bug. None has committed the fix yet. Alphonse will additionally publish a `test_avg/mae_surf_p_excluding_bad_sample` baseline from `hqj9bt84` (advisor-authorized in cycle 4).
- 8/8 student pods 1/1 Ready (3h33m uptime). The host-side harvest/kill is expected to manage final fleet shutdown after pods are Ready.
