<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State — `icml-appendix-willow-pai2g-24h-r2`

- **Date / time:** 2026-05-12 22:00 UTC
- **Advisor branch:** `icml-appendix-willow-pai2g-24h-r2`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`
- **Most recent human direction:** none.

## Current research focus

Round 2 of the 24h Willow logging ablation on TandemFoilSet. Single-run hypothesis tests under a hard 30-min wall-clock cap. Primary decision metric: `val_avg/mae_surf_p` (lower is better). Paper-facing `test_avg/mae_surf_p` is structurally NaN on this branch (cruise test split overflows in nearly every run including baseline).

## Cycle-2 update — noise floor is much bigger than first thought

Three alphonse baseline runs span **119.64 → 132.73 → 131.79** — a 13-point range (~10%) under identical config. The single-run noise floor on val_avg/mae_surf_p is therefore ~10%, not 0.5–1% as initially recorded. **Most hypotheses to date are inside this noise band.** This recalibrates the merge bar substantially.

## Cycle-5 update — First terminal results + major milestone: first finite test_avg

**Two PRs reached review-ready this cycle.**

### #1480 thorfinn (bf16 + grad_accum=2): SENT BACK — code not committed

Thorfinn posted terminal results (run `5wvm7na2`): **val_avg=116.30, test_avg=104.96 (finite)**. This is:
- Best val_avg in the cohort (tied with frieren's 116.34)
- **First finite `test_avg/mae_surf_p` in the project** — enabled by the cruise-NaN workaround in evaluate_split
- 18 epochs in 30 min (vs ~14 baseline) — throughput hypothesis confirmed

However, the PR branch has only the empty assign commit — no code pushed. The squash-merge would be empty. **Sent back** asking for 3 commits: bf16 autocast, grad_accum=2 loop, evaluate_split workaround. This is the highest-priority merge once code lands.

### #1461 alphonse (baseline): CLOSED — deliverables captured

Run `ztb0ri42`: val_avg=140.01, test_avg=NaN. Delivered the key deliverable: `test_avg/mae_surf_p_excluding_bad_sample = 126.20` — canonical pre-fix test comparator. BASELINE.md updated. Alphonse closed and re-assigned.

### New assignment: PR #1631 alphonse — cruise-NaN eval workaround

Alphonse assigned the `train.py:evaluate_split` sanitize-and-gate workaround as a clean baseline-hardening PR. Stock config + bug fix only. Expected result: finite `test_avg/mae_surf_p` around 126. Once merged, all subsequent PRs get finite test_avg for free.

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

## Current leaderboard (post cycle-5)

Baseline best: val_avg=119.64 (noise band ±17%). The current bar is 131.79 (canonical baseline run `hqj9bt84`).

| Student / PR | Best val_avg | test_avg (best) | Status | Code committed? | Notes |
|---|---|---|---|---|---|
| thorfinn #1480 (bf16+accum) | **116.30** | **104.96** ✓ | WIP (sent back) | no | Highest-priority merge; needs code commit |
| frieren #1471 (p_weight=2+clip) | **116.34** | NaN | WIP | partial (p_weight=3 arm only) | Best val tied; needs updated code + SENPAI-RESULT |
| fern #1469 (lr=2e-3+clip) | 118.77 | NaN | WIP | no | Within noise of baseline-best; needs code + SENPAI-RESULT |
| alphonse #1631 (cruise-nan eval fix) | TBD | TBD | WIP (new) | no | Baseline hardening; expected ~126 test_avg |
| askeladd #1465 (surf_w=30) | 127.53 | NaN | WIP (stale) | no | Moderate signal |
| tanjiro #1476 (per-field heads) | 137.21 | NaN | WIP (stale) | no | Stability issues |
| nezuko #1475 (wider 256/8h) | 176.37 | NaN | WIP (stale) | no | Under-trained under 30-min cap |
| edward #1466 (Huber per-sample) | 275.04 | NaN | WIP | no | Direction broken; cruise-NaN workaround coded locally |

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

## Next directions (post cycle-5)

1. **Merge #1480 thorfinn immediately when code commits land.** This is the highest-leverage action in the round — bf16+accum lands throughput win AND cruise-NaN workaround in one PR. test_avg=104.96 is the paper-facing number to beat.
2. **Drive frieren #1471 to completion.** Frieren has val_avg=116.34 from W&B but only the old p_weight=3 code committed. Need updated code (p_weight=2+clip) + terminal SENPAI-RESULT.
3. **Drive fern #1469 to completion.** fern 118.77 is 3rd-best and within noise of baseline-best; needs code + SENPAI-RESULT.
4. **PR #1631 alphonse (cruise-nan eval fix)** — baseline hardening. Once merged, every subsequent PR gets finite test_avg. Timing with thorfinn's send-back: whichever of the two gets code committed first becomes the bug-fix vehicle.
5. **Stack winners** — after bf16+accum, p_weight=2+clip, and lr=2e-3+clip are individually confirmed and merged, build a stacked-arm combining all three. Likely synergistic.
6. **Schedule recalibration** — with bf16+accum reaching 18 epochs in 30 min, `T_max=50` means the LR never anneals. A `T_max=18` (or throughput-matched value) should give free improvement from full cosine anneal.
7. **Close edward #1466** when their SENPAI-RESULT arrives — Huber per-sample direction is broken (275 val_avg = 130%+ regression).
8. **Close nezuko #1475** when SENPAI-RESULT arrives — wider model is under-trained under 30-min cap.
9. **Close tanjiro #1476** when SENPAI-RESULT arrives — per-field heads shown stability issues and +15% regression.

## Operational notes

- **Cycle 5 decisions:** #1480 sent back (code not committed), #1461 closed (deliverables captured), #1631 new assignment to alphonse.
- **7 of 8 students still have no real code commits** on their branches. The harvest workflow is the primary driver. The GraphQL rate-limit pressure from pod logs persists.
- Frieren's branch has the p_weight=3 code only (old arm); p_weight=2+clip results are in W&B only.
- 8/8 student pods 1/1 Ready. Host-side harvest/kill controls fleet shutdown.
- The cruise-NaN workaround has two parallel paths: #1631 alphonse (new baseline-hardening) and #1480 thorfinn (sent back for code commit). Whichever lands first wins.
