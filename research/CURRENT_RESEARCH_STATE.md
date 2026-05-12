<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State — `icml-appendix-willow-pai2g-24h-r2`

- **Date / time:** 2026-05-12 22:20 UTC
- **Advisor branch:** `icml-appendix-willow-pai2g-24h-r2`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`
- **Most recent human direction:** none.

## Current research focus

Round 2 of the 24h Willow logging ablation on TandemFoilSet. Single-run hypothesis tests under a hard 30-min wall-clock cap. Primary decision metric: `val_avg/mae_surf_p` (lower is better). Paper-facing `test_avg/mae_surf_p` is structurally NaN on this branch (cruise test split overflows in nearly every run including baseline).

## Cycle-2 update — noise floor is much bigger than first thought

Three alphonse baseline runs span **119.64 → 132.73 → 131.79** — a 13-point range (~10%) under identical config. The single-run noise floor on val_avg/mae_surf_p is therefore ~10%, not 0.5–1% as initially recorded. **Most hypotheses to date are inside this noise band.** This recalibrates the merge bar substantially.

## Cycle-6 update — MAJOR MILESTONE: First merge! val=116.30, test=104.96

### PR #1480 thorfinn — MERGED ✓

Code committed and squash-merged. **New baseline: val_avg=116.30, test_avg=104.96.**

This merge simultaneously:
- Landed the bf16+accum=2 throughput win (2.5× epoch throughput; 18 epochs in 30 min)
- Landed the cruise-NaN evaluate_split workaround — all future runs now produce finite `test_avg` automatically

### PR #1466 edward — CLOSED (broken direction)

Huber direction: val=324.66 (~150% worse than baseline). Closed.

### PR #1631 alphonse — CLOSED (redundant after #1480 merge)

Cruise-NaN workaround targeted by this PR has landed via #1480.

### Three new assignments

| PR | Student | Hypothesis | Expected val |
|---|---|---|---|
| **#1651** | thorfinn | `epochs=18` cosine T_max recalibration | Free win from full anneal |
| **#1654** | edward | EMA model weights (`decay=0.9995`) | ~113-115 (1-4% gain) |
| **#1655** | alphonse | `OneCycleLR(max_lr=2e-3)` | TBD vs cosine-2e-3 |

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

### New assignment: PR #1631 alphonse — cruise-NaN eval workaround (now closed)

Was assigned but superseded by #1480 merge. Closed when #1480 landed.

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

## Current leaderboard (post cycle-6)

**New baseline: val=116.30 / test=104.96** (PR #1480, thorfinn, merged). Beat 116.30 to merge.

| Student / PR | Best val_avg | test_avg | Status | Notes |
|---|---|---|---|---|
| **thorfinn #1651** (cosine T18) | TBD | TBD | WIP (new) | Expects free improvement from full anneal |
| **edward #1654** (EMA weights) | TBD | TBD | WIP (new) | 1-4% gain expected at eval time |
| **alphonse #1655** (OneCycleLR) | TBD | TBD | WIP (new) | Warmup+anneal schedule at max_lr=2e-3 |
| frieren #1471 (p_weight=2+clip) | 116.34 (W&B only) | NaN | WIP (stale) | Needs code commit + SENPAI-RESULT |
| fern #1469 (lr=2e-3+clip) | 118.77 (W&B only) | NaN | WIP (stale) | Needs code commit + SENPAI-RESULT |
| askeladd #1465 (surf_w=30) | 127.53 (W&B only) | NaN | WIP (stale) | Needs code + SENPAI-RESULT |
| tanjiro #1476 (per-field heads) | 137.21 (W&B only) | NaN | WIP (stale) | Stability issues; likely close |
| nezuko #1475 (wider 256/8h) | 176.37 (W&B only) | NaN | WIP (stale) | Under-trained; likely close |

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

## Next directions (post cycle-6)

1. **Merge thorfinn #1651** when code + SENPAI-RESULT arrive. Expect improvement from full cosine anneal.
2. **Merge edward #1654** when EMA results arrive. 1-4% gain expected.
3. **Merge alphonse #1655** if OneCycleLR beats 116.30.
4. **Drive frieren #1471 and fern #1469 to completion** — both have W&B results beating baseline but no code committed. Both need to push code + terminal SENPAI-RESULT.
5. **Stack winners** — once the top confirmed hypotheses (bf16+accum ✓, cosine-T18, lr=2e-3, p_weight=2+clip) are individually merged, build a stacked-arm combining all. Synergistic gains likely.
6. **Close tanjiro #1476 and nezuko #1475** when SENPAI-RESULTs arrive — both show regression and no clear path to improvement.
7. **Research next frontier hypotheses** — need researcher-agent to survey new directions once current pipeline clears.

## Operational notes

- **Cycle 6 decisions:** #1480 merged (first merge!), #1466 closed, #1631 closed, #1651/#1654/#1655 assigned.
- **Frieren's branch** still has the p_weight=3 code only; the p_weight=2+clip W&B result (116.34) is better than the new 116.30 baseline by a sliver but needs proper code commit to be mergeable.
- GraphQL rate limit hit during cycle-6 assignment creation — workaround: used REST API for #1654 and #1655 label additions. Rate limit should recover before next cycle.
- 8/8 student pods still 1/1 Ready. Host-side harvest/kill controls fleet shutdown.
- **cruise-NaN fix is now landed** on the advisor branch (via #1480). All test_avg values will be finite going forward.
