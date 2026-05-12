<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research State — `icml-appendix-willow-pai2g-24h-r2`

- **Date / time:** 2026-05-12 23:05 UTC
- **Advisor branch:** `icml-appendix-willow-pai2g-24h-r2`
- **W&B project:** `wandb-applied-ai-team/senpai-charlie-wilson-willow-g-24h-r2`
- **Most recent human direction:** none.

## Current research focus

Round 2 of the 24h Willow logging ablation on TandemFoilSet. Single-run hypothesis tests under a hard 30-min wall-clock cap. Primary decision metric: `val_avg/mae_surf_p` (lower is better). Paper-facing `test_avg/mae_surf_p` is structurally NaN on this branch (cruise test split overflows in nearly every run including baseline).

## Cycle-2 update — noise floor is much bigger than first thought

Three alphonse baseline runs span **119.64 → 132.73 → 131.79** — a 13-point range (~10%) under identical config. The single-run noise floor on val_avg/mae_surf_p is therefore ~10%, not 0.5–1% as initially recorded. **Most hypotheses to date are inside this noise band.** This recalibrates the merge bar substantially.

## Cycle-7 update — stale_wip cleanup + 2 new architectural/loss-shape arms

After PR #1480 merge, the 4 stale_wip PRs were triaged:

- **CLOSED #1475 nezuko (wider 256/8h):** under-trained at 30-min cap (val≈176 on old baseline → ~51% worse than new 116.30). Capacity > training budget.
- **CLOSED #1476 tanjiro (per-field heads):** val≈137 on old baseline (~18% worse than new), no code pushed, 4h silent. Shared backbone + per-channel weighting (frieren's `p_weight` direction) is a cleaner channel-prioritization mechanism.
- **SENT BACK #1471 frieren (p_weight=2+clip):** had merge conflict from #1480 base change. Asked for rebase + redirect from cycle-3 (`p_weight=2.0` + grad clip 1.0).
- **SENT BACK #1465 askeladd (surf_weight=30):** no code committed in 4h. Asked for rebase, commit, run on new baseline.

Two new assignments cover the orthogonal axes still unexplored:

| PR | Student | Hypothesis | Mechanism |
|---|---|---|---|
| **#1665** | nezuko | `n_layers: 5 → 6` (one more Transolver block) | Adds one round of slot-mixing/attention. Throughput headroom from bf16+accum should keep it within 30-min cap (~15 epochs). Acceptance: val<116.30 AND test doesn't materially regress vs 104.96. |
| **#1666** | tanjiro | `smooth_l1` (Huber, β=1) loss replaces MSE | Aligns training loss shape with eval metric (MAE). Bounds gradient magnitude per element — should reduce p-channel long-tailed sample dominance without per-sample reweighting (which broke in #1466). Orthogonal to all in-flight directions. |

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

## Current leaderboard (post cycle-7)

**Active baseline: val=116.30 / test=104.96** (PR #1480, thorfinn, merged). Beat 116.30 to merge.

| Student / PR | Best val_avg | test_avg | Status | Notes |
|---|---|---|---|---|
| **nezuko #1665** (n_layers=6) | TBD | TBD | WIP (new, cycle-7) | Single block added; capacity bump within budget |
| **tanjiro #1666** (smooth_l1 loss) | TBD | TBD | WIP (new, cycle-7) | Train/eval shape alignment (MSE→L1-ish) |
| **thorfinn #1651** (cosine T18) | TBD | TBD | WIP (cycle-6) | Expects free improvement from full anneal |
| **edward #1654** (EMA weights) | TBD | TBD | WIP (cycle-6) | 1-4% gain expected at eval time |
| **alphonse #1655** (OneCycleLR) | TBD | TBD | WIP (cycle-6) | Warmup+anneal schedule at max_lr=2e-3 |
| frieren #1471 (p_weight=2+clip) | 116.34 (W&B only) | NaN | WIP (sent-back, rebase needed) | Cycle-7 send-back: rebase against merged base + retry |
| fern #1469 (lr=2e-3+clip) | 118.77 (W&B only) | NaN | WIP | Needs code commit + SENPAI-RESULT |
| askeladd #1465 (surf_w=30) | 127.53 (W&B only) | NaN | WIP (sent-back) | Cycle-7 send-back: commit code + rebase + retry |
| ~~tanjiro #1476~~ (per-field heads) | 137.21 (W&B) | — | **CLOSED** cycle-7 | Direction not worth chasing vs new baseline |
| ~~nezuko #1475~~ (wider 256/8h) | 176.37 (W&B) | — | **CLOSED** cycle-7 | Under-trained; capacity > budget |

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

## Next directions (post cycle-7)

1. **Merge winners as they arrive.** Sorted by expected ETA: thorfinn #1651 (cosine T18) and edward #1654 (EMA — pure eval-time change, no retraining) should land first. alphonse #1655 (OneCycleLR), nezuko #1665 (deeper), and tanjiro #1666 (smooth_l1) need one full training run each.
2. **Drive frieren #1471 to completion** — strongest pre-merge signal (W&B val=116.34, essentially tied with merged baseline). Now sent back with rebase + p_weight=2+clip retry on new baseline. If it lands, p_weight axis is locked in.
3. **Drive askeladd #1465 to completion** — sent back with code-commit nudge. Direction was 10% worse than merged baseline on old W&B run, but never properly tested on new bf16+accum baseline.
4. **Drive fern #1469 to completion** — W&B result (118.77, lr=2e-3+clip) is the cleanest LR direction. Still needs code commit + SENPAI-RESULT.
5. **Stack winners** — once the top confirmed hypotheses (bf16+accum ✓, cosine-T18, lr=2e-3, EMA, smooth_l1, deeper, OneCycleLR, p_weight=2+clip) are individually merged, build a stacked-arm combining all. Synergistic gains likely.
6. **Research next frontier hypotheses** — invoke researcher-agent when current pipeline clears. Open axes: weight_decay sweep, attention dropout, slice_num scaling, AdamW betas, alternative positional encodings.

## Operational notes

- **Cycle 7 decisions:** #1475 closed, #1476 closed, #1471 sent back (rebase needed), #1465 sent back (commit needed), #1665 nezuko assigned (n_layers=6), #1666 tanjiro assigned (smooth_l1).
- **Frieren's branch (#1471)** has merge conflict from #1480 base change. Rebase instructions posted at 22:55 UTC.
- 8/8 student pods still 1/1 Ready. Host-side harvest/kill controls fleet shutdown.
- 6 active WIP PRs at cycle-7 close, 0 idle students. Fleet fully utilized.
- **cruise-NaN fix is now landed** on the advisor branch (via #1480). All test_avg values will be finite going forward.
