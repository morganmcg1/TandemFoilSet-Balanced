<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI Research Results — `icml-appendix-willow-pai2g-24h-r2`

Primary metric: `val_avg/mae_surf_p` (lower is better; baseline ~132).
Test metric (`test_avg/mae_surf_p`) is structurally NaN on this branch due to a cruise-test overflow systemic to the codebase; rankings use val_avg + the three finite per-test-split metrics.

---

## 2026-05-12 20:00 — PR #1471: frieren — pressure channel weight=3 in loss (sent back, not merged)

- Branch: `willowpai2g24h2-frieren/p-channel-weight-3`
- Hypothesis: up-weight pressure (dim 2) inside per-channel `sq_err` by `p_weight=3.0` to direct more gradient at the ranking metric.
- Result: monotone val descent 241 → 130.98 over 14 epochs; W&B runs `ftuclvqz` (first arm, 148.57) and `ph14bsim` (second arm, 130.98, canonical).

### Metrics

| Metric | Value | Baseline | Delta |
|---|---|---|---|
| `val_avg/mae_surf_p` (best `ph14bsim`) | 130.98 | 131.79 / 132.73 | ~-0.6 to -1.3% |
| `val_single_in_dist/mae_surf_p` | 166.71 | 136.34 | +22% (worse) |
| `val_geom_camber_rc/mae_surf_p` | 140.45 | ~130 | +8% (worse) |
| `val_geom_camber_cruise/mae_surf_p` | 98.51 | 117.71 | **-16% (better)** |
| `val_re_rand/mae_surf_p` | 118.23 | 121.79 / 117.71 | mixed |
| `test_single_in_dist/mae_surf_p` | 140.14 | ~135 | mixed |
| `test_geom_camber_rc/mae_surf_p` | 126.71 | TBD | n/a |
| `test_geom_camber_cruise/mae_surf_p` | **NaN** | **NaN** (systemic) | n/a |
| `test_re_rand/mae_surf_p` | 116.95 | TBD | n/a |

### Analysis

The val_avg gain (~1%) is inside noise (the two baseline arms differ by 0.7%) and the first frieren arm regressed by 13% — high variance. The mean-improvement signal is dominated by `val_geom_camber_cruise` (-16%), which is precisely the split where the test counterpart blew up. Up-weighting pressure made cruise val better but pushed the model's p output into overflow territory on the larger test cruise set (200 samples vs 100 val).

Student's diagnostic is correct: the cruise NaN traces to `accumulate_batch` propagating `inf - y` through `mask` arithmetic. The systemic cruise-test NaN affects every run including baseline, so it's not a frieren-specific veto — but the *magnitude* of the p-output blowup at `p_weight=3` is what makes this hypothesis risky.

### Decision

Sent back to student with two changes: drop `p_weight` to 2.0 (less aggressive) and add `clip_grad_norm_(model.parameters(), 1.0)` as a baseline-hardening numerical safety. Same `--wandb_group "willow-r2-p-weight"` so the arms remain comparable. Acceptance criterion for re-review: val_avg cleanly below baseline AND no regression on the three finite per-test-splits.

### Update — 2026-05-12 21:00 (cycle 2)

Frieren ran the re-run in W&B per the send-back. New best `val_avg/mae_surf_p` = **116.34** (run `18f9jjzt`), which is the best in the entire cohort across all students/arms. The configuration delivered as asked: `p_weight=2.0` + `clip_grad_norm_=1.0`. However: student has not yet committed/pushed the updated `train.py` (commit `acf88af` on the branch still reflects the original `p_weight=3.0` change) and has not posted the updated SENPAI-RESULT comment. Awaiting student-side workflow completion before final adjudication.

---

## 2026-05-12 21:00 — Cycle-2 advisor-side observations (no formal submissions)

All 8 students have completed multiple W&B runs but only frieren has any code commit on their branch beyond the empty assign commit. The other 7 PRs are `stale_wip`. Pod logs show students are alive and polling but throttled by GitHub GraphQL API rate limits.

### Live W&B leaderboard (latest best per student)

| Rank | Student / hypothesis | Best W&B run | val_avg/mae_surf_p | Δ vs baseline-median (~131) |
|---|---|---|---|---|
| 1 | frieren / p_weight=2 + clip (re-run) | `18f9jjzt` | **116.34** | -11.2% |
| 2 | fern / lr=2e-3+clip | `m7xp2x4b` | 118.77 | -9.3% |
| 3 | alphonse / baseline (3rd rep) | `z2ls7ol1` | 119.64 | -8.7% |
| 4 | thorfinn / bf16+accum=2 | `zg3qckt7` | 124.60 | -4.9% |
| 5 | askeladd / surf_weight=30 | `dqey3vto` | 127.53 | -2.6% |
| 6 | tanjiro / per-field heads | `0bh0u3h1` | 137.21 | +4.7% |
| 7 | nezuko / wider 256/8h | `shqqxayq` | 176.37 | +34.6% |
| 8 | edward / Huber per-sample | `wxpj1e4u` | 275.04 | +110% |

### Re-calibrated noise floor

The 3 alphonse baseline runs span 119.64–132.73 (13 points, ~10%). The previous "0.5–1%" noise estimate was wrong (it was the spread between 131.79 and 132.73, ignoring the third run). With the true noise band ~10%, frieren (-11%) and fern (-9%) are at the edge of, but plausibly within, noise. Without repeat seeds we can't fully separate signal from noise.

### No formal decisions made this cycle

No PRs were marked ready for review. No merges, send-backs, or closes happened (beyond the pre-existing frieren send-back from cycle 1). The advisor branch was updated with the recalibrated noise floor and cycle-2 W&B observations.

---

## 2026-05-12 21:30 — Cycle-3 observations

### Important discovery: cruise-test NaN root cause + workaround

Two students independently diagnosed the systemic `test_geom_camber_cruise/mae_surf_p = NaN` issue in detailed PR comments:

- **#1466 (edward)** at 21:00 UTC and **#1480 (thorfinn)** at 20:56 UTC both identified that `data/scoring.py:accumulate_batch` has a `0 * Inf = NaN` propagation bug when a batch contains a sample with non-finite `y` values. Specifically: `test_geom_camber_cruise` sample 20 has 761 nodes with `y_p = -Inf`. The per-sample skip logic in `accumulate_batch` is defeated by the masked-multiply at the end.

- Both students implemented identical workarounds in `train.py:evaluate_split` (sanitize `y` and gate `mask` per-sample before calling `accumulate_batch`).

- Edward verified the workaround on their best checkpoint (run `wxpj1e4u`): `test_avg/mae_surf_p = 257.22` (was NaN), `test_geom_camber_cruise/mae_surf_p = 156.58` (was NaN).

- **`data/scoring.py` is read-only per `program.md`** — neither student modified it. Both fixes live in `train.py`.

**Implication:** when these fixes are committed and merged, every future run on this branch should produce a finite `test_avg`. This unlocks the paper-facing metric. The fix is hypothesis-agnostic and should be merged as a baseline-hardening change even if the surrounding hypothesis (edward's Huber, thorfinn's bf16+accum) doesn't win on val.

### Cycle-3 leaderboard (live W&B, unchanged in ranks since cycle 2)

| Student | Best val_avg | Δ vs best baseline (119.64) | Running now |
|---|---|---|---|
| frieren | 116.34 | -2.8% | fshtpt6z (new arm) |
| fern | 118.77 | -0.7% | 2ny5alj3 (after j6ugv3ik crash) |
| alphonse | 119.64 | reference | 7uv601md (after ytujykqu crash) |
| thorfinn | 124.60 | +4.1% | 5wvm7na2 |
| askeladd | 127.53 | +6.6% | 3cv4bxtr + qhnzquax (2 arms) |
| tanjiro | 137.21 | +14.7% | 2wlx399x |
| nezuko | 176.37 | +47.4% | wp67vqws |
| edward | 275.04 | +130% | 4bplylk3 |

- All 8 students have currently-running W&B runs (active iteration).
- 2 crashes since 20:00 UTC: `j6ugv3ik` (fern), `ytujykqu` (alphonse). Both already have follow-up arms running.
- No new finished runs have produced a finite `test_avg` yet — the bug fixes haven't been committed/pushed to PR branches, so new training runs aren't using the workaround.

### No decisions this cycle

All 8 PRs remain draft `status:wip`. No code commits on 7 of 8 branches (frieren has a partial commit). No SENPAI-RESULT terminal markers. Advisor held back on per-student nudge comments to avoid further burning the shared GraphQL rate-limit budget (visible in pod logs as 6-retry token exhaustion per heartbeat).
