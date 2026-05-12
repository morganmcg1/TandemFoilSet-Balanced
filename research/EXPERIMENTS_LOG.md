# SENPAI Research Results — charlie-pai2g-48h-r1

## 2026-05-12 21:20 — Round-2 dispatch (5 PRs)

After establishing L1 as the new measured baseline (PR #1355, val_avg/mae_surf_p=94.29),
five round-2 PRs were dispatched to compound the L1 winner with orthogonal levers that
are cheap (no extra params, no compute hit) and target known weak spots of
short-budget training in normalized space.

| PR | Student | Lever | Arms |
|----|---------|-------|------|
| #1581 | frieren | L1 + OneCycleLR compound | peak_lr=1e-3 vs 2e-3 |
| #1582 | alphonse | surf_weight sweep on L1 | surf_weight=5 / 10 (control) / 20 |
| #1601 | thorfinn | EMA of model weights on L1 | decay=0.999 vs 0.9999 |
| #1602 | fern | Gradient clipping on L1 | max_norm 0 (control) / 0.5 / 1.0 |
| #1605 | edward | asinh transform on pressure target with L1 | scale=100 (aggressive) vs 680 (~σ_p, gentle) |

All five dispatched against the L1 baseline (`--loss l1` is the recipe-level
default in every PR body). The three still-running round-1 PRs (#1381 wider
askeladd, #1399 nezuko channel-weight corrected replan, #1405 tanjiro bf16)
remain in flight; if any beats 94.29 it will be merged ahead of the round-2
results.

---

## 2026-05-12 21:15 — PR #1385: Finer physics attention (slice_num 64→128, n_head 4→8) ❌ CLOSED

- **Student branch:** `charliepai2g48h1-edward/slices128-heads8`
- **Hypothesis:** Doubling `slice_num` (64→128) and `n_head` (4→8) gives the
  Transolver physics attention finer slicing and better feature interaction.
  Predicted -2% to -5% on `val_avg/mae_surf_p`.

### Result

| Arm | slice_num | n_head | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4) |
|-----|-----------|--------|---------|---------------------|---------------------------|
| A   | 128       | 8      | 9       | 151.92              | 144.78                    |
| B   | 128       | 4      | 9       | 156.83              | 149.51                    |

### Action: CLOSED — 61% regression vs the new L1 baseline (94.29)

Both arms regressed massively. The finer attention slicing combined with
more heads roughly **doubled** attention compute per layer; under the
hard 30-min wall-clock cap, only 9 epochs of 15 ran — same schedule-mismatch
pathology as the deeper-net experiment (#1389). Even discounting the
schedule artifact, the gap to L1 baseline is too large for further sweeps
under the current epoch budget. Closing the slice/head lever for this arm.

---

## 2026-05-12 21:12 — PR #1389: Deeper Transolver (n_layers 5→8) ❌ CLOSED

- **Student branch:** `charliepai2g48h1-fern/deeper-8-layers`
- **Hypothesis:** Going from 5 → 8 layers gives more iterations of slice
  attention → MLP refinement.

### Final result (after Arm C `--epochs 9` rerun)

| Arm | lr | epochs | best ep | val_avg/mae_surf_p |
|-----|------|--------|---------|---------------------|
| A   | 5e-4 | 9 / 15 | 8       | 153.48              |
| B   | 3e-4 | 9 / 15 | 8       | 147.40              |
| C   | 3e-4 | 9 / 9  | 8       | ~142–145 (schedule-matched, still 50%+ worse than L1) |

### Action: CLOSED — 42%+ regression vs L1 baseline

Arm C (cosine T_max=9 matching the realized epoch count) closed only ~3
points vs Arm B's schedule-mismatched 147.40. Even with the schedule
artifact removed, depth=8 is ~50% worse than the 94.29 L1 baseline at
this epoch count. Closing the depth lever for the current epoch budget;
re-test when total compute budget allows full 15-epoch realization of
deeper nets.

Reusable finding: more depth raises per-step compute by ~55% (so realized
epochs drop from 14 → 9 under the 30-min cap), and the cosine schedule
needs to be retuned to T_max=realized_epochs whenever the per-step cost
materially changes.

---

## 2026-05-12 21:08 — PR #1410: Multi-scale Fourier features for (x,z) coords ❌ CLOSED

- **Student branch:** `charliepai2g48h1-thorfinn/fourier-position-features`
- **Hypothesis:** Adding learned/fixed sinusoidal frequency encodings of (x,z)
  coordinates should give the network a richer geometric representation
  than the raw 2D coords alone, particularly helping the geometry-OOD splits.
  Predicted -3% to -8% on `val_avg/mae_surf_p`.

### Result

| Arm | n_freq_bands | scale_max | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4) |
|-----|--------------|-----------|---------|---------------------|---------------------------|
| A   | 8            | 10        | 13      | 105.05              | 102.78                    |
| B   | 16           | 32        | 12      | 109.20              | 106.94                    |

### Action: CLOSED — 11% regression vs L1 baseline (94.29)

Both Fourier-feature arms regressed against L1 baseline. The geometry-OOD
splits (`val_geom_camber_rc/cruise`) did not improve relative to the L1
baseline either — the synthetic frequency bands seem to add noise without
adding geometric information that the raw coords don't already provide
to the Transolver's slice attention.

This is informative: it suggests the Transolver's physics-aware attention
on raw coords is already extracting geometric structure effectively, and
the next geometric-representation experiment should attack the slicing
mechanism itself rather than the input coordinate encoding.

---

## 2026-05-12 20:50 — PR #1355: Smooth L1 / pure L1 loss vs MSE on normalized residuals ✅ MERGED

- **Student branch:** `charliepai2g48h1-alphonse/smooth-l1-loss`
- **Hypothesis:** L1-family losses (Smooth L1 / Huber β=1.0, pure L1) align
  training objective with the eval metric (MAE in original space), unlike MSE
  which over-penalizes outliers. Predicted -2% to -8% on `val_avg/mae_surf_p`.

### Result

| Arm | Loss | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4 finite) | best ep |
|-----|------|---------------------|----------------------------------|---------|
| **B (winner)** | Pure L1 | **94.291** ⭐ | **91.859** | 14 |
| A | Smooth L1 / Huber β=1 | 97.791 | 94.393 | 14 |
| baseline MSE | MSE | 218.388 @ ep3 (partial) | — | — |

Metrics: `models/model-pure-l1-20260512-191540/{metrics.jsonl,metrics.yaml}` and
`models/model-charliepai2g48h1-alphonse-smooth-l1-huber-20260512-175942/{...}` on advisor branch.

### Action: MERGED as new baseline

Pure L1 is the winner by 3.6% over Smooth L1 (and ~15% over implied full-epoch MSE). The
metric-objective alignment argument is confirmed: pure L1 in normalized space ≈ MAE
in original space, so the training gradient points exactly at what we measure.

Key finding: Pure L1's advantage grows through training (Smooth L1 briefly leads at ep3 by
convergence smoothness, but Pure L1 overtakes by ep9 and dominates). The `val_geom_camber_cruise`
split improved most dramatically (71.66 vs 79.99 vs implied ~90+ MSE) — the hardest OOD split
benefits most from better loss alignment.

**New baseline as of 2026-05-12 20:52:** `val_avg/mae_surf_p = 94.291`. All future experiments
use `--loss l1`.

---

## 2026-05-12 20:04 — PR #1393: OneCycleLR with warmup replacing CosineAnnealingLR

- **Student branch:** `charliepai2g48h1-frieren/onecycle-lr`
- **Hypothesis:** OneCycleLR (warmup → peak → cosine wind-down, per-batch
  stepping) at peak_lr=1e-3 should beat vanilla CosineAnnealingLR for short
  (14-epoch) training runs. Predicted -2% to -6% on `val_avg/mae_surf_p`.

### Result

| Arm | peak_lr | epochs | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4 splits) |
|-----|---------|--------|---------|---------------------|----------------------------------|
| **A (winner)** | 1e-3 | 14 / 15 | 14 | **111.2984** ⭐ | 107.54 |
| B   | 5e-4 | 14 / 15 | 14 | 113.8337            | 108.63                           |

Per-epoch ~131 s, peak GPU memory 42.12 GB. Per-batch LR stepping confirmed
firing as intended (lr 4e-5 → 9.97e-4 by ep2 warmup → 1.34e-5 by ep14 for
Arm A). Arm A wins by 2.5 points (-2.2% rel) — well inside the predicted
band. The full 15-epoch schedule didn't complete because the 30-min cap
cut at epoch 14 (~7% of schedule tail wasted, but LR was already in deep
decay so unlikely to matter much).

### Action: SEND BACK to push (not yet mergeable)

**The student branch is empty.** Only the `assign` commit is on
`charliepai2g48h1-frieren/onecycle-lr` — no `train.py` diff, no
`models/model-onecycle-*` artifacts. Cause: the GitHub API rate-limit
storm (18:30–19:50 UTC) let `gh pr comment` retries succeed (the
SENPAI-RESULT comment posted at 20:04Z), but the separate `git push` for
code + metrics artifacts never landed and was not retried. The result is
real and the methodology checks out — but the PR has no diff to merge.

Sent back at 20:0Xz with concrete push commands. Will merge as new
baseline once the diff is on the branch (this is the first cleanly
terminal round-1 result with no loss-formulation caveat — Arm A becomes
the new `val_avg/mae_surf_p` floor).

### Pre-existing issue (not this PR)

`test_geom_camber_cruise/mae_surf_p` came back NaN on **both** arms; pre-existing
(reproduces identically across arms, confirmed by alphonse PR #1355). Same
`+Inf in y` sample 000020.pt. `data/scoring.py` stays read-only.

---

## 2026-05-12 20:02 — PR #1389: Deeper Transolver (n_layers 5 → 8)

- **Student branch:** `charliepai2g48h1-fern/deeper-8-layers`
- **Hypothesis:** Going from 5 → 8 layers gives more iterations of slice
  attention → MLP refinement. Predicted -2% to -6% on `val_avg/mae_surf_p`,
  bigger gains on tandem OOD splits.

### Result

| Arm | lr | epochs | best ep | val_avg/mae_surf_p | test_avg (3/4 splits, fern's local recompute) |
|-----|------|--------|---------|---------------------|------------------------------------------------|
| A   | 5e-4 | 9 / 15 | 8       | 153.4759            | 139.91                                         |
| B   | 3e-4 | 9 / 15 | 8       | **147.3969**        | 134.38                                         |

Per-epoch ~205 s (≈55% slower than baseline due to extra blocks), peak GPU
memory 64.5 GB. `n_params=1,025,827` (~1.0M, lower than the predicted
1.6-1.8M because Transolver blocks add ~125k each, not 200k). **Both arms
realized only 9 of the 15 configured epochs** before the 30-min cap → the
cosine schedule was set for T_max=15 so the LR decayed for 60% of where it
"thought" it was → effective LR was still elevated at the cut.

### Action: SEND BACK with push + Arm C (not a winner at current state)

**Same push problem as #1393** — the branch has no `train.py` diff, no
metrics. Need the push first regardless of merge decision.

On the result itself: Arm B's 147.40 is ~32% worse than the round-1 leader
(frieren OneCycle Arm A: 111.30). But the comparison is contaminated by the
schedule mismatch (cosine T_max=15 vs realized 9 epochs). Asked fern to run
one more arm with `--epochs 9` so the cosine fully completes within the
wall-clock — that isolates "does depth help once the schedule matches
realized epochs?" from "did the depth lever just get a half-decayed LR?".

If Arm C lands at <120 then depth is salvageable for round 2 compounding;
if it stays >135 we close the depth lever as a confirmed regression.

The hypothesis-aligned finding (Arm B improves `val_geom_camber_rc` by 42.5
over Arm A) is genuine and consistent with the "more layers help tandem
OOD" theory — it's just dominated by the schedule artifact in the average.

---

## 2026-05-12 19:09 — PR #1399: Surface loss pressure-channel weight 2× + surf_weight sweep

- **Student branch:** `charliepai2g48h1-nezuko/surf-channel-pressure-weight`
- **Hypothesis:** Per-channel surface-loss weighting (`[Ux, Uy, p] = [1, 1, 2]`)
  should improve `val_avg/mae_surf_p` because the ranking metric is surface
  pressure. Predicted -3% to -8%.

### Result

| Arm | surf_weight | CHANNEL_W | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p (3/4 splits) |
|-----|-------------|-----------|---------|---------------------|----------------------------------|
| A   | 10          | [1,1,2]   | 13      | **111.7978**        | 110.9876                         |
| B   | 20          | [1,1,2]   | 13      | 126.2973            | 125.9050                         |

Metrics: `models/model-surf-pw2-sw10-20260512-175612/{metrics.jsonl,metrics.yaml}`
and `models/model-surf-pw2-sw20-20260512-183156/{...}` on the student branch.

### Action: send back (not merged)

The student's results comment uncovered a normalization error in the PR's loss
formulation. With denominator `surf_mask.sum() * surf_channel_weight.sum()`,
the new `surf_loss` is ~3× smaller in magnitude than the baseline `surf_loss`
when channel weights are `[1,1,1]`. So Arm A's *effective* surf:vol ratio is
`10/3 ≈ 3.3` (in baseline-equivalent units) and Arm B's is `~6.7` — both
**below** the baseline's `10`. That makes A-vs-B mostly a sweep of effective
surf_weight, not the per-channel-weighting hypothesis we wanted to test.

Sent back with a 3-arm replan (all `surf_weight=10`, fixed denominator using
`surf_channel_weight.mean()`):

- Arm A control: `CHANNEL_W=[1,1,1]` — exactly recovers baseline; first
  true baseline measurement on this branch.
- Arm B: `CHANNEL_W=[1,1,2]` — corrected version of the original hypothesis.
- Arm C (if time): `CHANNEL_W=[1,1,3]` — dose-response.

### Pre-existing issue (not this PR)

`test_geom_camber_cruise/mae_surf_p` came back NaN on both arms while the
matching `val_geom_camber_cruise/mae_surf_p` was finite (87.41 for Arm A).
This affects the pressure channel only on that one test split. Pre-existing —
likely a numerical instability in model predictions on at least one extreme
cruise test sample, not a scoring bug (since val_finite + same split's
Ux/Uy_test were finite). Logged; will revisit if other PRs hit the same
NaN. Test_avg reported here is the partial mean over the 3 finite splits.

### Trajectory (Arm A)

| epoch | val_avg/mae_surf_p | seconds | peak_mem_GB |
|-------|---------------------|---------|-------------|
|  1 | 223.35 | 133 | 41.7 |
|  5 | 163.80 | 130 | 42.1 |
| 10 | 139.59 | 132 | 42.1 |
| 13 | **111.80** ⭐ best | 132 | 42.1 |
| 14 | 112.80 | 130 | 42.1 |

Per-epoch ~2.2 min; 14 of 15 configured epochs ran before the 30-min cap.
Peak memory ~42 GB of 96 GB — large headroom for wider/deeper models.
