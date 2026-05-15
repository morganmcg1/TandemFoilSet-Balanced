# SENPAI Research Results — charlie-pai2i-24h-r2

## 2026-05-15 14:05 — PR #3208: Replace MSE with SmoothL1 (Huber) loss
- Branch: `charliepai2i24h2-fern/huber-loss`
- Student: charliepai2i24h2-fern
- Hypothesis: Replace MSE in normalized target space with SmoothL1 (Huber, β=1.0) in both training and validation; expect 4–10% improvement on `val_avg/mae_surf_p` from reducing high-Re sample domination.

### Results table

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best @ epoch 13) | **116.611** |
| `test_avg/mae_surf_p` | NaN (pre-existing infra bug; 3 clean splits avg 114.59) |
| Epochs completed | 14/50 (cut at 30-min cap) |
| Peak VRAM | 42.1 GB |
| Wall-clock | ~30.6 min |
| Per-split val mae_surf_p | single 161.69 \| geom_rc 117.56 \| geom_cruise 85.67 \| re_rand 101.53 |
| Per-split test mae_surf_p | single 139.80 \| geom_rc 104.38 \| geom_cruise NaN \| re_rand 99.60 |
| Metrics artifact | `models/model-charliepai2i24h2-fern-huber-loss-20260515-130151/metrics.{jsonl,yaml}` |

### Analysis
- First completed experiment on this branch — establishes the de facto baseline at `val_avg/mae_surf_p` = 116.61 under the 30-min/14-epoch wall-clock cap.
- Trajectory still improving at the cap (epoch 1: 229.6 → epoch 13: 116.6). Headroom from longer schedules is real.
- Per-split pattern is consistent with the hypothesis: cruise (smallest pressure magnitudes) is easiest at 85.7; high-Re raceCar single is hardest at 161.7.
- **Important: no clean MSE companion run exists on this branch**, so the Huber-vs-MSE delta is not isolated. The 7 other in-flight round-1 PRs all carry MSE plus one other change, so we will get indirect signal.
- Two-line diff exactly matched the prescription — clean execution.

### Pre-existing NaN bug surfaced
- `test_geom_camber_cruise` sample 20 has non-finite `y` ground-truth values. `data/scoring.py:accumulate_batch` has a sample-level skip, but it computes `err = |pred - y|` *before* masking, and `NaN * 0 = NaN` so the masked sum propagates NaN. The same pattern exists in the normalized-space loss path of `train.py:evaluate_split`.
- `data/scoring.py` is read-only per `program.md`. The workaround is to sanitize `y` (zero-out NaN, mask out the sample) *before* the loss/scoring calls in `train.py:evaluate_split`.
- Routed to next fern assignment (gradient-clip + selective-decay PR also carries the NaN guard fix in `evaluate_split`).

### Decision
- **Merge.** The PR establishes the first concrete baseline. Huber loss carries forward to subsequent experiments. NaN bug is independent of this change.
- Updated `BASELINE.md` with the new reference numbers.
