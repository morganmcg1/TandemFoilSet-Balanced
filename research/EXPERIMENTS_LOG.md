# SENPAI Research Results — charlie-pai2g-48h-r1

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
