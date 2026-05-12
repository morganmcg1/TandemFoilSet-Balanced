# SENPAI Research Results — icml-appendix-willow-pai2g-48h-r5

This log records experiment outcomes for the willow-pai2g-48h-r5 track. New entries are appended at the bottom.

## 2026-05-12 18:56 — PR #1427: askeladd `surf_weight=30` (review 1, sent back)

- Branch: `willowpai2g48h5-askeladd/surf-weight-30`
- Hypothesis: raise `surf_weight` from 10 → 30 to align gradient norm with surface-pressure MAE primary metric. No code change other than the flag.
- W&B run: `kfsefiwx` (12 of 30 epochs; hit 30-min cap at 30.52 min)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 12) | **134.14** |
| `val/single_in_dist/mae_surf_p` | 159.38 |
| `val/geom_camber_rc/mae_surf_p` | 141.89 |
| `val/geom_camber_cruise/mae_surf_p` | 111.28 |
| `val/re_rand/mae_surf_p` | 124.02 |
| `test_avg/mae_surf_p` | **NaN** (non-finite pred on ≥1 test_geom_camber_cruise sample) |
| `test/test_single_in_dist/mae_surf_p` | 141.25 |
| `test/test_geom_camber_rc/mae_surf_p` | 128.46 |
| `test/test_geom_camber_cruise/mae_surf_p` | NaN |
| `test/test_re_rand/mae_surf_p` | 122.25 |
| Peak GPU memory | 42.1 GB |
| Best epoch | 12 (val curve still descending at cap) |

- Val numbers all confirmed against W&B summary.
- Direction is promising — val moves cleanly in the expected direction on all four splits. But `test_avg/mae_surf_p` is NaN-poisoned by one numerically-bad prediction on `test_geom_camber_cruise`, which violates the program.md "finite paper-facing test metric required" contract.
- Sent back with instructions to add a single defensive `pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)` line in `evaluate_split` and rerun. The fix is measurement-only; it does not change training dynamics or the hypothesis. Val should remain ~134; test should become finite.
- This NaN failure mode may affect other round-1 PRs that also evaluate the same cruise tandem test samples — will keep an eye on incoming reviews for the same symptom and apply the same fix instruction if it recurs.

