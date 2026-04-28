# SENPAI Research Results — `icml-appendix-willow-pai2d-r3`

## 2026-04-28 00:30 — PR #320: Linear warmup + higher peak LR (5e-4 → 1e-3, 2-epoch warmup) — **MERGED**

- Branch: `willowpai2d3-nezuko/higher-lr-warmup`
- **Hypothesis:** Bare cosine annealing with `lr=5e-4` is conservative for transformer training; a 2-epoch linear warmup unlocks a higher peak LR (1e-3), giving faster early convergence and a deeper minimum.
- **Predicted Δ on `val_avg/mae_surf_p`:** −3% to −7%.
- **Observed Δ on `val_avg/mae_surf_p`:** **−21.5%** (147.55 → 115.84). Far exceeded prediction.

### Sweep results (group `lr-warmup-sweep`, all with 2-epoch warmup)

| peak_lr | best epoch | val_avg/mae_surf_p | mean test (3 valid splits) | W&B run |
|---|---:|---:|---:|---|
| 5e-4 | 13 | 147.5538 | 150.67 | `liddr8ce` |
| **1e-3** | **14** | **115.8379** | **112.78** | `w3mjq2ua` |
| 2e-3 | 9 | 151.1338 | 151.16 | `4zc03997` |

### Per-split val MAE at best epoch (`mae_surf_p`)

| Split | 5e-4 | **1e-3 (winner)** | 2e-3 |
|---|---:|---:|---:|
| `val_single_in_dist` | 182.68 | **131.06** | 181.14 |
| `val_geom_camber_rc` | 157.13 | **129.57** | 165.60 |
| `val_geom_camber_cruise` | 111.21 | **92.55** | 115.88 |
| `val_re_rand` | 139.19 | **110.17** | 141.92 |
| **val_avg** | 147.55 | **115.84** | 151.13 |

### Analysis & conclusions

- **Peak LR is the dominant effect, not warmup itself.** With warmup held fixed at 2 epochs, the 5e-4 control (147.55) and 2e-3 (151.13) sit nearly on top of each other — only 1e-3 peels away from the pack. So warmup at the *old* peak LR doesn't capture the win.
- **Improvement is uniform across all 4 val splits** (−28%, −18%, −17%, −21%) — not a single-split fluke. Generalizes across the in-distribution sanity, both OOD-camber tracks, and the Re-stratified track.
- **2e-3 was too aggressive** — best epoch fell to 9 and val plateaued/regressed, consistent with overshoot once cosine engaged.
- **Why the prediction was so far off:** with the 30-min timeout cutting at ~epoch 14, training ends in the *early* cosine regime where lr is still ~95% of peak. Faster early convergence dominates and the lower-lr runs never get to "anneal into a sharp minimum." Prediction assumed the full 50 epochs.

### Pre-existing issue surfaced (not introduced by this PR)

All three runs produced `test_avg/mae_surf_p = NaN` because `test_geom_camber_cruise` returns `loss=inf, mae_surf_p=NaN` on every checkpoint. Root cause: model emits non-finite predictions on at least one of the 200 test samples in that split, and `data/scoring.accumulate_batch` filters non-finite ground truth but not non-finite predictions. The val analogue of that split is sane (92.55 in the winner) — so the failure is test-sample-specific. Same NaN at all three peak_lrs → lr-independent, deterministic failure mode.

### Decision: **MERGED**

- Val improvement is large (−21.5%), consistent across splits, and matches the expected direction.
- Test NaN is pre-existing and orthogonal to this PR's lever.
- New baseline established: `peak_lr=1e-3, warmup_epochs=2`. All in-flight Round-1 PRs (#294, #315, #316, #317, #319, #322, #323) now compare against this baseline; their winners must beat **115.84**.
- Follow-up assigned to nezuko to investigate the NaN-prediction issue in test_geom_camber_cruise.
