# SENPAI Research Results — charlie-pai2g-48h-r5

---

## 2026-05-12 18:58 — PR #1444: Swap MSE → Smooth-L1 (Huber, beta=1.0)

- **Branch:** `charliepai2g48h5-thorfinn/smooth-l1-loss`
- **Student:** charliepai2g48h5-thorfinn
- **Hypothesis:** Replace squared-error loss with Smooth-L1 (Huber, β=1.0) in
  normalized space for both training and evaluation losses. The ranking metric is
  MAE in original space; MSE in normalized space over-weights extreme high-Re
  samples. Smooth-L1 is linear outside |err|>β, providing bounded gradients.
  Both vol_loss and surf_loss use the same substitution; `surf_weight=10.0` and
  `data/scoring.py` MAE unchanged.

### Results

| Split | val mae_surf_p | val mae_surf_Ux | val mae_surf_Uy | test mae_surf_p |
|---|---:|---:|---:|---:|
| `single_in_dist` | 135.16 | 1.719 | 0.769 | 120.38 |
| `geom_camber_rc` | 129.08 | 2.104 | 0.988 | 119.47 |
| `geom_camber_cruise` | 77.70 | 1.047 | 0.555 | NaN (bug) |
| `re_rand` | 101.10 | 1.607 | 0.740 | 97.36 |
| **avg** | **110.76** | — | — | NaN / 112.40 (3-split) |

- **Best epoch:** 14 of 50 configured (wall-clock-bound; monotonically improving)
- **Epochs/30-min:** ~14 at default model size (~131 s/epoch)
- **Peak GPU:** 42.1 GB (Blackwell RTX PRO 6000)
- **Artifacts:** `models/model-charliepai2g48h5-thorfinn-smooth-l1-loss-20260512-180133/metrics.{jsonl,yaml}`
- **Status:** MERGED → round-5 baseline floor

### Analysis

This is the first terminal result on the round-5 branch, so we cannot yet compare
against an MSE baseline on the same branch. The absolute val_avg = 110.76 sets
the floor. Key observations:

1. **Under-convergence.** The run was strictly monotonically improving at epoch 14
   when the 30-min cap hit (~14 epochs in 30 min for n_hidden=128). The floor is
   a loose lower bound on what the model could achieve with more compute.
2. **Split pattern consistent with hypothesis.** `val_geom_camber_cruise` (77.70)
   and `val_re_rand` (101.10) — the two splits the PR predicted would benefit most
   from bounded gradients at high-Re — are the best-performing splits. The raceCar
   splits (`single_in_dist` 135.16, `geom_camber_rc` 129.08) are noisier
   epoch-to-epoch, consistent with the loss being driven by the wide-Re tail.
3. **Scoring NaN bug discovered.** `test_geom_camber_cruise/000020.pt` has ±Inf
   values in the `p` channel. The `data/scoring.py` sample-skip logic misses this
   due to `0 × Inf = NaN` (IEEE-754). This affects all PRs in round 5 that run
   the test step. Round-5 ranking decision: **val_avg/mae_surf_p only**. The fix
   (filter the bad sample in `train.py`'s `evaluate_split` before calling
   `accumulate_batch`) will be rolled into an upcoming student assignment.

### Conclusions

- Smooth-L1 is a viable baseline for round 5. Whether it beats MSE requires the
  other in-flight arms (which use MSE) to finish and post results.
- The binding constraint is wall-clock convergence speed: ~14 epochs in 30 min.
  The highest-leverage next move is anything that increases epochs/wall-clock
  (bf16 AMP, smaller batch, smaller model, compile) rather than per-epoch quality.
- `val_geom_camber_cruise` is the easiest split (lowest MAE). The hardest splits
  are the raceCar ones.
