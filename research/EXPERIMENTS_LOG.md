# SENPAI Research Results — willow-pai2i-24h-r4

## 2026-05-15 14:10 — PR #3257: Surface MAE loss + pressure-channel weight 3×

- **Student/branch:** willowpai2i24h4-frieren / `willowpai2i24h4-frieren/surf-mae-p-weight`
- **Hypothesis:** Switch surface loss from MSE to MAE and weight the p channel 3× to align the training signal with `test_avg/mae_surf_p`. Volume loss kept as MSE.
- **W&B run:** `zz2r70lt` (https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/zz2r70lt)

### Result (best checkpoint at epoch 13 of 14; timeout cap)

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------:|----------------:|
| `single_in_dist`        | 115.42 | 106.18 |
| `geom_camber_rc`        | 119.27 | 104.56 |
| `geom_camber_cruise`    |  73.02 | **NaN** (non-finite p preds) |
| `re_rand`               |  91.93 |  86.33 |
| **avg**                 | **99.91** | **NaN** |

Peak GPU: 42.1 GB / 96 GB. Wall time: 30.8 min (hit cap). Val curve still descending at termination (ep 1 → 228.01, ep 13 → 99.91, ep 14 → 122.76).

### Decision: send back to student (#3257-comment-4460628326)

`test_avg/mae_surf_p = NaN` is disqualifying per advisor protocol — the primary ranking metric for the paper-facing comparison must be finite.

### Root cause

`data/scoring.py:accumulate_batch` skips samples with non-finite **ground truth** but does not guard against non-finite **predictions** — one runaway pred poisons the running sum. `data/scoring.py` is read-only, so the fix has to live in `train.py:evaluate_split`. Sent back with explicit `torch.nan_to_num(pred_orig, nan=0.0, posinf=1e6, neginf=-1e6)` patch + `n_nonfinite_pred` per-split diagnostics, plus instructions to rerun the same arm.

### Analysis

- Per-split val shape matches the hypothesis prediction (cruise/re_rand carry the gain), suggesting the loss change is doing what we wanted — we just can't confirm the test-side number until the rerun.
- Training was wall-clock-capped at epoch 14/50 — the cosine schedule was set for `T_max=50` but only ~14 epochs run. The model never saw the low-LR end of the schedule. This is a systemic issue affecting every PR in this round; will address in a follow-up hypothesis family.
- No baseline (unmodified Transolver) measurement exists yet on this branch — the clean rerun of this PR will be the first credible point.
