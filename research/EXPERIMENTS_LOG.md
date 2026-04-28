# SENPAI Research Results

## 2026-04-28 19:30 — PR #771: Learnable per-channel uncertainty weighting (Kendall & Gal 2018)

- Branch: `willowpai2e1-edward/uncertainty-weighting`
- Hypothesis: Replace fixed MSE loss with Kendall-Gal learnable uncertainty weighting. A scalar log-variance per output channel (Ux, Uy, p) is jointly learned with the model. The natural gradient signal redistributes loss capacity toward channels with the highest residual variance, which we expected to be the pressure channel given its larger dynamic range.

| W&B run | surf_weight | val_avg/mae_surf_p | test_avg/mae_surf_p | Epoch | Notes |
|---------|------------|---------------------|---------------------|-------|-------|
| [1tvvwlux](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/1tvvwlux) | 10 | 123.243 | 111.227 | 14/50 | Timeout hit |
| [6gjtvi4h](https://wandb.ai/wandb-applied-ai-team/senpai-charlie-wilson-willow-e-r1/runs/6gjtvi4h) | 1 | 123.887 | 113.698 | 14/50 | Timeout hit |

**Note:** No unmodified baseline exists yet. These numbers cannot be interpreted as improvement or regression.

**Analysis and conclusions:**

The UW mechanism is fundamentally misaligned with the task objective. Kendall-Gal UW assigns lower effective loss weight to the channel with the *highest* per-element MSE — because high residual variance is interpreted as high aleatoric uncertainty, not as something the model should work harder on. Since the pressure channel has the largest dynamic range (and hence the highest absolute MSE), `log_var[p]` converged to -1.0 while `log_var[Ux/Uy]` converged to ~-2.25, giving pressure approximately 3.5x *less* effective loss weight than velocity. This is precisely backwards: our ranking metric is `mae_surf_p`, so we need more focus on p, not less.

The approach is a mathematical dead-end for this metric and objective. Inverse/fixed channel weighting (upweighting p explicitly) remains a valid idea worth testing separately.

**Bug found and fixed by student (credited to willowpai2e1-edward):**
`test_geom_camber_cruise/000020.pt` contains 761 non-finite pressure node values. The existing per-sample `y_finite` guard in `data/scoring.py` correctly excluded sample 20 from the accumulation mask, but `NaN * 0.0 = NaN` in IEEE-754 meant the error tensor still poisoned the running sum. The same NaN propagated through `y_norm` into the monitoring loss in `train.py`'s `evaluate_split()`. Both bugs were fixed in commit `49c55ed` on the advisor branch with `torch.nan_to_num(err, nan=0.0)` guards. All future runs on this track will have correct `test_geom_camber_cruise` metrics.

**PR closed** as a dead end.

---
