STUDENT willowpai2i48h2-alphonse:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["rxd1s8lt","4tkbs850"],"primary_metric":{"name":"val_avg/mae_surf_p","value":80.8537},"test_metric":{"name":"test_avg/mae_surf_p (3-split, cruise=NaN)","value":78.8663}}

## Results

**Arm A (`--huber_delta 0.5`) wins on both val and test.** The asinh-compression argument from the PR body is empirically confirmed: cutting δ in half from 1.0 → 0.5 improves `val_avg/mae_surf_p` by **−1.37%** and the 3-split test by **−3.07%**. Arm B at δ=0.3 regresses past Arm A but stays slightly under the baseline on test_3split. δ=0.5 is the new sweet spot for the asinh-pressure target.

### Per-arm summary

| Arm | huber_delta | run_id | val_avg/mae_surf_p | test_avg/mae_surf_p (3-split, cruise=NaN) | best epoch | end `train/loss_l1_frac` |
|---|---|---|---|---|---|---|
| Baseline (PR #3475) | 1.0 | j5214ii4 | 81.9754 | 81.3654 | — | — |
| **A (winner)** | **0.5** | `rxd1s8lt` | **80.8537** | **78.8663** | 14 | 0.0294 |
| B | 0.3 | `4tkbs850` | 83.3495 | 80.6279 | 14 | 0.1300 |

`test_avg/mae_surf_p` here is the mean of the three valid test splits — `test_geom_camber_cruise/mae_surf_p` is NaN in every run (same as the merged baseline) because its surface-pressure evaluator returns NaN/inf by data construction. W&B's auto-aggregated `test_avg/mae_surf_p` is therefore `None`; the 3-split mean is computed manually below.

### Per-split metrics — Arm A (δ=0.5, winner)

- **val** (best epoch 14):  single_in_dist 95.858 | geom_camber_rc 91.638 | geom_camber_cruise 58.615 | re_rand 77.303 → **val_avg 80.8537**
- **test**:  single_in_dist 82.640 | geom_camber_rc 84.003 | geom_camber_cruise NaN | re_rand 69.956 → **test_3split mean 78.8663**

For reference, baseline PR #3475 per-split val was 101.013 | 90.717 | 59.909 | 76.263. Arm A's main wins are on `single_in_dist` (−5.16) and `geom_camber_rc` (+0.92 vs baseline — slight regression here). The `cruise` and `re_rand` splits each shift ~+1 relative to baseline.

### Per-split metrics — Arm B (δ=0.3)

- **val**:  single_in_dist 98.677 | geom_camber_rc 95.047 | geom_camber_cruise 60.558 | re_rand 79.116 → **val_avg 83.3495**
- **test**:  single_in_dist 85.184 | geom_camber_rc 84.679 | geom_camber_cruise NaN | re_rand 72.021 → **test_3split mean 80.6279**

### Diagnostic — `train/loss_l1_frac`

The PR predicted ~5–15% l1_frac for a well-calibrated δ on the post-asinh distribution. Actual end-of-training fractions:

| Arm | δ | l1_all | l1_p (pressure) | l1_p_surf (surface p) |
|---|---|---|---|---|
| A | 0.5 | **2.94%** | 1.41% | **1.25%** |
| B | 0.3 | **13.00%** | 5.10% | **6.55%** |

Interpretation: δ=0.5 produces a much smaller l1_frac than the PR's prediction range — only ~3% of residuals exceed δ, vs the expected 5–15%. By the PR's mechanistic argument, this would say "δ is still too big, push smaller." But empirically the smaller δ=0.3 (whose l1_frac=13% lands squarely in the predicted range) regresses. So the residual-fraction heuristic is **directionally inverted** for this problem: with asinh-compression, the residual distribution is so tight that the right δ leaves the L1 region nearly empty and lets the L2-smooth gradients do the work on the bulk of residuals.

In short: asinh + δ=0.5 means Huber is operating in effectively-MSE mode on >97% of residuals while still capping gradient on the tail 3%. That's exactly the regime where you keep MSE's bias-reduction benefit while preventing the worst outliers from blowing up gradients.

### What happened

- **Hypothesis confirmed direction-wise**: δ should drop after introducing asinh. δ=0.5 beats δ=1.0 on both val and test.
- **Hypothesis mis-predicts the magnitude**: the PR predicted "val_avg ∈ [79.5, 81.5]" for δ=0.5. Actual: 80.854 — at the high end of the predicted range but inside it.
- **Hypothesis wrong about the mechanism**: the "5–15% l1_frac is the goldilocks zone" prediction does not hold. The winner has l1_frac = 2.94%, well below the predicted range. The 13% l1_frac of δ=0.3 (in-range by PR logic) regresses. The actual mechanism seems to be: post-asinh residuals are so tightly concentrated that the optimal Huber operates as near-pure-MSE except for a tiny robust-tail. Pushing δ smaller forces too many "central" residuals into L1 mode, weakening the bulk gradient signal.
- **Volume vs surface tension confirmed**: pressure surface l1_frac (1.25% at δ=0.5) is lower than Ux/Uy l1_frac (3.21%/4.20%), consistent with asinh compressing pressure residuals more than the un-transformed velocity residuals. The single-δ compromise still favors pressure here because the asinh tail-compression makes surface-pressure residuals stay well below δ=0.5.

### Suggested follow-ups

1. **Most direct next step — try `δ = 0.6` or `δ = 0.7`.** δ=0.5 wins but its l1_frac is well below the expected range; δ=1.0 (baseline) is too high; the optimum on val may lie slightly above 0.5. A single arm at δ=0.7 would resolve this (1 run, 30 min).
2. **Split-δ (different threshold for surface-p vs volume-Ux/Uy).** As the PR risk note flagged: surface pressure is asinh-compressed but velocities are not. A separate `huber_delta_p_surf=0.3` + `huber_delta_other=1.0` might unlock both regimes simultaneously. Implementation: split `elem_loss` by channel before applying the Huber kink, then re-combine. Moderate-complexity change.
3. **Stop pushing smaller δ.** δ=0.3 regresses, so the "go smaller" direction is exhausted.
4. **Try `δ=2.0` as a counter-test (low priority)** — the PR's risk note suggested this as a backup. Given δ=0.5 won and δ=0.3 lost, the surface is monotone in this region; δ=2.0 would likely regress toward MSE-like behavior. Not worth a run unless we want to confirm the curvature.
5. **Re-run the per-split l1_frac analysis on the merged baseline (δ=1.0).** That would tell us what l1_frac looks like at the current best, and whether the right "residual-fraction" rule of thumb for asinh targets is closer to 2–3% than 5–15%.

### Reproduce commands

```bash
# Arm A (winner, val_avg=80.8537, test_3split=78.8663)
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.5 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --wandb_group huber-delta-on-asinh --wandb_name huber-delta-0.5 \
  --agent willowpai2i48h2-alphonse

# Arm B (regressed, val_avg=83.3495, test_3split=80.6279)
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 0.3 --ema_decay 0.99 --asinh_p_scale 1.0 \
  --wandb_group huber-delta-on-asinh --wandb_name huber-delta-0.3 \
  --agent willowpai2i48h2-alphonse
```

### Run details

- **W&B group**: `huber-delta-on-asinh`
- **W&B run IDs**: `rxd1s8lt` (δ=0.5, winner), `4tkbs850` (δ=0.3)
- **Best epoch**: 14 for both arms; both hit the 30-min wall-clock cap mid-training (50-epoch budget never reached).
- **Peak GPU memory** (wandb system monitor): δ=0.5 ~74.9 GB; δ=0.3 ~93.2 GB. Both within 96 GB limit.
- **No instability** observed at either δ. grad_clip=5.0 was sufficient — no NaN, no loss explosion, even at δ=0.3 where the L1 region is much wider.
- **Code change**: added `train/loss_l1_frac` and channel-wise variants (`_Ux`, `_Uy`, `_p`, `_p_surf`) to step logging in `train.py` lines 558–590 — diagnostic only, gated on `cfg.huber_delta > 0`. No effect on training or loss computation.
