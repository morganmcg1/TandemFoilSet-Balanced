## Hypothesis

**H87: Hold the cosine LR floor above zero — every recent run is still descending at the wall-cut.**

H73 baseline (val=42.98), H75 (LR sweep), and H81 (RMSNorm) all show the val_avg trajectory still monotonically descending at epoch 15 (the wall-cut). With CosineAnnealingLR(T_max=15, eta_min=0 default), the LR is ~0 by ep 15, so the final 1-3 epochs are nearly frozen. Holding the floor above zero (eta_min > 0) gives every epoch a meaningful step.

This is the **inverse** of H84 (T_max compression). H84 lets the model fine-tune at LR=0 for several epochs; H87 prevents the LR from collapsing to zero at all. The two together bracket the schedule-tail question.

Mechanistic prior: under Lion's sign-update, the effective per-parameter step at LR=0 is exactly 0 (sign(g)·0 = 0). With AdamW you can still have small drifts from m̂/v̂ residuals; under Lion the freeze is absolute. So eta_min > 0 should matter *more* under Lion than it would under AdamW.

Two arms:
- **Arm A: eta_min=3e-5** (lr/10) — moderate floor; final epochs run at ~10% of peak LR.
- **Arm B: eta_min=1e-5** (lr/30) — light floor; final epochs run at ~3% of peak LR.

**Predicted:**
- Arm A: ~40-43 val_avg (1-3 pt improvement if late-epoch gradient signal matters).
- Arm B: ~41-44 val_avg (smaller change; closer to H73 behavior).
- **Need Δ ≥ 3 pts** to clearly beat the 2.6 pt seed noise floor.

**Risk:** Holding LR too high prevents the implicit fine-tuning of the cosine tail. If both arms regress, the late-epoch fine-tune was actually useful and we should chase H84's direction instead.

## Baseline

H73 Arm B val=42.9784 / test=41.5455 (PR #4055).

⚠ Noise floor ~2.6 pts (from H74). Aim for ≥3 pt improvement.
