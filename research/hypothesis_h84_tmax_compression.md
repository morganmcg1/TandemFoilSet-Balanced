## Hypothesis

**H84: Compress cosine schedule below the wall budget — spend more time at sub-1e-5 LR for fine-tuning.**

H74 showed extending T_max=20 beyond the 15-epoch wall budget is harmful. Askeladd's own follow-up suggestion: try T_max < epochs_actual. With T_max=12 and 15 actual epochs, the cosine reaches LR=0 by epoch 12, then trains 3 epochs at LR=eta_min for fine-tuning.

Two arms:
- **Arm A: T_max=12, epochs=15** — 3 epochs of LR fine-tune.
- **Arm B: T_max=10, epochs=15** — 5 epochs of LR fine-tune.

**Predicted:** Arm A: ~41-44 val_avg; Arm B: ~42-45 val_avg. Need ≥3 pt improvement to exceed seed noise.

## Baseline
H73 Arm B val=42.9784 / test=41.5455 (PR #4055).

⚠ Noise floor ~2.6 pts (from H74).
