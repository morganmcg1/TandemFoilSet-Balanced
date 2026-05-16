## Hypothesis

**H88: Refine the β₂ optimum around 0.995 (H78 winner).**

H78 found a non-monotonic β₂ response: 0.99 → 0.995 (wins) → 0.999 (regresses). With only 3 sparse samples, the peak is undercharacterized — 0.995 may be local-optimum or a slightly different value may be marginally better.

Two arms:
- **Arm A: β₂=0.992** — interpolates between 0.99 (old baseline) and 0.995 (new best).
- **Arm B: β₂=0.997** — interpolates between 0.995 and 0.999.

**Predicted:**
- Arm A: ~42-43 val_avg (close to baseline; β₂ probably above 0.99 is needed).
- Arm B: ~42-44 val_avg (between 0.995 and 0.999, likely close to 0.995).
- **Need Δ ≥ 1 pt vs current best (42.30) to be considered a refinement.**

**Risk:** Within seed noise; refinement may be a local-optimum hunting waste. But single-flag with zero complexity cost.

## Baseline

H78 Arm B val=42.3048 / test=40.5564 (PR #4097, MERGED).

⚠ Noise floor ~2.6 pts (from H74). Within-noise refinements are still candidates for merge if both metrics improve.
