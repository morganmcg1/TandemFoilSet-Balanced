## Hypothesis

**H85: Test FFN activation alternatives under Lion+slice=96.**

H48 (AdamW+slice=64) found GEGLU > SwiGLU > vanilla. The optimum may shift under Lion+slice=96.

Two arms:
- **Arm A: ffn_act=swiglu** — H48 runner-up; smoother gate gradient may suit Lion better.
- **Arm B: ffn_act=vanilla** — no gating sanity floor.

**Predicted:** Arm A: ~43-47 val_avg; Arm B: ~46-50 val_avg. Both likely lose; the value is in the mechanism check.

## Baseline
H73 Arm B val=42.9784 / test=41.5455 (PR #4055).

⚠ Noise floor ~2.6 pts (from H74).
