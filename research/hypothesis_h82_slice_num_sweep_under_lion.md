## Hypothesis

**H82: Slice_num sweep under Lion+H73 baseline — does the slice_num optimum shift with Lion?**

H66 (AdamW) tested slice_num={64, 96, 128} and found 96 wins (56.75 vs 57.85 at 128). Under AdamW, slice=128 regressed. But H73 showed Lion has fundamentally different optimization dynamics (super-additive with slice=96, gradient surface widens, sign-update removes magnitude info). The slice_num optimum under Lion may differ from the AdamW finding.

Two arms:
- **Arm A: slice_num=128** — direct retest of AdamW's regression point under Lion.
- **Arm B: slice_num=80** — intermediate hedge.

**Predicted:** Arm A: ~40-46 val_avg; Arm B: ~42-46 val_avg.
**Risk:** Slice=128 may cause divergence at higher slice. Memory ~55-60 GB possible.

Run both arms with H73 winning config except `--slice_num` swapped. Stop if val_avg at ep 3 > 250.

## Baseline
H73 Arm B val_avg=42.9784 / test=41.5455 (PR #4055).
