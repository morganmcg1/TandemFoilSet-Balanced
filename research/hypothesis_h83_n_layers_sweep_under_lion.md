## Hypothesis

**H83: Retune n_layers under Lion+slice=96 (H73 baseline).**

H60 (AdamW) found n_layers=4 wins over n_layers={3, 5, 6} at slice=64+GEGLU baseline. Under Lion at slice=96, the depth optimum may shift.

Two arms:
- **Arm A: n_layers=5** — deeper, more capacity.
- **Arm B: n_layers=3** — shallower, faster per epoch (more cosine steps).

**Predicted:** Arm A: ~41-44 val_avg; Arm B: ~42-46 val_avg.

Run both arms with H73 winning config except `--n_layers` swapped. Stop if val_avg at ep 3 > 250.

## Baseline
H73 Arm B val_avg=42.9784 / test=41.5455 (PR #4055).
