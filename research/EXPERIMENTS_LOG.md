# SENPAI Research Results — willow-pai2g-48h-r3

Round 1 of a fresh launch on advisor branch `icml-appendix-willow-pai2g-48h-r3`.
All eight hypotheses (PRs #1504–#1511) dispatched 2026-05-12; results recorded below as they land.

## 2026-05-12 19:30 — PR #1510: Fourier positional encoding (L=6) for (x, z)

- **Student:** willowpai2g48h3-tanjiro
- **Branch:** willowpai2g48h3-tanjiro/fourier-pos-enc
- **Hypothesis:** Add NeRF-style Fourier features (`sin(2^k π x), cos(2^k π x)` for k=0..5, scale=1.0) on the spatial coords prepended to the input. Predicted Δ on `val_avg/mae_surf_p`: −3% to −10%.

### Results

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| `single_in_dist` | 138.27 | 121.63 |
| `geom_camber_rc` | 146.15 | 140.23 |
| `geom_camber_cruise` | 93.25 | **NaN** ⚠️ |
| `re_rand` | 107.96 | 108.53 |
| **`avg/mae_surf_p`** | **121.41** | **NaN** (3-split mean = 123.46) |

W&B run: `fp227kem`. Best checkpoint at epoch 13/14 (run hit the 30-min wall-time cap mid-epoch-14). Total params: 0.67M. Peak VRAM: 42.3 GB.

### Conclusion

**Not merged — sent back.** `test_avg/mae_surf_p` is NaN because at least one sample in `test_geom_camber_cruise` produced inf/NaN on the pressure channel during the end-of-run test eval. Per the no-NaN-on-primary-metric rule, this is disqualifying for merge even though val_avg is finite.

The student's diagnosis (Fourier max freq 32π → slice_norm collapse on one outlier sample → pressure-head amplification to inf) is well-reasoned. The cleanest fix is the already-in-PR fallback: re-run with `pos_scale=0.1`, dropping the max frequency to 3.2π ≈ 10. Sent back with that instruction.

### Follow-up

PR #1510 returned to WIP with explicit instructions to retry only the `pos_scale=0.1` variant. Acceptance criterion for the retry: `test_avg/mae_surf_p` must be finite across all four test splits.

## 2026-05-12 21:30 — PR #1510 (closed): Fourier pos enc retry with `pos_scale=0.1`

Retry results (W&B run `qziefxht`, 30.8 min, 14 epochs, best at epoch 12):

| Split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| `single_in_dist` | 169.58 | 137.17 |
| `geom_camber_rc` | 134.73 | 122.34 |
| `geom_camber_cruise` | 108.11 | **NaN** ⚠️ |
| `re_rand` | 117.81 | 115.11 |
| **`avg/mae_surf_p`** | **132.56** | **NaN** (3-split mean = 124.87) |

**Closed.** The cruise NaN persisted at a 10× softer Fourier spectrum, meeting the pre-stated stop condition. Combined with the val regression vs. `scale=1.0` (132.6 vs 121.4 — the higher-frequency features were doing useful representational work), conclusion is:

1. The cruise pressure-head blowup is a **model-level robustness issue**, not a Fourier-spectrum issue.
2. Fourier pos enc (any scale) cannot be evaluated fairly until the cruise instability is independently resolved.

Tanjiro reassigned to PR #1589 (AdamW betas tuning) — a clean optimizer-only hypothesis on a different axis.

Future work (separate PRs, not bolted onto Fourier):
- Per-sample `(slice_norm.min, |fx|.max)` instrumentation on cruise test samples
- Compare cruise test eval behavior on the unmodified baseline (rules in/out whether the blowup is Fourier-specific)
- Output-head magnitude bounding (slice_norm clamp from below, or LayerNorm on residual)
