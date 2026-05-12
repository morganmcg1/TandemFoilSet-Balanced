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

## 2026-05-12 21:15 — Cross-PR observation: cruise test NaN is a baseline correctness issue

W&B audit of all round-1 finished runs (snapshot ~21:15 UTC) shows `test_geom_camber_cruise/mae_surf_p` returns `None` for every finished run **except** alphonse's `xqrz8bjw` (mask-aware PhysicsAttention, PR #1504):

| Student / PR | wandb_id | val_avg/mae_surf_p | test_avg/mae_surf_p | cruise_test present |
|---|---|---:|---:|---|
| alphonse #1504 (mask-aware) | xqrz8bjw | 128.97 | **117.62** | **Yes** |
| edward #1506 (wider 192) | 1o90ujme | 148.45 | None | No |
| frieren #1508 (surf_weight 25) | zjxmwjhs | 140.47 | None | No |
| thorfinn #1511 (deeper 7) | i14s7xxp | 152.83 | None | No |
| tanjiro #1510 (Fourier, both scales) | fp227kem, qziefxht | 121.41, 132.56 | None | No |

Combined with the already-stated PR #1510 conclusion ("cruise blowup is a model-level robustness issue, not a Fourier-spectrum issue"), this is a strong correctness signal: the unmodified PhysicsAttention slice softmax produces inf/NaN on the cruise test eval, and **PR #1504's mask-aware fix appears to resolve it**.

Implications for round-1 review:
- PR #1504 just got materially more important — it's both a metric improvement and a correctness fix on the paper-facing metric.
- Other round-1 PRs that don't change the slice softmax mask cannot beat baseline on `test_avg/mae_surf_p` until the mask fix is in place (their test_avg will be None).
- Once #1504 merges, the rest of round 1 should be re-evaluated against the new mask-aware baseline.
- alphonse's seed-comparison run `hg135fap` is in flight to confirm `xqrz8bjw` isn't an RNG fluke.

Action: wait for alphonse to post `SENPAI-RESULT` once `hg135fap` finishes (~15 min ETA from 21:07Z), then prioritize PR #1504 for merge.

## 2026-05-12 21:52 — PR #1504 MERGED: Mask-aware PhysicsAttention (round-1 baseline)

- **Student:** willowpai2g48h3-alphonse
- **Branch:** willowpai2g48h3-alphonse/mask-aware-physics-attn
- **Merge commit:** `a6981e1`
- **W&B runs:** `hg135fap` (submitted), `xqrz8bjw` (seed-2)

### Final numbers (best-val checkpoint → test eval)

| metric | hg135fap | xqrz8bjw |
|---|---:|---:|
| `val_avg/mae_surf_p` | **119.450** | 128.966 |
| `test_avg/mae_surf_p` | **109.669** | 117.623 |
| best_epoch | 14 | 12 |
| runtime (min) | 31.9 | 32.1 |

### Per-split (hg135fap best-val)

| split | val mae_surf_p | test mae_surf_p |
|---|---:|---:|
| `single_in_dist` | 140.20 | 123.97 |
| `geom_camber_rc` | 133.10 | 121.92 |
| `geom_camber_cruise` | 93.08 | 81.06 |
| `re_rand` | 111.42 | 111.73 |

### Implementation note (intentional deviation from PR instructions)

PR proposed masking *before* slice softmax with `-inf`. Alphonse caught that the softmax is over `slice_num` (last dim), not `N` — applying `-inf` along N would produce `softmax([-inf, ...])` = NaN. They masked *after* softmax instead (`slice_weights * mask[:,None,:,None]`), which yields the same effect (padded positions contribute zero to numerator and denominator) without the NaN trap. Empirical validation: both seeds trained cleanly, finite metrics on all four splits including cruise.

### Implications for round 1

The 5 other in-flight stale_wip PRs (#1505-1509, #1511) and #1589 are running on the pre-merge train.py without the mask fix. If their results return `test_geom_camber_cruise=None`, the per-PR test_avg will be invalid against the new merged baseline. They should be allowed to finish their current runs; if the metrics-fidelity rule kicks in (NaN/None on primary), the PR will need rebase + re-run on the merged code. Send-back rather than close, since the underlying hypothesis is still untested.

### Follow-ups (for alphonse's next assignment)

Alphonse becomes idle on merge. Next axis: `mlp_ratio` (FFN capacity per block) — the one architectural lever not in flight in round 1 (width, depth, slice count all already assigned).
