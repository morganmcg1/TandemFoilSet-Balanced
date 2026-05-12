# SENPAI Research Results

## 2026-05-12 19:XX — PR #1396: Double Transolver slice tokens (slice_num 64 -> 128)
- willowpai2g24h4-frieren/more-slice-tokens
- **Hypothesis:** Doubling slice_num from 64 to 128 gives PhysicsAttention finer spatial resolution for boundary-layer pressure capture.
- **W&B:** `5qh8pj8v`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best epoch 9) | **146.2510** ✓ NEW BASELINE |
| test_avg/mae_surf_p | NaN (scoring bug — see below) |
| test 3-split avg (excl. cruise) | 147.07 |
| Epochs completed | 11 / 50 (30-min cap) |
| Sec/epoch | ~172s |
| Peak GPU | 54.5 GB |

Per-split val (epoch 9):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 175.68 |
| val_geom_camber_rc | 158.18 |
| val_geom_camber_cruise | 115.62 |
| val_re_rand | 135.53 |
| **avg** | **146.25** |

**Conclusion:** MERGED. Cleanest improvement — single-knob change, val still descending at cutoff. Cruise split is the strongest (115.62). Baseline raised to 146.25.

**Bug found:** `test_geom_camber_cruise` sample 20 has NaN GT (p-channel only). `err * mask` in `data/scoring.py:49` produces `NaN * 0 = NaN` even though `sample_mask` is False for that sample. Blocks `test_avg/mae_surf_p` for all experiments. Bug-fix PR #1521 assigned to frieren.

---

## 2026-05-12 19:XX — PR #1404: OneCycleLR (max_lr=1e-3, pct_start=0.1)
- willowpai2g24h4-nezuko/onecycle-lr
- **Hypothesis:** OneCycleLR super-convergence should reach deeper minimum within 30-min cap.
- **W&B:** `nsrzmqb5`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best epoch 10) | 151.1062 |
| test_avg/mae_surf_p | NaN (scoring bug) |
| Epochs completed | 10 / 50 (30-min cap) |
| Sec/epoch | ~131s |

Per-split val (epoch 10):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 170.48 |
| val_geom_camber_rc | 165.43 |
| val_geom_camber_cruise | 131.89 |
| val_re_rand | 136.63 |

**Conclusion:** SENT BACK. val=151.11 does not beat new baseline 146.25. Structural flaw: `total_steps = 50 * 375` sized the schedule for 50 epochs, but the 30-min cap reached only ~20% of total steps. Schedule never entered cosine decay tail — not a fair super-convergence test. Retry with total_steps=12*steps_per_epoch and rebase on updated baseline.

---

## 2026-05-12 19:XX — PR #1409: Bigger Transolver (n_hidden 128->192, n_head 4->6)
- willowpai2g24h4-tanjiro/bigger-hidden
- **Hypothesis:** Wider model (~2× params) reduces val_avg/mae_surf_p on irregular mesh.
- **W&B:** `mymcr3v4`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best epoch 9) | 160.3114 |
| test_avg/mae_surf_p | NaN (scoring bug) |
| test 3-split avg (excl. cruise) | 154.76 |
| Epochs completed | 9 / 50 (30-min cap) |
| Sec/epoch | ~202s |
| Peak GPU | 63.0 GB |

**Conclusion:** CLOSED (9.6% above new baseline 146.25). Underlying hypothesis still valid — model only got 9 epochs due to throughput hit. Retried as PR #1522 on updated baseline (slice_num=128 + n_hidden=192, + grad_clip). Tanjiro correctly identified scoring bug root cause.
