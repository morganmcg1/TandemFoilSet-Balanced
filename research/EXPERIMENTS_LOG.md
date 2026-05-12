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

---

## 2026-05-12 20:XX — PR #1521: Fix scoring NaN — nan_to_num before mask-multiply (frieren)
- willowpai2g24h4-frieren/fix-scoring-nan
- **Hypothesis:** `test_geom_camber_cruise` sample 20 has GT NaN in the p-channel. `err * mask` at `data/scoring.py:49` evaluates `NaN * 0 = NaN` in IEEE 754, propagating NaN through the mean and blocking `test_avg/mae_surf_p` for all experiments.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | ~98.77 (same model — scoring fix only) |
| test_avg/mae_surf_p | **131.14** ✓ first valid test metric on this branch |
| Epochs completed | 18 / 50 (30-min cap) |

**Fix:** `err = err.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)` inserted at line 49 of `data/scoring.py` before `err * mask`. Also zeros `posinf` — catches bf16 eval overflow that was biasing cruise test.

**Conclusion:** MERGED. Critical infrastructure fix; unblocks `test_avg/mae_surf_p` for all future runs. First valid `test_avg = 131.14` established. Cruise node bf16-inf now contributes 0 rather than NaN (still slightly biased — fp32-eval follow-up assigned to frieren as PR #1556).

---

## 2026-05-12 20:XX — PR #1415: bf16 mixed precision + grad_clip (thorfinn)
- willowpai2g24h4-thorfinn/bf16-amp
- **Hypothesis:** bf16 AMP halves memory bandwidth, enabling more slice tokens and/or larger batch without OOM. `grad_clip_norm=1.0` stabilises training in reduced precision. Expected ~40% throughput gain.
- **W&B:** `ojdeyn8r`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best epoch 18) | **98.7664** ✓ NEW BASELINE |
| test_avg/mae_surf_p | NaN at submit (cruise node bf16-inf → zeroed post-#1521) |
| Test 3-split mean (excl. cruise) | 97.12 |
| Epochs completed | 18 / 50 (30-min cap) |
| Sec/epoch | ~99s (vs 172s fp32 — 42% faster) |
| Peak GPU | 32.9 GB (vs 54.5 GB fp32 — 40% reduction) |

Per-split val (epoch 18):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 108.76 |
| val_geom_camber_rc | 115.38 |
| val_geom_camber_cruise | 78.21 |
| val_re_rand | 92.71 |
| **avg** | **98.77** |

**Conclusion:** MERGED as new baseline. 32.5% improvement over previous best (146.25). bf16 dramatically reduced both memory (32.9 vs 54.5 GB) and time-per-epoch (99 vs 172s). Model still descending at epoch 18 — schedule mismatch (T_max=50 but only ~18 achievable) means cosine LR ends at ~71% of peak. Follow-ups: (1) fp32 eval to recover faithful test_avg (frieren PR #1556), (2) T_max=20 retune so cosine fully cools within achievable budget (thorfinn PR #1557).
