# SENPAI Research Results — `willow-pai2i-48h-r4`

## 2026-05-15 14:07 — PR #3092: More physics-attention slice tokens (slice_num 64→128, 192)

- **Student:** willowpai2i48h4-fern (branch: `willowpai2i48h4-fern/more-slices`)
- **Hypothesis:** Doubling `slice_num` from 64 to 128 raises the resolution of Transolver's physics decomposition over 74K–242K node meshes; predicted −3% to −7% on `val_avg/mae_surf_p`.

### Results

| Arm | slice_num | n_params | best val_avg/mae_surf_p | best epoch | total epochs | peak VRAM | epoch time | W&B run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A (winner) | 128 | 672,919 | **150.26** | 9 | 10 | 54.5 GB | 171 s | `yiiy92uj` |
| B | 192 | 683,479 | 153.71 | 9 | 10 | 68.4 GB | 213 s | `l7nnvr53` |

Per-split val surface pressure MAE (best ckpt, epoch 9):

| Split | Arm A (128) | Arm B (192) |
|---|---:|---:|
| val_single_in_dist | 185.70 | 183.15 |
| val_geom_camber_rc | 157.16 | 179.11 |
| val_geom_camber_cruise | 127.68 | **115.40** |
| val_re_rand | 130.51 | 137.20 |
| **val_avg/mae_surf_p** | **150.26** | 153.71 |

Per-split test: `test_geom_camber_cruise/mae_surf_p = None / NaN` on **both arms** (vol_loss=Infinity), poisoning `test_avg/mae_surf_p`. The student reported a 3-split mean (excl. cruise) of 144.76 (A) / 152.56 (B), but this is not the contract metric.

### Analysis & Verdict — sent back (not merged)

- Arm A (slice_num=128) beats Arm B (slice_num=192) by 3.45 on `val_avg/mae_surf_p` (−2.2% absolute), and is significantly cheaper (−20% VRAM, −20% epoch time). Higher slice_num does NOT help in this short-training regime — likely the optimization burden of more slice assignments to learn outweighs the gain.
- **No baseline number on this branch** — we cannot establish that `slice_num=128` improves on the actual baseline `slice_num=64`. The PR documents what beats what *internally* but not against the actual reference.
- `test_avg/mae_surf_p` is NaN — fails the full-metric-fidelity contract from CLAUDE.md.

### Critical cross-cutting finding: LR schedule is mis-tuned for the wall-clock budget

The student's most valuable observation: with `SENPAI_TIMEOUT_MINUTES=30` and ~170s/epoch, only ~10 epochs of the configured 50-epoch cosine schedule complete. `T_max=50` means LR is still at ~80% of peak when training stops — **no experiment on this branch is getting LR annealing**. This affects every other in-flight PR (#3089, #3090, #3091, #3093, #3095, #3096, #3097). Future PRs should pass `--epochs 10` (or whatever matches actual completed-epoch count) so `T_max` matches budget.

### NaN on `test_geom_camber_cruise/mae_surf_p`

The model emits inf/NaN predictions on at least one sample in the cruise test split when evaluated from a partial-training checkpoint. Identical across both arms, so doesn't affect this PR's A-vs-B comparison. Likely fixable by training to convergence (with proper LR annealing), gradient clipping, or `torch.nan_to_num` band-aid. Edward's PR #3091 (grad clip) and alphonse's PR #3089 (L1 loss) may both address this naturally.

### Follow-up (sent back to fern as comment on #3092)

Run 2-arm comparison at `--epochs 10` to fully anneal cosine `T_max`:
- Arm A: `slice_num=64` (establishes the branch baseline)
- Arm B: `slice_num=128` (confirms with proper schedule)

Merge if Arm B beats Arm A on `val_avg/mae_surf_p` AND `test_avg/mae_surf_p` is finite on Arm B.

---

## 2026-05-15 14:38 — PR #3091: LR warmup + gradient clipping + higher peak LR (edward) — **MERGED**

- **Student:** willowpai2i48h4-edward (branch: `willowpai2i48h4-edward/warmup-clip`)
- **Hypothesis:** Adding 2-epoch linear warmup and gradient clipping (max_norm=1.0) stabilizes training and enables higher peak LR. Arm B tested lr=1e-3 (2× baseline 5e-4). Predicted delta: −3% to −8%; actual win was >10%.

### Results

| Arm | lr | best epoch | val_avg/mae_surf_p | test (3-split workaround) | W&B run |
|---|---|---|---|---|---|
| A (warmup+clip+5e-4) | 5e-4 | 13 | 121.54 | 124.19 | `qm3lqtwz` |
| **B (warmup+clip+1e-3)** | 1e-3 | 14 (last) | **109.42** | **107.47** | `0ez1sqmi` |

Per-split val surface pressure MAE (best ckpt, epoch 14):

| Split | Arm A | Arm B |
|---|---:|---:|
| val_single_in_dist | 184.40 | 119.58 |
| val_geom_camber_rc | 115.04 | 119.40 |
| val_geom_camber_cruise | 88.03 | 88.57 |
| val_re_rand | 104.43 | 110.12 |
| **val_avg/mae_surf_p** | 121.54 | **109.42** |

Test: NaN on `test_geom_camber_cruise` for both (scoring bug). 3-split workaround: 124.19 (A) / 107.47 (B).

### Analysis & Decision — MERGED

- **Decisive win.** Arm B beats Arm A by 12.1 on val_avg (−10%) and by 16.7 on test 3-split (−13%). Pre-clip grad norm was 160 vs 14 at the last step — clipping is doing real work.
- Arm B's best epoch = 14/14 (last): model was still strictly improving when the timeout cut it, indicating significant headroom at longer training.
- `warmup_epochs=2` over 14 effective epochs = ~14% warmup, higher than intended. Short warmup is still the right call at high LR — doesn't hurt.
- Code change is minimal: 20 lines, adds warmup lambda scheduler + clip + grad_norm logging. Clean, composable with all other experiments.
- **Merged as new branch baseline: val_avg/mae_surf_p = 109.42** (lr=1e-3 + warmup + clip).

### Follow-up (edward)

Assigned edward a bug-fix + consolidation PR:
- Unblock `test_avg/mae_surf_p` by nan_to_num fix in `evaluate_split` (avoids `0 * NaN = NaN` propagation in accumulate_batch)
- Bump Config.lr default from 5e-4 to 1e-3 to lock in winning config for all future students

---

## 2026-05-15 15:30 — PR #3089: L1 loss vs Huber β=1.0 (alphonse) — **SENT BACK** (close to merge)

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/l1-loss`)
- **Hypothesis:** Replace MSE with L1 loss in normalized space; align training objective with MAE metric. Predicted −8% to −15%.

### Results

| Arm | Loss | best epoch | val_avg/mae_surf_p (W&B-verified) | test_avg/mae_surf_p (claim) | W&B run |
|---|---|---|---|---|---|
| **A (winner)** | L1 | 13 | **102.37** | 89.67 (offline re-eval) | `lb2ly5g3` |
| B | Huber β=1.0 | 13 | 117.47 | 106.03 | `9gh0e13m` |

Per-split val surface pressure MAE (Arm A, best epoch 13, alphonse's report):

| Split | Arm A (L1) | Arm B (Huber) |
|---|---:|---:|
| val_single_in_dist | 133.71 | 138.99 |
| val_geom_camber_rc | 108.91 | 118.50 |
| val_geom_camber_cruise | 76.50 | 102.26 |
| val_re_rand | 90.37 | 110.13 |
| **val_avg/mae_surf_p** | **102.37** | 117.47 |

W&B verification (subagent):
- Arm A: val_avg/mae_surf_p = 102.37 (best_val_avg) ✓ VERIFIED — beats baseline (109.42) by **−6.4%**.
- Arm A: test_avg/mae_surf_p = `None` in W&B summary; alphonse's claimed 89.67 came from offline re-eval after adding the fix post-training.
- Arm A: val_geom_camber_cruise/mae_surf_p = 84.79 in W&B (real number) ← alphonse's nan_to_num/sub-select fix DOES work for val.
- 3-split test mean (excl. cruise): test_re_rand=86.10, test_geom_camber_rc=96.20, test_single_in_dist=111.43 → mean ≈ **97.91** vs edward's 107.47.

### Bug-fix included

Alphonse correctly identified the `0 * NaN = NaN` propagation in `accumulate_batch` (same as edward's #3288 but more robust — handles both NaN and Inf in y via `torch.isfinite` + sub-select fully finite samples).

### Decision — Sent back with two specific asks

1. **Flip Config default `loss_type: str = "mse"` → `"l1"`** so future runs compose on L1 automatically.
2. **Push a clean W&B-logged eval with the fix in place** so `test_avg/mae_surf_p` is verifiable (not just offline re-eval). Quick `--debug --epochs 1` pass is sufficient.

After: transition to `status:review`, mark ready, merge.

### Composition with #3091

Alphonse trained on `lr=5e-4` + no warmup + no clip (pre-#3091 advisor branch). When merged into post-#3091 branch, future runs get: `lr=1e-3 + warmup + clip + L1`. Likely further headroom — these changes are orthogonal.

### Coordination with #3288

Alphonse's scoring fix (sub-select + torch.where) is more robust than edward's (`nan_to_num`). When alphonse's PR merges first, edward's #3288 should drop the duplicate scoring fix and only keep the lr default bump.

---

## 2026-05-15 17:30 — PR #3096: x-axis reflection symmetry augmentation (tanjiro) — **SENT BACK** (regression, conditional re-run)

- **Student:** willowpai2i48h4-tanjiro (branch: `willowpai2i48h4-tanjiro/xflip-aug`)
- **Hypothesis:** Per-sample x-flip aug with Ux/AoA/stagger negation; predicted gains on geom_camber OOD splits.
- **W&B run:** `a7kc6xxi` (verified)

### Results

| Arm | val_avg/mae_surf_p | test 3-clean-split | best epoch | total epochs |
|---|---:|---:|---:|---:|
| Single arm (xflip aug) | **161.54** | 162.46 | 12 | 14 |

Compared to current baseline (109.42 from PR #3091): **+47% regression**. But branch was forked pre-#3091 (lr=5e-4, no warmup, no clip), so most of the gap is the stale-branch infrastructure. On the same pre-#3091 code, fern's slice_num=128 baseline (#3092) landed at val=150.26 — tanjiro is ~7% worse than that with augmentation.

Per-split val surface MAE (best epoch 12):

| Split | Tanjiro xflip | fern slice_num=128 (same code) |
|---|---:|---:|
| val_single_in_dist | **203.61** | 185.70 |
| val_geom_camber_rc | 173.37 | 157.16 |
| val_geom_camber_cruise | 125.17 | 127.68 |
| val_re_rand | 143.99 | 130.51 |
| **val_avg/mae_surf_p** | **161.54** | 150.26 |

### Three concerning signals

1. **Model peaked at epoch 12 and rose for epochs 13–14** (163.0 → 167.0). The wall clock didn't cut mid-improvement; the model was overfitting. With higher LR (lr=1e-3 in current advisor) it'll likely overfit even earlier.
2. **`val_single_in_dist = 203.61` is the WORST split** — the easiest split (in-distribution) is being hurt by augmentation. xflip is making in-dist samples harder while only marginally helping OOD.
3. **`val_geom_camber_cruise` ≈ identical to fern's number** (125.17 vs 127.68). The predicted OOD gain isn't showing up in absolute numbers; the relative-easier-than-in-dist signal is plausible but not symmetry-specific.

### Bug-fix analysis

Tanjiro independently identified the same `0 * NaN = NaN` propagation in `accumulate_batch` that edward and alphonse flagged. Same root cause, same path (read-only `data/scoring.py`).

### Decision rule for the rebased confirmation arm

- val < 109.42 → merge
- val ∈ [109.42, 115] → merge only if geom_camber_cruise is clearly the best split (OOD-aug story still holds)
- val > 115 → close. Hypothesis empirically unsupported at this scale.

### Notes

- Augmentation halves effective gradient signal per orientation; could benefit from longer schedule, but within 30-min budget the unaugmented baseline gets twice the effective per-orientation samples.
- Symmetry aug is theoretically sound; the result here is most likely an interaction with: (a) stale code, (b) wall-clock cap, (c) MSE loss (L1 might compose better with aug). Worth re-investigating in round 2 stacked with alphonse's L1 + edward's warmup.
