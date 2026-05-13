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

---

## 2026-05-12 20:57 — PR #1557: Retune cosine T_max=50 → 20 (thorfinn)
- willowpai2g24h4-thorfinn/tmax-retune-20
- **Hypothesis:** With T_max=50 but only ~18 epochs achievable, cosine LR ends at ~71% of peak — schedule never enters its low-LR refinement tail. Resizing T_max=20 should let LR cool fully to ~0 within achievable budget. Predicted: 3-8% val improvement.
- **W&B:** `iycgna1l`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best epoch 13) | **112.9602** (−14.4% worse than baseline 98.77) |
| test_avg/mae_surf_p (4-split, bf16 eval) | **101.4561** ✓ first faithful 4-split test on this branch |
| Epochs completed | 13 / 20 (30-min cap) |
| Sec/epoch | ~139s (not the predicted ~99s — throughput variance) |
| Peak GPU | 93.76 GB |

Per-test-split surface MAE (bf16 eval, scoring fix in place):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 130.32 |
| test_geom_camber_rc | 106.18 |
| test_geom_camber_cruise | 70.73 |
| test_re_rand | 98.59 |
| **mean** | **101.46** |

**Conclusion:** CLOSED. Hypothesis disproven — at this budget, T_max=50 was already near-optimal because the model is compute-bottlenecked (still descending at termination) rather than schedule-bottlenecked. At epoch 13 LR was 27% of peak (vs 74% under T_max=50), so the run trained at roughly half the LR throughout, slowing convergence. The achievable epoch count (13 here) was lower than predicted because throughput came in at 139s/epoch not 99s.

**Key learning:** First faithful 4-split test_avg = 101.46 on this branch. Compute throughput, not LR schedule shape, is the bottleneck — directly motivates torch.compile follow-up (PR #1584).

---

## 2026-05-12 20:55 — PR #1522: Bigger hidden (n_hidden=192, n_head=6) on slice_num=128 baseline (tanjiro)
- willowpai2g24h4-tanjiro/hidden192-on-slice128
- **Hypothesis:** Wider model (~2.25× params, 0.65M → 1.46M) reduces val_avg/mae_surf_p with slice_num=128 baseline + grad clipping.
- **W&B:** `q9ja73zj`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best epoch 7) | **144.9091** |
| test 3-split avg (excl. cruise) | 145.75 |
| Epochs completed | 7 / 50 (30-min cap) |
| Sec/epoch | ~264s |
| Peak GPU | 95.37 GB (vs 96 GB cap — very tight) |

**Conclusion — SENT BACK.** Tanjiro built on top of the **old** baseline (PR #1396, pre-bf16) — they branched before PR #1415 was merged and never picked up bf16. Their 144.91 was compared against the old baseline of 146.25 (claimed −0.9% improvement), but the **current** baseline is 98.77. Asked to rebase on the merged baseline (which now includes bf16 + grad_clip), then re-run hidden192/n_head6 on top. With bf16, memory should drop ~40% (95 GB → ~55-60 GB) and epoch count should rise from 7 → ~15-17, giving the wider model a real shot at the new baseline. Directional signal interesting: cruise −8.6%, re_rand −9.5% on val (vs old baseline) suggests width helps OOD geometry/Reynolds; needs re-test with full bf16 budget.

---

## 2026-05-12 22:00 — PR #1556: fp32 eval (bf16 train only) — faithful test_avg (frieren)
- willowpai2g24h4-frieren/fp32-eval
- **Hypothesis:** Remove bf16 autocast from `evaluate_split` — train stays bf16, eval reverts to fp32. Recovers faithful `test_avg/mae_surf_p` for cruise node that overflows to inf in bf16.
- **W&B:** `m4gdwz80`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best epoch 13) | 114.9255 (regressed from baseline 98.77) |
| test_avg/mae_surf_p (4-split, fp32 eval) | **101.9851** ✓ first faithful fp32 test_avg |
| test_geom_camber_cruise | **74.52** ✓ finite (was NaN/inf-biased) |
| Epochs completed | 13 / 50 (30-min cap) |
| Sec/epoch | ~144s (vs 99s baseline — +45% overhead) |
| Peak GPU | 74.2 GB |

**Conclusion — SENT BACK.** Hypothesis confirmed: cruise test now finite (74.52) and test_avg=101.99 is faithful. But fp32 eval costs 45s/epoch, dropping from 18 epochs to 13 — an unacceptable wall-clock regression that would drag down all future baselines. Sent back with guidance to gate eval frequency via `eval_every_n_epochs=3` so average eval overhead = 15s/epoch (restore ~16 epochs, val near baseline). The fp32-no-autocast code change itself is correct — just needs the frequency lever before merging.

---

## 2026-05-12 21:56 — PR #1584: torch.compile(model, dynamic=True) — free throughput (thorfinn)
- willowpai2g24h4-thorfinn/torch-compile
- **Hypothesis:** torch.compile fuses kernels and eliminates Python overhead. On a transformer-style bf16 model, expected 20-40% speedup → more epochs in the 30-min cap → direct val improvement.
- **W&B:** `t0zwgi1n`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best epoch 27) | **76.4310** ✓ NEW BASELINE (−22.6% vs 98.77) |
| test_avg/mae_surf_p (4-split, bf16 eval) | 68.7604 (cruise biased-low by nan_to_num posinf-zero) |
| Test 3-split mean (excl. cruise) | 74.84 |
| Epochs completed | 29 / 50 (30-min cap) |
| Sec/epoch | 62.6s median (epoch 3+) — **1.58× throughput** |
| Peak GPU | 50.78 GB |

Per-val-split (best epoch 27):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 99.46 |
| val_geom_camber_rc | 93.02 |
| val_geom_camber_cruise | 51.21 |
| val_re_rand | 73.14 |
| **avg** | **76.43** |

**Conclusion: MERGED.** Free win on every axis — 1.58× throughput (not 1.2-1.4× predicted), 29 epochs vs 18 (+61%), val=76.43 (−22.6% improvement). Compile overhead is one-shot (~12s epoch 1 only), amortized over 28 subsequent epochs. dynamic=True confirmed correct — tight epoch-to-epoch variance (σ=0.7s) shows no recompilation jitter. No correctness regressions. New baseline raised to 76.43. Follow-up: T_max=30 retune (PR #1628) to align cosine schedule to new 29-epoch budget.

---

## 2026-05-12 23:XX — PR #1373: lr=1e-3 + 3-epoch linear warmup + cosine (alphonse)
- willowpai2g24h4-alphonse/lr-warmup-1e-3
- **Hypothesis:** Higher peak LR (1e-3 vs 5e-4) with linear warmup enables faster early descent and deeper final minimum within the 30-min cap. Predicted 5-15% val improvement.
- **W&B:** `waeuuqkw`

| Metric | Baseline #1584 | PR #1373 | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best ckpt) | 76.4310 | **75.8473** | **−0.58 (−0.76%)** |
| test_avg/mae_surf_p (4-split) | 68.7604 | **67.3037** | **−1.46 (−2.12%)** |
| Test 3-split mean (excl. cruise) | 74.84 | 74.80 | −0.04 |
| Best epoch | 27/29 | 27/29 | same |
| Peak GPU | 50.78 GB | 48.60 GB | −2.18 GB |
| Sec/epoch | ~62.6s | ~61.2s | −1.4s |

Per-split val (best epoch 27):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 86.08 |
| val_geom_camber_rc | 91.75 |
| val_geom_camber_cruise | 52.57 |
| val_re_rand | 72.99 |
| **avg** | **75.85** |

Per-split test (all 4 splits clean):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 76.74 |
| test_geom_camber_rc | 83.08 |
| test_re_rand | 64.59 |
| test_geom_camber_cruise | 44.80 (nan_to_num biased low) |
| **avg (4-split)** | **67.30** |

**Diff vs prior baseline:** lr 5e-4 → 1e-3; replaced `CosineAnnealingLR(T_max=50)` with `SequentialLR([LinearLR(start_factor=0.1, total_iters=3), CosineAnnealingLR(T_max=47)], milestones=[3])`.

**Conclusion: MERGED.** Both val (−0.76%) and test (−2.12%) improved in the same direction. The val delta alone is within the ±5% RNG variance alphonse measured (single-seed; cannot conclude lr=1e-3 categorically better), but the test improvement of 2.1% is more meaningful and the directions are consistent. Per "when in doubt, merge" principle and "small improvements compound" guideline, took it. New baseline: val=75.85, test=67.30. Note: the warmup prevented the high-LR instability seen in earlier 1e-3 trials. T_max retune (thorfinn #1628) and the warmup are now orthogonal — may stack.

---

## 2026-05-12 23:XX — PR #1390: Raise surf_weight 10 → 25 (fern)
- willowpai2g24h4-fern/higher-surf-weight
- **Hypothesis:** Increasing the surface-node loss weight from 10× to 25× biases gradients toward surface-pressure prediction, improving val_avg/mae_surf_p.
- **W&B:** `73qlj0l3`, `h4cghevp`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p best (run 1) | **128.75** |
| val_avg/mae_surf_p best (run 2) | 131.82 |
| Run-to-run variance | ~2.4% |
| Epochs completed | 13-14 / 50 (30-min cap) |
| Sec/epoch | ~128s |
| Peak GPU | 91.6 – 94.9 GB |

**Problem: Stale baseline.** Fern's branch was created before PR #1415 (bf16), #1584 (compile), and #1373 (lr warmup) were merged. The run is on pre-compile fp32 code: ~128 s/epoch, 14 epochs, 91+ GB peak — none of these match the current compile+bf16 baseline (~62s/epoch, 29 epochs, ~50 GB). val=128.75 is not comparable to current baseline of 75.85.

**Notable finding:** `test_geom_camber_cruise` showed NaN in both runs. Fern correctly diagnosed this: higher surf_weight forces more aggressive pressure predictions; on OOD cruise samples the normalised MSE can overflow fp32, sending vol_loss to inf and contaminating the split metric. On the compile+bf16 baseline the nan_to_num scoring fix (#1521) should mitigate this, but it may still NaN if predictions genuinely overflow before scoring. Worth monitoring on rebase.

**Conclusion: SENT BACK.** Rebase on current advisor branch (gets compile + bf16 + warmup + lr=1e-3 automatically), keep only surf_weight=25 change, re-run. Once rebased, surf_weight=25 gets 29 epochs at 61s/epoch — a real head-to-head test. Also flagged the cruise NaN failure mode to watch after rebase.

---

## 2026-05-13 00:00 — PR #1404: OneCycleLR (max_lr=1e-3, SCHEDULER_EPOCHS=29, per-batch) — nezuko
- willowpai2g24h4-nezuko/onecycle-lr
- **Hypothesis:** OneCycleLR with `SCHEDULER_EPOCHS` correctly sized to the compile budget (29 epochs) should outperform warmup+cosine by fully firing the decay tail (LR → 1e-7) within the 30-min wall.
- **W&B:** `wd9na4r7`

| Metric | Baseline #1373 | PR #1404 (OneCycleLR) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best ckpt) | 75.8473 | **70.9449** | **−4.90 (−6.5%)** |
| test_avg/mae_surf_p (4-split) | 67.3037 | **61.8276** | **−5.48 (−8.1%)** |
| Test 3-split mean (excl. cruise) | 74.80 | **68.12** | **−6.68 (−8.9%)** |
| Best epoch | 27/29 | **29/29 (last)** | Still descending! |
| Median sec/epoch | ~61.2s | 62.5s | +1.3s negligible |
| Peak GPU | 48.60 GB | 50.97 GB | +2.4 GB |

Per-split val (best epoch 29):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 80.87 |
| val_geom_camber_rc | 80.80 |
| val_geom_camber_cruise | 51.75 |
| val_re_rand | 70.36 |
| **avg** | **70.94** |

Per-split test (all 4 clean):

| Split | mae_surf_p |
|---|---|
| test_single_in_dist | 70.98 |
| test_geom_camber_rc | 72.75 |
| test_geom_camber_cruise | 42.96 (nan_to_num biased low) |
| test_re_rand | 60.63 |
| **avg (4-split)** | **61.83** |

**LR curve:** 1e-4 (start) → 1e-3 (peak, ~ep3) → 1e-7 (final, ep29). Decay tail fully fired. 10875 per-batch LR updates (vs 29 for epoch-level cosine). Noisy phase ep11-13 (during early decay) then smooth descent from ep14-29.

**Conclusion: MERGED. Massive −6.5%/−8.1% win on both metrics.** All 4 splits improved consistently. Best epoch = last epoch = still descending at cutoff. Key mechanism: per-batch LR stepping + full decay tail gives the model much better convergence structure than per-epoch warmup+cosine. This is now the default stack. T_max retune (thorfinn #1628) and warmup+cosine LR sweep (alphonse #1687) are both superseded — redirecting to OneCycleLR-based follow-ups.

---

## 2026-05-13 00:55 — PR #1628: T_max=27 via epochs=30 (SENT BACK — stale base, single seed)
- willowpai2g24h4-thorfinn/tmax-compile-retune
- **Hypothesis:** Aligning T_max to achievable epoch budget (epochs=30 → T_max=27) eliminates the late-epoch LR-noise uptick observed in PR #1584 (76.43→79.21 between ep27-29).
- **W&B:** `lm4n59dm`

| Metric | Run #1628 (1 seed) | Ref baseline #1373 (T_max=47) | Δ vs #1373 | Current OneCycleLR #1404 | Δ vs #1404 |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p (best ckpt) | **69.2638** | 75.8473 | −6.58 (−8.7%) | 70.9449 | **−1.68 (−2.4%)** |
| test_avg/mae_surf_p (4-split) | **59.6119** | 67.3037 | −7.69 (−11.4%) | 61.8276 | **−2.22 (−3.6%)** |
| Test 3-split mean (excl. cruise) | 65.58 | 74.80 | −9.22 | 68.12 | −2.54 |
| Best epoch | 29/30 (last) | 27/29 | — | 29/29 | — |
| Sec/epoch | 62.6 | 61.2 | — | 62.5 | — |
| Peak GPU | 29.8 GB | 48.60 GB | — | 50.97 GB | — |
| LR at termination | 3.38e-6 | — | — | ~1e-7 | — |

**Hypothesis: directionally CONFIRMED on the old stack.** Val curve became monotonically decreasing through epoch 29 (no late-epoch uptick): 78.05 → 76.09 → 73.26 → 71.74 → 70.61 → 70.07 → **69.26**. Best epoch is the last. The LR-noise theory holds: residual LR ~3e-6 at termination is well below the kick-out threshold, vs ~1.9e-4 in PR #1584.

**Procedural issue (why SENT BACK rather than merged):**
- PR base was commit `23df5a0` (before OneCycleLR #1404 merged at `b1c91fb`). The student didn't rebase, so their training run used the OLD SequentialLR(warmup+cosine, T_max=27) stack, NOT the new OneCycleLR baseline.
- Diff vs current advisor tip is ONLY `epochs: 50 → 30`. Squash-merging this onto the new OneCycleLR baseline yields OneCycleLR(SCHEDULER_EPOCHS=29) + epochs=30, which runtime-equivalent to current baseline (wall clock caps both at 29 epochs).
- The merged code would NOT reproduce 69.26 — the result depended on the SequentialLR scheduler family, not the epochs change.

**What the data point actually tells us:**
- Single-seed comparison: SequentialLR(T_max=27, per-epoch) vs OneCycleLR(SCHEDULER_EPOCHS=29, per-batch). On this seed, SequentialLR(T_max=27) wins by 2.4% val / 3.6% test.
- Test delta of 3.6% is just above the 2% reliability floor — but this is a single seed, RNG variance is ±5%, AND a different scheduler architecture entirely. Multi-seed confirmation is essential before considering reverting OneCycleLR.

**Conclusion: SENT BACK** for proper rebase + intentional SequentialLR revert + 2-seed confirmation. If 2-seed mean still beats OneCycleLR by >2% test, this becomes a baseline revert candidate. The single-seed lucky-draw possibility is real.

---

## 2026-05-13 01:00 — PR #1522: hidden-192 + n_head=6 on OneCycleLR baseline (CLOSED — clear regression)
- willowpai2g24h4-tanjiro/hidden-192-on-slice-128
- **Hypothesis:** Wider hidden dim (128→192) + more heads (4→6) compounds with merged slice_num=128 to give better representational capacity for surface pressure.
- **W&B:** `smyi09m6` (iter2 on rebased OneCycleLR baseline)

| Metric | OneCycleLR baseline #1404 | This run (hidden-192, OneCycleLR) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 70.9449 | 79.0117 | **+8.07 (+11.4%)** ❌ |
| test_avg/mae_surf_p (4-split) | 61.8276 | 69.0873 | **+7.26 (+11.7%)** ❌ |
| Test 3-split mean (excl. cruise) | 68.12 | 76.83 | +8.71 (+12.8%) |
| Epochs completed | 29 | 20 | −9 epochs |
| Sec/epoch | 62.5 | 92.4 | +47.9% slower |
| Peak GPU | 50.97 GB | 76.87 GB | +25.9 GB |

All 4 splits regressed (in-dist +17.1%, rc +14.5%, cruise +5.4%, re_rand +5.6%). The wider model is bottlenecked by the 30-min cap: it only sees 20 epochs vs baseline's 29, and OneCycleLR's per-batch decay schedule (sized for 29 epochs) only consumes 7500/10875 steps (69%) — the LR decay tail to 1e-7 never fires (terminal LR was 2.7e-4).

**Mixed per-split signal:** Wider hidden hurts in-distribution + rc-camber (likely needs full schedule for fine-detail fitting) but partially helps OOD-Re and cruise even at truncated training. Direction may be valid but isolated test under the cap is impossible.

**Conclusion: CLOSED.** Tanjiro's analysis identified all three correct levers (right-size OneCycleLR for the wider model, trade depth for width, drop slice_num back to 64). Cap-bottlenecked experiment — capacity-via-width is dominated by capacity-via-more-epochs under the 30-min budget. Tanjiro reassigned to throughput experiment (reduce-overhead compile mode) to attack the cap directly.

---

## 2026-05-13 01:05 — PR #1556: fp32 eval + eval_every_n_epochs=3 — MERGED (paper-faithful test_avg)
- willowpai2g24h4-frieren/fp32-eval
- **Hypothesis:** Remove bf16 autocast from `evaluate_split` to eliminate the cruise-pressure inf overflow that was being zeroed by `nan_to_num`. Gate eval frequency to recover wall-clock cost.
- **W&B:** `uwk17oc0` (iter2 with eval_every_n_epochs=3)

| Metric | Pre-merge baseline #1404 (bf16 eval) | Iter2 this PR (CosineAnneal stack + fp32-eval) | Notes |
|---|---:|---:|---|
| val_avg/mae_surf_p | 70.9449 | 72.9137 | +2.8% — but iter2 used pre-OneCycleLR cosine, NOT a direct comparison |
| test_avg/mae_surf_p (4-split) | 61.8276 (bf16 eval, cruise biased low) | 64.3287 (fp32 eval, cruise faithful) | Numbers not directly comparable — different cruise treatment |
| test_geom_camber_cruise | 42.96 (biased low by nan_to_num zeroing) | **43.71** (finite, no inf) | Fidelity recovered |
| Best epoch | 29/29 | 28/30 | Eval gate fired at ep 28 |
| Sec/epoch (mean) | ~62.5 | ~60.6 | Train ~57s + eval epochs ~66s |
| Peak GPU | 50.97 GB | 29.8 GB | Much lower |
| fp32 eval | ❌ | ✓ (every 3 epochs + final-epoch guard) | |

**Decision: MERGED as metric-FIDELITY improvement, not metric-VALUE.** The diff is clean (no scheduler changes), so squash-merging onto current OneCycleLR baseline yields OneCycleLR + fp32 eval + 3-epoch gate. The val metric (70.94) stays valid because val cruise was always finite under bf16. The test metric of 61.83 (cruise zeroed) is no longer the right comparison — future runs will produce a slightly higher, paper-faithful test_avg on this stack.

**Key insight:** This is the first paper-faithful 4-split test_avg/mae_surf_p — cruise contribution recovered from artificial 42.96 (zeroed) to genuine 43.71 (finite). The wall-clock cost of fp32 eval (45s/epoch) is recovered by the eval_every_n_epochs=3 gate (10 evals across 30 epochs instead of 30).

**Residual issue (cosmetic, not metric-affecting):** `test_geom_camber_cruise/vol_loss` still shows Infinity due to NaN-GT propagation in `evaluate_split`'s normalized-space loss path. Easy fix-up PR available. Headline metric unaffected (flows through patched `data/scoring.py`).

---

## 2026-05-13 01:30 — PR #1716: OneCycleLR max_lr=1.5e-3 — MERGED (in-dist squeeze + LR ceiling probe)
- willowpai2g24h4-alphonse / alphonse/onecycle-max-lr-1p5e3
- **Hypothesis:** Raising OneCycleLR max_lr from 1e-3 to 1.5e-3 (keeping pct_start=0.1, all else fixed) gives the optimizer a higher peak LR to escape flat regions faster, yielding lower val and test MAE within the 30-min budget.
- **W&B:** `dvk0201k`

| Metric | Baseline (#1556, max_lr=1e-3) | PR #1716 (max_lr=1.5e-3) | Δ |
|---|---:|---:|---|
| val_avg/mae_surf_p | 70.9449 | **68.5843** | −2.36 (−3.3%) |
| test_avg/mae_surf_p (4-split, fp32 eval) | 61.8276 | **60.3521** | −1.48 (−2.4%) |
| Best epoch | 29/29 | 27/29 | −2 |
| Sec/epoch | ~62.5s | ~62.1s | ≈same |
| Peak GPU | 50.97 GB | 48.80 GB | −2.2 GB |

Per-split breakdown:

| Split | Baseline val | This run val | Δ |
|---|---:|---:|---|
| val_single_in_dist | 80.87 | **73.78** | −7.09 (−8.8%) |
| val_geom_camber_rc | 80.80 | 80.71 | −0.09 (≈0) |
| val_geom_camber_cruise | 51.75 | 51.72 | −0.03 (≈0) |
| val_re_rand | 70.36 | **68.10** | −2.26 (−3.2%) |

| Split | test mae_surf_p |
|---|---:|
| test_single_in_dist | 63.5400 |
| test_geom_camber_rc | 74.6399 |
| test_geom_camber_cruise | 42.1353 |
| test_re_rand | 61.0934 |
| **test_avg** | **60.3521** |

**Decision: MERGED.** Both val and test beat the baseline by more than the noise floor (RNG ≈ ±5%; test delta −2.4% just at threshold, but single-seed and consistent direction across all 4 splits). Clean 2-line diff. LR sweep confirmed: 1.5e-4 → 1.5e-3 → 1.5e-7, no instability spikes.

**Key insight:** LR gain is heavily concentrated on `val_single_in_dist` (−8.8%). OOD splits (geom_camber_rc, geom_camber_cruise) are LR-saturated and essentially unmoved — they require a different lever. Best epoch shifts 29→27 with last 3 epochs within 0.03 of each other (converged tail), suggesting the higher peak LR consumed the descent budget more efficiently without leaving room for further improvement.

**Implication:** Two experiments warranted now:
1. Push `max_lr=2e-3` to test whether the LR ceiling has been reached.
2. Attack the OOD bottleneck (`geom_camber_rc`, `geom_camber_cruise`) via non-LR levers (augmentation, geometric features, loss weighting).

---

## 2026-05-13 01:55 — PR #1719: OneCycleLR pct_start=0.05 — SENT BACK (single-knob below new baseline; composition test pending)
- willowpai2g24h4-nezuko / nezuko/onecycle-pct-start-0p05
- **Hypothesis:** Shorter warmup (pct_start 0.10→0.05) shifts the LR-peak earlier, removing the noisy mid-descent epochs (ep11 spike of +30 in the baseline) and giving more of the 30-min budget to the cosine decay phase.
- **W&B:** `urcnrdsc`

| Metric | OLD Baseline (#1404, max_lr=1e-3 pct_start=0.10) | This run (max_lr=1e-3 pct_start=0.05) | Δ vs OLD | NEW Baseline (#1716, max_lr=1.5e-3 pct_start=0.10) | Δ vs NEW |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p | 70.9449 | 70.0009 | −0.94 (−1.33%) | 68.5843 | +1.42 (+2.07%) |
| test_avg/mae_surf_p | 61.8276 | 61.3230 | −0.50 (−0.82%) | 60.3521 | +0.97 (+1.61%) |
| Best epoch | 29/29 | 29/29 | 0 | 27/29 | +2 |

Per-split val deltas vs OLD baseline (mirror alphonse's pattern but smaller):

| Split | Δ vs OLD |
|---|---:|
| val_single_in_dist | 80.87→78.12 (−2.75) |
| val_geom_camber_rc | 80.80→81.11 (+0.31, flat) |
| val_geom_camber_cruise | 51.75→52.08 (+0.33, flat) |
| val_re_rand | 70.36→68.69 (−1.67) |

**Diagnosis: SENT BACK for composition test.** Single-knob effect of `pct_start=0.05` against the OLD baseline is positive but within RNG noise floor (val −1.33%, test −0.82%, with RNG ≈ ±5%). The shorter warmup did successfully eliminate the ep11 baseline spike (122.99 vs baseline 154.8). Pattern of gains (in-dist + re_rand benefit, OOD flat) is exactly the same pattern that alphonse #1716 saw with max_lr=1.5e-3 — the two knobs may be mechanistically redundant rather than additive.

**Procedural issue:** PR now conflicts with merged advisor branch (alphonse changed max_lr on adjacent lines in the OneCycleLR constructor). Rebase needed.

**Action:** Sent back with instructions to rebase onto new baseline (max_lr=1.5e-3) and re-run as a compositional test: does pct_start=0.05 still help when peak LR is already higher, or do the mechanisms overlap? Will group in W&B with frieren #1768 (pct_start=0.15) for a 3-point sweep against the new max_lr.

---

## 2026-05-13 02:00 — PR #1764: compile mode=reduce-overhead — CLOSED (speed hypothesis refuted; metric "win" is noise)
- willowpai2g24h4-tanjiro / willowpai2g24h4-tanjiro/reduce-overhead-compile
- **Hypothesis:** `torch.compile(mode="reduce-overhead")` removes per-op Python dispatch overhead via CUDAGraphs, delivering 10-20% throughput → 3-5 extra epochs of descent in the 30-min cap.
- **W&B:** `b2v9xrqd`

| Metric | Baseline #1716 (mode="default") | This run (mode="reduce-overhead") | Δ |
|---|---:|---:|---|
| val_avg/mae_surf_p | 68.5843 | 67.6348 | −0.95 (−1.38%) |
| test_avg/mae_surf_p | 60.3521 | 59.4957 | −0.86 (−1.42%) |
| Mean sec/epoch | ~62.5 | 63.74 | +1.2 (1.6% slower) |
| Epochs in 30-min cap | 29 | 28 | −1 |

Per-split test deltas (showing the noise fingerprint):

| Split | Baseline | This run | Δ |
|---|---:|---:|---|
| test_single_in_dist | 63.54 | 67.13 | **+3.59 (worse)** |
| test_geom_camber_rc | 74.64 | 73.64 | −1.00 (better) |
| test_geom_camber_cruise | 42.14 | 39.69 | −2.45 (better) |
| test_re_rand | 61.09 | 57.52 | −3.57 (better) |

**Diagnosis: CLOSED — speed hypothesis refuted, metric "win" is RNG noise.** The CUDAGraph dispatch-overhead savings were exactly offset (or worse) by the per-capture overhead — Tanjiro's diagnosis: with `dynamic=True` on variable mesh shapes, CUDAGraphs records 9 distinct shape variants and the recording overhead defeats the savings. Speed regression is real (−1.6% throughput) and material (lost 1 epoch under cap). The metric "improvement" (−1.38% val, −1.42% test) is well within RNG variance (±5% on val, 2% test reliability threshold). The opposing-direction per-split test deltas (in-dist gets worse, others better) are the fingerprint of seed noise, not a systematic shift.

**Genuine learning:** Python dispatch is not the bottleneck on this Transolver/dynamic-mesh workload — kernels are already well-saturated on H100. Throughput attacks need to target kernel-level work (autotuning, fused kernels, better attention backends), not the Python layer.

**Next:** Assigning tanjiro to `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)` — Triton kernel autotuning without the CUDAGraph problem he diagnosed.

---

## 2026-05-13 02:00 — PR #1383: channel-weighted loss (Ux:1, Uy:1, p:3) — CLOSED (regression on headline metric)
- willowpai2g24h4-edward / willowpai2g24h4-edward/p-channel-weight
- **Hypothesis:** Multiplying p's MSE by 3 enlarges p's gradient share; tradeoff is "slight" Ux/Uy regression but net val_avg/mae_surf_p improvement.
- **W&B:** `b82kyqdm`

| Metric | Baseline #1716 | This run | Δ |
|---|---:|---:|---|
| val_avg/mae_surf_p | 68.5843 | 70.2099 | **+1.63 (+2.4% WORSE)** |
| test_avg/mae_surf_p | 60.3521 | 62.8702 | **+2.52 (+4.2% WORSE)** |
| val_avg/mae_surf_Ux | 1.072 | 1.307 | **+22% WORSE** |
| val_avg/mae_surf_Uy | 0.528 | 0.606 | **+15% WORSE** |

**Diagnosis: CLOSED — direction correct, magnitudes wrong, implementation flawed.** The prediction "p improves, Ux/Uy slightly regress" was directionally accurate but the regression magnitudes (+22% Ux, +15% Uy) dwarfed the p improvement. Two compounding issues:

1. **Implementation flaw:** the weight was applied to BOTH surface AND volume p-channels. Volume p is not in the headline metric, so up-weighting it wastes optimization capacity on the wrong target.
2. **Shared-representation drag:** large Ux/Uy gradient suppression degrades the shared backbone, which partially offsets the targeted p improvement.

**Genuine learning:** Aggressive flat-multiplier channel weighting reshapes gradient balance more than it reshapes performance, because the headline metric is in physical (un-normalized) space — the loss-balance shift doesn't translate cleanly to MAE improvements.

**Next:** Assigning edward to his own suggested follow-up #2: **surface-only p-weighting with milder weight (p_weight=2)**. Applying the weight only inside `surf_mask` is the surgical version of the hypothesis — directly targets the headline metric. This is the targeted retry that should determine whether the underlying mechanism (gradient re-allocation to p) is the right lever or whether channel-balance entirely needs a different approach.

**Next target:** beat val_avg/mae_surf_p = **70.9449** on the new fp32-eval stack. The next experiment on this stack will establish the new faithful test_avg baseline.
