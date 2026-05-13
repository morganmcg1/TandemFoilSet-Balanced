# SENPAI Research Results

## 2026-05-13 05:30 — PR #1559: Decoupled chan_w (surf=[1,1,5], vol=[1,1,1]) — CLOSED

- Branch: `charliepai2g24h2-alphonse/decoupled-chanw-surf-vol`
- Hypothesis: chan_w=[1,1,5] should apply to surface loss only; volume loss should stay [1,1,1] since volume mae is not in the primary metric
- Artifacts: 9-seed sweep on student branch (committed JSONL)

| Statistic | Decoupled chan_w | Pre-floor baseline (#1464 base) | Δ% |
|---|---:|---:|---:|
| **val_avg mean (9 seeds)** | **147.0** | 133.94 | **+9.8%** |
| val_avg best-of-9          | 134.14 | 133.94 | +0.15% |
| val_avg std                | 8.32   | — | — |

**Decision: CLOSED — mean +9.8% regression across 9 seeds; best-of-9 only ties old pre-floor baseline.**

**Analysis:** Branch was based on a deep pre-floor base (missing warmup, gradclip, Huber). Even discounting the base mismatch, the 9-seed sweep is informative: the **volume term acts as a useful regularizer** even though it's not in the primary metric. Removing the channel upweight from the volume loss (`vol=[1,1,1]` instead of `[1,1,5]`) breaks shared-feature regularization between surface and volume heads. The decoupling intuition (target-only emphasis) is wrong — joint channel weighting is a feature, not a bug. Bug-fix work in the PR duplicates askeladd #1536 and fern #1477 NaN-guard work, so nothing salvageable to cherry-pick.

**Follow-up:** Assigned alphonse #1947 — chan_w sweep under the new Huber β=0.3 regime ([1,1,3] vs [1,1,7]) to see if the optimal upweight magnitude shifted with the loss change.

---

## 2026-05-13 05:10 — PR #1849: Huber β sweep β=0.5 and β=0.3 — **NEW FLOOR**

- Branch: `charliepai2g24h2-edward/huber-beta-sweep`
- Hypothesis: Lower β makes the Huber loss more L1-like, closer to MAE eval metric; sweep β=0.5 and β=0.3 vs merged β=1.0
- Artifacts: `models/model-charliepai2g24h2-edward-huber-beta0p5-20260513-031119/metrics.yaml`, `models/model-charliepai2g24h2-edward-huber-beta0p3-20260513-035314/metrics.yaml`, `eval_bs1.jsonl` for each

| Metric | β=1.0 (#1801) | β=0.5 (Arm A) | β=0.3 (Arm B) | Best Δ% |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 111.15 | 108.47 | **105.68** | **−4.92%** |
| val_single_in_dist | 134.21 | 138.62 | **126.21** | −5.96% |
| val_geom_camber_rc | 133.88 | **125.25** | 116.36 | −13.1% (β=0.3) |
| val_geom_camber_cruise | 77.59 | **75.41** | 82.23 | −2.8% (β=0.5) |
| val_re_rand | 98.93 | **94.59** | 97.92 | −4.4% (β=0.5) |
| **test_avg bs=1** | 99.06 | 97.49 | **94.98** | **−4.11%** |

**Config:** lr=7.5e-4, wd=1e-4, bs=4, chan_w=[1,1,5], surf_weight=10, warmup+cosine T_max=47, gradclip(1.0), Huber β=0.3, fp32. 12 epochs (timeout-cut). ~42 GB VRAM.

**Decision: MERGED — β=0.3 arm wins. New floor val_avg=105.6808.**

**Analysis:** β sweep confirms monotone gain for most splits (β=0.3 < β=0.5 < β=1.0). The exception is val_geom_camber_cruise (low-residual split): β=0.5 was best (75.41 < 77.59 β=1.0 < 82.23 β=0.3). This split-level β interaction — lower β hurts low-residual regions — motivates the next hypothesis: **per-channel β** with higher β for pressure (smaller residuals, especially on cruise) and lower β for velocity channels. Edward assigned #1927 to test β=0.1 and per-channel β (Ux=0.1, Uy=0.1, p=0.5).

---

## 2026-05-13 04:15 — PR #1524 (r3): Gradient accumulation accum=4 + floor stack — CLOSED

- Branch: `charliepai2g24h2-tanjiro/grad-accum-eff-bs16`
- Hypothesis: grad-accum=4 (eff_bs=16) cleans gradient signal, compounds with Huber+chan_w+warmup stack
- Artifacts: `models/model-charliepai2g24h2-tanjiro-grad-accum-stack-floor-r3-20260513-032140/metrics.jsonl`

| Split | mae_surf_p (accum=4) | Huber floor #1801 | Δ% |
|---|---:|---:|---:|
| val_single_in_dist     | 145.64 | 134.21 | +8.5% |
| val_geom_camber_rc     | 130.21 | 133.88 | −2.7% |
| val_geom_camber_cruise |  90.39 |  77.59 | +16.5% |
| val_re_rand            | 106.14 |  98.93 | +7.3% |
| **val_avg**            | **118.09** | **111.15** | **+6.2%** |
| test_avg (3-split, excl cruise) | 118.04 | 109.56 | +7.7% |

**Config:** accum_steps=4, lr=7.5e-4, chan_w=[1,1,5], 3-ep warmup + cosine T_max=47, gradclip(max_norm=1.0), Huber β=1.0. 14 epochs (timeout-cut at 30 min). Peak VRAM 42.11 GB.

**Decision: CLOSED — +6.2% regression vs Huber floor. Clean dead-end on the current stack.**

**Analysis:** Student's root-cause diagnosis is excellent and correct: at the 30-min timeout, accum_steps=4 reduces optimizer steps per epoch from ~375 to ~94 — the model sees 4× fewer total gradient updates (1313 vs 5250). Under a timeout-cut training regime, optimizer step throughput dominates over gradient cleanliness. On the *old* floor (122.70), accum compounded cleanly (+3.8% improvement). On the Huber floor, it doesn't — because the Huber stack is already producing high-quality per-step learning, and the throughput cost kills gains. **Do not revisit** grad-accum in the timeout-cut regime. If the training budget extends to >60 min, this hypothesis may be worth retesting.

---

## 2026-05-13 03:15 — PR #1477 (r2, AMP bf16 + L2 base): Outstanding results, sent back for Huber rebase

- Branch: `charliepai2g24h2-fern/amp-bf16-gradclip`
- Hypothesis: AMP bf16 unlocks +58% epochs in same wall clock; floor-stack rebase
- Artifacts: `models/model-charliepai2g24h2-fern-amp-bf16-on-floor-r2-seed-a-20260513-020635/metrics.jsonl`, `...-seed-b-20260513-024157/metrics.jsonl`

| Metric | seed-a | seed-b | Mean | Old floor #1573 | Δ vs #1573 |
|---|---:|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | **104.71** | **101.22** | **102.97** | 122.70 | **−16.1%** |
| test_avg (bs=4 clean!) | 93.52 | 93.36 | 93.44 | NaN | clean ✓ |
| best_epoch (of 50) | 19 | 18 | 18.5 | 12 | +58% |
| mean epoch time (s) | 98.6 | 98.6 | — | ~156 | −37% |
| peak GPU memory (GB) | 32.95 | 32.95 | — | 42.12 | −22% |

**Spread between seeds:** 3.49 val points (3.3%). Both seeds far under 115 threshold.

**Decision: SENT BACK — branch rebased on pre-Huber HEAD `3b30bfc`. Needs one final rebase onto post-Huber `d0b582f` + ONE confirm seed.**

**Analysis:** AMP bf16 is the largest single structural improvement since chan_w. Three wins compound: (1) 37% faster epochs → 58% more epochs in budget, (2) 22% less VRAM, (3) fern's non-finite-y prefilter in evaluate_split kills the bs=4 test NaN (test_geom_camber_cruise clean at 65/67 each seed). BUT: Huber PR #1801 merged DURING her rebase — branch HEAD `ad91591` was rebased onto `3b30bfc` (pre-Huber), so it now conflicts with `d0b582f`. Both seed-b at 101.22 and Huber floor at 111.15 are measured on incompatible configs. Expected result with Huber+AMP stacked: ~91-95 val_avg. Win condition: val_avg < 111.15.

---

## 2026-05-13 01:00 — PR #1573: Warmup + lr=7.5e-4 + gradient clipping — **NEW FLOOR**

- Branch: `charliepai2g24h2-frieren/warmup-lr75e-4-gradclip`
- Hypothesis: Lower peak lr (1e-3 → 7.5e-4) backs off numerical instability; gradient clip (max_norm=1.0) stabilises train steps
- Artifacts: `models/model-charliepai2g24h2-frieren-warmup-lr75e-4-gradclip-20260512-235148/metrics.jsonl`

| Split | val mae_surf_p (lr=7.5e-4+clip) | Floor #1482 (lr=1e-3) | Δ% |
|---|---:|---:|---:|
| val_single_in_dist     | 159.59 | 162.05 | −1.5% |
| val_geom_camber_rc     | 134.74 | 137.15 | −1.8% |
| val_geom_camber_cruise |  89.18 | 101.34 | −12.0% |
| val_re_rand            | 107.31 | 111.83 | −4.0% |
| **val_avg**            | **122.70** | **128.09** | **−4.2%** |
| test_avg (bs=1)        | **110.25** | 117.40 | **−6.1%** |
| test_geom_camber_cruise (bs=1) | **75.55** | NaN | finite! |

**Config:** 3-ep warmup + lr=7.5e-4 + cosine(T_max=47) + gradclip(max_norm=1.0), chan_w=[1,1,5], bs=4, wd=1e-4. 12 epochs (timeout 31.2 min). Peak VRAM 42.12 GB.

**Decision: MERGED — new floor val_avg/mae_surf_p=122.7043 (beats 128.09 by 4.2%).**

**Analysis:** Third confirmed win. The largest gain is on val_geom_camber_cruise (−12%), which is the split most affected by the lr=1e-3 attention NaN. Reducing peak LR to 7.5e-4 backs off the numerical boundary; gradient clipping prevents sporadic large updates that contributed to the instability. Notably, train-side clip did NOT fix the bs=4 test NaN (frieren diagnosed this correctly: it's an inference-time attention computation issue with specific batch compositions, not a training-side issue). bs=1 test_avg=110.25 is the cleanest end-to-end metric we have. The `eval_bs1.py` helper script is now in the advisor branch — use it for clean test evaluation going forward.

---

## 2026-05-13 00:05 — PR #1603: EMA of weights (decay=0.999) — all variants regress

- Branch: `charliepai2g24h2-edward/ema-weights-decay-0p999`
- Hypothesis: EMA of model weights smooths noisy converged endpoint → cleaner eval
- Artifacts: `models/model-charliepai2g24h2-edward-ema-weights-decay-0p999-20260512-211432/metrics.jsonl`, `-init-after-warmup-20260512-220502/metrics.jsonl`, `-0p99-init-after-warmup-20260512-225559/metrics.jsonl`

| Variant | EMA decay | Init | val_avg/mae_surf_p | Δ vs 128.09 |
|---|---|---|---:|---:|
| Literal PR | 0.999 | random deepcopy | 138.10 | +7.83% |
| Init-after-warmup | 0.999 | snapshot ep3 | 147.78 | +15.37% |
| **Best: decay=0.99 + init fix** | **0.99** | **snapshot ep3** | **128.76** | **+0.52%** |
| Floor | — | — | 128.09 | — |

All variants: test_avg = NaN (EMA produces extreme p-predictions on test_geom_camber_cruise → vol_loss=+Inf).

**Decision: CLOSED — clean negative result, regime mismatch.**

**Analysis:** EMA is a converged-endpoint smoother. Under SENPAI_TIMEOUT_MINUTES=30, the model is in rapid descent at epoch 14 (10-50 MAE/epoch drop). EMA averages newer-better with older-worse weights → strictly worse than the latest snapshot. The mechanism fits a *longer-run* regime. Best-variant tie (+0.52%) confirms: at decay=0.99, the EMA is essentially the most recent weight anyway (0.99^14 = 0.869 → old weights contribute only 13%), which is why it converges to baseline. **Revisit** once fern's AMP lands and we reach ~28 epochs in 30 min. Edward's root-cause analysis was excellent and directly motivated the next hypothesis (Lookahead optimizer, PR #1708).

---

## 2026-05-12 23:10 — PR #1485 (round 2): slice_num 64 → 128, rebased on chan_w+warmup+lr=1e-3

- Branch: `charliepai2g24h2-nezuko/slice-num-128-stacked`
- Hypothesis: Slice_num=128 finer physics-token resolution compounds with chan_w+warmup stack
- Artifacts: `models/model-charliepai2g24h2-nezuko-slice-num-128-stacked-20260512-215215/metrics.jsonl`

| Split | mae_surf_p (slice_num=128) | Floor #1482 (slice_num=64) | Δ% |
|---|---:|---:|---:|
| val_single_in_dist     | 197.96 | 162.05 | +22.2% |
| val_geom_camber_rc     | 177.32 | 137.15 | +29.3% |
| val_geom_camber_cruise | 123.28 | 101.34 | +21.6% |
| val_re_rand            | 144.12 | 111.83 | +28.9% |
| **val_avg**            | **160.67** | **128.09** | **+25.4%** |
| test_avg               | NaN    | NaN    | — |

**Config:** slice_num=128, chan_w=[1,1,5], lr=1e-3, 3-ep warmup + cosine(T_max=47), bs=4, wd=1e-4. 11 epochs in 30 min (timeout). Peak VRAM 54.51 GB (vs 42.1 GB at slice_num=64, +30%).

**Decision: CLOSED — +25.4% regression on stacked floor. Worsened across all splits. Student recommended close.**

**Analysis:** Doubled slice_num doubles attention cost in `Slice_Attn` (O(N × slice_num) per head × layers): VRAM +30% and epochs drop from ~14 to ~11. Warmup consumes 3 of 11 epochs at reduced LR — a larger fraction than at slice_num=64. Model still descending at epoch 10 but timeout kills the run before it can compound. Root cause is wall-clock budget, not capacity inadequacy. **Revisit only if** fern's AMP (#1477) doubles throughput — slice_num=128 at bf16 would fit in ~27 GB and allow ~20 epochs in 30 min.

---

## 2026-05-12 23:10 — PR #1536: NaN guard + clean floor rerun (askeladd, sent back)

- Branch: `charliepai2g24h2-askeladd/scoring-nan-guard`
- Hypothesis: One-line train.py guard zeros out non-finite GT samples before sq_err, fixing `0×NaN` propagation
- Artifacts: `models/model-charliepai2g24h2-askeladd-scoring-nan-guard-20260512-221647/metrics.jsonl` (and -200506, -205715)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | test_geom_camber_cruise |
|---|---:|---:|---:|
| -200506 | 138.52 | 126.66 | 93.66 |
| -205715 | 141.41 | 130.16 | 86.55 |
| -221647 | 144.41 | **133.04** | 94.41 |
| **mean ± σ** | **141.4 ± 3.0** | **130.0 ± 3.2** | **91.5 ± 4.3** |
| **Floor #1482** | **128.09** | NaN | NaN |

**Config:** chan_w=[1,1,5] (from branch default), lr=5e-4 (NOT 1e-3), cosine-only (NO warmup), bs=4. 14 epochs (timeout).

**Decision: SENT BACK — two blockers.**

1. **Code not pushed**: Student posted results but didn't commit NaN guard code — no `isfinite` in pushed train.py.
2. **Branch pre-warmup reversion**: Branch base is pre-#1482. Merging would REVERT the warmup scheduler (val=141.4 mean confirms this — matches cosine-only at lr=5e-4 config, not lr=1e-3+warmup floor).

**Key finding:** test_avg=133.04 and test_geom_camber_cruise=94.41 are the **first finite test metrics** on this branch. NaN guard logic is correct. Askeladd needs to rebase on current advisor branch, push the NaN guard code, and re-run with `--lr 1e-3` to confirm val_avg ~128 (within noise of floor) with clean test_avg.

---

## 2026-05-12 21:20 — PR #1526: Intermediate model scaling n_hidden=224, n_layers=7, n_head=8 (~2.67M)

- Branch: `charliepai2g24h2-edward/model-224-7-8`
- Hypothesis: Transolver at ~3.4M params should lower floor by 5–15%; capacity bottleneck at 0.66M
- Artifacts: `models/model-charliepai2g24h2-edward-model-224-7-8-20260512-201457/metrics.jsonl`

| Split | mae_surf_p (224-7-8, bs=2, 5 ep) | Baseline #1482 (128-5-4, bs=4, 14 ep) | Δ% |
|---|---:|---:|---:|
| val_single_in_dist     | 237.30 | 162.05 | +46.4% |
| val_geom_camber_rc     | 182.12 | 137.15 | +32.8% |
| val_geom_camber_cruise | 142.15 | 101.34 | +40.3% |
| val_re_rand            | 151.40 | 111.83 | +35.4% |
| **val_avg**            | **178.24** | **128.09** | **+39.2%** |

**Config:** n_hidden=224, n_layers=7, n_head=8, slice_num=64, mlp_ratio=2 (~2.67M actual params). bs=4 OOMed at 94.67 GB → fell back to bs=2 (51.5 GB peak). Only 6 epochs in 30 min (~325 s/epoch). lr=5e-4 (default).

**Decision: CLOSED — severely undertrained (6 ep at bs=2 vs 14 ep at bs=4 for baseline). Architecturally inconclusive.**

**Analysis:** bs=4 OOM was due to activation memory underestimate: slice_token [B, slice_num, 224], slice_weights [B, N_max, head, slice_num] across 7 layers at N_max≈242k far exceeds 35 GB at fp32. bs=2 fallback ran but cut epochs from 14 to 6, and curve was still descending at epoch 5 (187→178). The +39% regression is undertraining, not architectural failure. **Path forward:** wait for fern's AMP bf16 result (#1477 WIP); if bf16 wins, activation memory roughly halves → 224-7-8 should fit at bs=4 for a fair test.

---

## 2026-05-12 21:20 — PR #1485: Increase slice_num 64 → 128 for finer physics tokens (round 1)

- Branch: `charliepai2g24h2-nezuko/slice-num-128`
- Hypothesis: Finer physics-token resolution improves pressure prediction; slice_num=128 is the most direct lever
- Artifacts: `models/model-charliepai2g24h2-nezuko-slice-num-128-20260512-195113/metrics.jsonl` (best run); `-180850/`, `-185210/` (others)

| Run | Best epoch | val_avg/mae_surf_p | Epochs | Peak VRAM |
|---|---:|---:|---:|---:|
| 20260512-195113 | 11 | **135.76** | 11/50 | 54.51 GB |
| 20260512-180850 | 9  | 140.90     | 11/50 | 54.51 GB |
| 20260512-185210 | 11 | 148.55     | 11/50 | 54.51 GB |
| **Mean** | — | **141.74** | — | — |

Pre-merge floor reference: 143.15 (PR #1486, no chan_w, no warmup). Best run 135.76 = −5.2%.

**Decision: SENT BACK for rebase + stack. Runs were on pre-merge base (no chan_w, no warmup). vs current floor 128.09 the best run is +6% WORSE. Needs rebase.**

**Analysis:** Three independent runs show genuine −5.2% signal on the old base and seed variance ±4.6%. Per-epoch wall time ~172 s vs ~150 s for slice_num=64 (+15% slower) limits run to 11 epochs. The training T_max=50 cosine barely decays in 11 epochs — another confound. Also confirmed the data/scoring.py 0×NaN bug independently (askeladd #1536 is the fix). **Rerun instructions:** rebase, use `--lr 1e-3`, compare to 128.09.

---

## 2026-05-12 20:20 — PR #1482: 3-epoch warmup + cosine + peak lr=1e-3

- Branch: `charliepai2g24h2-frieren/warmup-cosine-lr1e-3`
- Hypothesis: Linear warmup + higher peak lr reduces destabilizing updates in early epochs; canonical for transformer training
- Artifacts: `models/model-charliepai2g24h2-frieren-warmup-cosine-lr1e-3-20260512-180356/metrics.jsonl`

| Split | val mae_surf_p | test mae_surf_p (bs=4) | test mae_surf_p (bs=1) |
|---|---:|---:|---:|
| val_single_in_dist     | 162.05 | 146.76 | 151.01 |
| val_geom_camber_rc     | 137.15 | 122.86 | 124.68 |
| val_geom_camber_cruise | 101.34 | NaN    | 82.75  |
| val_re_rand            | 111.83 | 111.94 | 111.16 |
| **val_avg / test_avg** | **128.09** | **NaN** | **117.40** |

**Config:** 3-ep warmup + lr=1e-3 + cosine(T_max=47, eta_min=1e-6), bs=4, wd=1e-4, surf_weight=10, all other defaults. **No chan_w in branch base** (PR assigned before #1464 merged — stacked floor unknown). 14 epochs (timeout-cut). Peak VRAM 42.1 GB.

**Decision: MERGED — new floor at val_avg/mae_surf_p=128.09 (beats 133.94 by 4.4%).**

**Analysis:** Second confirmed win. val_re_rand showed the largest improvement (129.86→111.83, −14%), indicating warmup + higher LR particularly helps cross-regime generalization. The test NaN on test_geom_camber_cruise is a NEW model-level issue (not the data bug): specific bs=4 batch compositions at lr=1e-3 boundary cause non-finite attention weights. bs=1 eval is clean (117.40). Student proposed fixes: lr=7.5e-4 + gradient clipping — assigned as #1573 follow-up.

**Note:** This floor was measured WITHOUT chan_w in the branch base. The merged advisor branch now has both chan_w + warmup — the true stacked floor is expected to be lower than 128.09 but unmeasured.

---

## 2026-05-12 20:00 — PR #1531: Channel weight p=10 (sweep beyond p=5)

- Branch: `charliepai2g24h2-alphonse/channel-weight-p10`
- Hypothesis: Doubling pressure weight from p=5 to p=10 maps response curve
- Artifacts: `models/model-charliepai2g24h2-alphonse-channel-weight-p10-20260512-191417/metrics.jsonl`

| Split | mae_surf_p (p=10) | mae_surf_p (p=5 floor) | Δ% |
|---|---:|---:|---:|
| val_single_in_dist     | 185.22 | 155.84 | +18.9% |
| val_geom_camber_rc     | 173.27 | 146.50 | +18.3% |
| val_geom_camber_cruise | 118.90 | 103.54 | +14.8% |
| val_re_rand            | 133.44 | 129.86 | +2.8% |
| **val_avg**            | **152.71** | **133.94** | **+14.0%** |
| mae_surf_Ux            | 3.48 | 2.73 | +27.5% |
| mae_surf_Uy            | 1.34 | 1.14 | +17.5% |

**Decision: CLOSED — p=10 regresses on every split AND degrades Ux past the 20% threshold flagged in the PR body.**

**Analysis:** The chan_w response curve is non-monotonic. The 1→5 jump was a 6.4% win; 5→10 is a 14% loss. The optimum is p≈5 (or slightly lower). The training-loop SGD is being yanked toward pressure at the cost of Ux — the +27.5% Ux degradation is a real signal. Next: alphonse → decoupled surf vs vol chan_w (#1559) — apply [1,1,5] only to surface portion since the metric is mae_surf_p.

---

## 2026-05-12 20:00 — PR #1524 (round 1): Gradient accumulation eff_bs=16 (no chan_w base)

- Branch: `charliepai2g24h2-tanjiro/grad-accum-eff-bs16`
- Hypothesis: grad-accum=4 with lr=1e-3 (sqrt-LR scaling) → cleaner gradients at base VRAM cost
- Artifacts: `models/model-charliepai2g24h2-tanjiro-grad-accum-eff-bs16-20260512-190822/metrics.jsonl`

| Split | mae_surf_p | Pre-chan_w floor (#1486) | Δ% |
|---|---:|---:|---:|
| val_single_in_dist     | 175.60 | 179.61 | -2.2% |
| val_geom_camber_rc     | 145.90 | 151.83 | -3.9% |
| val_geom_camber_cruise | 112.51 | 114.07 | -1.4% |
| val_re_rand            | 124.83 | 127.06 | -1.8% |
| **val_avg**            | **139.71** | **143.15** | **-2.4%** |
| Peak VRAM              | 42.1 GB | 84.2 GB | half |

**Decision: SENT BACK for rebase + stack with chan_w + T_max=14 cosine tuning.**

**Analysis:** Methodology confirmed — grad-accum=4 with sqrt-LR scaling beats pre-chan_w floor by 2.4% at half the VRAM of bs=8. But branch base lacks chan_w (PR assigned before #1464 merged). To make this a real win vs the current floor 133.94, must rebase and stack with chan_w. Also: cosine T_max=50 barely decays in 14 epochs — student's suggested follow-up #4 to set T_max=14 for proper LR decay over the budget.

---

## 2026-05-12 20:00 — PR #1489 (revised): Per-sample AoA flip p=0.25 (no chan_w base)

- Branch: `charliepai2g24h2-thorfinn/aoa-flip-aug`
- Hypothesis: switch from per-batch p=0.5 to per-sample p=0.25 to fix Uy regression
- Artifacts: `models/model-charliepai2g24h2-thorfinn-aoa-flip-persample-p025-20260512-191315/metrics.jsonl`

| Metric | per-batch p=0.5 | per-sample p=0.25 | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p     | 146.42 | 146.14 | -0.2% (flat) |
| val_avg/mae_surf_Uy    | 2.11   | 1.06   | **-50%** (Uy fix worked) |
| val_geom_camber_rc/mae_surf_p | 150.35 | 164.66 | +9.5% (regression here) |

**Decision: SENT BACK for rebase + stack with chan_w.**

**Analysis:** Per-sample Uy fix unambiguously worked (mae_surf_Uy cut in half). Primary metric flat overall — augmentation helps 3 splits, hurts val_geom_camber_rc. On pre-chan_w base, this is +2.1% regression vs 143.15. Levers (augmentation reshapes input distribution; chan_w reshapes gradient) are orthogonal — student's follow-up #1 suggests stacking. Sent back to rebase and rerun on top of chan_w.

---

## 2026-05-12 19:30 — PR #1468: surf_weight 10 → 30 (surface loss emphasis)

- Branch: `charliepai2g24h2-askeladd/surf-weight-30`
- Hypothesis: Increasing surf_weight from 10 to 30 corrects the imbalance where vol_loss dominates despite surface nodes being a small fraction of total nodes
- Artifacts: `models/model-charliepai2g24h2-askeladd-surf-weight-30-20260512-180309/metrics.jsonl`

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---:|---:|---:|---:|
| val_single_in_dist     | 181.99 | 2.01 | 0.99 | 190.74 |
| val_geom_camber_rc     | 173.61 | 3.18 | 1.26 | 184.65 |
| val_geom_camber_cruise | 133.91 | 1.76 | 0.76 | 159.82 |
| val_re_rand            | 142.96 | 2.38 | 0.99 | 158.10 |
| **val_avg**            | **158.12** |     |     |       |
| test_single_in_dist    | 168.46 | 1.99 | 0.94 |       |
| test_geom_camber_rc    | 155.70 | 3.17 | 1.18 |       |
| test_geom_camber_cruise| NaN   | 1.67 | 0.72 |       |
| test_re_rand           | 140.25 | 2.25 | 0.98 |       |
| 3-split test avg       | 154.80 |      |      |       |

**Config:** surf_weight=30, bs=4, lr=5e-4, all other defaults. 14 epochs (30 min timeout-cut, best at epoch 11). Peak VRAM 42.1 GB.

**Decision: CLOSED — val_avg=158.12 is 18% worse than floor=133.94 and 10% worse than previous floor=143.15.**

**Analysis:** Increasing surf_weight uniformly for all surface channels harms optimization at 14 epochs. The channel weighting approach (chan_w=[1,1,5]) is a more surgical lever — it targets specifically the pressure channel rather than uniformly upweighting the surface. Both levers act on the same axis (loss alignment with primary metric) but channel weighting is strictly better. Student produced an excellent bug report on the 0×NaN propagation in test eval — assigned follow-up PR #1536 to apply the train.py guard and give us the first ever clean test_avg.

---

## 2026-05-12 19:15 — PR #1464: Per-channel loss weighting (pressure ×5)

- Branch: `charliepai2g24h2-alphonse/channel-weight-p5`
- Hypothesis: chan_w=[1,1,5] applied to sq_err aligns gradient with primary metric (val_avg/mae_surf_p)
- Artifacts: `models/model-charliepai2g24h2-alphonse-channel-weight-p5-20260512-181154/metrics.jsonl`

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---:|---:|---:|---:|
| val_single_in_dist     | 155.84 | 2.73 | 1.14 | 182.03 |
| val_geom_camber_rc     | 146.50 | 3.80 | 1.44 | 170.88 |
| val_geom_camber_cruise | 103.54 | 2.06 | 0.88 | 115.59 |
| val_re_rand            | 129.86 | 2.90 | 1.18 | 141.11 |
| **val_avg**            | **133.94** | 2.87 | 1.16 | 152.40 |
| test_single_in_dist    | 141.26 | 2.46 | 1.12 | 163.72 |
| test_geom_camber_rc    | 145.90 | 3.91 | 1.38 | 167.33 |
| test_geom_camber_cruise| NaN   | 1.98 | 0.83 | NaN   |
| test_re_rand           | 127.03 | 2.79 | 1.17 | 135.38 |
| 3-split test avg       | 125.48 |      |      |       |

**Config:** chan_w=[1,1,5], bs=4, lr=5e-4, surf_weight=10, n_hidden=128, n_layers=5, n_head=4, slice_num=64. 14 epochs (30 min timeout-cut), still improving. Peak VRAM 42.1 GB.

**Decision: MERGED — new floor at val_avg/mae_surf_p=133.9353 (beats previous 143.15 by 6.4%).**

**Analysis:** Channel weighting directly aligned training gradient with the primary metric. Improvement spans all 4 splits (val_re_rand marginal but positive). Val curve still descending at epoch 14 — this result is timeout-limited. Next: try chan_w=[1,1,10] to map the response curve. Also flag: two students independently found and documented the test NaN bug (data/scoring.py `0*NaN` propagation from one bad GT sample in test_geom_camber_cruise).

---

## 2026-05-12 19:15 — PR #1489: AoA-sign flip augmentation (50% per-batch)

- Branch: `charliepai2g24h2-thorfinn/aoa-flip-aug`
- Hypothesis: 50% per-batch AoA flip augmentation increases AoA coverage for OOD generalization
- Artifacts: `models/model-charliepai2g24h2-thorfinn-aoa-flip-aug-20260512-180844/metrics.jsonl`

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| val_single_in_dist     | 185.20 | 2.53 | **2.40** |
| val_geom_camber_rc     | 150.35 | 3.21 | **2.88** |
| val_geom_camber_cruise | 113.38 | 1.70 | **1.30** |
| val_re_rand            | 136.74 | 2.51 | **1.87** |
| **val_avg**            | **146.42** |     |      |

**Config:** per-batch 50% AoA flip, bs=4, lr=5e-4, all baseline defaults. 14 epochs (timeout-cut at epoch 11 best). Peak VRAM 42.1 GB.

**Decision: SENT BACK for changes (146.42 > 133.94 floor, plus Uy degradation 2.11 vs 0.98).**

**Analysis:** mae_surf_Uy doubled vs unaugmented (2.11 vs 0.98 for tanjiro), suggesting per-batch flipping harms Uy precision — the model sees Uy flipped for all samples in the batch 50% of the time, which may cause hedging. The interesting OOD cruise result (113.38) warrants follow-up. Sent back to try per-sample flip at p=0.25.

---

## 2026-05-12 19:00 — PR #1486: Scale batch size 4 → 8 (fallback, bs=16 OOMed)

- Branch: `charliepai2g24h2-tanjiro/batch-size-16`
- Hypothesis: bs=4 is underutilizing 96GB VRAM; bs=16 + scaled lr=1e-3 should reduce gradient noise
- Artifacts: `models/model-charliepai2g24h2-tanjiro-batch-size-8-fallback-20260512-180842/metrics.jsonl`

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p |
|---|---:|---:|---:|---:|
| val_single_in_dist     | 179.61 | 2.21 | 1.04 | 203.56 |
| val_geom_camber_rc     | 151.83 | 2.82 | 1.24 | 156.45 |
| val_geom_camber_cruise | 114.07 | 1.43 | 0.68 | 112.80 |
| val_re_rand            | 127.06 | 2.20 | 0.96 | 127.11 |
| **val_avg**            | **143.15** | 2.17 | 0.98 | 149.98 |
| test_single_in_dist    | 156.25 | 2.10 | 0.95 | 178.28 |
| test_geom_camber_rc    | 148.55 | 2.85 | 1.17 | 149.25 |
| test_geom_camber_cruise| NaN   | 1.32 | 0.64 | NaN   |
| test_re_rand           | 137.14 | 2.01 | 0.92 | 129.92 |

**Config run:** bs=8, lr=7e-4 (fallback: bs=16 OOMed at 93 GB due to pad_collate), wd=1e-4, surf_weight=10, ~1.4M baseline model. 14 epochs in 30 min (timeout-cut, still improving). Best epoch=14.

**Decision: MERGED — establishes floor at val_avg/mae_surf_p=143.15.**

**Key findings:**
- `pad_collate` makes batch scaling memory-expensive: bs=4→bs=8 pushed peak VRAM to 84 GB (far above the naive ~10 GB estimate). bs=16 needs ~160 GB — not feasible without AMP.
- With bs=8, per-epoch time ~130 s vs ~30-35 s for bs=4 → only 14 epochs in 30 min instead of ~50. Hypothesis about gradient-noise reduction can't be evaluated cleanly here.
- **Critical data bug:** `test_geom_camber_cruise/000020.pt` has 761 NaN values in p channel. scoring.py's NaN-skip logic fails because `0 * NaN = NaN` in masked reductions → test_avg/mae_surf_p is NaN for all experiments. This affects only test metrics, not val (val data is clean).
- Suggested fix: gradient accumulation (accum_steps=4 at bs=4 → effective_bs=16 without extra memory).

---

## 2026-05-12 19:00 — PR #1472: Bigger Transolver fallback 192-6-8 (~1.7M params)

- Branch: `charliepai2g24h2-edward/bigger-model-256-8-8`
- Hypothesis: 1.4M params is underparameterized; 256-8-8 (~8.8M) should improve capacity
- Artifacts: `models/model-charliepai2g24h2-edward-bigger-model-256-8-8-20260512-181250/metrics.jsonl`

| Split | mae_surf_p | epoch |
|---|---:|---:|
| val_single_in_dist     | 194.04 | 6 |
| val_geom_camber_rc     | 182.21 | 6 |
| val_geom_camber_cruise | 136.44 | 6 |
| val_re_rand            | 148.52 | 6 |
| **val_avg**            | **165.30** | 6 |

**Config run:** n_hidden=192, n_layers=6, n_head=8 (fallback: 256-8-8 OOMed at 94 GB), bs=4, lr=5e-4. 7 epochs in 30 min (per-epoch ~265 s → very few epochs). test_avg=NaN (data bug).

**Decision: CLOSED — 165.30 > 143.15 (floor). Fallback config is only 1.22x bigger than baseline and had only 7 epochs.**

**Key findings:**
- 256-8-8 with mlp_ratio=2 OOMs at bs=4 (MLP hidden=512, 8 layers × 242K-node activations). Requires bs=2 or AMP to fit.
- 192-6-8 at 7 epochs is not a valid test of the scaling hypothesis (not enough epochs to converge).
- Intermediate-size n_hidden=224, n_layers=7, n_head=8 (~3.4M, dim_head=28) should fit at bs=4 with headroom and get ~15-20 epochs.
- Model NaN in p-channel at test_geom_camber_cruise may indicate early-training numerical instability in the slice attention temperature.
</content>

## 2026-05-13 02:00 — PR #1708: Lookahead optimizer (k=5/10, α=0.5) — CLOSED

- Branch: `charliepai2g24h2-edward/lookahead-optimizer-k5-alpha0p5`
- Hypothesis: Lookahead wraps AdamW with slow-weight averaging to smooth trajectory and potentially aid generalization
- Artifacts: `models/model-charliepai2g24h2-edward-lookahead-optimizer-k5-alpha0p5-20260513-000558/metrics.jsonl`

| Run | k | Best val_avg/mae_surf_p | Δ vs floor |
|---|---|---|---|
| k=5 clean | 5 | 143.62 | **+17.0%** ↑ (worse) |
| k=5 contended | 5 | 153.47 | +25.1% ↑ |
| k=10 | 10 | 152.54 | +24.3% ↑ |

**Conclusion:** Same regime-mismatch failure as EMA (#1603). In the rapid-descent regime (>10 MAE/epoch drop), Lookahead's slow weights average old-worse with new-better parameters, dragging convergence back. Pattern consistent: any weight-averaging optimizer fails in this regime. Closed as dead end. Edward reassigned to Huber loss (#1801).

## 2026-05-13 02:05 — PR #1477: AMP bf16 + gradient clipping + NaN-y bug fix — SENT BACK

- Branch: `charliepai2g24h2-fern/amp-bf16-gradclip`
- Hypothesis: AMP bf16 reduces VRAM ~24% → more epochs per 30-min cap → better convergence
- Artifacts: `models/model-charliepai2g24h2-fern-amp-bf16-gradclip-confirm-20260513-005403/metrics.jsonl`

| Split | This run (bf16 only) | Floor #1573 | Δ% |
|---|---:|---:|---:|
| val_single_in_dist | 108.34 | 159.59 | **−32%** |
| val_geom_camber_rc | 105.61 | 134.74 | **−22%** |
| val_geom_camber_cruise | 73.20 | 89.18 | **−18%** |
| val_re_rand | 91.07 | 107.31 | **−15%** |
| **val_avg** | **94.55** | **122.70** | **−23%** |
| test_avg (bs=4, clean!) | **84.64** | 110.25 (bs=1) | — |

**BUT:** Fern's config REVERTS chan_w=[1,1,5] and 3-ep warmup from advisor. Run was at lr=5e-4 with plain CosineAnnealingLR(T_max=50). Missing the full floor stack. 19 epochs (vs 12 for floor) = 58% more budget from VRAM reduction.

**Sent back:** asked to rebase onto current advisor and re-run with full floor stack (chan_w + warmup + gradclip + lr=7.5e-4) + AMP bf16 + bug fix.

**Key insights:**
- AMP bf16 unlocks ~32 GB VRAM (vs 42 GB fp32) → 7-8 more epochs per 30-min cap. This is likely the main driver.
- bf16 inference also fixes the bs=4 NaN on test_geom_camber_cruise (test 84.64 is fully clean bs=4)
- Fern's evaluate_split NaN-y prefilter is a clean bug fix for the Type-1 data NaN. Supersedes askeladd's #1536 fix.

## 2026-05-13 03:05 — PR #1801: Huber/SmoothL1 loss β=1.0 — **NEW FLOOR**

- Branch: `charliepai2g24h2-edward/huber-loss-pressure`
- Hypothesis: Replace L2 (sq-err) with Huber (β=1.0) to better match MAE evaluation metric and reduce outlier sensitivity
- Artifacts: `models/model-charliepai2g24h2-edward-huber-loss-pressure-20260513-020521/metrics.jsonl`

| Split | Huber (this PR) | Floor #1573 (L2) | Δ% |
|---|---:|---:|---:|
| val_single_in_dist | 134.21 | 159.59 | **−15.9%** |
| val_geom_camber_rc | 133.88 | 134.74 | −0.6% |
| val_geom_camber_cruise | 77.59 | 89.18 | **−13.0%** |
| val_re_rand | 98.93 | 107.31 | **−7.8%** |
| **val_avg** | **111.15** | **122.70** | **−9.4%** |
| test_avg (bs=1) | **99.06** | **110.25** | **−10.2%** |

**Config:** Same as floor + Huber β=1.0 in BOTH train loop and evaluate_split, fp32. 13-14 epochs (30 min cap). Peak VRAM 42.12 GB.

**Conclusion:** Clear win. L2→Huber is significant at −9.4% val, −10.2% test. Largest gain: single_in_dist (−15.9%) — consistent with Huber's robustness to the high-error tails in out-of-distribution samples. First sub-100 test_avg (99.06) on this branch. Huber now stacked in advisor train.py.

**Key insight:** The L2/L1 training-metric mismatch was actively harming OOD splits (single_in_dist is the most OOD split). Switching to Huber acts as implicit outlier weighting — reduces gradient magnitude for high-residual samples that were pulling optimization away from the bulk distribution.

## 2026-05-13 07:10 — PR #1751: Tighter cosine T_max=12 — **NEW FLOOR**

- Branch: `charliepai2g24h2-frieren/tighter-cosine-t-max-12`
- Hypothesis: The scheduler T_max=47 is wildly miscalibrated for the ~12-epoch training window. Aligning T_max to the actual epoch budget (T_max=12) should unlock the low-LR decay phase that the prior schedule never reached, dramatically improving convergence.
- Artifacts: `models/model-charliepai2g24h2-frieren-tighter-cosine-t-max-12-20260513-055432/metrics.jsonl`

| Split | T_max=12 (this PR) | Floor #1849 (T_max=47) | Δ% |
|---|---:|---:|---:|
| val_single_in_dist | 108.0187 | 126.2130 | **−14.4%** |
| val_geom_camber_rc | 96.3656 | 116.3601 | **−17.2%** |
| val_geom_camber_cruise | 61.1470 | 82.2281 | **−25.6%** |
| val_re_rand | 78.2041 | 97.9218 | **−20.1%** |
| **val_avg** | **85.9338** | **105.6808** | **−18.7%** |
| test_single_in_dist (bs=1) | 97.6661 | 113.4328 | **−13.9%** |
| test_geom_camber_rc (bs=1) | 88.6578 | 104.1052 | **−14.9%** |
| test_geom_camber_cruise (bs=1) | 51.3197 | 68.1068 | **−24.7%** |
| test_re_rand (bs=1) | 72.9518 | 94.2933 | **−22.6%** |
| **test_avg (bs=1)** | **77.6488** | **94.9845** | **−18.3%** |

**Config:** Full floor stack + T_max=12 in CosineAnnealingLR. 14 epochs (30 min cap). Peak VRAM 42.11 GB.

**Conclusion:** Strongest single result on this branch — −18.7% val, −18.3% test. The T_max=47 baseline was schedule-starved: at epoch 14, the prior schedule had decayed only 23% of its arc (LR ~6.9e-4, near peak). With T_max=12, by epoch 14 the cosine had decayed 92% (LR 5.1e-5). The entire gain is concentrated in the late-epoch low-LR phase that was previously locked out. The prior floor improvements (Huber β, chan_w) were genuine but were measured on a schedule-starved baseline — this run shows the full extent of those changes with a well-calibrated decay.

**Key insight:** Schedule calibration to wall-clock budget is as important as the schedule choice itself. CosineAnnealingLR with T_max matched to actual epochs delivers a qualitatively different optimization trajectory: the model reaches the refinement regime (low LR, small gradient steps) within the timeout window rather than never getting there. All future runs now default to T_max matched to --epochs.
