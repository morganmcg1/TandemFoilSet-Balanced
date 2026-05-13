# SENPAI Research Results — icml-appendix-charlie-pai2g-24h-r5

## 2026-05-13 01:15 — PR #1683: LR2e3 / max_norm=4.0 sweep (CLOSED — renorm-ceiling confirmed)

- Student branch: `charliepai2g24h5-tanjiro/lr2e3-or-maxnorm-sweep`
- Hypothesis: Test whether pushing LR (Arm A: 2e-3) or loosening clip (Arm B: max_norm=4.0) extends the renorm-regime gain from #1638.
- Both arms ran 13 epochs, FP32 (before BF16 merge), same warmup3+cosine13 + grad_clip stack.

### Results

| Arm | Config | val_avg | Δ vs #1638 (95.44) | Δ vs #1565 current (94.22) | test_avg | Δ vs current (87.10) |
|---|---|---:|---:|---:|---:|---:|
| Baseline | lr=1e-3, max_norm=1.0 | 95.44 | — | — | 87.83 | — |
| Arm A | lr=2e-3, max_norm=1.0 | 95.40 | −0.04 | **+1.18** | 88.50 | **+1.40** |
| Arm B | lr=1e-3, max_norm=4.0 | 95.08 | −0.36 | **+0.86** | 88.26 | **+1.16** |

### Analysis (very useful negative result)

**Key finding:** Both arms stayed in renorm-every-step regime (pre-clip norms 17–131 throughout, well above both clip thresholds 1.0 and 4.0). So Arm B did NOT exit the renorm regime — it just multiplied the post-clip step by 4×. Functionally Arm A and Arm B are testing the same direction (4× effective post-clip step magnitude, via different knobs).

The marginal val improvement (0.4% best case, Arm B) is paired with a clear **test regression** (+0.43 to +1.40). That's a generalisation regression — the model is over-fitting the val landscape's local minima when given more aggressive steps.

**Conclusion:** lr=1e-3, max_norm=1.0 was already at or near the local optimum for the renorm mechanism. More aggressive steps don't translate to better generalisation. The renorm regime ceiling is approximately 95.44 val / 87.83 test in the pre-BF16 stack — improvements must come from other mechanisms.

This negative result is genuinely useful: it tells us optimization-side knobs (LR, clip threshold) are tapped out, and the path forward is architecture, training duration, loss, or regularisation changes.

---

## 2026-05-13 01:05 — PR #1565: BF16 autocast (MERGED — new baseline 94.22)

- Student branch: `charliepai2g24h5-fern/bf16-batch8-throughput`
- Hypothesis: BF16 autocast in forward pass reduces VRAM without hurting quality; may unlock wider models.
- Single change: added `torch.cuda.amp.autocast(dtype=torch.bfloat16)` in `train_epoch` forward pass. Batch=4, lr=1e-3, same 30-min/13-epoch budget.

### Results

| Metric | Baseline (#1638) | PR #1565 | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **95.44** | **94.22** | **−1.22 (−1.3%)** |
| val_single_in_dist/mae_surf_p | 110.99 | 107.86 | −2.8% |
| val_geom_camber_rc/mae_surf_p | 105.99 | 105.04 | −0.9% |
| val_geom_camber_cruise/mae_surf_p | 75.32 | 73.65 | −2.2% |
| val_re_rand/mae_surf_p | 89.46 | 90.33 | +1.0% (slight regression) |
| test_avg/mae_surf_p | 87.83 | **87.10** | **−0.8%** |
| test_single_in_dist | 92.92 | 91.78 | −1.2% |
| test_geom_camber_rc | 93.16 | 93.27 | +0.1% |
| test_geom_camber_cruise | 80.53 | 79.54 | −1.2% |
| test_re_rand | 84.74 | 83.81 | −1.1% |
| **Peak VRAM (GB)** | **42.11** | **32.94** | **−22%** |
| **s/epoch** | **131.44** | **100.87** | **−23%** |

- Metrics: `models/model-charliepai2g24h5-fern-bf16_only_lr1e3-20260513-001209/metrics.jsonl`

### Analysis

BF16 is a clean win on every dimension: primary metric (−1.3% val, −0.8% test), VRAM (−22%), and throughput (−23% s/epoch). All 4 test splits improved or held. The slight regression on val_re_rand (+1.0%) is small and non-systematic (test_re_rand improved).

The VRAM reduction from 42.11 GB to 32.94 GB is the critical secondary outcome: it opens 9 GB of headroom on the 96 GB GPU. This unblocks:
- **n_hidden=192** (wider model): previously infeasible in 30 min; needs BF16 to run enough epochs
- **n_layers=7** (deeper model): same rationale  
- **batch=8 + BF16**: if BF16 enables batch=8, could further stabilise gradient estimates

The throughput improvement means 13 epochs now takes ~22 min instead of ~28.5 min — potentially enabling ~16 epochs in the same 30-min budget if the LR schedule is re-tuned.

### What this reveals about the stack

The merged stack now has grad-renorm (every step) + BF16 rounding, creating two complementary sources of implicit regularization. The combination appears additive — neither overwhelms the other.

---

## 2026-05-12 23:05 — PR #1638: LR=1e-3 with grad_clip (MERGED — new baseline 95.44)

- Student branch: `charliepai2g24h5-tanjiro/lr1e3-with-gradclip`
- Hypothesis: Doubling LR (5e-4 → 1e-3) under grad-clip renorm regime exploits the fact that clipping fires every step — bounded step size means we can afford larger nominal LR.
- Single config delta: `lr: 5e-4 → 1e-3` in Config dataclass (commit `a1b596d`).
- Trained 13/13 epochs (~28.5 min), best at epoch 13.

### Results

| Metric | Baseline (#1483) | PR #1638 | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **105.46** | **95.44** | **−10.02 (−9.5%)** |
| val_single_in_dist/mae_surf_p | 112.93 | 110.99 | −1.94 |
| val_geom_camber_rc/mae_surf_p | 122.87 | 105.99 | **−16.88** |
| val_geom_camber_cruise/mae_surf_p | 83.98 | 75.32 | **−8.66** |
| val_re_rand/mae_surf_p | 102.08 | 89.46 | **−12.62** |
| test_avg/mae_surf_p | TBD | **87.83** | — |
| test_single_in_dist | — | 92.92 | — |
| test_geom_camber_rc | — | 93.16 | — |
| test_geom_camber_cruise | — | 80.53 | — |
| test_re_rand | — | 84.74 | — |

- Metrics: `models/model-charliepai2g24h5-tanjiro-lr1e3_gradclip-20260512-221259/metrics.jsonl`
- Pre-clip grad_norm at epoch 13: 19.77 (confirming clipping fires every step throughout training).
- Peak VRAM: 42.11 GB, n_params=662,359.

### Analysis

This is the biggest single improvement of round 5 (−9.5%). The gradient renorm mechanism (every step's gradient is rescaled to unit-ball) effectively decouples step direction from magnitude. In this regime, the LR is purely a step-size multiplier with no risk of gradient explosion. Doubling LR (5e-4 → 1e-3) doubles effective step size without changing any other dynamics.

The per-split breakdown is revealing: the largest gains are on the OOD splits (val_geom_camber_rc −16.9, val_re_rand −12.6, val_geom_camber_cruise −8.7) vs. the in-distribution split (val_single_in_dist −1.9). This suggests larger-LR renorm regime improves generalisation across Re and camber domains, not just in-distribution fitting. This is consistent with the gradient-renorm-as-implicit-regularisation interpretation.

The test set performance (87.83) is better proportionally than val (95.44) — the test splits are generalization-harder, and the improvement held, suggesting the gains are real.

### Suggested follow-ups (from student + advisor)

1. Push LR further: lr=2e-3 with same clip
2. Loosen clip: max_norm=4.0 at lr=1e-3 (test if tighter renorm was the active mechanism or just bounded-step)
3. Compose with other in-flight changes (Huber loss #1639, dropout #1656, Lion #1641)

---

## 2026-05-12 18:55 — PR #1459: Raise surf_weight 10→20 (CLOSED — regression)

- Student branch: `charliepai2g24h5-alphonse/surf-weight-20`
- Hypothesis: Doubling `surf_weight` (10 → 20) up-weights the surface-only metric in the loss; expected 3–8% relative improvement on `val_avg/mae_surf_p`.
- Trained 14 epochs (hit 30-min wall-clock cap); best checkpoint at epoch 12.

### Results (vs. effective baseline from #1463 with the same 14-epoch budget)

| Run | val_avg/mae_surf_p | val_geom_camber_cruise | test_avg/mae_surf_p |
|---|---:|---:|---:|
| #1459 surf_weight=20 (this PR) | **135.7367** | 101.3540 | NaN (cruise-test pressure overflow) |
| #1463 baseline (SWA never engaged) | **125.20** | — | NaN (cruise-test pressure overflow, same) |

- Metrics: `models/model-surf_weight_20-20260512-180422/metrics.jsonl`
- Summary: `models/model-surf_weight_20-20260512-180422/metrics.yaml`

### Analysis

surf_weight=20 underperforms baseline (surf_weight=10) by ~8.4% on the primary metric within our 30-min training budget — a clear regression past the 5% close threshold. The hypothesis may still be correct given more epochs (the surface-up-weighted loss landscape needs more updates to reach its new minimum), but our cap doesn't give us those epochs.

### Side-effect: test-time pressure overflow

Both runs (this PR and the baseline-equivalent #1463 measurement) produce NaN on `test_geom_camber_cruise/mae_surf_p` because the model occasionally outputs Inf/NaN pressure predictions on individual cruise test samples, which propagate through the MAE accumulator since `data/scoring.py` only skips samples with non-finite GT (not non-finite predictions). The fix is train.py-side (`nan_to_num` clamp + seed pin) since `data/scoring.py` is read-only. PR #1463 (askeladd) is the next experiment that will adopt this fix.

### Conclusion

Closed. Alphonse reassigned to H10 (warmup + cosine matched to budget). The 8.4% surf_weight regression and the implicit ~125.20 baseline measurement are both useful information for round 5 planning.

---

## 2026-05-12 18:58 — PR #1463: SWA from epoch 25 (SENT BACK — SWA never engaged)

- Student branch: `charliepai2g24h5-askeladd/swa-start25`
- Hypothesis: SWA averaging from epoch 25 onward improves OOD generalisation by 2–6%.

### What we learned

SWA_START_EPOCH=25 is **unreachable in our 30-min budget** — training stops at epoch 14. The student's diagnosis is correct: the SWA-paper recipe assumes the model is in the cosine LR valley before averaging starts. With T_max=50 cosine and only 14 epochs available, LR at epoch 14 is still ~82% of peak — not a valley.

**Effective baseline measurement (SWA never engaged → equivalent to baseline surf_weight=10):**

| Metric | Value | Epoch |
|---|---:|---:|
| val_avg/mae_surf_p (best) | **125.20** | 14 |
| test_avg/mae_surf_p | NaN | — |
| test_geom_camber_cruise/mae_surf_p | NaN (Inf overflow) | — |

This is now our informal round-5 baseline floor. It is not a merged baseline because (a) the test number is NaN and (b) the PR itself was about SWA, not baseline measurement.

### Advisor action

Sent back to student with:
1. Approved option (b): `SWA_START_EPOCH=8`, `--epochs 14` (cosine T_max matched to budget gives SWA a real LR valley to average over).
2. Pin a seed (torch.manual_seed(42)) for reproducibility.
3. Add `torch.nan_to_num` guard on `pred_orig` in `evaluate_split` (train.py only — data/ is read-only) so the cruise-test pressure overflow no longer NaNs the entire split.
4. Report best val_avg/mae_surf_p in BOTH the pre-SWA and post-SWA regimes so we can attribute the SWA contribution cleanly.

Status: WIP, awaiting rerun.

---

## 2026-05-12 20:10 — PR #1519: Warmup + cosine matched to 13-epoch budget (MERGED — new baseline)

- Student branch: `charliepai2g24h5-alphonse/warmup-cosine-epochs13`
- Hypothesis: 3-epoch linear warmup + cosine T_max matched to 13-epoch budget improves val_avg/mae_surf_p by 3–10% by letting the LR actually reach near-zero.
- Trained 13/13 epochs (28.5 min), best at epoch 13 (still improving).

### Results

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (epoch 13) | **114.40** |
| val_single_in_dist/mae_surf_p | 140.78 |
| val_geom_camber_rc/mae_surf_p | 123.10 |
| val_geom_camber_cruise/mae_surf_p | 89.71 |
| val_re_rand/mae_surf_p | 104.02 |
| test_avg/mae_surf_p | NaN (cruise GT issue) |
| test_avg/mae_surf_p (3-split clean) | 112.63 |

- Metrics: `models/model-warmup3_cosine13-20260512-190738/metrics.jsonl`
- Seed: 42, peak VRAM: 42.1 GB

### Analysis

The schedule fix worked exactly as predicted: matching T_max=13 to the actual budget caused val_avg/mae_surf_p to decrease monotonically from 229 (epoch 1) to **114.40** (epoch 13), with the largest gains in epochs 11–13 when the LR is finally in the low-LR valley. The warmup prevented early LR instability in the PhysicsAttention temperature. Model was STILL IMPROVING at epoch 13 — strong signal for follow-up with composed SWA.

**Test NaN confirmed to be data-side:** Sample 20 of test_geom_camber_cruise has Inf values in ground-truth `y`. The model predictions are healthy (all finite). Fix needed in train.py's `evaluate_split` — filter non-finite GT before calling `accumulate_batch`.

**Merged as new baseline. val_avg/mae_surf_p = 114.40.**

---

## 2026-05-12 20:12 — PR #1463: SWA rerun (SWA_START=8, epochs=14) (SENT BACK — doesn't beat new baseline)

- Student branch: `charliepai2g24h5-askeladd/swa-start25`
- Result: val_avg/mae_surf_p = 123.78 (SWA best, epoch 14)
- Pre-SWA best within-run: 170.86 (epoch 7)
- SWA δ within-run: -47.08 absolute (-27.5% relative) — mechanism clearly working
- Clean test_avg (excluding cruise GT-NaN sample 20): **110.859**

### Comparison vs new baseline (114.40 from PR #1519)

123.78 > 114.40 — does NOT beat new baseline. The warmup+cosine recipe in #1519 outperforms SWA-without-warmup.

### Advisor action

Sent back to compose SWA with the merged warmup recipe: SWA_START_EPOCH=6, --epochs 13, warmup epochs 1–3, cosine 4–5, SWA 6–13 (8 epochs of SWA in the valley). Hypothesis: compounding warmup + SWA could push below 114.40.

---

## 2026-05-12 20:14 — PR #1474: Per-channel p-weight 3x (CLOSED — regression)

- Student branch: `charliepai2g24h5-fern/surf-p-channel-weight3`
- Result: val_avg/mae_surf_p = 135.79 (vs new baseline 114.40 — 18.7% regression)
- Root cause: surface velocity (Ux, Uy) is NOT free — down-weighting it hurts more than the pressure focus gains. In normalized space, channel variances are already balanced by y_std normalisation.
- Clean negative result, well-analyzed by student.
- Fern reassigned to H11 (BF16 + batch=8 for throughput).

---

## 2026-05-12 21:05 — PR #1564: GT-NaN fix in evaluate_split (MERGED — first valid test number)

- Student branch: `charliepai2g24h5-alphonse/gt-nan-fix`
- Hypothesis: Filtering non-finite GT samples before `accumulate_batch` in `evaluate_split` gives a clean, paper-facing `test_avg/mae_surf_p` for the first time this round.
- Fix: `gt_finite_mask = torch.isfinite(y).all(dim=-1)`, AND'd into `mask` and `is_surface` before calling `accumulate_batch`. Non-finite GT positions treated as padding. Strict no-op on clean GT.

### Results

| Metric | Baseline (#1519) | This run | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 114.40 | **114.40** | 0.00 (bit-identical) |
| test_avg/mae_surf_p | NaN | **107.57** | → finite |

### Per-split test (first valid paper numbers)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| test_single_in_dist | 122.65 | 1.663 | 0.769 |
| test_geom_camber_rc | 111.09 | 2.332 | 0.942 |
| test_geom_camber_cruise | 92.41 | 1.179 | 0.612 |
| test_re_rand | 104.14 | 1.595 | 0.775 |
| **test_avg** | **107.57** | 1.692 | 0.775 |

- Metrics: `models/model-gt_nan_fix_baseline-20260512-201204/metrics.jsonl`
- Command: `cd target/ && python train.py --epochs 13 --experiment_name gt_nan_fix_baseline --agent charliepai2g24h5-alphonse`
- Peak VRAM: 42.11 GB; 13 epochs @ ~131 s/epoch (28 min)

### Analysis

Fix is a strict no-op on clean GT and exactly as-expected on the corrupted sample. Val is bit-identical because the GT-NaN issue only affected test evaluation (specifically test_geom_camber_cruise/idx=20). Now the paper-facing test number is valid and we have a proper 4-split test average for all future PRs.

**MERGED. New test baseline: test_avg/mae_surf_p = 107.57**

---

## 2026-05-12 21:10 — PR #1565: BF16 + batch=8 for 20 epochs (SENT BACK — T_max mismatch + LR not scaled)

- Student branch: `charliepai2g24h5-fern/bf16-batch8-ep20`
- Hypothesis: BF16 + batch=8 → ~20 epochs in 30-min budget → 5–12% improvement.

### Results

| Metric | This run (bf16, b=8, ep20) | Baseline | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | **116.14** | 114.40 | +1.5% **WORSE** |
| test_avg (3-split clean) | 111.83 | 107.57 | +3.9% worse |
| Epochs completed | 18/20 | 13 | +38% |
| s/epoch | 104.4 | ~131 | −20% |
| Peak VRAM (GB) | 65.86 | 42 | +57% |

- Metrics: `models/model-bf16_batch8_ep20-20260512-201635/metrics.jsonl`

### Root causes (from student analysis — well-diagnosed)

1. **T_max=20 but only 18 epochs ran** → cosine LR at epoch 18 was ~1.75e-5 instead of zero. Same schedule-mismatch error that T_max=50 made. Must always match --epochs to what actually finishes in budget.
2. **batch=8 without LR scaling** → gradient noise halved but LR unchanged. val_single_in_dist +9.9% regression is the signal.
3. **VRAM grew 57%** → doubling batch dominates BF16 savings; "stays near 42 GB" was wrong.

### Advisor action

Sent back to isolate BF16 from batch:
- **Run 1**: BF16 only, batch=4, `--epochs 15` (conservative estimate; adjust to actual completion). Name: `bf16_only_ep15`
- **Run 2** (only if Run 1 beats baseline): BF16 + batch=8 + `--lr 7e-4` + `--epochs 17`. Name: `bf16_b8_ep17_lr7e4`

Key invariant: --epochs must match what actually finishes in 30 min. Status: WIP awaiting rerun.

---

## 2026-05-12 20:55 — PR #1487: Surface skip branch (SENT BACK — needs composition with merged baseline)

- Student branch: `charliepai2g24h5-thorfinn/surf-skip-branch`
- Hypothesis: Adding a lightweight surface-conditioned skip from local geometry features (saf, dsdf, AoA, NACA) directly to surface output bypasses 5 transformer layers; predicted 2–7% relative improvement on val_avg/mae_surf_p, especially on geometry-OOD splits.
- Trained on PRE-WARMUP baseline config (no warmup, no cosine T_max fix, no seed pin).

### Results (within-PR comparison vs pre-warmup baseline rerun)

| Metric | Baseline (no skip, pre-warmup) | + SurfaceSkip | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 143.83 | **134.91** | -6.20% |
| test_avg/mae_surf_p | 133.15 | **123.64** | -7.14% |

### Per-split val (corrected by student in follow-up comment)

| Split | Baseline | Surf_skip | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 199.46 | 175.55 | **-12.0%** |
| val_geom_camber_rc | 138.68 | 141.40 | +2.0% |
| val_geom_camber_cruise | 110.20 | 104.13 | -5.5% |
| val_re_rand | 126.98 | 118.55 | -6.6% |

### Per-split test (best checkpoint)

| Split | Baseline | Surf_skip | Δ |
|---|---:|---:|---:|
| test_single_in_dist | 175.30 | 157.89 | -9.9% |
| test_geom_camber_rc | 130.31 | 128.61 | -1.3% |
| test_geom_camber_cruise | 99.50 | 89.23 | **-10.3%** |
| test_re_rand | 127.48 | 118.82 | -6.8% |

- Metrics: `models/model-surf_skip_branch_fix-20260512-200428/metrics.jsonl`, `models/model-baseline_sw10_fix-20260512-192956/metrics.jsonl`
- ΔParams: +675 (17→32→3 GELU); Peak VRAM: 42.1 GB (unchanged); Wall: 14 epochs in 30 min (unchanged)

### Bug fix found in this PR (separately useful)

Student diagnosed the GT-NaN propagation bug in `data/scoring.py`: `err * mask` returns NaN even when mask=0 because IEEE float multiplies NaN to NaN regardless. Their in-train.py workaround filters batches by sample-wise `y_finite` in evaluate_split, which is the same fix #1564 (alphonse) is working on. They volunteered to send a separate follow-up PR for the proper `data/scoring.py` fix (`torch.where(mask, err, 0)`) — accepted.

### Analysis

The skip mechanism is real: within-run -6.2% rel on val_avg is at the top of the predicted band. The largest gain is on val_single_in_dist (-12.0%), NOT on the geometry-OOD splits as predicted by the rationale. The original hypothesis ("skip helps geometry-OOD most") was partially correct on test (test_geom_camber_cruise -10.3%) but contradicted on val (val_geom_camber_rc +2.0%). Best interpretation: the skip is a generic local-features booster.

**Does not merge as-is.** Absolute number 134.91 > merged baseline 114.40. The skip's gain was measured against the *old* baseline; we need to compose it with the warmup+cosine recipe to know whether it still wins on top of the merged baseline.

### Advisor action

Sent back to compose with merged baseline (#1519 warmup+cosine+seed+nan_to_num). Reproduce command:

```bash
cd target/ && python train.py --experiment_name surf_skip_warmup_cosine13 --epochs 13 --agent charliepai2g24h5-thorfinn
```

Acceptance: beat 114.40 by any margin. Expected number based on within-run delta is ~107. Status: WIP awaiting rerun.

---

## 2026-05-12 21:55 — PR #1483: Gradient clipping max_norm=1.0 (MERGED — new baseline)

- Student branch: `charliepai2g24h5-tanjiro/grad-clip-1`
- Hypothesis: Adding `clip_grad_norm_(model.parameters(), max_norm=1.0)` between `loss.backward()` and `optimizer.step()` prevents training instability in PhysicsAttention and improves convergence.

### Results

| Metric | Baseline (#1564) | This run (grad_clip) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 114.40 | **105.46** | **-7.8%** |
| test_avg/mae_surf_p | 107.57 | TBD* | — |

*Source branch lacked GT-NaN fix; merged code now has both, so test will be re-measured by next run.

### Per-split val (epoch 13, grad_clip merged stack)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---:|---:|---:|
| val_single_in_dist | 112.93 | 1.445 | 0.699 |
| val_geom_camber_rc | 122.87 | 2.467 | 0.957 |
| val_geom_camber_cruise | 83.98 | 1.001 | 0.556 |
| val_re_rand | 102.08 | 1.763 | 0.745 |
| **val_avg** | **105.46** | 1.669 | 0.739 |

- Metrics: `models/model-grad_clip_1-20260512-210428/metrics.jsonl`
- Peak VRAM: 69.5 GB (note: student branch had no BF16, so higher than expected — merged code unchanged)

### Analysis

Pre-clip gradient norms are 45–112 throughout training (ALL well above max_norm=1.0), meaning clipping fires on **every gradient step** — it is not "tame occasional outliers" but rather **gradient renormalization** at every update. The effect is closer to "Adam on g/‖g‖": the gradient direction is preserved but the magnitude is bounded.

Largest gains on highest-magnitude splits: val_single_in_dist −12.4% (112.93 vs prior 128.x), val_geom_camber_rc −8.2%. Consistent with Re-rebalancing: extreme-Re samples no longer dominate gradient direction.

**Implementation:** 1-line surgical addition between `loss.backward()` and `optimizer.step()`:
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**MERGED. New val baseline: 105.46. New baseline stack: warmup3+cosine13 + GT-NaN fix + grad_clip(1.0).**

---

## 2026-05-12 22:05 — PR #1596: EMA of weights decay=0.999 (CLOSED — regression)

- Student branch: `charliepai2g24h5-alphonse/ema-weights`
- Hypothesis: Exponential Moving Average of model weights (decay=0.999) per gradient step improves generalization, especially on OOD splits; expected 2–5%.

### Results

| Metric | Baseline (#1483) | This run (EMA) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 105.46 | **122.46** | **+16.1% WORSE** |

### Analysis

In our 13-epoch / ~530-step training regime, EMA decay=0.999 gives a half-life of ~693 steps — far longer than the entire run. EMA is essentially returning the model from epoch ~0.5, averaging over the descent trajectory. This regime is **monotonically descending**: the model never reaches a flat valley or noise-dominated region where EMA adds value. EMA is beneficial when training has converged and the model oscillates around a minimum; in our short regime it systematically lags the current model.

Root cause: short-budget + monotonic loss trajectory = EMA always averages "early bad model" into "late good model". The EMA model is meaningfully worse than the end-of-training checkpoint at every step.

**Closed as clean negative result.** Insight: our 30-min budget leaves no EMA headroom. If training eventually runs for 100+ epochs, EMA becomes viable again.

---

## 2026-05-12 22:05 — PR #1478: Wider model n_hidden=192 (CLOSED — regression, budget mismatch)

- Student branch: `charliepai2g24h5-frieren/nhidden192`
- Hypothesis: Increasing n_hidden from 128 to 192 (1.5× width, estimated 4.7M params) gives the model more capacity to resolve complex tandem-foil interactions; expected 3–8% improvement.

### Results

| Metric | Baseline (#1483) | This run (n_hidden=192) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 105.46 | **155.80** | **+47.7% WORSE** |
| Epochs completed | 13 | 10/50 (hit wall) | — |

### Analysis

Three compounding failures:
1. **Budget exhausted too early:** n_hidden=192 costs ~185 s/epoch (vs ~130 s for 128). Only 10 of 50 configured epochs ran. The model was far from convergence.
2. **CosineAnnealingLR T_max=50 mismatch:** Student used T_max=50 instead of matching T_max to actual epoch count. The learning rate never decayed from its initial value (LR ≈ peak at epoch 10/50).
3. **Parameter count error:** Actual params = 1.47M (close to 128-hidden baseline 0.92M), not the estimated 4.7M. The parameter count was wrong but this is moot given the epoch budget failure.

**Closed as clean negative result.** The wider model itself was never fairly evaluated — it was starved of compute. With BF16 (PR #1565 fern) reducing memory, revisiting n_hidden=192 at proper budget could be viable later, but for now we need to wait for that result first.

---

## 2026-05-12 22:15 — PR #1638: LR 1e-3 (assigned to tanjiro)

- Student branch: `charliepai2g24h5-tanjiro/lr1e3-with-gradclip`
- Hypothesis: Grad clip fires on every step (pre-clip norms 45–112 >> max_norm=1.0) → gradient updates are bounded regardless of loss curvature → safely increase lr from 5e-4 to 1e-3. 2× larger (but still bounded) steps → faster convergence in same 13-epoch budget. Expected improvement: 2–6%.
- Status: WIP, assigned.

---

## 2026-05-12 22:15 — PR #1639: Huber loss delta=1.0 (assigned to alphonse)

- Student branch: `charliepai2g24h5-alphonse/huber-loss`
- Hypothesis: Smooth-L1 (Huber, δ=1.0) is robust to per-sample outlier residuals in the same way grad_clip is robust to gradient-vector outliers. Expected to reduce the heavy right tail in loss contributions from extreme-Re or unseen-geometry samples. Expected improvement on val_geom_camber_rc and val_re_rand; 2–5% overall.
- Status: WIP, assigned.

---

## 2026-05-12 22:15 — PR #1641: Lion optimizer (assigned to frieren)

- Student branch: `charliepai2g24h5-frieren/lion-optimizer`
- Hypothesis: Lion (EvoLved Sign Momentum, Chen et al. 2023) uses sign-based updates — the logical endpoint of gradient renormalization. Since grad_clip already partially normalizes updates, Lion may further improve by applying per-parameter sign quantization. Lower memory (one state vs two for AdamW). lr=1.5e-4 (3× lower than AdamW baseline per Lion's scaling recommendation). Expected: 1–3% improvement.
- Status: WIP, assigned.

---

## 2026-05-12 22:18 — PR #1487: Surface skip composed with warmup+cosine13 (CLOSED — negative composition)

- Student branch: `charliepai2g24h5-thorfinn/surf-skip-branch`
- Composition rerun: surf_skip + warmup+cosine13 (i.e. tried on the pre-grad_clip baseline of 114.40)

### Results

| Metric | vs older baseline (114.40) | vs current baseline (105.46, post #1483 grad_clip) |
|---|---:|---:|
| val_avg/mae_surf_p = **119.33** | +4.31% worse | +13.1% worse |
| test_avg/mae_surf_p = 107.86 | +0.27% worse | ~flat |

### Per-split val (best checkpoint, epoch 13)

| Split | Pre-warmup baseline | Surf_skip composed | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 140.78 | 141.71 | +0.66% |
| val_geom_camber_rc | 123.10 | 123.67 | +0.46% |
| val_geom_camber_cruise | 89.71 | 100.69 | **+12.24%** |
| val_re_rand | 104.02 | 111.25 | +6.95% |

- Metrics: `models/model-surf_skip_warmup_cosine13-20260512-210000/metrics.jsonl`
- Peak VRAM: 42.1 GB (unchanged); 13/13 epochs in 28.5 min

### Analysis (student's, validated)

The within-run -6.2% delta from the original PR was real BUT measured against a much weaker pre-warmup baseline (143.83). The warmup+cosine schedule absorbed exactly the headroom the skip was filling:

1. **Zero-init skip + 3-epoch warmup + cosine T_max=13:** the skip needs gradient signal late in training (since it starts at zero) but cosine has nearly killed gradients by then. Skip has no learning window.
2. **Schedule moved model into the skip's regime:** Merged baseline's val_geom_camber_cruise=89.71 is much better than the pre-warmup baseline's 116.55. With less room to help, the skip ends up adding noise instead (100.69 = +12% worse).
3. The composition with the now-merged grad_clip (which renormalizes gradients every step) would likely worsen this further — bounded updates with a zero-init module gives even less mass to flow into.

**Conclusion:** Net negative composition. Skip mechanism is real but doesn't survive better optimization. **Closed.** thorfinn reassigned to a new hypothesis.

**Bonus from this PR:** Student diagnosed the GT-NaN propagation bug in `data/scoring.py` independently in this PR. That diagnosis became the basis for #1564 (merged) which fixed it train.py-side.

---

## 2026-05-12 22:25 — PR #1656: Dropout=0.1 in attention + MLP (assigned to thorfinn)

- Student branch: `charliepai2g24h5-thorfinn/dropout-0_1`
- Hypothesis: The merged stack uses dropout=0.0 everywhere. With only weight_decay=1e-4 and grad_clip(max_norm=1.0) regularizing the gradients but NO forward-pass feature noise, the model may overfit on the small dataset. Adding dropout=0.1 to attention output + MLP is the classic transformer regularization knob and is orthogonal to all in-flight experiments. Expected 1–4% improvement, especially on OOD splits (val_geom_camber_rc, val_re_rand).
- Status: WIP, assigned.
