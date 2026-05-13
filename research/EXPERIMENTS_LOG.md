# SENPAI Research Results — `icml-appendix-charlie-pai2g-24h-r4`

This log records every PR review on this advisor branch with the
hypothesis, the metrics pulled from the committed JSONL, and a short
commentary.

Entries are appended chronologically (newest at top). The metric of
record for ranking is `val_avg/mae_surf_p`; the paper-facing comparison
metric is `test_avg/mae_surf_p`.

## 2026-05-13 17:05 — PR #2370 (frieren learned-freqs-no-wd-10x-lr) — **MERGED** (17th compound win, −3.73% val)

- Branch: `charliepai2g24h4-frieren/learned-freqs-no-wd-10x-lr`
- Hypothesis: Learned Fourier frequencies (6 params) with no WD + 10× lr. Default WD=1e-4 + lr=5e-4 was keeping freqs stuck near dyadic init.
- Metric artifact: `models/model-charliepai2g24h4-frieren-learned-freqs-no-wd-10x-lr-20260513-151314/metrics.jsonl`

| Split | New baseline (#2370) | Prior baseline (#2360) | Δ vs prior |
|---|---:|---:|---:|
| val_single_in_dist | 70.235 | 67.276 | +4.39% (slight regression) |
| val_geom_camber_rc | 71.466 | 72.143 | **−0.94%** |
| val_geom_camber_cruise | **39.185** | 45.901 | **−14.62%** |
| val_re_rand | **57.372** | 62.181 | **−7.73%** |
| **val_avg** | **59.5645** | **61.875** | **−3.73%** |
| test_avg | **51.6141** | 54.117 | **−4.63%** |

Best epoch: 12 (timeout). n_params: 831,197 (+6 freqs). sec/epoch: ~150s.

**Analysis:** Bottom 3 freqs [1,2,4] drifted to [0.75, 1.46, 3.44] (−15–27% off init); top 3 [8,16,32] stayed within ±1.5% (gradient-magnitude limited). The mechanism is real: camber_cruise −14.62% and re_rand −7.73% show the adapted low-frequency basis strongly benefits the dominant spatial pressure gradients. single_in_dist slight regression (+4.39%) is the only cost — high-freq features important for boundary layer details may be slightly de-emphasized.

Student's key insight: "The top freqs (8, 16, 32) remain pinned even at 10× lr. Their gradient signal really is much smaller." This confirms the gradient-magnitude limitation hypothesis.

**Compound progress: 17 merges, 100.957 → 59.5645 = −41.0%**

Closed simultaneously:
- #2385 thorfinn AbsGLU (Outcome C, +10.35%): gate axis confirmed fully closed at ReGLU
- #2386 fern inner_dim=320 (Outcome C, +6.00%): inner_dim axis confirmed closed at 288
- #2391 nezuko OOD-upsampling (Outcome C, +12.45%): camber_rc is extrapolation gap, not data density gap

New assignments: #2434 frieren freq-init-equilibrium, #2435 thorfinn learned-freqs-50x-lr, #2436 fern layerscale-lr-10x, #2437 nezuko slice-temp-lr-10x

---

## 2026-05-13 16:55 — PR #2377 (tanjiro qk-norm-attention v1) — **CLOSED** (Outcome B vs old baseline / +1.86% vs new; init confound identified, retry queued)

- Branch: `charliepai2g24h4-tanjiro/qk-norm-attention`
- Hypothesis: QK-normalization (unit-norm Q,K + per-head learnable log temperature). Predicted: faster convergence, smaller/stabler grad-norm, better OOD generalization.
- Metric artifact: `models/model-charliepai2g24h4-tanjiro-qk-norm-attention-20260513-151354/metrics.jsonl`

| Split | ReGLU baseline (#2304 62.949) | New baseline (61.875) | QK-norm v1 | Δ vs new |
|---|---:|---:|---:|---:|
| val_single_in_dist | 69.925 | 67.276 | 70.943 | +5.45% |
| val_geom_camber_rc | 74.845 | 72.143 | 74.855 | +3.76% |
| val_geom_camber_cruise | 44.262 | 45.901 | 44.423 | −3.22% |
| val_re_rand | 62.765 | 62.181 | 61.879 | −0.49% |
| **val_avg** | 62.949 | **61.875** | **63.025** | **+1.86%** |
| test_avg | 54.221 | 54.117 | **53.893** | −0.41% (mild test gain) |

Best epoch: 12 (timeout). n_params: 831,211 (+20 over baseline). Peak GPU: 52.14 GB.

**Mechanism diagnosis (student's, excellent):** Init `log_temp = log(1/√d_k) = -1.733` → max attention logit = exp(-1.733) = 0.177 over 64 slice tokens → **softmax near-uniform at init**. Per-block log_temp moved at most ~17% from init across training (block 4 essentially frozen, std=0.005). The mechanism never activated because the cold init starved the gradient signal on tau, and default WD=1e-4 on tau pulled it further toward init. **This was an init failure, not a mechanism failure.**

Grad-norm hypothesis falsified: ep 1 was smaller (79 vs 134 ReGLU), but ep 12 was LARGER (101 vs 27) and the trace was more volatile (range 13–101) — exactly what one would expect when softmax is too soft to provide useful credit assignment.

Test gain (−0.41%) and re_rand val improvement (−0.49%) are real signals — QK-norm has some benefit even with the broken init. Strongly motivates retry.

**Reassigning tanjiro to QK-norm v2 (qk-norm-temp-init-0)** — init `log_temp = 0` (qk_scale=1, max logit≈1, matches ViT-22B/DiT/SD3 standard); exclude `qk_log_temperature` from weight decay (literature recommendation).

The needs_rebase state on this branch is moot since v2 starts fresh from advisor HEAD post-#2360.

---

## 2026-05-13 16:35 — PR #2371 (alphonse n-hidden-144) — **CLOSED** (Outcome C; n_hidden axis closed at 128)

- Branch: `charliepai2g24h4-alphonse/n-hidden-144`
- Hypothesis: Residual-stream width 128→144 unlocks additional capacity for OOD-camber/Re regimes. Predicted +12.5% n_params; head_dim 32→36.
- Metric artifact: `models/model-charliepai2g24h4-alphonse-n-hidden-144-20260513-145259/metrics.jsonl`

| Split | ReGLU baseline (#2304 62.949) | n_hidden=144 | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 69.925 | 85.345 | +22.05% |
| val_geom_camber_rc | 74.845 | 84.980 | +13.54% |
| val_geom_camber_cruise | 44.262 | 57.293 | +29.44% |
| val_re_rand | 62.765 | 72.468 | +15.46% |
| **val_avg (primary)** | **62.949** | **75.022** | **+19.18%** |
| test_avg | 54.221 | 66.627 | +22.88% |

Best epoch: 9 / ~9 trained (timeout-truncated, 3 epochs lost); n_params: 1,047,799 (+26.07%, far higher than predicted 935k); sec/epoch: ~183s median + 2 outlier epochs at 332/247s.

**Analysis:** Quadratic param scaling reality (QKV proj, output proj, MLP all scale as n_hidden² when inner_dim=2·n_hidden) drove n_params +26% vs predicted +12.5%. (128/144)² = 1.266 matches observation. The wider model lost 3 effective epochs AND requires more steps per epoch to converge — under the 30-min cap, the comparison was unwinnable. val_avg trajectory still descending steeply at ep 9 (−10% between ep 8→9), confirming severe under-training.

All 4 splits regressed (val +13% to +29%, test +18% to +30%) — no OOD-specific benefit emerges; if anything the wider net struggles MORE on previously-saturated splits (camber_cruise +29.4%).

**n_hidden axis permanently closed at 128.** Width is the wrong knob to pull under 30-min/run — fern's #2360/#2386 MLP-only inner_dim path is the right capacity axis since it leaves attention and the residual stream alone.

**Reassigning alphonse to dual LayerScale init** — separate the attention pathway γ (test 0.05) from the MLP pathway γ (keep 0.025). Compute-neutral 1-line change.

---

## 2026-05-13 16:25 — PR #2361 (nezuko stoch-depth-0.05-reglu) — **CLOSED** (Outcome B; stoch-depth axis confirmed at 0.10)

- Branch: `charliepai2g24h4-nezuko/stoch-depth-0.05-reglu`
- Hypothesis: Reduce max stoch-depth from 0.10 to 0.05 — ReGLU's exact-zero sparsity may reduce the need for explicit stochastic depth regularization.
- Metric artifact: `models/model-charliepai2g24h4-nezuko-stoch-depth-0.05-reglu-20260513-142034/metrics.jsonl`

| Split | ReGLU baseline (#2304 62.949/54.221) | Stoch-depth 0.05 | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 69.925 | **69.086** | **−1.20%** |
| val_geom_camber_rc | 74.845 | 75.971 | +1.50% |
| val_geom_camber_cruise | 44.262 | **43.996** | −0.60% |
| val_re_rand | 62.765 | 63.207 | +0.70% |
| **val_avg (primary)** | **62.949** | 63.065 | **+0.18%** (wash) |
| test_avg | 54.221 | **53.930** | **−0.54%** (mild test improvement) |

Best epoch: 12, n_params: 831,191 (unchanged).

**Analysis:** Stoch-depth 0.05 max shows the canonical tradeoff: lighter drop improves in-dist (single_in_dist −1.20%) but worsens camber_rc (+1.50%) — exactly what stoch-depth was designed to prevent (OOD overfitting). The cross-split effects roughly cancel. **Stoch-depth axis confirmed closed at 0.10 under ReGLU.** ReGLU's sparsity is NOT a substitute for stoch-depth regularization on this architecture/dataset.

Test avg mildly improved (−0.54%) but val gating metric +0.18% → does not meet Outcome A. The val→test improvement is noted but insufficient to merge at below-baseline val.

Vs new inner_dim=288 baseline (61.875): +1.92% worse.

**Reassigning nezuko to OOD-upweighted sampling** — the camber_rc and re_rand splits dominate the val error; 2.5× and 2× sampling weights target the bottleneck directly.

---

## 2026-05-13 16:05 — PR #2360 (fern reglu-inner-dim-288) — **MERGED** (16th compound win)

- Branch: `charliepai2g24h4-fern/reglu-inner-dim-288`
- Hypothesis: Bisect inner_dim between 256 (ReGLU baseline) and 320 (compute-bound Outcome B on GeGLU). At +4.7% sec/epoch, inner_dim=288 stays in 12-epoch window. ReGLU's exact-zero gate creates dead channels; extra width compensates.
- Metric artifact: `models/model-charliepai2g24h4-fern-reglu-inner-dim-288-20260513-141957/metrics.jsonl`

| Split | ReGLU baseline (#2304) | inner_dim=288 | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 69.925 | **67.276** | **−3.79%** |
| val_geom_camber_rc | 74.845 | **72.143** | **−3.61%** |
| val_geom_camber_cruise | 44.262 | 45.901 | +3.70% (lone regression) |
| val_re_rand | 62.765 | **62.181** | **−0.93%** |
| **val_avg (primary)** | **62.949** | **61.875** | **−1.71%** |
| test_single_in_dist | 61.108 | **60.873** | −0.38% |
| test_geom_camber_rc | 66.196 | **65.103** | **−1.65%** |
| test_geom_camber_cruise | 36.305 | 37.112 | +2.22% |
| test_re_rand | 53.276 | 53.380 | +0.20% |
| **test_avg** | **54.221** | **54.117** | **−0.19%** |

Best epoch: 12 (timeout-truncated, descending). n_params: 892,631 (+61,440). sec/epoch: +4.7%.

**Analysis:** Epoch-budget hypothesis confirmed — val_single_in_dist IMPROVES −3.79% (vs +11.2% regression at 320), proving that 320's regression was schedule truncation (11 epochs), not capacity overfit. At +4.7% sec/epoch, 288 preserves the 12-epoch window. val_geom_camber_cruise slight regression (+3.70%) is the easiest split, already at 44.3, likely saturated. 3/4 val splits and 2/4 test splits improve — clean Outcome A.

**New baseline**: val=61.875, test=54.117. Compound progress: 16 merges, **100.957 → 61.875 = −38.7%**.

**Natural follow-up**: inner_dim=320 on ReGLU — fern's own suggestion. At ReGLU's ~+10-12% sec/epoch vs GeGLU's +10.2%, 320 might now stay in 12-epoch window (ReGLU is slightly faster than GELU due to simpler gate computation). Assigning fern to this.

---

## 2026-05-13 16:05 — PR #2359 (thorfinn squared-relu-gate) — **CLOSED** (Outcome C; gate axis confirmed closed at ReLU)

- Branch: `charliepai2g24h4-thorfinn/squared-relu-gate`
- Hypothesis: `F.relu(x)^2` gate — Primer (So et al. 2021) — test whether gate-sharpness monotonicity extends past ReLU.
- Metric artifact: `models/model-charliepai2g24h4-thorfinn-squared-relu-gate-20260513-141913/metrics.jsonl`

| Split | ReGLU baseline (#2304) | Squared ReLU | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 69.925 | 79.593 | **+13.83%** (catastrophic in-dist) |
| val_geom_camber_rc | 74.845 | 77.313 | +3.30% |
| val_geom_camber_cruise | 44.262 | 45.360 | +2.48% |
| val_re_rand | 62.765 | 62.590 | −0.28% (flat) |
| **val_avg (primary)** | **62.949** | 66.214 | **+5.19%** |
| **test_avg** | **54.221** | 56.870 | +4.89% |

Best epoch: 12 (timeout). n_params: 831,191 (unchanged). Pre-clip grad-norm peaked at 120.6 (ep4), vs clip cap 25.

**Analysis:** Gate-sharpness monotonicity ends at ReLU. The SiLU→GELU→ReLU wins were all about hardening the negative-input threshold (sparser gating); Squared ReLU is different — it leaves sparsity identical to ReLU (same dead zone) but *amplifies* positive-input channels quadratically (x>1 grows, x<0.1 shrinks). In the CFD regime: high-magnitude pressure features (stagnation/wake) have the largest x values; quadratic amplification explodes their gradient, saturates grad-clip for the first half of training, and catastrophically overfits in-distribution. The student's analysis is excellent. LayerScale asymmetry (γ_attn shrinks to 0.015-0.021 from 0.025 init, γ_mlp grows to 0.037-0.044) shows the optimizer fighting the gate dynamics.

**Gate axis permanently closed at ReLU**: any gate that breaks the identity-slope-for-positive invariant (Squared, Cube, etc.) will have this failure mode. ReGLU = `max(0, x)` is exactly right: maximum sparsity + identity slope for survivors.

---

## 2026-05-13 15:55 — PR #2281 (tanjiro swiglu-inner-dim-320) — **CLOSED** (Outcome A vs old baseline, C vs new ReGLU baseline)

- Branch: `charliepai2g24h4-tanjiro/swiglu-inner-dim-320`
- Hypothesis: Bisect between inner_dim=256 (#2175 won) and inner_dim=384 (#2200 closed) at 320, expecting +10-11% compute cost (vs 384's +21%) preserving enough epochs for cosine schedule completion.
- Metric artifact: `models/model-charliepai2g24h4-tanjiro-swiglu-inner-dim-320-20260513-135742/metrics.jsonl`

| Metric | Old SwiGLU baseline #2175 (67.381/57.800) | inner_dim=320 (SwiGLU) | New ReGLU baseline #2304 (62.949/54.221) |
|---|---:|---:|---:|
| val_avg | 67.381 | **66.876 (−0.75%)** | 62.949 (**+6.24% worse**) |
| test_avg | 57.800 | **57.400 (−0.69%)** | 54.221 (+5.86% worse) |
| n_params | 831,191 | 954,071 (+14.8%) | 831,191 |
| sec/epoch | ~138 | ~161 (+16.7%) | ~150 |
| best_epoch | 13 | 12 (timeout) | 12 (timeout) |
| val_geom_camber_rc | 80.673 | 76.952 (−4.61%) | 74.845 |
| val_re_rand | 66.834 | 65.010 (−2.73%) | 62.765 |

**Mechanism finding (real and useful):** The bisect hypothesis confirmed — inner_dim=320 wins over 256 on SwiGLU stack because it stays just inside the schedule-completeness threshold (12 epochs at +16.7% sec/epoch vs 384's 11 epochs at +21%). val_geom_camber_rc and val_re_rand show the OOD capacity gain that 384 only partially captured. Student's analysis was excellent.

**Why this can't merge:** PR #2304 (ReGLU) merged on the advisor branch while this experiment ran on pre-ReGLU SwiGLU. The ReGLU gate-sharpness change produced −4.75% val gain on the same width axis — far outweighing this experiment's +1.94% from widening alone. Absolute numbers are obsolete; mechanism extrapolates if ReGLU+inner_dim=320 is tested later.

**Conclusion:** Outcome C in absolute terms. fern's #2360 (ReGLU + inner_dim=288) will give the direct comparable measurement on the current stack. If 288 wins on ReGLU, a future ReGLU+inner_dim=320 test would be the natural next step. Reassigning tanjiro to **QK-normalization on PhysicsAttention** — a fresh untested architectural axis (DiT/ViT-22B/SD3 standard) orthogonal to all in-flight Wave 16 experiments.

---

## 2026-05-13 15:40 — PR #2308 (alphonse cosine-tmax-12) — **CLOSED** (Outcome C; schedule axis confirmed closed at T_max=14)

- Branch: `charliepai2g24h4-alphonse/cosine-tmax-12`
- Hypothesis: Reduce cosine T_max from 14→12 epochs to land LR≈0 at epoch 12-13 (the actual 30-min compute budget on the ReGLU stack).
- Metric artifact: `models/model-charliepai2g24h4-alphonse-cosine-tmax-12-20260513-132330/metrics.jsonl`

| Split | ReGLU baseline (#2304 62.949/54.221) | T_max=12 | Δ vs new baseline |
|---|---:|---:|---:|
| val_single_in_dist | 69.925 | 71.714 | +2.56% |
| val_geom_camber_rc | 74.845 | 77.031 | +2.92% |
| val_geom_camber_cruise | 44.262 | 46.374 | +4.77% |
| val_re_rand | 62.765 | 64.843 | +3.31% |
| **val_avg (primary)** | **62.949** | 64.991 | **+3.24%** |
| test_avg | 54.221 | 56.451 | +4.11% |

Best epoch 12 (timeout-truncated at 30 min; baseline got 13 epochs at ~138s/epoch, alphonse got 12 at ~151s/epoch).

**Mechanism analysis (student's analysis, correct):** "Compressing the cosine schedule to T_max=12 would let the model do a near-zero LR fine-tuning final epoch ... Instead of adding a low-LR fine-tuning epoch we removed the moderate-LR descent epoch the baseline used. Net: model is slightly less converged on val." Per-epoch runtime jitter (138→151 s/epoch, ~10% slowdown) caused the schedule to miss its design point; T_max=14 absorbs this jitter gracefully while T_max=12 does not.

**Per-split pattern is informative**: 3/4 test splits actually IMPROVED vs baseline (camber_rc −1.6%, camber_cruise −3.1%, re_rand −1.5%), so the schedule isn't catastrophically wrong. But val_single_in_dist regressed (+5.6% vs old baseline) and re_rand was tied, dragging the average.

**Conclusion:** Schedule axis confirmed closed at T_max=14. The student's diagnosis is sound — runtime jitter robustness matters as much as design-point optimization. Reassigning alphonse to **n_hidden=144 width bump** (only unexplored capacity axis: depth and slice_num closed, inner_dim is fern's #2360 in-flight).

---

## 2026-05-13 15:30 — PR #2312 (frieren learned-fourier-freqs) — **CLOSED** (borderline B; under-trained freqs)

- Branch: `charliepai2g24h4-frieren/learned-fourier-freqs`
- Hypothesis: Make `FourierCoordEnc.freqs` an `nn.Parameter` (dyadic init 1,2,4,8,16,32) so the model can discover its own spectral basis instead of using fixed dyadic frequencies.
- Metric artifact: `results/icml-appendix-charlie-pai2g-24h-r4_charliepai2g24h4-frieren_learned-fourier-freqs.jsonl`

| Split | ReGLU baseline (#2304 62.949/54.221) | Learned freqs | Δ vs new baseline |
|---|---:|---:|---:|
| val_single_in_dist | 69.925 | 73.665 | +5.35% (regression) |
| val_geom_camber_rc | 74.845 | 72.751 | **−2.80%** |
| val_geom_camber_cruise | 44.262 | **42.121** | **−4.84%** (best on board) |
| val_re_rand | 62.765 | 61.222 | **−2.46%** |
| **val_avg (primary)** | **62.949** | 63.580 | **+1.00%** (borderline) |
| **test_avg** | **54.221** | 55.389 | +2.15% |

Best epoch 12 (timeout-truncated, run still descending). n_params: 831,197 (+6).

**Mechanism analysis:** Learned freqs barely moved — bottom freqs converged to (1.084, 1.819, 3.868) from dyadic init (1, 2, 4); top freqs essentially fixed at (7.992, 15.996, 31.983) from init (8, 16, 32). Only the bottom 3 freqs received meaningful gradient signal; high-freq freqs stayed pinned. AdamW with default wd=1e-4 + lr=5e-4 + post-step `clamp(0.1, 100)` effectively froze the freqs at init. Split pattern is informative: 3/4 OOD splits improved (camber_cruise hit a new best 42.121), but in-dist regressed +5.35% and dragged the average.

**Conclusion:** Mechanism shows real signal — spectral adaptation moves OOD-cruise. The blocker is under-training of the 6-parameter freq vector. Follow-up assigning to frieren: per-block independent learned freqs (5 blocks × 6 = 30 params) for finer spectral carving, OR no-wd param group + 10× lr multiplier on freqs to actually let them explore. The student's diagnosis was correct.

---

## 2026-05-13 15:30 — PR #2309 (edward hybrid-fourier-dyadic-rff) — **CLOSED** (Outcome C; σ=1.0 redundant)

- Branch: `charliepai2g24h4-edward/hybrid-fourier-dyadic-rff`
- Hypothesis: Concatenate dyadic L=6 + Gaussian RFF m=6 σ=1.0 Fourier encoders to combine high-freq pressure structure (dyadic) with smooth OOD coverage (RFF).
- Metric artifact: `results/icml-appendix-charlie-pai2g-24h-r4_charliepai2g24h4-edward_hybrid-fourier-dyadic-rff.jsonl`

| Split | ReGLU baseline (#2304) | Hybrid σ=1.0 | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 69.925 | 78.504 | +12.27% (large regression) |
| val_geom_camber_rc | 74.845 | 78.012 | +4.23% |
| val_geom_camber_cruise | 44.262 | 44.766 | +1.14% (NOT the −7.9% from #2225) |
| val_re_rand | 62.765 | 67.022 | +6.78% |
| **val_avg (primary)** | **62.949** | 68.076 | **+8.14%** |
| **test_avg** | **54.221** | 57.878 | +6.75% |

Best epoch 12 (~9% slower per epoch). n_params: 834,263 (+3,072).

**Mechanism analysis:** σ=1.0 RFF generates frequencies clustered around the LOW end of dyadic's already-covered range (dyadic spans π to 32π ≈ 3.14 to 100.5; σ=1.0 RFF median freq ≈ 6.28). Net effect: capacity dilution — preprocess MLP input grew from 46-d to 58-d (+26%), wasting parameters learning to ignore redundant low-freq features rather than capturing complementary structure. The OOD-cruise win from RFF-only #2225 (at σ=3.0) did NOT transfer — σ=3.0 was the actual high-freq complement that worked.

**Conclusion:** The hybrid was correct in principle but used the wrong σ. The student's analysis identified the issue: "σ=3.0 hybrid would directly test whether high-freq RFF (not low-freq) is the actual OOD-cruise ingredient." Follow-up assigning to edward: hybrid retest with σ=3.0.

---

## 2026-05-13 15:30 — PR #2286 (askeladd flow-cond-fourier-re-aoa) — **CLOSED** (Outcome C; class falsified)

- Branch: `charliepai2g24h4-askeladd/flow-cond-fourier-re-aoa`
- Hypothesis: Apply Fourier features `[sin(2^k π x), cos(2^k π x)] for k=0..3` to log_Re, AoA0, AoA1 (3 per-sample scalar dims); fun_dim 44→56.
- Metric artifact: `results/icml-appendix-charlie-pai2g-24h-r4_charliepai2g24h4-askeladd_flow-cond-fourier-re-aoa.jsonl`

| Metric | Branch-point baseline (#2175 67.381) | Flow-cond Fourier | Δ vs new ReGLU baseline (62.949) |
|---|---:|---:|---:|
| val_avg | 67.381 | 70.172 | **+11.47%** |
| test_avg | 57.800 | 60.988 | +12.49% |
| val_re_rand | — | (worst hit) | +13.84% |

All splits regressed. n_params: 834,263 (+3,072).

**Mechanism analysis (quoting student's correct diagnosis):** "Spatial Fourier decomposes a signal that varies across nodes — sin(2^k π · x_node) generates progressively higher-frequency modes that capture spatial detail. Flow-cond Fourier operates on per-sample constants: every node in a sample sees the same log_Re, the same AoA. sin(2^k π · AoA) is just k constant features per sample. There is no spectral structure to decompose — only redundant rescalings of a single scalar."

**Conclusion:** Class of `(sin/cos)(α·scalar)` features for per-sample condition scalars is fundamentally mismatched — Fourier features are useful when there is per-node variation along a coordinate axis, not for global conditioning. **Permanently closing this axis (any variant of per-sample scalar Fourier).** Follow-up assigning to askeladd: FiLM-style γ/β = MLP(condition_scalars) applied to TransolverBlock activations — the correct mechanism for global conditioning.

---

## 2026-05-13 14:10 — PR #2304 (thorfinn reglu-gate) — **MERGED** (15th compound win)

- Branch: `charliepai2g24h4-thorfinn/reglu-gate`
- Hypothesis: Replace GELU with ReLU in the GLU gate. Closes gate-sharpness monotonicity axis (SiLU<GELU<ReLU). Zero param cost.
- Metric artifacts: `models/model-charliepai2g24h4-thorfinn-reglu-gate-20260513-132007/metrics.jsonl`

| Split | Baseline (#2266 GeGLU 64.182/56.523) | ReGLU | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 67.894 | 69.925 | +2.04% (lone regression) |
| val_geom_camber_rc | 76.235 | **74.845** | **−1.83%** |
| val_geom_camber_cruise | 47.790 | **44.262** | **−7.38%** |
| val_re_rand | 64.808 | **62.765** | **−3.15%** |
| **val_avg (primary)** | **64.182** | **62.949** | **−1.92%** |
| test_single_in_dist | 60.676 | 61.108 | +0.71% (near-neutral) |
| test_geom_camber_rc | 70.778 | **66.196** | **−6.46%** |
| test_geom_camber_cruise | 39.001 | **36.305** | **−6.91%** |
| test_re_rand | 55.636 | **53.276** | **−4.24%** |
| **test_avg** | **56.523** | **54.221** | **−4.07%** |

Best epoch: 12 (hit 30-min timeout, val still descending 76.4→71.5→68.2→62.9 across last 4 epochs). n_params: 831,191 (unchanged). Grad-norm trace: 133.8→47.1→37.5→27.3 (clean, no instability). LayerScale γ_attn means 0.017-0.024, γ_mlp 0.042-0.057 (in normal ranges).

**Analysis:** Gate-sharpness monotonicity hypothesis fully confirmed. Each step wins: SiLU (−6.96% val), GELU (−4.75%), ReLU (−1.92%). The OOD gain pattern is striking: test improves −4.07% vs val −1.92%, and OOD splits gain the most (test_camber_rc −6.46%, test_camber_cruise −6.91%). ReLU's exact-zero gate provides maximum cross-channel contamination suppression at stagnation/wake features. The hard zero also acts as implicit regularization, explaining why OOD generalization benefits more than in-dist. The run was cut at ep 12 with val still strongly descending — full potential likely 1-2% higher with more compute.

**Compound progress:** 15 merges, **100.957 → 62.949 = −37.7%**

**Gate-sharpness axis status:** Confirmed monotonic through SiLU<GELU<ReLU. Next test: Squared ReLU (F.relu²) — Primer (So et al. 2021). If that wins too, the axis is still open. If not, ReLU is the gate optimum.

---

## 2026-05-13 14:10 — PR #2306 (fern geglu-inner-dim-320) — **CLOSED** (B outcome; compute-bound; below new ReGLU baseline)

- Branch: `charliepai2g24h4-fern/geglu-inner-dim-320`
- Hypothesis: Inner_dim=320 (from 256) with GeGLU stack — test capacity expansion.

| Metric | GeGLU baseline (#2266) | GeGLU inner_dim=320 | vs ReGLU baseline (#2304) |
|---|---:|---:|---:|
| val_avg | 64.182 | 64.286 (+0.16%) | 62.949 (**+2.13%**) |
| test_avg | 56.523 | 55.965 (−0.99%) | 54.221 (+3.24%) |
| best_epoch | 13 | 12 (timeout) | — |

3/4 val splits improve (camber_rc −4.1%, camber_cruise −3.5%, re_rand −3.7%), but val_single_in_dist regresses +11.2%. Model capacity-limited at ep 12 (every epoch was a new best). Compute overhead +10.2%/epoch.

**Analysis:** Capacity expansion trades in-dist precision for OOD performance. The OOD signal is real (val_geom_camber_rc −4.08%, confirming PR #2200's signature). But val_single_in_dist regression (+11.2%) dominates the average. With ReGLU now merged (much better than GeGLU), GeGLU-320 is obsolete. **Closed against new ReGLU-256 baseline. Next: test ReGLU+inner_dim=288 (less compute-heavy step).**

---

## 2026-05-13 14:10 — PR #2310 (nezuko lr-bracket-up-7e-4) — **CLOSED** (B outcome; LR axis exhausted)

- Branch: `charliepai2g24h4-nezuko/lr-bracket-up-7e-4`
- Hypothesis: Peak lr=7e-4 (upper bracket from lr=5e-4).

| Metric | GeGLU baseline (#2266) | lr=7e-4 | vs ReGLU baseline (#2304) |
|---|---:|---:|---:|
| val_avg | 64.182 | 63.628 (−0.86%) | 62.949 (**+1.07%**) |
| test_avg | 56.523 | 56.278 (−0.43%) | 54.221 (+3.78%) |

Mixed split pattern: camber_cruise −7.5% (strong OOD win), single_in_dist +3.1% (regression). Best epoch 12 (timeout).

**Analysis:** Marginal near-tie on GeGLU stack, clearly worse than new ReGLU baseline. LR axis bracket complete: 3e-4=+5.95% (closed), 5e-4=optimal (baseline), 7e-4=marginal near-tie. The per-split pattern (OOD improves, in-dist regresses) mirrors ReGLU's own pattern — but at a weaker magnitude. **LR axis closed at lr=5e-4.**

---

## 2026-05-13 14:00 — PR #2266 (thorfinn geglu-gate-comparison) — **MERGED** (14th compound win)

- Branch: `charliepai2g24h4-thorfinn/geglu-gate-comparison`
- Hypothesis: Replace SiLU with GELU in the GLU gate: `F.silu → F.gelu` in `SwiGLUMLP.forward()`. GELU's harder switch in the negative-input regime should suppress cross-channel contamination at high-magnitude pressure features. Zero parameter cost.
- Metric artifacts: `models/model-charliepai2g24h4-thorfinn-geglu-gate-comparison-20260513-121756/metrics.jsonl`

| Split | Baseline (#2175 67.381/57.800) | GeGLU | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 73.341 | **67.894** | **−7.43%** |
| val_geom_camber_rc | 80.673 | **76.235** | **−5.50%** |
| val_geom_camber_cruise | 48.675 | **47.790** | **−1.82%** |
| val_re_rand | 66.834 | **64.808** | **−3.03%** |
| **val_avg (primary)** | **67.381** | **64.182** | **−4.75%** |
| test_single_in_dist | 64.685 | **60.676** | **−6.20%** |
| test_geom_camber_rc | 69.035 | 70.778 | +2.53% (lone regression) |
| test_geom_camber_cruise | 40.356 | **39.001** | **−3.36%** |
| test_re_rand | 57.121 | **55.636** | **−2.60%** |
| **test_avg** | **57.800** | **56.523** | **−2.21%** |

Best epoch: 13 (identical to baseline). n_params: 831,191 (unchanged). Peak GPU: 57.1 GB.

**Analysis:** Largest single-PR gain from a 1-character change in the programme. All 4 val splits beat baseline; 3 of 4 test splits beat baseline. The lone test_geom_camber_rc regression (+2.53%) is attributed to val/test camber_rc decorrelation rather than a GeGLU artifact (val_geom_camber_rc improved strongly −5.50%). The student's grad-norm trace is clean — GELU's harder gate did not destabilize early training.

**Mechanism:** GELU < SiLU for x < 0, suppressing the gate path in negative-valued features. Under L1 + surf-ch-weight [0.5,0.5,2.0], gradient signal is dominated by extreme pressure values at stagnation points and wakes — precisely where the gate sharpness matters most. Per-channel pressure gains (mae_surf_p) exceed velocity gains, exactly matching the mechanism prediction. **Gate activation axis open — ReGLU (ReLU gate) is the natural next test to close the monotonicity question.**

**New compound progress:** 14 merges, **100.957 → 64.182 = −36.4%**

---

## 2026-05-13 14:00 — PR #2262 (alphonse slice-num-96) — **CLOSED** (representation dilution)

- Branch: `charliepai2g24h4-alphonse/slice-num-96`
- Hypothesis: slice_num=64→96: increase Transolver physics-slice token capacity (+50% inter-token states).

| metric | Baseline (#2175) | slice_num=96 | Δ |
|---|---:|---:|---:|
| **val_avg** | 67.381 | 73.322 | **+8.82%** |
| **test_avg** | 57.800 | 64.110 | **+10.92%** |
| val_single_in_dist | 73.341 | 86.259 | +17.61% |
| val_geom_camber_rc | 80.673 | 82.265 | +1.97% |

Best epoch: 11. Sec/epoch: 169.5 vs 137 (+23.7%). n_params: 836,471 (+5,280 — note: in_project_slice is single Linear per layer, not per-head).

**Analysis:** Striking inversion of the prediction — val_single_in_dist (easiest, in-dist) is hit hardest (+17.6%), while val_geom_camber_rc (hardest OOD) barely moves (+2.0%). The hypothesis predicted OOD splits would benefit most. The actual pattern is representation dilution: at 831k params / 5 layers / 1499 train samples, 96 slices over-partitions the representation. Extra slices dilute per-slice gradient signal across more slots than the model can usefully populate. **Slice_num axis exhausted: 64 is the operating point at this model scale.**

---

## 2026-05-13 14:00 — PR #2244 (fern n-layers-6-depth) — **CLOSED** (compute-bound overfitting)

- Branch: `charliepai2g24h4-fern/n-layers-6-depth`
- Hypothesis: n_layers=5→6: depth increase on SwiGLU stack.

| metric | Baseline (#2175) | n_layers=6 | Δ |
|---|---:|---:|---:|
| **val_avg** | 67.381 | 71.039 | **+5.43%** |
| **test_avg** | 57.800 | 62.755 | **+8.57%** |
| val_single_in_dist | 73.341 | 82.117 | +11.97% |
| val_geom_camber_rc | 80.673 | 80.601 | **−0.09%** (tie) |
| val_geom_camber_cruise | 48.675 | 52.620 | +8.10% |
| val_re_rand | 66.834 | 68.817 | +2.97% |

Best epoch: 11 (vs 13 baseline). Sec/epoch: 178 vs ~145 (+22.8%). n_params: 984,987 (+153,796 / +18.5%). Peak GPU: 67.57 GB. Trajectory still descending at ep 11 cap (−4.8 from ep10).

**Notable:** Block-5 (new) has the largest attn γ of all blocks (0.0247), meaning the model actively uses the new layer — yet generalization degrades. Classic capacity-vs-epoch-budget overfitting on 1499 train samples. Two compounding effects: per-epoch compute grew +23% → 2 fewer epochs; and model has more parameters than it can fit in this epoch/data budget. **Depth axis exhausted: n_layers=5 is sweet-spot at this data scale and 30-min budget.**

---

## 2026-05-13 14:00 — PR #2239 (nezuko lr-3e-4-bracket-post-swiglu) — **CLOSED** (schedule mismatch)

- Branch: `charliepai2g24h4-nezuko/lr-bracket-down-3e-4-post-swiglu`
- Hypothesis: Peak lr=5e-4→3e-4 on post-SwiGLU stack (SwiGLU gating may prefer smaller lr).

| metric | Baseline (#2175) | lr=3e-4 | Δ |
|---|---:|---:|---:|
| **val_avg** | 67.381 | 71.335 | **+5.95%** |
| **test_avg** | 57.800 | 62.055 | **+7.36%** |
| val_single_in_dist | 73.341 | 84.437 | +15.10% |

Best epoch: 12 (still improving at cap). Ep5 grad norm: 115.3 (counter-hypothesis — no suppression).

**Analysis:** lr=3e-4 simply can't complete enough cosine schedule descent in 30 min. LR at ep12 bottom: 3.27e-05 — model hasn't descended far enough. The warmup+cosine T_max=14 already provides the early-training stability the hypothesis was reaching for. Grad norm trace disproves the mechanism: ep5 norm=115.3 with lr=3e-4, not suppressed. **LR axis: 5e-4 confirmed optimal on the downward side. Upper bracket (7e-4) still untested.**

---

## 2026-05-13 14:00 — PR #2225 (edward gaussian-rff-sigma-calibrated) — **CLOSED** (low-pass filter, in-dist regression)

- Branch: `charliepai2g24h4-edward/gaussian-rff-sigma-calibrated`
- Hypothesis: Calibrated Gaussian RFF σ∈{1.0, 3.0} to match std-normalized coordinate scale. Two-arm comparison vs dyadic L=6.

| arm | val_avg | test_avg | vs baseline |
|---|---:|---:|---:|
| Arm A σ=1.0 | 70.283 | 62.129 | **+4.31% / +7.49%** |
| Arm B σ=3.0 | 67.939 | 58.954 | **+0.83% / +2.00%** |
| Baseline (dyadic L=6) | 67.381 | 57.800 | — |

Best epoch both arms: 12 (vs 13 baseline, 1 epoch shorter). Arm B still descending at cap (Δ=−2.93 ep11→12).

**Key split pattern:** Both arms improve val_geom_camber_cruise and val_re_rand vs baseline, but lose hard on val_single_in_dist (+13.1 for A, +5.6 for B). Dyadic's broadband octave structure covers up to 32π ≈ 100 rad/unit; 12 random frequencies from N(0,9·I) reach max ~5-7 rad/unit, missing the high-freq pressure structure near leading/trailing edges.

**Mechanism confirmed:** Gaussian RFF acts as a low-pass filter vs dyadic. Better OOD-cruise generalization (smoother features) but loses in-dist precision. **Pure Gaussian RFF direction closed. Hybrid dyadic+RFF encoder is the promising follow-up** (keep dyadic high-freq structure + add RFF smooth coverage). Learned frequencies (B as nn.Parameter) also noted as natural next step.

---

## 2026-05-13 14:00 — PR #2098 (frieren lion-optimizer post-SwiGLU) — **CLOSED** (mechanism redundancy + schedule mismatch)

- Branch: `charliepai2g24h4-frieren/lion-optimizer` (rebased, 2 seeds)
- Hypothesis: Lion optimizer (sign-momentum, lr=1e-4, wd=3e-4, betas=(0.9,0.99)) on post-SwiGLU stack. Pre-SwiGLU Lion won −8.83% val on an older baseline.

| run | val_avg | test_avg | vs post-#2175 baseline |
|---|---:|---:|---:|
| Seed A (-110800, primary) | 69.406 | 59.068 | **+3.00% / +2.19%** |
| Seed B (-115847, confirmation) | 69.561 | 59.040 | +3.25% / +2.15% |
| Current baseline (GeGLU) | **64.182** | **56.523** | — |

Best epoch: 12 (both seeds, vs 14 for pre-SwiGLU Lion). Sec/epoch: 151 vs 136 (+11%, SwiGLU compute overhead). n_params: 831,191.

**Critical insight from student analysis:** Pre-SwiGLU Lion's gain (−8.83%) accrued almost entirely in epochs 13-14 (val 73.24 → 67.85 = −7.4% in final 2 epochs). SwiGLU's +11% compute cost means only 12 epochs fit → those 2 crucial low-lr epochs are cut. Additionally, SwiGLU's gated forward path provides per-channel scale adaptation (forward path), while Lion provides scale invariance in the backward path — they address the same underlying issue (per-channel grad heterogeneity from LayerScale sign-flip channels). **Partial redundancy + schedule mismatch.**

**Suggested follow-up (from student):** T_max=11 adjustment — cosine schedule calibrated to actual SwiGLU/GeGLU epoch budget (12 epochs) rather than pre-SwiGLU budget (14 epochs). Worth testing as a scheduler-only PR. **Lion optimizer direction closed on this stack.**

---

## 2026-05-13 13:30 — PR #2221 (askeladd rmsnorm-post-swiglu) — **CLOSED** (catastrophic regression)

- Branch: `charliepai2g24h4-askeladd/rmsnorm-post-swiglu`
- Hypothesis: Replace LayerNorm with RMSNorm (Zhang & Sennrich 2019) in all TransolverBlocks. Motivation: LLaMA/Mistral pair SwiGLU+RMSNorm successfully; removing mean-centering + β bias reduces params by 1,408.

| metric | Baseline (#2175) | RMSNorm | Δ |
|---|---:|---:|---:|
| **val_avg** | 67.381 | 80.997 | **+20.2%** |
| **test_avg** | 57.800 | 71.398 | **+23.5%** |
| val_single_in_dist | 73.341 | 103.066 | **+40.5%** |
| val_geom_camber_rc | 80.673 | 87.473 | +8.4% |
| val_geom_camber_cruise | 48.675 | 57.770 | +18.7% |
| val_re_rand | 66.834 | 75.679 | +13.2% |

Best epoch: 10 (vs 13 baseline — slower per-epoch convergence). n_params: 829,783 (−1,408 from LN bias removal).

**Mechanism**: LayerNorm's mean-centering is essential for SwiGLU gate stability. Without it, W_gate·x carries DC component → SiLU asymmetry. LayerNorm β bias was load-bearing additive DOF (3 DOF with LN: γ_LN + β_LN + γ_l; only 2 with RMSNorm: γ_RMS × γ_l). Uniform regression across all splits — generic training-dynamics failure, not domain-specific. val_single_in_dist (+40.5%) worst-hit: high-magnitude p features sensitive to gate asymmetry without mean centering. **Normalization axis closed: LayerNorm + affine is the optimum.**

## 2026-05-13 13:30 — PR #2200 (tanjiro swiglu-inner-dim-384) — **CLOSED** (compute-bound)

- Branch: `charliepai2g24h4-tanjiro/swiglu-inner-dim-384`
- Hypothesis: SwiGLU inner_dim 256→384 (1.5× wider gate/up/down projections). Following 256's win, test next bisect.

| metric | #2175 (256) | inner_dim=384 | Δ |
|---|---:|---:|---:|
| **val_avg** | 67.381 | 68.687 | **+1.94%** |
| **test_avg** | 57.800 | 60.389 | **+4.48%** |
| val_single_in_dist | 73.341 | 78.506 | +7.04% |
| val_geom_camber_rc | 80.673 | 78.742 | **−2.39%** ✓ |
| val_re_rand | 66.834 | 66.581 | **−0.38%** ✓ |

n_params: 1,076,951 (+245K). Best epoch: 11 (vs 13 baseline). Sec/epoch: 167 vs 138 (+21%).

**Mechanism**: Compute-bound failure. +21% per-epoch cost → only 11 epochs vs 13 baseline. Cosine LR T_max=14 not fully traversed. Model still capacity-limited (best=last epoch trained, monotone improving). Key finding: val_camber_rc IMPROVED (−2.39%) despite average regression — extra capacity helps hardest OOD. In-dist splits regressed because fewer LR steps, not because width hurts. **Bisect at 320 (H39) assigned.** Inner_dim width axis may still open if compute budget decoupled.

## 2026-05-13 13:00 — PR #2075 (thorfinn layerscale-init-0.0125) — **CLOSED** (amplitude floor)

- Branch: `charliepai2g24h4-thorfinn/layerscale-init-0.0125`
- Hypothesis: LayerScale init bracket-down from 0.025 → 0.0125. Tests whether lower γ_l amplitude drives further per-channel sign-flip diversification and metric improvement.
- Note: ran on pre-SwiGLU branch (merge base 64bc9fc, baseline #2018 val=74.415)

| metric | Baseline (#2018, init=0.025) | init=0.0125 | Δ |
|---|---:|---:|---:|
| **val_avg** | 74.415 | 76.299 | **+2.53%** |
| **test_avg** | 65.524 | 66.388 | +1.32% |
| val_single_in_dist | 80.907 | 83.728 | +3.49% |
| val_geom_camber_rc | 84.613 | 86.655 | +2.41% |
| val_geom_camber_cruise | 58.100 | 59.755 | +2.85% |
| val_re_rand | 74.039 | 75.060 | +1.38% |

LayerScale sweep (block-0 attn std/mean): 0.1 (38.8%), 0.05 (70.7%), 0.025 (110.5%), **0.0125 (156.6%)** — diversification continues growing but absolute γ amplitude (mean ~0.014–0.017 attn) drops below useful residual-gate conditioning floor. MLP std/mean stuck at 78.2%, never crossed sign-flip threshold.

**Conclusion**: LayerScale init axis exhausted at 0.025. Per-channel diversification mechanism (std/mean) keeps growing as amplitude shrinks, but absolute γ drops below conditioning floor. val_single_in_dist breakdown (+3.49%) confirms even in-dist splits can't be rescued at this amplitude. **Axis closed.**

## 2026-05-13 13:00 — PR #2136 (alphonse n-head-2-wider-attention) — **CLOSED** (regression, axis exhausted)

- Branch: `charliepai2g24h4-alphonse/n-head-2-wider-attention`
- Hypothesis: n_head=4→2 (dim_head=32→64): wider attention heads to preserve OOD cross-geom features, motivated by n_head=8 failure (+7.81%) from head fragmentation.
- Note: ran on pre-SwiGLU branch (merge base b6f687e, baseline #1754 val=73.958)

| metric | Baseline (#1754) | n_head=2 | Δ |
|---|---:|---:|---:|
| **val_avg** | 73.958 | 74.873 | **+1.24%** |
| **test_avg** | 64.502 | 66.190 | **+2.62%** |
| val_single_in_dist | 81.293 | 83.549 | +2.77% |
| val_geom_camber_rc | 85.285 | 83.978 | **−1.53%** |
| val_geom_camber_cruise | 56.390 | 57.573 | +2.10% |
| val_re_rand | 72.862 | 74.391 | +2.10% |

Mechanism: Only val_camber_rc shows predicted improvement (−1.53%); all other splits regress. n_head=8 failed from head fragmentation (dim_head=16); n_head=2 fails from insufficient head diversity (3/4 splits regress). n_head=4 / dim_head=32 is a tight local optimum — the OOD-geom mechanism is head-count-side, not per-head-width-side. **Axis closed: n_head=4 is the optimum at n_hidden=128.**

## 2026-05-13 12:30 — PR #2157 (fern vol-ch-weight-pressure) — **CLOSED** (net regression)

- Branch: `charliepai2g24h4-fern/vol-ch-weight-pressure`
- Hypothesis: Add per-channel weighting to vol_loss: `vol_loss = mean([1.0, 1.0, 2.0] * |y_pred - y_vol| / y_std)`. Mirror the pressure emphasis already in surf_loss [0.5, 0.5, 2.0], motivated by the gap between surf and vol MAE on pressure channel.
- Metric artifacts: `models/model-charliepai2g24h4-fern-vol-ch-weight-pressure-*/metrics.jsonl`

| Split | Baseline (#2175 67.381/57.800) | vol-ch-weight [1,1,2] | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 73.341 | 78.338 | **+6.81%** |
| val_geom_camber_rc | 80.673 | 80.048 | **−0.77%** |
| val_geom_camber_cruise | 48.675 | 48.034 | **−1.32%** |
| val_re_rand | 66.834 | 66.069 | **−1.14%** |
| **val_avg (primary)** | **67.381** | **68.248** | **+1.28%** (regression) |
| test_single_in_dist | 64.685 | 67.855 | **+4.90%** |
| test_geom_camber_rc | 69.035 | 68.442 | **−0.86%** |
| test_geom_camber_cruise | 40.356 | 39.803 | **−1.37%** |
| test_re_rand | 57.121 | 56.638 | **−0.85%** |
| **test_avg** | **57.800** | **58.302** | **+0.87%** (regression) |

**Analysis:** Multi-objective tension between vol and surf pressure channels. 3/4 val splits and 3/4 test splits improve by 0.77–1.37%, but val_single_in_dist regresses sharply +6.81% (test +4.90%). This is the dominant in-distribution split — its regression drives the net average to +1.28% val, just over the 1% close threshold.

**Mechanism**: The vol-pressure upweight ([1,1,2]) conflicts with the surf-loss pressure emphasis ([0.5,0.5,2.0]). Both try to steer the pressure channel in different directions through the shared decoder: vol_loss pressure gradient pushes toward volume-field accuracy while surf_loss pressure gradient (already 2× weighted) pushes toward surface accuracy. The model has to compromise, and the compromise is worse on val_single_in_dist (in-dist full-domain pressure patterns) while marginally better on OOD splits (where the vol-pressure signal overlaps with geom/Re variation).

**Axis-wide finding**: vol-loss per-channel pressure weighting conflicts with merged surf-ch-weight [0.5,0.5,2.0]. Axis closed. The `[1.0,1.0,1.5]` follow-up would suffer the same mechanism at smaller magnitude. Per-channel vol weighting would need a fundamentally different architecture (separate vol/surf decoders) to avoid the interference.

## 2026-05-13 11:20 — PR #2175 (tanjiro swiglu-inner-dim-256) — **MERGED** (13th compound win)

- Branch: `charliepai2g24h4-tanjiro/swiglu-inner-dim-256`
- Hypothesis: Expand SwiGLU inner_dim from 176 (round_up8(256×2/3)) to 256 (full hidden_dim), removing the param-matched budget constraint. Param cost +22.6% (677,591 → 831,191). 5×mlp_ratio=2 with full hidden_dim projection in gate/up/down paths.
- Metric artifacts: `models/model-charliepai2g24h4-tanjiro-swiglu-inner-dim-256-20260513-102355/metrics.jsonl`

| Split | Baseline (#2105 68.812/59.410) | inner_dim=256 | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 76.377 | 73.341 | **−3.97%** |
| val_geom_camber_rc | 79.291 | 80.673 | +1.74% (lone regression) |
| val_geom_camber_cruise | 52.005 | 48.675 | **−6.40%** |
| val_re_rand | 67.573 | 66.834 | **−1.09%** |
| **val_avg (primary)** | **68.812** | **67.381** | **−2.08%** |
| test_single_in_dist | 67.134 | 64.685 | **−3.65%** |
| test_geom_camber_rc | 69.308 | 69.035 | **−0.39%** |
| test_geom_camber_cruise | 42.352 | 40.356 | **−4.72%** |
| test_re_rand | 58.848 | 57.121 | **−2.94%** |
| **test_avg** | **59.410** | **57.800** | **−2.71%** |

**Analysis:** Expanding SwiGLU inner_dim from 176 to 256 removes the 2/3-ratio budget constraint. Win is modest (−2.08% val, −2.71% test) but consistent: all 4 test splits improve, 3/4 val splits improve. The lone val regression (camber_rc +1.74%) reverses on test (−0.39%), suggesting noise on a 100-sample slice. Every epoch was a new best (best=ep 13, last trained), indicating the wider model is in the capacity-limited regime — not overfitting. Run was timeout-bound at 13/50 epochs; longer schedule would likely widen the gap. Consistent with Shazeer's original argument: 2/3 ratio is budget-neutral, not capacity-optimal. Compound: 100.957 → 67.381 = **−33.3% over 13 merges**.

## 2026-05-13 11:15 — PR #2105 (tanjiro swiglu-activation) — **MERGED** (12th compound win)

- Branch: `charliepai2g24h4-tanjiro/swiglu-activation`
- Hypothesis: Replace GELU MLP with SwiGLU (W_down · (SiLU(W_gate·x) ⊙ W_up·x)), inner_dim=176=round_up8(256×2/3) for ~param-match with original 2-matrix GELU MLP. Per-token gated feature routing.
- Metric artifacts: `models/model-charliepai2g24h4-tanjiro-swiglu-activation-20260513-091304/metrics.jsonl`

| Split | Baseline (#2018 74.415) | SwiGLU | Δ vs #2018 | Δ vs #1754 (73.958) |
|---|---:|---:|---:|---:|
| val_single_in_dist | 80.907 | 76.377 | **−5.60%** | — |
| val_geom_camber_rc | 84.613 | 79.291 | **−6.29%** | — |
| val_geom_camber_cruise | 58.100 | 52.005 | **−10.49%** | — |
| val_re_rand | 74.039 | 67.573 | **−8.73%** | — |
| **val_avg (primary)** | **74.415** | **68.812** | **−7.53%** | **−6.96%** |
| test_single_in_dist | 70.626 | 67.134 | **−4.94%** | — |
| test_geom_camber_rc | 73.856 | 69.308 | **−6.16%** | — |
| test_geom_camber_cruise | 49.491 | 42.352 | **−14.43%** | — |
| test_re_rand | 68.125 | 58.848 | **−13.62%** | — |
| **test_avg** | **65.524** | **59.410** | **−9.33%** | **−7.89%** |

**Analysis:** Largest single-PR gain in the entire compound: −7.53% val / −9.33% test. By far exceeds all prior wins. Mechanism confirmed: per-token gated feature routing allows selective amplification/suppression of Fourier features per geometry. OOD splits gain most (re_rand −8.73%/−13.62%, camber_cruise −10.49%/−14.43%) — the gate adapts to out-of-distribution input distributions where fixed GELU cannot. Best epoch advances 14→12 (faster convergence). LayerScale γ_l attn means stable (0.021–0.027), MLP means 0.041–0.053 (slightly higher than attn, consistent with gating adding more representational energy per MLP block).

⚠ Note: Run measured on pre-#1754 baseline (cosine T_max=15, no warmup). Post-merge code includes LR warmup. Re-validation in-flight (#2175).

Compound progress: **100.957 → 68.812 = −31.8%** over 12 merges.

---

## 2026-05-13 11:00 — PR #2099 (fern weight-decay-3x) — **CLOSED** (regression, wd axis closed)

- Branch: `charliepai2g24h4-fern/weight-decay-3x`
- Hypothesis: wd=1e-4 → 3e-4 (3× bracket). Compound regularization stack (stoch-depth + grad-clip + LayerScale) may have effectively raised regularization load, making wd=1e-4 too weak.
- Metric artifacts: `models/model-charliepai2g24h4-fern-weight-decay-3x-20260513-091704/metrics.jsonl`

| Metric | wd=3e-4 (#2099) | Old Baseline (#2018) | New Baseline (#1754) | Δ vs old | Δ vs new |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | 75.615 | 74.415 | **73.958** | **+1.61%** | **+2.24%** |
| test_avg/mae_surf_p (4-split) | 66.227 | 65.524 | **64.502** | **+1.07%** | **+2.67%** |

Per-split val: single_in_dist=83.453 (+3.15%) / camber_rc=86.665 (+2.43%) / camber_cruise=57.428 (**−1.16%**) / re_rand=74.915 (+1.18%). In-dist and rc regress most; only camber_cruise improves.

**Analysis:** Three clean mechanism findings:
1. **Fit/generalization gap already tight** at wd=1e-4 (surf 17%, vol 14% val/train ratio) — no overfitting slack for extra L2 to recover.
2. **In-dist splits regress worst** (single_in_dist val +3.15%, test +6.43%) — opposite of the overfitting prediction. Extra wd damages fitting capacity when generalization gap is already tight.
3. **Only camber_cruise improves** (−1.16%/−4.01%) — easiest OOD split benefits from more constraint; harder splits lose more than they gain. Textbook signature of wd already at/past optimum.

LayerScale γ_l final values at ep 14 (l0=0.052 → l4=0.028) healthy — wd=3e-4 didn't collapse γ_l. Trajectory shows run still descending at ep 14 (75.62 = best), suggesting schedule/wd coupling problem.

**Closed axis.** No wd=5e-4 bracket-up; no wd=5e-5 bracket-down. wd=1e-4 confirmed as optimum for current compound.

---

## 2026-05-13 10:10 — PR #1754 (nezuko LR warmup H19) — **MERGED** (11th compound win)

- Branch: `charliepai2g24h4-nezuko/lr-warmup-h19`
- Hypothesis: Linear LR warm-up over epoch 1 (per-batch LinearLR) + cosine T_max=14 (SequentialLR). Addresses ep1 grad-norm spike on compound stack; orthogonal to all prior merged components.
- Metric artifacts: `models/model-charliepai2g24h4-nezuko-lr-warmup-h19-20260513-075103/metrics.jsonl`

| Metric | LR warmup H19 (#1754) | Baseline (#2018) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **73.958** | 74.415 | **−0.61% WIN** |
| test_avg/mae_surf_p (4-split) | **64.502** | 65.524 | **−1.56% WIN** |

Per-split val: single_in_dist=81.293 (+0.48%) / camber_rc=85.285 (+0.79%) / camber_cruise=56.390 (**−2.94%**) / re_rand=72.862 (**−1.59%**). Split pattern inverted vs original pre-rebase run (old: single_in_dist led; new: camber_cruise and re_rand lead). 

**Mechanism — warmup rescues LayerScale-0.025 OOD degradation:**

The compound stack's latest merge (#2018, LayerScale init=0.025 with block-0 attn sign-flip at 110.5% std/mean) had regressed on the OOD splits: camber_cruise +1.93%, re_rand +1.34% vs #1896. These same splits are exactly where warmup provides the most gain (−2.94%, −1.59%). The per-batch linear ramp over epoch 1 reduces ep1 grad-norm, giving γ_l parameters a stable starting trajectory before the cosine LR peaks — the sign-flip channel dynamics resolve more cleanly on OOD splits as a result.

Test gain (−1.56%) again exceeds val gain (−0.61%) — consistent with original pre-rebase H19 signal. Orthogonality to LayerScale, Fourier, and surf-ch-weight confirmed.

New compound progress: 100.957 → **73.958 = −26.7%** over 11 merges. Zero new parameters (schedule change only).

---

## 2026-05-13 10:10 — PR #2078 (edward Gaussian Fourier σ=10) — **CLOSED** (+36.6% regression; σ scale mismatch)

- Branch: `charliepai2g24h4-edward/gaussian-fourier-sigma-10`
- Hypothesis: Gaussian RFF (Tancik NeurIPS 2020) with σ=10. Tests whether continuous Fourier distribution sidesteps the dyadic-L=8 val_re_rand/surf-ch-weight interaction.
- Metric artifacts: `models/model-charliepai2g24h4-edward-gaussian-fourier-sigma-10-20260513-082642/metrics.jsonl`

| Metric | Gaussian σ=10 | Baseline (#2018, at submission time) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **101.620** | 74.415 | **+36.6% REGRESSION** |
| test_avg/mae_surf_p (4-split) | **90.775** | 65.524 | **+38.5% REGRESSION** |

Still descending at ep14 (val trajectory: 221.9 → 185.9 → ... → 101.6) — not converged.

**Root cause (student's diagnosis confirmed): σ=10 badly miscalibrated for standardized coords.**

Our coords are `x_norm = (x - x_mean)/x_std` with std~1. The projected argument `2π · coord · B` has expected magnitude ~`2π × σ_coord × σ_B = 2π × 1 × 10 ≈ 63` — placing Fourier features in extreme aliasing territory. Tancik's σ recommendation targets coords in [-1,1]. For our standardized space, correct σ range is ~[0.3, 1.5].

**Axis NOT closed** — Gaussian RFF with calibrated σ is a valid direction. Follow-up PR #2135 assigned with σ=1.0 (arm A) + σ=0.5 (arm B) to find the correct operating point.

---

## 2026-05-13 10:10 — PR #2059 (alphonse n_head=4→8) — **CLOSED** (+7.81% val regression; head-fragmentation mechanism confirmed)

- Branch: `charliepai2g24h4-alphonse/n-head-4-to-8`
- Hypothesis: Increase n_head from 4 to 8 at zero param cost. Tests whether finer-grained slice attention patterns improve feature diversity.
- Metric artifacts: `models/model-charliepai2g24h4-alphonse-n-head-4-to-8-20260513-081035/metrics.jsonl`

| Metric | n_head=8 | Baseline (#1896, at submission time) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 11) | **80.290** | 74.476 | **+7.81% REGRESSION** |
| test_avg/mae_surf_p (4-split) | **71.949** | 66.014 | **+9.00% REGRESSION** |
| epochs completed | **11** | 14-15 | confound: 30-min cap |
| wall-time/epoch | 172.8s | ~120s | **+42%** |
| n_params | 652,651 | 669,271 | −2.5% (unexpected!) |

**Wall-time confound:** n_head=8 is 42% slower/epoch. Only 11 epochs completed in 30 min; val curve still descending. Even so, val=80.3 at ep11 vs baseline ~74 at the same epoch: gap is ~8 units, unlikely to close in 3 more epochs.

**Head-fragmentation mechanism confirmed:**

| Split | val Δ | test Δ |
|---|---:|---:|
| camber_rc | **+9.56%** | **+14.65%** ← largest both |
| re_rand | +8.66% | +7.36% |
| camber_cruise | +6.32% | +6.87% |
| single_in_dist | +6.36% | +6.22% |

OOD-geom split camber_rc leads by far on test (+14.65%). This is the same pattern as per-channel decoder experiments (#1811, #2020): when the per-head feature space shrinks from 32 to 16 dims, the cross-geometry correlations that enable OOD transfer degrade. A single 32-dim head can't fit the full cross-airfoil-shape correlation it needs to generalize, fragmented into 16-dim subspaces that are even more constrained.

**Param count discrepancy:** n_head=8 dropped 2.5% params (dim_head=16 → Q/K/V projections shrink). The "zero param cost" assumption in the PR was wrong — useful calibration.

**Follow-up: n_head=2 (#2136, alphonse).** Opposite direction: 2 heads × 64-dim each. Wider heads should preserve cross-geometry correlations better than the baseline 4×32.

---

## 2026-05-13 09:40 — PR #2060 (tanjiro coord-jitter-std=0.002) — **CLOSED** (+1.47% val regression; jitter axis closed)

- Branch: `charliepai2g24h4-tanjiro/coord-jitter-std-0.002`
- Hypothesis: Bracket-down from #1852 (std=0.005): halve jitter intensity to std=0.002. Tests whether reduced std preserves in-dist gain while shrinking OOD-geom regression.
- Metric artifacts: `target/models/model-charliepai2g24h4-tanjiro-coord-jitter-std-0.002-20260513-081718/metrics.jsonl`

| Metric | std=0.002 (#2060) | std=0.005 (#1852) | Baseline (#1896) |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **75.569** | 75.159 | **74.476** |
| Δ vs baseline | **+1.47% REGRESSION** | +0.92% | — |
| test_avg/mae_surf_p | **67.273** | 66.445 | **66.014** |
| Δ vs baseline | **+1.91% REGRESSION** | +0.65% | — |

Per-split val (bracket comparison):

| Split | std=0.002 | std=0.005 | Baseline |
|---|---:|---:|---:|
| val_single_in_dist     | 80.442 (−5.45%) | 80.960 (−4.84%) | 85.075 |
| val_geom_camber_rc     | 88.115 (+6.47%) | 86.831 (+4.91%) | 82.764 |
| val_geom_camber_cruise | 59.128 (+3.73%) | 58.914 (+3.35%) | 57.002 |
| val_re_rand            | 74.592 (+2.09%) | 73.931 (+1.19%) | 73.063 |
| **val_avg**            | **75.569 (+1.47%)** | **75.159 (+0.92%)** | **74.476** |

**Hypothesis falsified — direction-inverted, not parameterized:** Both OOD-geom splits (camber_rc, camber_cruise) regress MORE as std decreases (camber_rc: +4.91% → +6.47%), while the in-dist split improves MORE as std decreases (−4.84% → −5.45%). This is the opposite of a linear-shrinkage parameterization — the mechanism conflict is structural, not intensity-dependent.

The in-dist split sees jitter as noise-regularization (benefiting from any amount); the OOD-geom splits see jitter as a corrupting perturbation of the spatial position representation that they need clean for geometry extrapolation. No std setting resolves this split.

**Axis closure (2 experiments):** Both std=0.005 and std=0.002 falsify the coord-jitter direction on the post-Fourier compound stack. Closed with mechanism finding: "coord jitter is a position-conditioned regularizer that conflicts with Fourier L=6's high-frequency spatial features; the conflict grows as jitter std decreases, indicating a structural coupling that bracket-down cannot resolve."

---

## 2026-05-13 09:40 — PR #1753 (askeladd adaptive-grad-clip K=100 α=1.5) — **CLOSED** (+3.30% val regression)

- Branch: `charliepai2g24h4-askeladd/adaptive-grad-clip-q50-a1.5`
- Hypothesis: Replace static max_norm=25 with adaptive clip at 1.5× running q50 (K=100 steps) + warmup_threshold=100. Tests whether the gradient distribution's median is a better scale reference than a global max_norm constant.
- Metric artifacts: `models/model-charliepai2g24h4-askeladd-adaptive-grad-clip-q50-a1.5-20260513-075927/metrics.jsonl`

| Metric | Adaptive clip (#1753) | Baseline (#2018) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **76.931** | **74.415** | **+3.30% REGRESSION** |
| test_avg/mae_surf_p (4-split) | **68.005** | **65.524** | **+3.79% REGRESSION** |

Per-split val: single_in_dist=84.902 (+4.83%) / camber_rc=84.969 (+0.27%) / camber_cruise=61.892 (+7.74%) / re_rand=75.961 (+1.65%). Uniform regression across all splits — consistent with global regularization.

**Mechanism analysis:** Adaptive clip at 1.5× q50 is significantly more aggressive than static max_norm=25 on the LayerScale-attenuated stack. With LayerScale γ_l=0.025 and block-0 sign-flip channels (std/mean=110.5%), per-step gradient norm distributions are highly variable per-channel. The median q50 of these distributions sits far below the static max_norm=25 — meaning the adaptive ceil clips the majority of normal training steps where static does not. This over-regularization explains the uniform regression across all splits.

**Context:** The static grad-clip max_norm=25 (PR #1637) was explicitly tuned on this architecture. Adaptive variants require separate calibration; no obvious parameterization wins. Gradient-clip axis closed.

---

## 2026-05-13 09:40 — PR #1828 (frieren SmoothL1 β-bracket, β=0.005 close-out) — **CLOSED** (+0.38% val; mechanism absorbed by LayerScale)

- Branch: `charliepai2g24h4-frieren/smooth-l1-loss-beta-001`
- Hypothesis (H25): Replace L1 loss with SmoothL1 (Huber, β=0.01 → β=0.005 close-out). Tests whether smooth-near-zero gradient eliminates late-training zigzag oscillation that L1's subgradient discontinuity causes.
- Metric artifacts: `models/model-charliepai2g24h4-frieren-smooth-l1-loss-beta-0.005-20260513-075416/metrics.jsonl`

**β-bracket trajectory summary:**

| Stack | γ_l init | β | val Δ% | test Δ% | grad@ep14 |
|---|---:|---:|---:|---:|---:|
| post-#1548 (L=4, no LS) | n/a | 0.01 | **−0.97% WIN** | **−1.82% WIN** | 13.9 |
| post-#1799 (L=6, LS=0.1) | 0.1 | 0.01 | +0.05% flat | −1.03% | 28.5 |
| post-#2018 (L=6, LS=0.025) | 0.025 | 0.005 | **+0.38% REGRESSION** | −0.39% | 68.22 |

**Key mechanism finding — SmoothL1 absorbed by LayerScale:**

The grad@ep14 trace (13.9 → 28.5 → 68.22) directly demonstrates the absorption: as LayerScale γ_l init decreases (0.1 → 0.05 → 0.025), per-channel residual gating becomes more aggressive, absorbing the late-cooldown gradient-pinning effect that SmoothL1 provides. At γ_l init=0.025 with sign-flip channels (std/mean=110.5%), the per-channel gradient flow is too variable for SmoothL1's smooth-near-zero window to dominate late-epoch behavior.

The per-split signature also inverted between post-#1799 (test_single_in_dist was the biggest winner at −3.40%) and post-#2018 (test_single_in_dist became the biggest loser at +3.01%). Consistent with LayerScale's sign-flip channels placing different splits' residuals inside/outside the β-window.

**Axis closure:** Loss-landscape smoothing (SmoothL1) on the post-LayerScale stack is absorbed by the per-channel residual gating mechanism. Both β values (0.01 and 0.005) land in plateau territory. Tighter β-windows don't re-engage the mechanism. The loss-landscape direction is closed on this compound stack.

---

## 2026-05-13 09:40 — PR #1549 (fern FiLM global conditioning) — **CLOSED** (stalled, no terminal result)

- Branch: `charliepai2g24h4-fern/film-global-cond`
- Hypothesis: FiLM (Feature-wise Linear Modulation) conditioning on global flow params (Re, AoA) in each Transolver block. Tests whether explicit global-flow awareness improves cross-regime generalization.
- Metric artifacts: None committed on the current stack (stalled at pre-merge baseline from 2026-05-12 21:55 UTC)

**Status:** 4 rebase requests over 10+ hours with no terminal result. Last student commit was at 21:55 UTC on 2026-05-12; pod had 2 restarts in 15h. Baseline has moved 8 compound wins since the one preliminary result (81.291 on baseline 90.294).

**The original pre-merge result** (val=81.291 on baseline 90.294, 10.5% improvement) was among the strongest single-direction signals in early rounds but was never confirmed on the compounding stack.

**Hypothesis note:** FiLM conditioning on flow parameters is a theoretically strong direction (Perez et al. 2018, tested in many conditional regression settings). The lack of a terminal result is a student-execution issue, not a direction issue. If reassigned with simpler instructions and cleaner implementation, this direction could still be tested.

**Closed due to execution failure, not mechanism failure.** FiLM axis is NOT conclusively closed — reassignment with a fresh, simpler implementation approach recommended.

---

## 2026-05-13 08:35 — PR #2018 (thorfinn LayerScale init=0.025) — **MERGED** (10th compound win)

- Branch: `charliepai2g24h4-thorfinn/layerscale-init-0.025`
- Hypothesis: Continue LayerScale operating-point sweep: drop init=0.05 → **init=0.025**. Tests whether per-channel diversification has additional headroom.
- Metric artifacts: `models/model-charliepai2g24h4-thorfinn-layerscale-init-0.025-20260513-070817/metrics.jsonl`

| Metric | init=0.025 (#2018) | Baseline (#1896) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **74.415** | 74.476 | **−0.08% WIN** |
| test_avg/mae_surf_p (4-split) | **65.524** | 66.014 | **−0.74% WIN** |

Per-split val: single_in_dist=80.907 (−4.90%) / camber_rc=84.613 (+2.23%) / camber_cruise=58.100 (+1.93%) / re_rand=74.039 (+1.34%). Single split dominates the average improvement; OOD splits regress slightly.

**Mechanism breakthrough — γ_l sign-flip threshold reached:**

| init | val_avg | Block-0 attn std/mean |
|---|---:|---:|
| 0.1 | 78.260 | 38.8% |
| 0.05 | 74.476 | 70.7% |
| **0.025** | **74.415** | **110.5%** |

Block-0 attn std now EXCEEDS mean (>100%) — ~half of per-channel γ_l entries have learned **negative** residual scale (sign-flipping residuals). MLP-side std/mean 47–73%, not yet at sign-flip. This is structurally novel: the model uses the residual gate not just to attenuate but to flip specific channels.

Gain-per-halving: -1.21% (0.1→0.05) → **-0.08% (0.05→0.025)**. Diminishing returns confirmed. Test signal (-0.74%) stronger than val, driven by single_in_dist (-5.86%).

Merged as 10th compound win per "merge even marginal improvements" principle. New compound progress: 100.957 → 74.415 = −26.3% over 10 merges.

---

## 2026-05-13 08:35 — PR #1830 (edward Fourier coords L=8, rebased-3) — **CLOSED** (+0.745% val regression; val_re_rand interaction)

- Branch: `charliepai2g24h4-edward/fourier-coords-L8`
- Hypothesis: Bracket up Fourier L=6 → L=8; test whether dyadic frequency spectrum continues to improve.
- Metric artifacts: `models/model-charliepai2g24h4-edward-fourier-coords-L8-rebased-2-20260513-072103/metrics.jsonl`

| Metric | L=8 rebased (#1830) | Baseline (#1896) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **75.031** | 74.476 | **+0.745% REGRESSION** |
| test_avg/mae_surf_p (4-split) | **65.563** | 66.014 | −0.683% improvement |

Per-split val: single_in_dist=80.904 (−1.36%) / camber_rc=86.085 (−2.80%) / camber_cruise=58.167 (−0.84%) / **re_rand=74.968 (+9.19%)**. 3/4 splits improve; val_avg sign set entirely by val_re_rand.

**val_re_rand interaction with surf-ch-weight — key mechanism finding:**
- On post-#1799 (no surf-ch-weight): L=8 gave val_re_rand **−5.31%** improvement
- On post-#1896 (with surf-ch-weight [0.5,0.5,2.0]): L=8 gives val_re_rand **+9.19% regression**
- The 14.5% swing is caused by the surf-ch-weight 4× velocity de-emphasis. Velocity gradients are most important for Reynolds-OOD (val_re_rand). At L=8, the high-frequency Fourier bands capture fine-grained velocity boundary-layer features, but these features are under-weighted by the surf-ch-weight 4× pressure emphasis. Result: L=8's highest-band capacity is available but not exploitable.

**Dyadic Fourier direction now closed:** L=4 merged, L=6 merged, L=8 blocked. Follow-up: Gaussian Fourier features (Tancik 2020, σ=10) to test whether a continuous frequency distribution sidesteps the val_re_rand interaction.

---

## 2026-05-13 08:15 — PR #2020 (alphonse per-channel-decoder-heads) — **CLOSED** (+4.20% regression)

- Branch: `charliepai2g24h4-alphonse/per-channel-decoder-heads`
- Hypothesis: Replace shared `mlp2` decoder with 3 fully independent `Sequential(Linear(128→128), GELU, Linear(128→1))` per-channel heads. Full-capacity design (128-hidden each) to fix the capacity confound in failed PR #1811 (half-capacity 128→64→1).
- Metric artifacts: `target/models/model-charliepai2g24h4-alphonse-per-channel-decoder-heads-20260513-071517/metrics.jsonl`

| Metric | Per-channel heads (#2020) | Baseline (#1896) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13) | **77.605** | 74.476 | **+4.20% REGRESSION** |
| test_avg/mae_surf_p (4-split) | **67.329** | 66.014 | +1.99% REGRESSION |

Per-split val: single_in_dist=85.863 (+0.93%) / camber_rc=90.964 (**+9.91%**) / camber_cruise=58.140 (+1.99%) / re_rand=75.454 (+3.27%). **All 4 val splits regress; camber_rc worst by far.**
Per-split test: single_in_dist=76.076 (+1.40%) / camber_rc=76.844 (+3.88%) / camber_cruise=48.277 (−0.34%) / re_rand=68.119 (+2.25%).

**Sanity checks passed:** param count exact (702,295 vs 702,000 predicted), val_single_in_dist regression collapsed from +5.34% (#1811 half-capacity) → +0.93% (full capacity), VRAM 49.15 GB. The capacity confound was real — fixing it reduced the in-dist regression 6×.

**BUT per-channel specialization still regresses overall.** Key mechanism: camber_rc regressed +9.91% — geometry-interpolation OOD splits need the **cross-channel features** (Ux↔Uy↔p correlations) that the shared decoder learns through its single 128→3 projection. Per-channel heads sacrifice these cross-channel correlations for per-head capacity, which is net-negative on OOD generalization.

**Axis closure (2 experiments):** Both half-capacity (#1811) and full-capacity (#2020) per-channel-output directions tested. Both regressed. The shared cross-channel decoder is load-bearing and should not be split. If per-channel granularity is desired, do it on the **loss-side** (already done, #1711) or **input-side** (unexplored).

---

## 2026-05-13 08:10 — PR #1852 (tanjiro coord-jitter-aug std=0.005) — **CLOSED** (+0.92% regression with direction-inverted split pattern)

- Branch: `charliepai2g24h4-tanjiro/coord-jitter-aug-0.005`
- Hypothesis: Gaussian noise (std=0.005) on input spatial dims [x, z] during training, applied before Fourier encoding. Forces mesh-position perturbation invariance, regularizes high-freq Fourier features.
- Metric artifacts: `models/model-charliepai2g24h4-tanjiro-coord-jitter-aug-0.005-20260513-071736/metrics.jsonl`

| Metric | Coord-jitter std=0.005 (#1852) | Baseline (#1896) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **75.159** | 74.476 | **+0.92% REGRESSION** |
| test_avg/mae_surf_p (4-split) | **66.445** | 66.014 | +0.65% REGRESSION |

**Direction-inverted per-split pattern (key finding):**

| Split | val Δ | test Δ |
|---|---:|---:|
| `val_single_in_dist` | **−4.83%** | **−3.88%** |
| `val_geom_camber_rc` | +4.92% | +5.23% |
| `val_geom_camber_cruise` | +3.35% | +1.69% |
| `val_re_rand` | +1.19% | −0.09% |

**Mechanism analysis (student's insight, confirmed):** Coord jitter only perturbs the 2 spatial dims [x, y]. OOD-geom splits hold out *new airfoil cambers* distinguished by **shape features** in dims 2–11 (saf, dsdf) and NACA params dims 15–17 — not spatial positions. So coord jitter is a **position-conditioned regularizer** that helps where in-dist position is densely sampled, but doesn't help when the underlying shape function is OOD.

**Strong in-dist signal preserved:** val_single_in_dist = −4.83% (val + test both robust). This is real regularization, not noise.

**Net negative:** 3 OOD splits regress enough to overwhelm the 1 in-dist gain. val_avg +0.92%.

**Implication:** Input regularization works but on the wrong axis at std=0.005. Follow-up: bracket-down to std=0.002. At 2.5× smaller amplitude, in-dist gain may partially survive while OOD damage shrinks toward zero. Sweet spot may exist.

---

## 2026-05-13 07:05 — PR #1896 (thorfinn LayerScale init=0.05) — **MERGED** (9th compound win)

- Branch: `charliepai2g24h4-thorfinn/layerscale-init-0.05`
- Hypothesis: bracket the merged LayerScale init=0.1 downward to init=0.05. Tests "per-channel granularity hypothesis" — does the model converge to the same γ_l plateau regardless of init, or does the operating-point magnitude matter?
- Metric artifacts: `models/model-charliepai2g24h4-thorfinn-layerscale-init-0.05-20260513-061249/metrics.jsonl`

| Metric | init=0.05 (#1896) | Baseline (#1711) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **74.476** | 75.391 | **−1.21%** |
| test_avg/mae_surf_p (4-split) | **66.014** | 66.608 | **−0.89%** |

Per-split val: single_in_dist=85.075 (+3.39% ← only regression) / camber_rc=82.764 (−3.35%) / camber_cruise=57.002 (−2.78%) / re_rand=73.063 (−2.60%). **3/4 val splits improve.**
Per-split test: single_in_dist=75.023 (+3.40%) / camber_rc=73.975 (−0.32%) / camber_cruise=48.442 (−4.78%) / re_rand=66.617 (−3.16%). **3/4 test splits improve.**

**Mechanism — hybrid Outcome A/B:**
- Headline metrics: Outcome A range (val -1.21%, test -0.89% — marginal win, compounding cleanly)
- γ_l means: scale ~linearly with init (0.05 → means ~0.043/0.063 vs 0.1 → means ~0.119/0.083) — Outcome B (model stays near init, NOT converging to a single plateau)
- Per-channel std/mean ratio: **block-0 attn jumps 38.8% → 70.7%** — model compensates for lower amplitude by widening per-channel diversity
- Depth-decreasing MLP trend preserved (blocks 0→4: 0.063→0.064→0.055→0.044→0.047)

**Key insight:** Per-channel granularity IS load-bearing, but the operating-point magnitude co-varies with init. Lower init = smaller residual amplitude = more per-channel diversification used for compensation. The `single_in_dist` regression (+3.4%) is likely an interaction with #1711's p:vel=4× weighting: reduced residual gain on the pressure-dominated single-foil split costs accuracy there, while geom/Re-shift splits benefit from the mild regularization.

**Compound progress: 100.957 → 74.476 = −26.2% over 9 merges** (#1397→#1552→#1611→#1637→#1548→#1772→#1799→#1711→#1896)
Thorfinn assigned follow-up: init=0.025 bracket to continue the operating-point sweep.

---

## 2026-05-13 07:08 — PR #1962 (alphonse surf-ch-weight [0.33,0.33,2.33] — H18-B) — **CLOSED** (Outcome C)

- Branch: `charliepai2g24h4-alphonse/surf-ch-weight-aggressive`
- Hypothesis: bracket-up from merged H18 `[0.5,0.5,2.0]` (4×) to `[0.33,0.33,2.33]` (7×). Pre-registered Outcome C protocol if val regresses.
- Metric artifacts: `models/model-charliepai2g24h4-alphonse-surf-ch-weight-aggressive-20260513-060655/metrics.jsonl`

| Metric | H18-B (7×) | Baseline H18 (4×) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | **76.890** | 75.391 | **+1.99% REGRESSION** |
| test_avg/mae_surf_p | **67.388** | 66.608 | +1.17% REGRESSION |

**Outcome C confirmed (bracket-up regresses).** Pressure regressed on all 4 val splits (+0.4% to +5.4%). Velocity degraded only +6-13% (vs predicted +30-40%), meaning the optimizer isn't redistributing capacity — it's descending less effectively overall.

**Mechanism:** At p:vel=7×, velocity-side gradient is too weak. Pressure prediction partially relies on velocity-field coherence (continuity-like relationships); starving Ux/Uy gradients harms pressure prediction back through the shared encoder. The two val splits with highest pressure errors (single_in_dist +5.4%, camber_rc +1.1%) regressed most sharply.

**Axis closure:** surf-loss-weighting axis mapped at 3 points: implicit [1,1,1]=78.260 / [0.5,0.5,2.0]=75.391 (optimum) / [0.33,0.33,2.33]=76.890. 4× is the local maximum. Direction fully closed. Alphonse reassigned to new direction.

---

## 2026-05-13 05:57 — PR #1711 (alphonse surf-ch-weight [0.5,0.5,2.0] — H18) — **MERGED** (8th compound win)

- Branch: `charliepai2g24h4-alphonse/surf-ch-weight-h18`
- Merge commit: `bd1d19a`; baseline update: `1d612fc`
- Hypothesis: per-channel surf-loss weighting `[w_Ux, w_Uy, w_p] = [0.5, 0.5, 2.0]` (p:vel ratio 4×) applied to `surf_loss` and `evaluate_split`. Mass-preserving (sum=3.0), zero new parameters. Expected -0.5% to -2.5% val.

| Metric | This PR (H18) | Baseline (#1799) | Δ vs current |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **75.391** | 78.260 | **−3.67%** |
| test_avg/mae_surf_p (4-split) | **66.608** | 69.903 | **−4.71%** |

Per-split val: single_in_dist=82.287 (−3.50%) / camber_rc=85.631 (−3.84%) / camber_cruise=58.630 (−6.34%) / re_rand=75.015 (−1.46%). **All 4 improve.**
Per-split test: single_in_dist=72.554 (−6.80%) / camber_rc=74.210 (−6.64%) / camber_cruise=50.877 (−1.60%) / re_rand=68.792 (−2.52%). **All 4 improve.**

Per-channel trade-off (mechanism confirmed): Ux +21%, Uy +12%, p −3.8% on val; Ux/Uy don't appear in primary metric so trade is pure win. Direction was the correct "fix the loss surface" pivot from failed #1610/#1636/#1675 prediction-side attempts. Test gains exceed val gains on in-dist (+6.80% vs 3.50%) — possible generalization bonus from reduced velocity-channel overfitting.

Exceeded pre-registered prediction band (-0.5% to -2.5%). Alphonse assigned #1962: bracket-up to [0.33, 0.33, 2.33] (p:vel ratio 7×) to test if there's remaining capacity to reallocate. Metric artifacts: `models/model-charliepai2g24h4-alphonse-surf-ch-weight-h18-20260513-050507/metrics.jsonl`.

## 2026-05-13 05:08 — PR #1828 (frieren SmoothL1 β=0.01 — H25, rebase 2) — **SEND-BACK** (β=0.005 close-out)

- Branch: `charliepai2g24h4-frieren/smooth-l1-loss-beta-001`
- Rebase commit: `a7f78c3` (onto `a675cb1` = post-#1799 LayerScale HEAD)
- Hypothesis: replace L1 with SmoothL1(β=0.01) for smooth gradient near zero — expected -0.3% to -1.5% val gain via late-cooldown noise reduction.

**Rebased results on current advisor HEAD (Fourier L=6 + LayerScale init=0.1 stack):**

| Metric | This PR (rebased) | Baseline (#1799) | Δ | Δ% |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **78.300** | 78.260 | +0.040 | **+0.05%** (essentially flat) |
| test_avg/mae_surf_p (4-split) | **69.185** | 69.903 | −0.718 | **−1.03%** |

Per-split val: 2 improve (single_in_dist -0.53%, re_rand -0.22%) / 2 small regress (camber_rc +0.43%, camber_cruise +0.62%) — all within ±0.7% (noise band).

Per-split test: 2 improve cleanly (test_single_in_dist **−3.40%**, test_re_rand −1.48%) / 2 small regress (camber_rc +0.68%, camber_cruise +0.53%). Net asymmetric.

**Mechanism finding (load-bearing for next-step design):** Compared to frieren's pre-rebase L=4 stack result (val −0.97%, test −1.82%), both effects compressed substantially on the post-LayerScale stack. Late-cooldown grad_norm rose from 13.9 (L=4 stack) to 28.5 (current stack) at epoch 14 — directly demonstrating that LayerScale's γ_l ≈ 0.1 per-channel residual attenuation absorbs part of what SmoothL1's smooth-near-zero gradient was buying on the unmodified residual stream. Both mechanisms reduce the late-training per-step update magnitude in the small-residual regime; the combination is sub-additive.

**Decision: REQUEST CHANGES → β=0.005 bracket close-out.** Val is essentially flat (+0.05% does not satisfy the primary-metric improvement merge criterion), but the direction isn't dead — test wins are clean and the mechanism is real, just attenuated. β=0.005 (one-line change) is the natural disambiguator: tighter window may match the now-LayerScale-attenuated residual distribution, or confirm the loss-landscape direction is tapped out on this stack. Three pre-registered outcomes: A win (8th compound), B flat (close direction), C regression (β=0.01 was the narrow sweet spot).

## 2026-05-13 05:57 — PR #1830 (edward Fourier L=8 — H26) — **SEND-BACK** (rebase onto post-#1711 baseline)

- Branch: `charliepai2g24h4-edward/fourier-coords-L8`
- Hypothesis: Fourier coord encoding N_FREQS=6→8, fun_dim=52. Expected val-win or val-plateau with Gaussian Fourier pivot.

| Metric | This PR (#1799 baseline) | Baseline (#1799) | Δ vs #1799 |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **78.144** | 78.260 | **−0.149%** (PLATEAU per verdict tree ±0.5%) |
| test_avg/mae_surf_p (4-split) | **68.168** | 69.903 | **−2.48%** (clean WIN) |

Mechanism checks passed: `val_geom_camber_cruise` showed smallest delta (plateau-leading-edge signal, as predicted); no aliasing on `val_single_in_dist`; best-epoch shifted 15→14 (soft overfit signal). LayerScale γ depth-decreasing pattern preserved (0.110→0.081 MLP branch).

**Send-back reason:** Measured against #1799 baseline (78.260). PR #1711 alphonse merged first (new baseline 75.391). Fourier L=8 at 78.144 doesn't beat 75.391. Sent back for rebase onto post-#1711 HEAD; mechanisms are orthogonal (input-side vs loss-side), expect clean compound.

## 2026-05-13 03:56 — PR #1799 (thorfinn LayerScale CaiT init=0.1 — H23) — **MERGED** (7th compound win)

- Branch: `charliepai2g24h4-thorfinn/layerscale-init-0.1`
- Merge commit: `4866280`; baseline update commit: `b8a7193`
- Hypothesis: per-channel learnable γ_l initialized to 0.1, gating each
  residual branch (attn + mlp) per CaiT (Touvron et al. 2021). Expected
  -1% to -3% on val; predicted ramp-up to [0.5, 1.5] range over training.
- Post-rebase results (on current Fourier L=6 stack, `d069290`):

| Metric | This PR (LS + L=6) | Baseline (#1772) | Δ vs current | Prior (L=4 result) |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | **78.260** | 82.311 | **−4.92%** | (77.629, −8.42% vs L=4 84.762) |
| test_avg/mae_surf_p (4-split) | **69.903** | 73.330 | **−4.67%** | (68.010, −8.91% vs L=4 74.659) |
| Best epoch | 14 | 15 | shifted earlier (preserved) | 14 |
| Param count | 669,271 | 667,991 | +1,280 (+0.19%) | 667,223 (smaller fun_dim) |
| Peak GPU memory | 47.17 GB | — | unchanged | 47.17 GB |
| Wall time | 31.4 min | — | 1 epoch past 30-min cap | 31.4 min |

Per-split val MAE @ best epoch 14 (vs #1772 baseline):

| Split | Baseline (L=6) | This PR | Δ |
|---|---:|---:|---:|
| val_single_in_dist     | 93.299 | 85.269 | **−8.61%** |
| val_geom_camber_rc     | 92.965 | 89.049 | **−4.21%** |
| val_geom_camber_cruise | 63.131 | 62.595 | −0.85% |
| val_re_rand            | 79.848 | 76.127 | **−4.66%** |
| **val_avg**            | **82.311** | **78.260** | **−4.92%** |

Per-split test MAE @ best val checkpoint (vs #1772 baseline):

| Split | Baseline (L=6) | This PR | Δ |
|---|---:|---:|---:|
| test_single_in_dist     | 83.323 | 77.850 | **−6.57%** |
| test_geom_camber_rc     | 81.867 | 79.485 | **−2.91%** |
| test_geom_camber_cruise | 54.094 | 51.705 | **−4.42%** |
| test_re_rand            | 74.038 | 70.573 | **−4.68%** |
| **test_avg**            | **73.330** | **69.903** | **−4.67%** |

- **Clean compound win.** All 4 val splits and all 4 test splits improve.
  Distribution: strongest on val_single_in_dist/test_single_in_dist
  (high-magnitude pressure regime) and Re axis, weakest on
  val_geom_camber_cruise (already-lowest-error split, hardest to move).
  Pattern mirrors L=4-stack result with smaller magnitudes — see margin
  analysis below.
- **Mechanism preserved across rebase from L=4 to L=6:**
  - γ_l means stay in [0.079, 0.119] range (range was [0.079, 0.115] on L=4 stack).
  - Per-channel std reaches **38.8%** of mean in block-0 attn (vs 33.9% on L=4) — slightly *higher* per-channel diversity with Fourier L=6 features.
  - Depth-decreasing mlp γ_l trend preserved: block-0 mlp mean 0.119 → block-4 mlp mean 0.083 (was 0.115 → 0.079 on L=4).
  - The "selectively preserve early blocks more, downweight later blocks already regularized by stoch-depth" interpretation holds across both stacks.
- **Margin analysis (gain shrinkage L=4 → L=6):** L=4 stack gave -8.42%/-8.91% (val/test); L=6 stack gives -4.92%/-4.67%. Both mechanisms partially overlap in "making the residual stream more useful at the right scale" — Fourier L=6 gives the input more frequency content (residual stream needs less amplification to preserve high-frequency info); LayerScale lets the model selectively preserve channels per-block. With both active, neither has to do as much alone, but they still leave a clear net gain because their levers are different (input encoding vs. per-channel gating). **Clean compound win, not a fight.**
- **Train-loss slow-start preserved**: Epoch-1 train loss = 0.733 (L=6) vs 0.714 (L=4) — essentially identical. The slow-start is structural (γ_l=0.1 attenuates residual stream from epoch 1) and unaffected by input encoding.
- **Why the gain was preserved (mechanism confirmed safe):** unlike #1608 EMA which fights Fourier features via spectral smoothing, LayerScale is a learnable per-channel gate post-attn/MLP. It doesn't smooth high-frequency information; it selects which channels to preserve. Fourier provides more useful channels, and LayerScale uses the richer signal effectively (per-channel std even increased slightly with L=6).
- **Compound progress**: #1397 → #1552 → #1611 → #1637 → #1548 → #1772 → **#1799 LayerScale** → val_avg 100.957 → **78.260** = **-22.5% over 7 merges**.
- **Student's pre-registered follow-ups (next steps assigned this iteration):**
  1. **Bracket init=0.05** (assigned to thorfinn this iteration) — model converged from init=0.1 to means in [0.079, 0.119]. If init=0.05 lands at similar plateau, confirms per-channel granularity is the load-bearing structure (init value doesn't matter much, the form does).
  2. Per-block init schedule [0.15, 0.12, 0.10, 0.08, 0.05] — natural follow-up reflecting observed depth-decreasing trend.
  3. Longer training with adjusted cosine T_max to see if γ_l drifts after cosine cooldown.
- Single arm. Wall time 31.4 min. Metrics: `models/model-charliepai2g24h4-thorfinn-layerscale-init-0.1-rebased-20260513-031524/metrics.{jsonl,yaml}`.

## 2026-05-13 03:24 — PR #1828 (frieren SmoothL1 β=0.01 — H25) — **REBASING** (sent back)

- Branch: `charliepai2g24h4-frieren/smooth-l1-loss-beta-001`
- Hypothesis: replace L1 with SmoothL1 (Huber) at β=0.01 — smooth the
  subgradient discontinuity at zero so the cosine-LR cooldown can pin
  small residuals without zigzag. Mechanism predicted orthogonal to
  Fourier features (changes loss landscape near r=0, not function
  representation). Different axis from EMA's weight-space smoothing
  which fought Fourier sharpening.
- Pre-rebase results (against stale baseline 84.762, Fourier L=4 stack):

| Metric | This PR | Stale baseline (#1548) | Current baseline (#1772) | Δ vs stale | Δ vs current |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 83.938 | 84.762 | 82.311 | **−0.97%** | +1.98% |
| test_avg/mae_surf_p (4-split) | 73.300 | 74.659 | 73.330 | **−1.82%** | −0.04% (~flat) |
| Param count | 666,247 | 665,943 | 667,991 | +304 (negligible) | -1744 |
| Peak mem | 42.2 GB | — | — | unchanged | unchanged |

Per-split val MAE @ best epoch 15 (vs stale baseline):

| Split | Stale base | This PR | Δ vs stale |
|---|---:|---:|---:|
| val_single_in_dist     | 97.074 | 96.090 | −1.01% |
| val_geom_camber_rc     | 94.997 | 94.271 | −0.76% |
| val_geom_camber_cruise | 63.711 | 64.088 | +0.59% |
| val_re_rand            | 83.266 | 81.303 | −2.36% |
| **val_avg**            | 84.762 | **83.938** | **−0.97%** |

Per-split test MAE @ best val checkpoint:

| Split | Stale base | This PR | Δ vs stale |
|---|---:|---:|---:|
| test_single_in_dist     | 85.819 | 84.846 | −1.13% |
| test_geom_camber_rc     | 83.023 | 80.750 | −2.74% |
| test_geom_camber_cruise | 54.879 | 54.416 | −0.84% |
| test_re_rand            | 74.916 | 73.186 | −2.31% |
| **test_avg**            | 74.659 | **73.300** | **−1.82%** |

- **Mechanism analysis — clean and well-instrumented.** All 4 test splits
  improve and 3/4 val splits improve. The single val regression
  (camber_cruise +0.59%) is contradicted by its test result (−0.84%) and
  is well within natural epoch-to-epoch variance. Late-epoch (10-15)
  train surface loss is monotonically decreasing with only a tiny ~+0.005
  bump at ep13 (in the noise band). Final-cooldown grad norms are notably
  small: **ep14=13.9, ep15=16.4 vs ep10-13 range 31-49** — the lowest of
  the whole run, directly demonstrating the predicted behavior of
  SmoothL1's smooth-near-zero gradient letting the LR cooldown actually
  pin small residuals instead of zigzagging them. Test improvement > val
  improvement is a positive generalization signal.
- **Fourier-gain conflict check:** student verified explicitly. No
  per-split pattern of consistent regression that would indicate conflict
  with merged Fourier coords (the single val regression is contradicted
  by test; not a mechanism conflict signal). Hypothesis explicitly
  predicted no spectral-bias conflict (SmoothL1 changes loss landscape
  near r=0, not function representation).
- **Why sent back for rebase (not merged):** student ran on pre-#1772
  baseline (84.762, Fourier L=4 stack, `fun_dim: 36`). Current baseline is
  82.311 (Fourier L=6, `fun_dim: 44`). On the current baseline, the
  pre-rebase result is +1.98% val regression and essentially flat on
  test. The 02:14 UTC run started 36 minutes BEFORE #1772 merged at 02:50
  UTC. Same rebase-confirm pattern as #1799 LayerScale (also sent back
  this iteration).
- **Big open mechanism question:** does Fourier L=6 already partially
  address what SmoothL1 was fixing? L=6 gives the model sharper feature
  representation, which could reduce the prevalence of small-residual
  zigzag regions that SmoothL1 smooths. If they overlap, the gain
  shrinks; if they don't, the gain holds. Expected post-rebase: val_avg
  in [81.0, 82.0] if clean compound; [82.0, 83.0] if partial overlap
  (marginal); > 82.311 if active interference (close with mechanism).
- **Operational note from student:** entrypoint re-invoked while a prior
  training was still running, creating duplicate model dirs at 02:54 and
  03:01. Student killed those duplicates and committed only the canonical
  02:14 run. Clean.
- **Student's pre-registered follow-ups (queued, contingent on rebase
  confirmation):**
  1. β=0.005 (tighter smooth window). If still wins, optimum is in
     [0.005, 0.01]. Tests "kill the zigzag at convergence" interpretation.
  2. β=0.05 (5× wider smooth window). Tests direction-not-magnitude. If
     also wins, mechanism is robust; if regresses, 0.01 is in a narrow
     sweet spot.
  3. Per-channel β — pressure has wider target distribution than velocity
     in y_norm space; could compound with surf_weight asymmetry.
- Single arm. Wall time confirmed within 30 min cap (14 epochs). Metrics:
  `models/model-charliepai2g24h4-frieren-smooth-l1-loss-beta-001-20260513-021419/metrics.{jsonl,yaml}`.

## 2026-05-13 03:15 — PR #1799 (thorfinn LayerScale CaiT init=0.1 — H23) — **REBASING** (sent back)

- Branch: `charliepai2g24h4-thorfinn/layerscale-init-0.1`
- Hypothesis: per-channel learnable γ_l initialized to 0.1, gating each
  residual branch (attn + mlp) per CaiT (Touvron et al. 2021). Expected
  -1% to -3% on val; predicted ramp-up to [0.5, 1.5] range over training.
- Pre-rebase results (against stale baseline 84.762):

| Metric | This PR | Stale baseline (#1548) | Current baseline (#1772) | Δ vs stale | Δ vs current |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | 77.629 | 84.762 | 82.311 | **−8.42%** | −5.69% |
| test_avg/mae_surf_p (4-split) | 68.010 | 74.659 | 73.330 | **−8.91%** | −7.26% |
| Param count | 667,223 | 665,943 | 667,991 | +1,280 (+0.19%) | -768 |
| Peak mem | 47.1 GB | — | — | unchanged | unchanged |

Per-split val MAE @ best epoch 14 (vs stale baseline):

| Split | Stale base | This PR | Δ vs stale |
|---|---:|---:|---:|
| val_single_in_dist     | 97.074 | 85.773 | −11.64% |
| val_geom_camber_rc     | 94.997 | 87.713 | −7.67% |
| val_geom_camber_cruise | 63.711 | 60.859 | −4.48% |
| val_re_rand            | 83.266 | 76.171 | −8.52% |
| **val_avg**            | 84.762 | **77.629** | **−8.42%** |

Per-split test MAE @ best val checkpoint:

| Split | Stale base | This PR | Δ vs stale |
|---|---:|---:|---:|
| test_single_in_dist     | 85.819 | 75.126 | −12.46% |
| test_geom_camber_rc     | 83.023 | 74.247 | −10.57% |
| test_geom_camber_cruise | 54.879 | 52.661 | −4.04% |
| test_re_rand            | 74.916 | 70.005 | −6.55% |
| **test_avg**            | 74.659 | **68.010** | **−8.91%** |

- **Strong result on stale stack.** Both val and test move together
  (~-8.4% / -8.9%), not val-overfit. All 4 splits improve. Gains
  concentrate on in-distribution / Re-axis splits (>7%) and are smaller on
  geom_camber_cruise (~4%, the already-lowest-error split).
- **Mechanism check — informative**. Final γ_l statistics per block:
  - Block 0 attn mean=0.087, std=0.029; mlp mean=0.115, std=0.039
  - Block 4 attn mean=0.084, std=0.020; mlp mean=0.079, std=0.020
  - Key observations: (1) model did NOT ramp γ_l up to CaiT-paper's
    expected [0.5, 1.5] range — means stay in [0.079, 0.115], essentially
    within ±15% of init=0.1. (2) Per-channel structure IS being used:
    std/mean ratio reaches 30% in block-0 attn — meaningful per-channel
    diversity. (3) MLP branches show clear depth-decreasing trend (0.115
    → 0.079) — earlier blocks contribute more, consistent with merged
    stoch-depth schedule already downweighting later blocks. **The win
    looks less like "channel-specific amplification of useful Fourier
    channels" (predicted) and more like "global attenuation of residual
    branches with per-channel preservation of useful signal" — a
    fine-grained learnable residual gate.**
- **Why sent back for rebase (not merged):** student ran on pre-#1772
  baseline (84.762, Fourier L=4 stack, `fun_dim: 36`). Current baseline is
  82.311 (Fourier L=6, `fun_dim: 44`). Despite `mergeable=CLEAN` (changes
  in TransolverBlock don't conflict with Fourier encoding changes), this
  branch has now sign-flipped two PRs after rebase: #1608 frieren EMA-0.999
  (-2.64% → +2.93%, low-pass smoothing fights Fourier spectral sharpening)
  and #1754 nezuko LR warmup (pending). Cost of a re-run is ~31 min of one
  GPU; benefit is a confirmed delta vs. current 82.311 baseline.
- **Mechanism expected to compound**: LayerScale per-channel residual gate
  is mechanistically orthogonal to Fourier input encoding. If anything,
  sharper input features should give *more* useful signal to gate per
  channel — strengthening rather than weakening the mechanism. Expected
  post-rebase: val_avg in [75, 78] if clean compound; [78, 81] if modest
  interference (still merge-eligible); >82.311 if active interference
  (unlikely).
- **Student's pre-registered follow-ups (queued, contingent on rebase
  confirmation):**
  1. Bracket init: try 0.05 (slower-ramp) and 0.3. If both converge to
     similar plateau, confirms the model chooses operating point and
     per-channel granularity is the load-bearing structure.
  2. Per-block init schedule `[0.5, 0.3, 0.2, 0.1, 0.05]` (Fixup-style) —
     interesting given the depth-decreasing trend in observed γ_l means.
  3. Longer training with adjusted cosine T_max to see if γ_l keeps
     drifting after the cosine cooldown finishes.
- Single arm. Wall time 31.4 min. Metrics: `models/model-charliepai2g24h4-thorfinn-layerscale-init-0.1-20260513-020358/metrics.{jsonl,yaml}`.

## 2026-05-13 03:05 — PR #1811 (tanjiro per-channel output head MLPs — H24) — **CLOSED**

- Branch: `charliepai2g24h4-tanjiro/output-head-per-channel-mlp`
- Hypothesis: replace shared `Linear(128→3)` final projection with three
  per-channel `Sequential(Linear(128→64), GELU, Linear(64→1))` heads;
  +3.72% params expected. Mechanism: per-channel decoding capacity for
  pressure vs velocity (different spatial-scale preferences).

| Metric | This PR | Current baseline (#1772) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 86.447 | 84.762 | **+1.99%** |
| test_avg/mae_surf_p (4-split) | 75.326 | 74.659 | +0.89% |
| Param count | 674,007 | 665,943 | +8,064 (+1.2%) |

(Note: this experiment ran on the #1548 baseline 84.762, before #1772
merged. The +1.99% delta is vs the baseline-at-submission, not the current
82.311. Either way, regression.)

- **Student found a critical confound in the PR spec.** The baseline
  `mlp2` was already `Sequential(Linear(128→128), GELU, Linear(128→3))`
  — a shared 128-hidden MLP, not a single Linear projection as the H24 PR
  spec assumed. Their replacement of that shared MLP with three 64-hidden
  per-channel MLPs effectively *halves* per-channel hidden capacity
  (128 → 64) while only adding +8K params (not the predicted +24K).
- **Per-split direction inverted the prediction**: `val_single_in_dist`
  (predicted to gain most as the highest-pressure-magnitude split) regressed
  **+5.34%** — the largest single-split hit. `val_geom_camber_cruise`
  +3.32%. Velocity-component MAE comparable to baseline; the regression
  is concentrated in pressure on in-dist.
- **No overfit signature**: train-val composite gap 10% at best ep,
  smooth monotonic descent.
- **Interpretation**: shared 128-hidden decoder is *better-suited* to
  high-magnitude pressure decoding than three 64-hidden specialists. The
  shared decoder finds cross-channel features (pressure-velocity
  correlations from physics) that channel specialists can't see. Decoder-
  side per-channel direction is closed regardless of capacity setting.
- **Axis-wide finding**: the merged ~666K stack is well-balanced at the
  decoder; capacity changes there are not the bottleneck. Future PRs
  should target different axes (input-side, loss-side, optimization-side).
- Follow-up assigned: PR #1852 (tanjiro coord jitter augmentation
  std=0.005) — fresh data-augmentation axis with zero parameter cost.

## 2026-05-13 02:50 — PR #1772 (edward Fourier coord encoding `n_freqs=6` — bracket up from merged L=4) — **MERGED**

- Branch: `charliepai2g24h4-edward/fourier-coords-L6`
- Hypothesis: pre-registered bracket-up from merged #1548 L=4. Tancik's curve
  predicts plateau at L=8-10; L=6 is the next probe on the upward slope.

**6th compound win on this branch.** New baseline 82.311 / 73.330.

| Metric | This PR | Previous baseline (#1548) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | **82.311** | 84.762 | **-2.89%** |
| test_avg/mae_surf_p (4-split) | **73.330** | 74.659 | **-1.78%** |
| Param count | 667,991 | 665,943 | +2,048 (+0.31%) |

- **All 4 val splits improve** (-0.91% to -4.10%): single_in_dist -3.89%,
  camber_rc -2.14%, camber_cruise -0.91%, re_rand -4.10%.
- **All 4 test splits improve** (-1.17% to -2.91%): single_in_dist -2.91%,
  camber_rc -1.39%, camber_cruise -1.43%, re_rand -1.17%.
- **Surprise finding**: `val_re_rand` improved -4.10% (pre-registered as
  "likely stays flat" since its OOD axis is Reynolds, not spatial frequency).
  Plausible mechanism: at L=4 the network was over-spending preprocess MLP
  capacity on encoding geometry in low-frequency bands; with L=6 it can push
  geometry into higher Fourier bands and free MLP capacity for Reynolds-
  dependent features. Test gain on test_re_rand (-1.17%) is smaller but
  consistent — argues against pure noise.
- **`val_geom_camber_cruise` only -0.91%**: pre-registered as a strong gainer
  (it was -7.94% at L=4); marginal gain is plateauing first on this split.
  Hypothesis: cruise split already extracted most spatial-freq info at L=4,
  and the marginal L=4→6 frequencies are beyond its dominant surface-pressure
  modes. Leading-edge plateau indicator.
- **No overfit signature**: best ep = 15 (cosine endpoint), wall time
  unchanged, no early plateau in val curve.
- **Magnitude at upper end of predicted band** (-0.5% to -2.5% predicted,
  -2.89% actual). Two reads: (a) L=4→L=6 jump is steeper than Tancik's
  curve at this dimensional regime; (b) L=4 result was on the noisy side
  of its run-to-run distribution. Either way, still on the upward slope.
- Compound progress: 100.957 → 84.762 → **82.311** = **-18.5% over 6 merges**.
- Follow-up assigned: PR #1830 (edward Fourier L=8 bracket-up to find plateau).

## 2026-05-13 02:35 — PR #1608 (frieren EMA-of-model-weights decay=0.999) — **CLOSED**

- Branch: `charliepai2g24h4-frieren/ema-weights-0.999`
- Hypothesis: EMA of model weights at decay=0.999, swap-in for val/save.
  Smooths the optimizer trajectory via exponential moving average; expected
  to compound with merged stoch-depth (orthogonal variance reduction).

**Pre-rebase run (vs old 98.353 baseline)**: -2.64% val, -3.08% test win.
**Rebased run (vs current 84.762 baseline)**: +2.93% val, +4.12% test
regression. Sign flipped after rebase onto the merged stack.

| Metric | This PR (rebased) | Current baseline (#1548) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 87.244 | 84.762 | **+2.93%** |
| test_avg/mae_surf_p (4-split) | 77.738 | 74.659 | **+4.12%** |
| Param count | 665,943 | 665,943 | unchanged (EMA is in-memory only) |

- **All 4 val splits regress or flat** (val_geom_camber_rc -0.66% only).
  Largest hits: `val_single_in_dist` +6.17%, `val_geom_camber_cruise` +5.63%.
- **All 4 test splits regress** uniformly (largest: test_single_in_dist
  +7.80%, test_geom_camber_cruise +4.95%).
- **Student's mechanism analysis is sharp and tracks the per-split pattern**:
  the Fourier-gain splits (#1548 saw val_single_in_dist -11.35%, test_single_in_dist
  also massive gain) regress most under EMA — clean inverse correlation.
  EMA's low-pass smoothing on weights smooths the high-frequency Fourier
  feature responses that gave us the -8.10% test gain.
- **EMA window vs cosine T_max=15 misalignment**: decay=0.999 → effective
  averaging window ≈ 2.7 epochs at batch=4. With T_max=15, the live model
  is in 5-50× LR cooldown for the final ~3 epochs; the EMA copy trails
  the live model into the cooldown rather than absorbing it.
- **Stoch-depth + cosine cooldown already absorb most optimizer variance**:
  EMA's variance-reduction effect is mostly double-counted on this stack.
- **Mechanism finding (axis-wide)**: weight-space smoothing on this compound
  is closed. Fights spectral-bias features. Future variance-reduction PRs
  must NOT operate on the weights directly; should target either the loss
  landscape (SmoothL1 — picked next) or trajectory features (e.g., SAM).
- Val curve was perfectly monotonic at every epoch (implementation correct;
  result is a clean negative not a bug).
- Follow-up assigned: PR #1828 (frieren SmoothL1 / Huber loss β=0.01).
  Loss-landscape smoothing rather than weight-space smoothing — should not
  fight spectral-bias features.

## 2026-05-13 02:15 — PR #1754 (nezuko linear LR warmup + cosine T_max=14 — H19) — **SENT BACK FOR REBASE**

- Branch: `charliepai2g24h4-nezuko/lr-warmup-h19`
- Hypothesis: linear LR warmup over epoch 1 (per-batch, total_iters=375)
  + CosineAnnealingLR(T_max=14*375=5250). Addresses ep1 pre-clip grad-norm
  spike (60-100) consistently observed in recent grad-clip experiments.

**This is a WIN on the old baseline** — but measured against pre-#1548
baseline (90.294), not current 84.762. Sent back for rebase + re-run.

| Metric | This PR (vs old) | Old baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 89.718 | 90.294 | **-0.64%** |
| test_avg/mae_surf_p (4-split) | 79.852 | 81.243 | **-1.71%** |

- **Per-split val MAE (3/4 splits improve)**:
  - val_single_in_dist     -1.94%
  - val_geom_camber_rc     -1.37%
  - val_geom_camber_cruise -0.33%
  - val_re_rand            +1.68% (small split-specific noise)
- **All 4 test splits improve** (avg -1.71%; test gain exceeds val gain).
- **Mechanism check passes**: ep1 last-batch pre-clip grad-norm dropped
  ~35% (99 → 65); LR trace matches design (peak at end ep1, half at ep8,
  zero at ep15).
- **Implementation refinement** from the student: PR draft suggested
  per-epoch SequentialLR (coarse 2-point step); student moved scheduler
  inside batch loop with `total_iters=375, milestones=[375], T_max=5250`
  for smooth per-batch ramp. Sharper than the original spec.
- **Sent back for rebase**: technically MERGEABLE (no textual conflict)
  but base is `0668de7` (pre-#1548 Fourier merge); current is `90b33ba`.
  Run must be re-measured against new baseline 84.762.
- Expected post-rebase: -0.3% to -1.5% on val_avg (warmup mechanism is
  orthogonal to Fourier input encoding).

## 2026-05-13 02:10 — PR #1756 (tanjiro stoch-depth drop_rate=0.15 — H bracket-up) — **CLOSED**

- Branch: `charliepai2g24h4-tanjiro/stoch-depth-0.15`
- Hypothesis: pre-registered bracket-up follow-up from closed #1612 at 0.05.
  Push schedule above merged 0.10 to `[0.0, 0.0375, 0.075, 0.1125, 0.15]`.

| Metric | This PR | Old baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 97.235 | 90.294 | **+7.69%** |
| test_avg/mae_surf_p (4-split) | 87.236 | 81.243 | +7.38% |
| Param count | 662,359 | 662,359 | unchanged |

- **All four val splits regress uniformly** (+5.31% to +12.14%); largest
  hit on val_single_in_dist +12.14%.
- **Outcome C confirmed**: the merged 0.10 is the genuine local optimum
  of the single-knob bracket. {0.05 → +13.7%, 0.10 → 0%, 0.15 → +7.69%}
  is a clear asymmetric V around 0.10.
- **Student's sharp finding**: both endpoints regress on val_geom_camber_rc
  (+13.33% at 0.05 from #1612; +5.31% at 0.15 here). The "OOD geometry
  wants more regularization" narrative is now falsified on BOTH sides of
  the bracket — should not appear in future regularization PR hypotheses.
- **Train-vs-val gap direction**: val > train (standard generalization
  gap), NOT train > val (which would have been the ensemble-dropout
  signature). Independent evidence that 0.15 is operating as just-more-noise,
  not stronger ensemble.
- **Mechanism limit**: with n_layers=5 and last block never dropped, the
  effective per-step drop variance at p=0.15 is still small — explains
  why higher drop rates don't unlock new ensemble behavior at this depth.
- Single-knob stoch-depth direction fully closed. Future regularization
  PRs should target different mechanism (per-layer schedule shape, weight
  decay, label smoothing, or output head reshaping).
- Follow-up assigned: PR #1811 (tanjiro per-channel output head MLPs).

## 2026-05-13 02:00 — PR #1773 (thorfinn AdamW betas (0.9, 0.95) — H22) — **CLOSED**

- Branch: `charliepai2g24h4-thorfinn/adamw-betas-0.95`
- Hypothesis: change AdamW `betas` from PyTorch default `(0.9, 0.999)` to
  `(0.9, 0.95)` for faster second-moment EMA adaptation. Mechanism: long
  EMA horizon (1000 steps) lags the cosine-annealed gradient regime; 20-step
  EMA tracks distribution shifts. Modern transformer recipe (LLaMA, PaLM).

| Metric | This PR | Baseline (#1548) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 86.427 | 84.762 | **+1.97%** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 75.863 | 74.659 | **+1.61%** |
| Best epoch | 15/15 | 15/15 | same |
| Param count | 665,943 | 665,943 | unchanged |

- **Per-split val MAE (non-uniform regression)**:
  - val_single_in_dist     +0.92%
  - val_geom_camber_rc     +1.97%
  - val_geom_camber_cruise **+5.47%** (largest hit — low-noise-floor regime)
  - val_re_rand            +0.50%
- **Per-split test MAE**:
  - test_single_in_dist     +0.71%
  - test_geom_camber_rc     +0.48%
  - test_geom_camber_cruise +3.41%
  - test_re_rand            +2.58%
- **Two falsified predictions confirm direction is closed**:
  1. Best epoch did NOT shift earlier (faster basin-finding falsified)
  2. Per-split direction NOT uniform (optimizer-as-global-mechanism framing falsified)
- **Deepest mechanism finding**: the merged stack already addresses the
  non-stationarity concerns that motivated H22. Grad-clip-25 truncates the
  epoch-1 99.48 spike *before* AdamW sees it. Cosine T_max=15 anticipates
  the LR regime change rather than asking AdamW to react. β₂=0.95 was
  solving a problem that no longer existed.
- **Regime gap**: LLaMA/PaLM use β₂=0.95 at batch=10³-10⁶ × ours, 10⁵-10⁶
  steps. Our regime (batch=4, 5,625 steps) doesn't benefit from short-EMA.
- Single-knob optimizer-betas direction closed on this dataset.
- Follow-up assigned: PR #1799 (thorfinn LayerScale CaiT-style init=0.1).

## 2026-05-13 01:15 — PR #1548 (edward Fourier coords L=4 — rebased) — **MERGED (new baseline)**

- Branch: `charliepai2g24h4-edward/fourier-coords-L4-rebased`
- Hypothesis: Tancik et al. (NeurIPS 2020) Fourier positional encoding on
  spatial `(x, z)` coords addresses spectral bias on the surface-pressure
  signal. Encode normalized coords with `sin/cos` at `2^k · π` for `k=0..3`
  (16 features), bumping `fun_dim` from 22 to 36.

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | **84.762** | 90.294 | **−6.13%** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **74.659** | 81.243 | **−8.10%** |
| n_params | 665,943 | 660,539 | +0.82% |
| peak GPU memory | 42.16 GB | 42.11 GB | +0.12% |
| wall time | ~31.5 min | ~30 min | cap-bound |

- **Per-split val MAE** (best ckpt):
  - val_single_in_dist     97.074 (−11.35% vs 109.497) — largest gain
  - val_geom_camber_rc     94.997 (−4.00% vs 98.952)
  - val_geom_camber_cruise 63.711 (−7.94% vs 69.208)
  - val_re_rand            83.266 (−0.30% vs 83.520) — flat, as predicted
- **Per-split test MAE** (best val ckpt):
  - test_single_in_dist     85.819 (−13.43%)
  - test_geom_camber_rc     83.023 (−7.47%)
  - test_geom_camber_cruise 54.879 (−4.03%)
  - test_re_rand            74.916 (−5.10%)
- **Mechanism confirmed**: split pattern matches spectral-bias hypothesis
  exactly — gains where high-frequency spatial structure dominates
  (in_dist, camber-OOD); minimal on val_re_rand whose OOD axis is Reynolds
  (flow-condition), not spatial coords.
- **Test gain exceeds val gain** (−8.10% vs −6.13%): the Fourier features
  generalize to held-out data better than they fit val. Strong signal for
  the paper-facing test metric.
- **Stacks cleanly with all 4 prior compound merges**: L1 → stoch-depth →
  cosine T_max=15 → grad-clip 25 → Fourier L=4. Compound progress over 5
  merges: val_avg 100.957 → 84.762 = **−16.0%**.
- Metrics: `models/model-charliepai2g24h4-edward-fourier-coords-L4-rebased-20260512-235326/metrics.jsonl`
- Follow-up assigned: PR #1772 (edward Fourier L=6 bracket-up).

## 2026-05-13 01:13 — PR #1699 (thorfinn attn+MLP dropout p=0.05) — **CLOSED**

- Branch: `charliepai2g24h4-thorfinn/attn-mlp-dropout-0.05`
- Hypothesis: standard ViT-style attn+MLP dropout p=0.05 orthogonal to
  block-level stoch-depth would add a fine-grained regularization layer.

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/15) | 92.342 | 90.294 | **+2.27%** |
| test_avg/mae_surf_p (4-split) | 83.382 | 81.243 | +2.63% |
| wall_time / epoch | 134s | 120s | **+12%** (not "free") |
| n_params | 662,359 | 660,539 | +0% (Dropout adds zero params) |

- **All four val splits regress uniformly** (+0.67% to +2.84%). Uniform
  direction confirms the "regularization is a global mechanism" framing,
  but the sign is the opposite of the hypothesis.
- **Three mechanisms drove the regression**, in order of impact (from
  student's own analysis):
  1. **Compute tax eats one epoch.** Dropout adds ~14s/epoch; run hit
     30-min cap at ep 14/15 instead of 15/15. The cosine T_max=15 hadn't
     fully decayed at ep 14 (lr≈5.5e-6, not zero). One additional cosine
     step would have closed part of the gap.
  2. **Stoch-depth was already at the regularization optimum.** Merged
     schedule averages 0.05 across blocks; adding p=0.05 inside surviving
     blocks pushed past the optimum for this 1499-sample/15-epoch budget.
  3. **Post-softmax slice-attention dropout disrupts unit-sum property.**
     `slice_weights` is used twice (soft binning + soft scatter); dropout
     zeros 5% then rescales surviving ones. The `slice_norm = slice_weights.sum(2)`
     renormalization partially mitigates but the double application of
     dropped weights amplifies effective noise above standard attention dropout.
- Single-knob fine-grained dropout direction closed on this baseline.
- Follow-up assigned: PR #1773 (thorfinn AdamW betas (0.9, 0.95) — clean
  pivot to orthogonal optimizer-recipe axis).

## 2026-05-13 01:00 — PR #1713 (askeladd grad-clip max_norm=10 — H15 bracket below) — **CLOSED**

- Branch: `charliepai2g24h4-askeladd/grad-clip-10`
- Hypothesis: bracket below merged max_norm=25 (single-line edit). Completes
  the fixed-threshold sweep around 25.

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 94.121 | 90.294 | **+4.24%** |
| test_avg/mae_surf_p (4-split) | 84.561 | 81.243 | +4.08% |
| Epochs with pre-clip norm > threshold | 15/15 (100%) | n/a | — |

- **Outcome B confirmed: max_norm=25 is the local optimum.** Bracket
  geometry: +5.4% (1.0) / +4.24% (10) / 0% (25) / +3.32% (50). Asymmetric
  — tighter direction (10 → +4.24%) costs more than looser direction
  (50 → +3.32%). Heavy 30–70 norms carry partial signal; clipping them
  to 25 helps via variance reduction, clipping all the way to 10 destroys
  signal.
- **All four val splits regress uniformly** (largest hit val_single_in_dist
  +6.16%, smallest val_re_rand +2.85%). Same uniform-direction pattern
  as #1637 (the merged win) and #1674 (the upper-bracket loss), confirming
  this is a global regularization mechanism.
- **15/15 epochs had pre-clip norms above the threshold** — the entire
  training trajectory was continuously clipped, basically destroying the
  per-step gradient direction. The 100% clipping is qualitatively
  different from #1674's 40% — too aggressive.
- **Fixed-threshold grad-clip direction now closed.** Bracket fully
  characterized. Next: adaptive clipping (running-quantile based) per
  the PR's pre-registered follow-up.

## 2026-05-13 00:55 — PR #1677 (nezuko H12 per-node adaptive temperature) — **CLOSED**

- Branch: `charliepai2g24h4-nezuko/per-node-temp-h12`
- Hypothesis: per-node deterministic temperature `τ_i = τ_0 + Linear(x_mid)_i`
  clamped at floor 0.1. Identity-init (zero linear weights). Attacks
  slice-collapse without sampling noise (clean pivot from #1553 Gumbel).

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/15) | 93.097 | 90.294 | **+3.11%** |
| test_avg/mae_surf_p (4-split) | 82.813 | 81.243 | +1.93% |
| n_params | 662,509 | 662,359 | +150 (+660 nominal, weight tying counted) |
| τ.std @ ep 14 | 0.340 | n/a | — |
| τ floor_fraction @ ep 8 | 0.361 | n/a | — |

- **Mechanism verified, outcome rejected.** τ head learned: identity-init
  → non-trivial spread (std=0.34, range [0.10, 1.80]) by best epoch.
  But 36% floor-fraction by ep8 indicates the model is pushing roughly
  a third of nodes to maximally sharp slice assignments. The PR pre-flagged
  this as a "binding too often" signal.
- **Per-split regression concentrated on splits the hypothesis predicted
  would benefit most** — val_single_in_dist (+5.29 absolute MAE),
  val_geom_camber_rc (+5.03). The cruise and re_rand splits are nearly
  neutral, as expected since they have less boundary-layer structure.
- **Student's mechanistic interpretation:** "aggressive sharpening on
  boundary-layer nodes commits the slice assignment too early, before
  the slice-token MLP has converged on good slice-level representations."
  Plausible and matches the per-split asymmetry.
- **Slice-collapse direction closed.** Three independent arms have now
  failed:
  - #1514 Ada-Temp v1/v2 (per-head scalar τ) — closed +3.4%
  - #1553 Gumbel-Softmax slice noise — closed +4.4% (3-run mean)
  - #1677 H12 per-node deterministic τ — closed +3.11%
- The wave-5-candidate H12-followup-floor-sweep is **dropped**: re-perturbing
  the same dimension we've now shown doesn't carry the signal would just
  burn GPU on a closed direction.

## 2026-05-13 00:50 — PR #1612 (tanjiro stoch-depth drop_rate=0.05) — **CLOSED**

- Branch: `charliepai2g24h4-tanjiro/stoch-depth-0.05`
- Hypothesis: halve the linear stoch-depth schedule from
  `[0,0.025,0.05,0.075,0.10]` to `[0,0.0125,0.025,0.0375,0.05]`. The
  PR body's original target was the +1.77% regression on val_re_rand
  vs pre-#1552 baseline.

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13/14) | 102.65 | 90.294 | **+13.7%** |
| test_avg/mae_surf_p (4-split) | 91.264 | 81.243 | +12.3% |

- **Note: writeup compared against pre-#1611/#1637 baseline (98.353), giving
  +4.4%.** Against the current advisor HEAD baseline (90.294), the
  regression is +13.7%. The mechanistic conclusions still hold — they're
  about per-split direction, not absolute level.
- **Split-asymmetric response — the key finding.** val_re_rand DID recover
  as hypothesised (-3.03% from the prior over-regularization), but
  val_geom_camber_rc blew up by +13.33% in the opposite direction. OOD
  geometry splits want MORE regularization; the Re sweep wants LESS.
  A single global drop rate can't satisfy both.
- **No overfitting under merged 0.10** — train surf_loss 0.282 ≈ val
  surf_loss 0.285 at the best epoch. Cutting drop in half didn't liberate
  any frozen useful capacity; it just produced noisier gradients with
  weaker implicit ensembling.
- **Reproducibility check:** independent launch at 23:03 produced val_avg
  101.5 — same regression direction, small run-to-run noise.
- **Pivot direction: pre-registered drop_rate=0.15 follow-up.** Test
  whether hard OOD geometry splits want even more regularization. If
  0.15 helps val_geom_camber_rc while keeping val_re_rand neutral, that's
  a winner. If 0.15 also regresses, the stoch-depth single-knob direction
  is at its local optimum at 0.10.

## 2026-05-13 00:08 — PR #1675 (alphonse H17 per-channel output γ, β) — **CLOSED**

- Branch: `charliepai2g24h4-alphonse/out-scale-bias-h17`
- Hypothesis: 6 learnable parameters `γ ∈ ℝ³, β ∈ ℝ³` on the output head,
  identity-init, attack per-channel pressure-vs-velocity calibration
  without compression (the closed log1p direction).

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 93.214 | 90.294 | **+3.24%** |
| test_avg/mae_surf_p (4-split) | 83.108 | 81.243 | +2.30% |
| n_params | 662,365 | 662,359 | +6 |

- **Mechanism behaved exactly as predicted, outcome did not.** `out_gamma[2]`
  (pressure) drifted +6.13% by ep 15, vs +1.43% on Ux and +3.21% on Uy —
  optimizer found pressure-channel scale up = local loss minimizer, as
  hypothesised. Drift was smooth and monotone across all 15 epochs
  (gradient flow healthy).
- **All four val splits regressed**, largest hit on `val_single_in_dist`
  (+5.07%) — exactly the split the hypothesis predicted would benefit
  most (highest p magnitudes). The inversion is informative: identity-init
  guarantees zero regression at step 0, but with no penalty toward
  identity, the optimizer drifts wherever the *training* gradient pulls.
  A 6% multiplicative drift on already-correct large-magnitude pressure
  predictions inflates their MAE by ~6%.
- **Output-side calibration is exhausted on this dataset.** Trio of
  closures: #1610 (full-channel log1p +1.18%), #1636 (pressure-only log1p
  +5.32%), #1675 (per-channel γ, β +3.24%). The existing pre-training
  normalization is more useful than any post-hoc multiplicative correction.
- **Pivot direction:** student's own suggestion #1 — per-channel surf-loss
  weighting. Attack the same imbalance upstream at the loss layer instead
  of letting the model mis-calibrate the prediction.

## 2026-05-13 00:08 — PR #1674 (askeladd grad-clip max_norm=50 — H15 bracket above) — **CLOSED**

- Branch: `charliepai2g24h4-askeladd/grad-clip-50`
- Hypothesis: bracket above merged max_norm=25 to test whether pure
  spike-only suppression (the 110-norm at ep 8 in #1637) is sufficient
  or whether bulk 30–70 norm clipping is the active mechanism.

| Metric | This PR | Baseline (#1637, max_norm=25) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 93.286 | 90.294 | **+3.32%** |
| test_avg/mae_surf_p (4-split) | 83.882 | 81.243 | +3.25% |
| Epochs above 50 (clipped) | 6/15 (40%) | n/a | — |

- **Outcome B (PR-predicted): bulk 30–70 norms are the active ingredient
  at threshold 25.** Spike-only suppression at threshold 50 captures the
  signature pattern (ep5→6, ep10→11 "spike-down" mapping correct) but
  recovers only a fraction of the #1637 gain.
- **Uniform regression across all four splits**, largest on
  `val_single_in_dist` (+7.08%), smallest on `val_geom_camber_cruise`
  (+1.43%) — same direction as the merged #1637 win was uniform, just
  reversed.
- **Implication:** the variance-reduction-on-heavy-steps reading is right.
  The 25-threshold is clipping not just outliers but moderately heavy
  (~30–50) steps, and removing that clipping when threshold = 50 is the
  cost.
- **Pivot direction:** student's suggestion #1 — `max_norm=10` (lower
  bracket). If 10 wins → bracket further below; if 10 regresses → 25 is
  at local optimum and the fixed-threshold sweep is complete (next would
  be adaptive clipping schemes).

## 2026-05-12 23:55 — PR #1555 (thorfinn tied projection + n_hidden=144 retune) — **CLOSED**

- Branch: `charliepai2g24h4-thorfinn/remove-in-project-fx`
- Hypothesis: keep the tied projection (in_project_fx removed, slice pool
  reuses x_mid) but widen `n_hidden` 128 → 144 to reinvest the freed
  parameter budget across all weights.
- Rebased onto current advisor HEAD `05a8b35` (post #1552 + #1611 + #1637).

| Metric | This PR | Baseline (#1637) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 12/13) | 102.668 | 90.294 | **+13.71%** |
| test_avg/mae_surf_p (4-split) | 91.739 | 81.243 | +12.92% |
| n_params | 730,423 | 662,359 | +10.3% |
| Wall time/epoch | 143s | 120s | +19% |
| Epochs in 30-min cap | 13 | 15 | -2 |

- **All four val splits regressed** by 5-22%, not just in-distribution.
  Rules out the "wider over-parameterizes a small training set"
  interpretation in favor of a structural "wider doesn't help here" signal.
- **Root cause (per student diagnostic)**: the original n_hidden=144 retune
  hypothesis was framed against the pre-cosine, pre-grad-clip baseline at
  val_avg=98.353 where single_in_dist sat at 129.4 (in-distribution
  underfit was real). After #1611 cosine + #1637 grad-clip both merged,
  single_in_dist dropped to 109.5 *via optimization fixes alone* — the
  underfitting the retune was meant to fix had already been resolved.
- **Wall-clock budget cost** the rest: wider model = +19%/epoch =
  -2 epochs in the cap = cosine arc cuts off at LR=2.2e-5 instead of
  5e-6, losing the late-epoch fine-tuning that the merged baseline relies on.
- **Reusable structural constraint reaffirmed**: under the 30-min cap, any
  capacity-add must be free or near-free on wall-clock. Tanjiro #1545
  asymmetric Q/K's +40% step cost set the same constraint; this PR's +19%
  step cost reconfirms it.
- Pivoting thorfinn to attention/MLP dropout=0.05 — orthogonal to merged
  stoch-depth (block-level), zero compute overhead, standard ViT recipe.

## 2026-05-12 22:55 — PR #1637: Grad-clip max_norm=25 — **MERGED, new baseline**

- Branch: `charliepai2g24h4-askeladd/grad-clip-25`
- Hypothesis: H15 from wave-3 candidate pool. Diagnostic-informed follow-up
  to closed #1529 (`max_norm=1.0`, +5.4% regression). At `max_norm=25`,
  clipping fires on the outlier spikes (training grad norms range 22-110)
  without touching the typical 30-70 norms.

| Metric | This PR | Baseline (#1611) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | **90.294** | 94.217 | **-4.16%** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **81.243** | 84.859 | **-4.26%** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 109.497 / 98.952 / 69.208 / 83.520 | 114.200 / 102.157 / 73.321 / 87.188 | **-4.12% / -3.14% / -5.61% / -4.21%** |

- **All four val splits improved uniformly** (-3.14% to -5.61%) — exactly as
  the hypothesis predicted ("stable descent helps everywhere"), no
  split-specific direction.
- **Mechanism confirmed by the new `train/last_grad_norm` log**: 14/15 epochs
  had end-of-epoch grad_norm > 25 (clipping active throughout). Largest
  spike: 110.04 at epoch 8. Per-step rate likely higher than the
  per-end-of-epoch rate.
- **Cosine cooldown phase shows the biggest payoff**: the single largest
  epoch-to-epoch val_avg drop (-13.7%) coincides with the only epoch
  where end-of-epoch norm fell below the clip (22.40 at ep12).
- Best epoch at the wall-clock cap (15/15) again — same monotonic-descent
  pattern as the cosine-T_max-15 winner. The model is still improving when
  time runs out.
- Brackets the grad-clip direction: clip=1.0 (#1529, +5.4%) too aggressive,
  clip=25 (#1637, -4.16%) the sweet spot or near it. Natural next:
  bracket at clip=50 to test if pure spike-suppression is sufficient.

## 2026-05-12 22:55 — PR #1636 (alphonse pressure-only log1p) — **CLOSED**

- Branch: `charliepai2g24h4-alphonse/log1p-p-only`
- Hypothesis: H16. Targeted follow-up to closed #1610 (full-target log1p,
  +1.18% regression). Apply log1p ONLY to the pressure channel (the only
  genuinely heavy-tailed channel per #1610's `log_y_std`), keep Ux/Uy raw.

| Metric | This PR | Baseline (post #1611) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | 99.227 | 94.217 | **+5.32%** |
| test_avg/mae_surf_p (4-split) | 88.264 | 84.859 | +4.01% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 130.667 / 111.589 / 64.848 / 89.803 | 114.200 / 102.157 / 73.321 / 87.188 | **+14.42% / +9.23% / -11.56% / +3.00%** |

- **Channel-attribution theory falsified.** The per-split asymmetry observed
  in full-channel #1610 was preserved AND amplified by pressure-only log1p.
  High-peak splits (single_in_dist, camber_rc) regressed *more* than under
  full log1p; low-peak cruise gained more.
- Mechanism (per the student's writeup): log compression flattens the tail
  relative spacing the model relies on to discriminate extreme-pressure
  samples (raceCar Re up to 5M, |p| up to ~30k). After inverse expm1,
  small relative errors blow up multiplicatively at the tails — which are
  exactly the high-peak splits that dominate val_avg.
- Closed per the explicit PR rubric: "If p-only log1p still regresses, the
  entire log-compression direction is dead on this dataset/metric — we
  close and pivot to other channel-rebalancing ideas (e.g. H17)."
- Pivoting alphonse to H17 (learnable per-channel scale+bias on output)
  — addresses pressure calibration *without* compression.

## 2026-05-12 22:55 — PR #1553 (nezuko Gumbel-Softmax slices, tau=1.0) — **CLOSED**

- Branch: `charliepai2g24h4-nezuko/gumbel-slice`
- Hypothesis: replace softmax over slice weights with Gumbel-Softmax during
  training (deterministic softmax at eval) to sharpen slice assignments
  and attack the slice-collapse failure mode.
- Note: branch was never rebased onto the current baseline (still on the
  pre-#1552/#1611 base). Student reported three runs with mean ± std.

| Run | val_avg/mae_surf_p | Δ vs old base (100.957) | Δ vs current base (90.294) |
|-----|---:|---:|---:|
| ...-201404 | 102.827 | +1.85% | +13.89% |
| ...-205438 | 109.970 | +8.93% | +21.79% |
| ...-215442 (canonical) | 103.490 | +2.51% | +14.62% |
| **Mean ± std** | **105.43 ± 3.16** | **+4.43%** | **+16.77%** |

- **Hypothesis falsified across 3 independent runs.** Variance (~3 MAE units)
  rules out single-seed effects; all 3 underperform even the old L1 baseline.
- Failure mode (per student diagnostic): Gumbel sampling noise slows early
  convergence enough that the 30-min cap binds before the model reaches
  the deterministic baseline's asymptote. The eval-time deterministic
  softmax can't recover because the slice weights were trained against
  noisy targets.
- Closed not just because of the negative result on the old base, but
  because the current baseline stack (stoch-depth + cosine T_max=15 +
  grad-clip max_norm=25) is *mechanistically antagonistic* to additional
  gradient noise: stoch-depth already adds variance via block drop, cosine
  cooldown relies on stable gradients in the late phase, and grad-clip
  suppresses spikes that Gumbel noise would create. Layering Gumbel on
  top would worsen, not improve, the gap.
- Pivoting nezuko to H12 (per-node adaptive temperature) — different attack
  on slice-collapse that *doesn't* inject sampling noise.

## 2026-05-12 22:50 — PR #1553 (nezuko Gumbel-Softmax slices) — **SENT BACK** for rebase + re-run

- Branch: `charliepai2g24h4-nezuko/gumbel-slice` (still at `bc30b0a` — pre-#1552, pre-#1611)
- WIP for ~3h with zero commits beyond the original assignment. Pod GPU
  showed a single ~30-min training window (22:00-22:30Z @ 99%/71GB), then
  back to 0% with no artifacts pushed. Likely combination of training
  completing but the post-run commit/push blocked by GH API rate limit
  errors in the student pod's polling loop.
- Even on a successful completion, the result would have been measured
  against the pre-#1552 baseline (val_avg=100.957), not the current 94.217.
  The Gumbel-Softmax slice-collapse hypothesis is still genuinely worth
  testing — it's mechanistically orthogonal to stoch-depth and cosine LR.
- Sent back with explicit rebase + re-run + commit-artifacts directive.
  See PR #1553 comment chain.

## 2026-05-12 21:17 — PR #1611: Cosine T_max=15 alignment — **MERGED, new baseline**

- Branch: `charliepai2g24h4-askeladd/cosine-tmax-15`
- Hypothesis: H14 from wave-2 candidate pool. Change `T_max=MAX_EPOCHS=50` to
  `T_max=15` so the cosine LR decay completes over the actual training
  horizon (~13-15 epochs under the 30-min cap), instead of being ~30% complete
  with LR still at ~80% of peak when training terminates.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | **94.217** | 98.353 | **-4.21% (largest wave-2 gain)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **84.859** | 87.995 | **-3.57%** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 114.200 / 102.157 / 73.321 / 87.188 | 119.16 / 111.09 / 73.32 / 89.84 | **-4.16% / -8.04% / -0.00% / -2.95%** |

- **All four val splits neutral-to-positive** — every split improved or stayed flat.
  camber_rc had the biggest gain (-8.04%); cruise was the only flat one (was already
  the easiest split at 73.32 in #1552, hard to push lower).
- **LR trace confirmed the mechanism**: epoch 1 LR = 4.945e-4, epoch 14 = 5.463e-6,
  epoch 15 = 0.0. The full cosine cooldown phase now happens — under the old
  `T_max=50` setting, LR at epoch 15 was still ~4.0e-4 (80% of peak),
  i.e. the model never entered the fine-tuning phase.
- **Val MAE descended monotonically every epoch** — still improving at the
  wall-clock cap. The cooldown helps without pulling the optimum forward.
- **Action: MERGED** as new canonical baseline. Single-line change, zero added
  compute, zero added params. Subsequent PRs are now compared against 94.217 /
  84.859. The two pending rebase PRs (edward #1548 Fourier, fern #1549 FiLM)
  and the strong wave-2 EMA result (frieren #1608, val_avg=95.761) are all
  affected by this baseline shift and need re-evaluation.

## 2026-05-12 21:17 — PR #1608: EMA of model weights (decay=0.999) — **REQUEST CHANGES** (sent back to frieren for rebase onto new cosine baseline)

- Branch: `charliepai2g24h4-frieren/ema-weights-0.999`
- Hypothesis: H13 from wave-2 pool. Exponential moving average of model weights;
  validate and checkpoint using the EMA copy.

| Metric | This PR | Baseline (#1552) | Δ vs #1552 | New baseline (#1611) | Δ vs #1611 |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15) | 95.761 | 98.353 | **-2.64%** | 94.217 | **+1.64% (worse vs new)** |
| test_avg/mae_surf_p (4-split) | 85.286 | 87.995 | -3.08% | 84.859 | +0.50% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 115.69 / 107.67 / 71.88 / 87.81 | 119.16 / 111.09 / 73.32 / 89.84 | -2.91% / -3.08% / -1.96% / -2.26% | 114.20 / 102.16 / 73.32 / 87.19 | +1.30% / +5.39% / -1.96% / +0.71% |

- **Mechanism worked.** All four splits improved vs the #1552 base, monotonic
  val descent every epoch. Implementation correct: EMA swap-in for val, live
  weights restored after, EMA `state_dict()` saved to checkpoint, test load
  picks up EMA weights automatically.
- **But the #1611 cosine merge shifted the baseline.** EMA's run was on
  T_max=50 base; the new baseline has T_max=15. Standalone EMA gain (-2.64%)
  no longer beats the new baseline.
- **Action: SENT BACK with rebase spec.** EMA and cosine T_max alignment are
  mechanistically orthogonal (different optimizer-trajectory variance vs
  LR-schedule shape), so they should stack. Expected stacked val_avg: ~92-93
  if EMA's -2.64% effect carries over to the new base. Re-evaluate after rebase.

## 2026-05-12 21:17 — PR #1549: FiLM conditioning on global flow params — **REQUEST CHANGES** (sent back to fern for rebase — extraordinary signal)

- Branch: `charliepai2g24h4-fern/film-global-cond`
- Hypothesis: H10 from round-2 list. FiLM (Feature-wise Linear Modulation) of
  per-block features by global flow parameters (Reynolds, AoA, etc. from
  metadata). Bug fix in conditioning extraction: use node-0 instead of mean-
  pool over padded zeros (which collapsed the conditioning signal).

| Metric | This PR | L1 baseline (#1397) | Δ vs L1 | Current baseline (#1611) | Δ vs current |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13) | **81.291** | 100.957 | **-19.5%** | 94.217 | **-13.7% (huge gap)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **71.731** | NaN | first finite | 84.859 | **-15.5%** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 94.72 / 103.94 / 52.13 / 74.38 | 127.37 / 110.83 / 77.35 / 88.27 | -25.6% / -6.2% / -32.6% / -15.7% | 114.20 / 102.16 / 73.32 / 87.19 | -17.1% / +1.7% / -28.9% / -14.7% |
| n_params | 677,719 | 660,000 | +2.7% | 662,359 | +2.3% |

- **Largest single-experiment signal of round 2 by a wide margin.** Beats the
  L1-only baseline by 19.5% on val, beats the current (cosine+stoch-depth)
  baseline by 13.7% **even without those two improvements stacked**. cruise
  OOD split dropped 73.32 → 52.13 (-29%) — exactly the regime where global
  flow params (Re, AoA) should carry the most information.
- **Caveat: no stoch-depth, no cosine T_max=15.** fern's branch is older than
  both #1552 and #1611. The 13.7% gap suggests FiLM is doing dominant
  conditioning work, but the comparison can't be confirmed without stacking
  on the full current baseline.
- **Action: SENT BACK with rebase spec.** This is the **top-priority pending
  rebase** of the round. If FiLM + stoch-depth + cosine all stack, projected
  val_avg lands in the 78-84 range — a massive new baseline. If interference
  is severe, we choose between FiLM-only and the current baseline; the
  FiLM-only result (81.291) would still be a -13.7% improvement.

## 2026-05-12 21:17 — PR #1610: log1p target reparameterization (H11) — **CLOSED**

- Branch: `charliepai2g24h4-alphonse/log1p-target`
- Hypothesis: H11. Sign-preserving log1p of the target across all 3 channels
  (Ux, Uy, p), inverse-transform at metric time. Compresses heavy-tailed
  distribution and rebalances per-sample gradient magnitude.

| Metric | This PR | Baseline (#1552) | Δ vs #1552 | New baseline (#1611) |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13) | 99.513 | 98.353 | **+1.18% (regression)** | 94.217 (+5.62% vs new) |
| test_avg/mae_surf_p (4-split) | 89.586 | 87.995 | +1.81% | 84.859 |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 125.01 / 114.83 / 69.34 / 88.87 | 119.16 / 111.09 / 73.32 / 89.84 | +4.91% / +3.37% / -5.43% / -1.08% | — |

- **Diagnostic value is high.** log1p helps the lower-peak splits (cruise -5.43%,
  re_rand -1.08%) but hurts the high-peak ones (single_in_dist +4.91%,
  camber_rc +3.37%). The pressure channel's log_y_std=4.64 is ~4× the
  other two channels (1.12, 1.53) — it's the only heavy-tailed channel, and
  full log-compression flattens the surface stagnation peaks that
  `mae_surf_p` rewards.
- **Implementation was correct**: signed_log1p / signed_expm1 wired properly,
  stats recomputed on log-space targets, sanity checks pass (epoch-1
  surf_loss=1.28 in log space as expected, physical-unit MAE reported
  correctly at O(100)).
- **Action: CLOSED**. Pressure-only log1p (H16) is the natural targeted variant
  and is being assigned as alphonse's next hypothesis — the heavy-tailed
  channel that benefits from compression is isolated, while Ux/Uy stay in
  physical units.

## 2026-05-12 21:13 — PR #1548: Fourier coord encoding (L=4) — **REQUEST CHANGES** (sent back to edward for rebase onto stoch-depth baseline)

- Branch: `charliepai2g24h4-edward/fourier-coords-L4`
- Hypothesis: H7 from round-2 list. Add Fourier positional encoding to the (x,z)
  coords with L=4 frequency bands. Captures geometric structure that raw coords miss.

| Metric | This PR | L1 baseline (#1397) | Current baseline (#1552) |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/15) | **92.053** | 100.957 | 98.353 |
| Δ vs L1-only | **-8.82%** | — | -2.58% |
| Δ vs current best | **-6.40%** (numerical) | +2.65% | — |
| test_avg/mae_surf_p (4-split) | 83.980 | NaN | 87.995 |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 106.553 / 102.895 / 71.689 / 87.076 | 127.371 / 110.832 / 77.353 / 88.273 | 119.16 / 111.09 / 73.32 / 89.84 |
| n_params | 665,943 | 660,000 | 662,359 |

- **Strongest single-experiment signal of round 2.** Every val split improves vs
  the L1-only baseline by 1.4% to 16.4%, with the biggest gain on val_single_in_dist
  (-16.3%) — exactly the split the merged stoch-depth baseline only partially fixed.
  Test 4-split (83.980) also better than current baseline (87.995).
- **Caveat: train.py is missing the stoch-depth code from #1552.** Edward's branch
  is 8 commits behind advisor base; no `stoch_depth_prob` anywhere. So the
  comparison is Fourier-without-stoch-depth (92.053) vs stoch-depth-without-Fourier
  (98.353). We don't yet know whether the two stack (likely lands ~89-90 → huge
  win) or interfere (could regress back toward ~95).
- **Action: SENT BACK with rebase spec.** Edward to pull current advisor HEAD
  (which includes both stoch-depth and the NaN-safe pre-filter) and re-run with
  Fourier encoding on top. Expected outcomes flagged in the PR comment:
  stacks (clear merge), partial interference (still likely merge as Fourier-dominant),
  severe interference (we then choose between Fourier-only and stoch-depth-only).
- This is the highest-EV in-flight signal — wave 2 results may need to be re-evaluated
  against a Fourier+stoch-depth baseline once edward's rebase lands.

## 2026-05-12 21:00 — PR #1555: Remove `in_project_fx` (Transolver++ tied projection) — **REQUEST CHANGES** (sent back to thorfinn for n_hidden=144 follow-up)

- Branch: `charliepai2g24h4-thorfinn/remove-in-project-fx`
- Hypothesis: H3 from round-2 list. Remove redundant `in_project_fx` from
  `PhysicsAttention`, re-using `x_mid` as the value source in the slice-pooling
  einsum (Transolver++, arXiv 2502.02414). Acts as a structural prior + frees VRAM.
- Run was on L1+stoch-depth base (post-#1552), so direct apples-to-apples vs current baseline.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/15) | 99.898 | 98.353 | **+1.57% (slightly worse)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 89.532 | 87.995 | +1.75% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 129.395 / 108.141 / 72.436 / 89.619 | 119.16 / 111.09 / 73.32 / 89.84 | **+8.60% / -2.65% / -1.21% / -0.25%** |
| n_params | 579,799 | 662,359 | **-12.5%** |
| Peak GPU memory | 39.63 GB | 42.11 GB | **-5.9%** |
| Wall time/epoch | ~125 s | ~123 s | ~unchanged |

- **Pattern is a classic capacity-vs-regularization tradeoff.** The three OOD-flavored
  splits (camber_rc, camber_cruise, re_rand) all improve modestly; single_in_dist
  regresses by +8.6%, pulling val_avg net negative. The tied projection acts as a
  structural regularizer that helps OOD but underfits the in-distribution mode.
  Efficiency gains are real: -12.5% params and -5.9% VRAM at identical wall time.
- **Action: sent back with re-tune spec** — keep the tied projection, but reinvest
  the freed parameter budget (~83k params) and VRAM headroom by widening
  `n_hidden=128 → 144`. This redistributes capacity across all weights rather than
  concentrating it in a single redundant projection. Expected: single_in_dist
  recovers toward 119, OOD gains preserved → net improvement vs 98.353. Student
  must rebase onto current HEAD to include the merged stoch-depth code.

## 2026-05-12 21:00 — PR #1514: Ada-Temp v2 (shared-across-heads Δτ) — **CLOSED**

- Branch: `charliepai2g24h4-alphonse/ada-temp` (v2 force-push)
- Hypothesis: v2 follow-up to test alphonse's own diagnosis that extra per-head
  Δτ capacity hurt cross-regime transfer. v2 uses `Linear(dim, 1)` (shared-heads).
- Run was on L1-only base (pre-#1552), so compared against 100.957 not 98.353.

| Metric | v2 | L1 baseline (#1397) | v1 (per-head) | Current baseline (#1552) |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p | 104.366 | 100.957 | 101.770 | 98.353 |
| Δ vs L1 baseline | **+3.4% (worse)** | — | +0.81% | -2.58% |
| Δ vs current best | **+6.1% (worse)** | +2.65% | +3.47% | — |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 122.77 / 114.35 / 85.47 / 94.88 | 127.37 / 110.83 / 77.35 / 88.27 | 118.02 / 114.13 / 78.35 / 96.58 | 119.16 / 111.09 / 73.32 / 89.84 |
| Δ vs L1 per-split | -4.60 / +3.51 / **+8.12** / +6.60 | — | -9.35 / +3.30 / +1.00 / +8.31 | — |
| test_avg/mae_surf_p (4-split, NaN-safe) | 93.936 | NaN | NaN | 87.995 |

- **Both Ada-Temp variants are now exhausted.** v1 (per-head) regressed by +0.81%;
  v2 (shared-heads) regresses harder by +3.4% on val_avg.
- **The capacity-overfit hypothesis is partially contradicted.** Shared-heads
  narrowed v1's val_re_rand regression (+8.31 → +6.60) and partially preserved
  v1's val_single_in_dist gain (-9.35 → -4.60). But v2 introduced a new
  large regression on val_geom_camber_cruise (+1.00 → +8.12), which v1 didn't
  have. Removing per-head freedom collapses head specialization on the cruise
  regime that needed it most.
- **Action: CLOSED.** The NaN-safe pre-filter from this PR was independently
  preserved via #1552 (now standard in baseline). Slice-collapse is also being
  attacked via a different mechanism in #1553 (Gumbel-Softmax, WIP under nezuko).
  Alphonse's suggested follow-up (Eidetic Slice Embedding) goes on the
  wave-3 candidate pile for later revival if Gumbel-Softmax doesn't pan out.

## 2026-05-12 21:00 — PR #1547: Kendall uncertainty weighting — **CLOSED**

- Branch: `charliepai2g24h4-askeladd/kendall-uncertainty`
- Hypothesis: H6 from round-2 list. Replace hand-tuned `surf_weight=10` with
  learnable per-task log-sigmas (Kendall et al., CVPR 2018) so the surf/vol
  balance becomes data-driven.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14) | 103.544 | 98.353 | **+5.28% (worse)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 94.524 | 87.995 | **+7.42% (worse)** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 132.989 / 115.872 / 73.899 / 91.417 | 119.16 / 111.09 / 73.32 / 89.84 | +11.6% / +4.3% / +0.8% / +1.8% |
| Learned `log_sigma_surf` / `log_sigma_vol` | -0.288 / -0.079 | — | — |
| **Effective `surf_weight`** | **1.518** | 10.0 | -85% lower than hand-tuned |

- **Key diagnostic finding: the Kendall MLE objective is fundamentally misaligned
  with the physical evaluation metric.** Learned sigmas converged to
  effective_surf_weight=1.518, ~7× lower than the hand-tuned value of 10 that
  the baseline uses. Cross-referencing closed PRs #1403 (surf_weight=30, +5.1%
  worse) and #1530 (effective surf×P_WEIGHT=30, +1.22% worse), the empirical
  optimum for surf_weight is at or near 10, and learnable per-task likelihood
  pulls it the wrong way.
- **Lesson: learnable loss-balance objectives must align with the physical
  eval metric, not just calibrated likelihoods.** This rules out the entire
  family of MLE-style balance learning (Kendall, GradNorm, dynamic weight
  averaging) unless they're constrained to optimize the evaluation surrogate
  directly.
- **Action: CLOSED.** Clean negative result. No reasonable variant of the
  Kendall objective recovers the gap; the objective is the problem, not the
  parameterization.

## 2026-05-12 21:00 — PR #1545: Asymmetric Q/K slice projections (LinearNO) — **CLOSED**

- Branch: `charliepai2g24h4-tanjiro/asymmetric-qk`
- Hypothesis: H2 from round-2 list. Independent V and K slice projections in
  PhysicsAttention (LinearNO-style) — separate the slice-assignment basis from
  the value basis to enable richer slice tokens.

| Metric | This PR | Baseline (#1552) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 10) | 116.940 | 98.353 | **+18.90% (worse)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | 105.058 | 87.995 | **+19.39% (worse)** |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 141.506 / 132.312 / 86.757 / 107.185 | 119.16 / 111.09 / 73.32 / 89.84 | +18.7% / +19.1% / +18.3% / +19.3% |
| n_params | 672,919 | 662,999 | +9,920 (+1.5%) |
| Epochs reached in 30-min cap | **10** | 15 | **-33%** |

- **Compute-bound failure mode.** The mechanism is empirically active (block-3
  slice cos-sim = 0.097 confirms slice divergence), but the extra `in_project_slice_k`
  projection adds ~40% wall-clock cost per epoch. Run terminated at epoch 10
  vs the baseline's 15 — same compute budget, fewer effective gradient steps.
- The trajectory was still descending at termination but needed ~17 additional
  MAE points of improvement to match baseline, which is implausible in the
  remaining 5 epochs even with monotonic descent.
- **Structural lesson: architectural changes that add >10% per-step compute
  are unviable in our 30-min training regime, even when the mechanism is
  theoretically sound.** Future architectural changes must be parameter-additions,
  not compute-additions, OR be paired with a complementary efficiency-saving
  (e.g., the tied-projection direction that thorfinn is iterating on in #1555).
- **Action: CLOSED.** Direction is dead within current budget constraints;
  asymmetric Q/K could only be re-attempted at higher budget or paired with a
  compute-saving change.

## 2026-05-12 20:52 — PR #1552: Stochastic depth (drop_rate=0.1, linear schedule) — **MERGED, new baseline**

- Branch: `charliepai2g24h4-frieren/stoch-depth-0.1`
- Hypothesis: H8 from round-2 list. Add stochastic depth (Huang et al., ECCV 2016)
  with linearly increasing per-block drop probs `[0.0, 0.025, 0.05, 0.075, 0.10]`.
  Implicit ensemble of shallower networks for OOD regularization. No-op at eval.
  Predicted 1-3% improvement on `val_avg/mae_surf_p`, primarily via OOD geometry splits.
- Also includes the NaN-safe pre-filter in `evaluate_split` (standardized in every
  round-2 PR after #1530/#1529 independently discovered it).

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 15/15) | **98.353** | 100.957 | **-2.58% (improvement)** |
| test_avg/mae_surf_p (4-split, NaN-safe) | **87.995** | NaN (data bug) | **first finite 4-split ref** |
| test_avg/mae_surf_p (3-split, ex-cruise) | 96.579 | 100.831 | -4.22% |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 119.159 / 111.093 / 73.323 / 89.837 | 127.371 / 110.832 / 77.353 / 88.273 | **-6.45% / +0.24% / -5.21% / +1.77%** |
| Per-split test: single_in_dist / camber_rc / camber_cruise / re_rand | 104.953 / 101.883 / 62.243 / 82.901 | — | new finite ref |

- **The hypothesis held, but the OOD-specific framing was only half-supported.**
  Predicted gains were on OOD geometry splits (camber_rc, camber_cruise).
  Observed: camber_cruise -5.21% (large), camber_rc +0.24% (flat),
  single_in_dist -6.45% (largest gain), re_rand +1.77% (small regression).
  Student's reading: single_in_dist was the worst split at baseline despite
  being in-distribution, so it had the most regularization headroom.
  Stoch-depth's implicit ensemble flattens split-specific overfit modes
  regardless of the OOD axis.
- **Training dynamics:** val trace is noisier than L1 baseline (epoch 13: 105.69
  → epoch 14: 113.91 → epoch 15: 98.35 = new best). Bernoulli-block-drop noise
  injects variance into val. Best epoch landed at the wall-clock cap; more
  training time would likely extend the gain. The L1 baseline plateaued earlier
  at the same wall-clock budget, so stoch-depth is also getting more out of
  each minute of training.
- **Cosmetic NaN caveat:** loss/surf_loss aggregates for `test_geom_camber_cruise`
  still show NaN/Inf in `metrics.yaml` because the normalized-space loss path
  runs before the §3 pre-filter; the §3 fix only protects `accumulate_batch`.
  All four `mae_surf_p`/`mae_vol_p` channels are finite, so the primary ranking
  metric is clean. Out of scope; one-line follow-up.
- **Decision: MERGED.** First post-L1 architectural improvement; -2.58% on the
  primary metric and establishes the first finite 4-split test reference
  (87.995). Stoch-depth is now part of the canonical config; all subsequent
  wave-1 PRs in flight will be compared to this stronger baseline.
- **Suggested follow-ups (student):**
  1. Run longer — not actionable (`SENPAI_TIMEOUT_MINUTES` is a hard bound).
  2. Sweep `drop_rate` ∈ {0.05, 0.15, 0.20} — 0.05 might be Pareto-better given
     val_re_rand +1.77% suggests slight over-regularization; 0.15-0.20 might
     bite harder on val_geom_camber_rc which barely moved.
  3. Combine with `dropout` inside PhysicsAttention/MLP at 0.05 — standard
     ViT recipe, may compound with stoch-depth.
  4. Loss-NaN cosmetic fix — pre-filter finite samples before `y_norm` is
     formed so the normalized-space loss aggregates report finite numbers
     for `test_geom_camber_cruise`.

## 2026-05-12 20:02 — PR #1514: Ada-Temp per-point adaptive slice temperature — **REQUEST CHANGES** (sent back to alphonse for v2)

- Branch: `charliepai2g24h4-alphonse/ada-temp`
- Hypothesis: H1 from round-2 list. Replace scalar `self.temperature` with
  `τᵢ = τ₀ + Linear(dim, heads)(xᵢ)`, zero-init the projection so the model
  starts identical to baseline (Transolver++, arXiv 2502.02414).

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 13/14) | 101.770 | 100.957 | +0.81% (slightly worse) |
| test_avg/mae_surf_p (3-split, ex-cruise) | 100.825 | 100.831 | -0.007 (effectively flat) |
| test_avg/mae_surf_p (4-split) | NaN (no NaN-safe fix in v1) | NaN | — |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 118.023 / 114.128 / 78.348 / 96.582 | 127.371 / 110.832 / 77.353 / 88.273 | **-9.3 / +3.3 / +1.0 / +8.3** |

- **Per-split signal is the key story.** Ada-Temp helps single-foil in-distribution
  by ~9.3 (~7.3% gain) but regresses on val_re_rand by ~8.3 (~9.4% loss). The
  geometry-camber splits drift slightly worse. Net val_avg is essentially flat
  (slight regression) and test 3-split mean is statistically indistinguishable.
- **Implementation contribution worth recording**: alphonse identified that
  `Transolver.__init__` calls `self.apply(self._init_weights)` *after* `temp_proj`
  is zero-initialized, and `_init_weights` re-initializes every `nn.Linear` with
  `trunc_normal_(std=0.02)`. This silently breaks the "Δτ = 0 at step 0" invariant.
  Fix: re-zero loop after `self.apply(...)`. Without the fix an earlier run
  diverged from baseline from epoch 1. The committed run is the corrected version.
- **Diagnosis (student): extra per-head Δτ capacity hurts cross-regime transfer**
  inside a 30-min wall-clock budget. Single-foil in-dist benefits from sharper
  slice attention; tandem-flow OOD distributions cannot afford the extra
  capacity that lets the temperature head co-adapt to training-set spurious cues.
- **Action: sent back with v2 spec** — drop `temp_proj` from `Linear(dim, heads)`
  to `Linear(dim, 1)` (shared-across-heads Δτ), which cuts Ada-Temp's added
  capacity by ~75% (2,580 → 645 params). Direct test of the student's own
  capacity-overfit hypothesis. Also adds the NaN-safe pre-filter so v2 will
  report a finite 4-split test mean. Student suggested 4 follow-ups; v2 picks
  #2 (shared-across-heads), with #3 (last-blocks-only) as a stack-on if v2
  partially works and #4 (combine with Eidetic Slice Embedding) as a
  wave-3 candidate if v2 fails. Suggestion #1 (length-budgeted retest)
  is not actionable (`SENPAI_TIMEOUT_MINUTES` is a hard bound).

## 2026-05-12 19:55 — Stale-WIP closures: 5 PRs branched off pre-L1 MSE base

Five round-1 PRs (#1407 wider/deeper, #1411 slice_num=128, #1417 lr-warmup=1e-3,
#1420 EMA weights, #1425 SwiGLU FFN) were assigned at 17:52 UTC, before L1 loss
(PR #1397) merged at 19:05. Student pods were stalled on GH API rate limits
through 19:50 and never started training. Closing because any result on those
branches would be measured against pre-L1 MSE base and not directly comparable
to the new L1 baseline. All five hypotheses remain valid avenues to revive
in a later round; they are deprioritized for round 2 in favour of architecture
and loss-formulation ideas from `RESEARCH_IDEAS_2026-05-12_round2.md`.

## 2026-05-12 19:50 — PR #1530: Per-channel L1 loss with pressure x3 weight — **CLOSED, worse than L1**

- Branch: `charliepai2g24h4-tanjiro/channel-weight-p3`
- Hypothesis: H4 from round-2 list. In L1 loss, multiply pressure channel by
  P_WEIGHT=3.0 to steer gradient flow toward the ranking metric `mae_surf_p`.
  Predicted 2-6% improvement on `val_avg/mae_surf_p`.

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 14/14) | 102.184 | 100.957 | **+1.22% (worse)** |
| test_avg/mae_surf_p (3-split, ex-cruise) | 100.696 | 100.831 | -0.13% |
| test_avg/mae_surf_p (**4-split, NaN-safe, new finite ref**) | **92.465** | NaN | — |
| Per-split val: single_in_dist / camber_rc / camber_cruise / re_rand | 126.233 / 112.645 / 77.502 / 92.356 | 127.371 / 110.832 / 77.353 / 88.273 | mostly noise except +4.6% on re_rand |

- Effective combined surface-pressure weight became `surf_weight × P_WEIGHT = 30`,
  the same regime as closed PR #1403 (`surf_weight=30`), which also regressed.
  Student diagnosed this directly. The 3× upweight is too aggressive on top
  of L1's already-amplified surface gradients.
- **Lasting deliverable:** the NaN-safe pre-filter in `train.py::evaluate_split`
  works as designed and produced the first finite 4-split test mean on this
  branch (92.465). Pre-filter pattern is now bundled into every round-2 PR
  assignment so subsequent runs land a comparable 4-split test reference.
- Per-channel surface MAE at best val: surf_Ux=1.43, surf_Uy=0.69, surf_p=102.18;
  vol_Ux=4.81, vol_Uy=2.22, vol_p=103.80. Predicted Ux/Uy uptick in exchange for
  p drop did NOT materialize — we got a p regression instead.
- Suggested follow-ups (lower P_WEIGHT, combined surf_weight+P_WEIGHT sweep)
  are deferred until higher-EV round-2 levers are explored.

## 2026-05-12 19:48 — PR #1529: Gradient clipping (max_norm=1.0) — **CLOSED, much worse than L1**

- Branch: `charliepai2g24h4-askeladd/grad-clip-1.0`
- Hypothesis: H5 from round-2 list. Add `clip_grad_norm_(max_norm=1.0)` to
  reduce variance from gradient spikes on variable mesh sizes / high-Re samples.
  Predicted 1-4% improvement on `val_avg/mae_surf_p` via smoother convergence.

| Metric | This PR | L1 baseline (#1397) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p (best @ ep 11/14) | 106.401 | 100.957 | **+5.4% (worse)** |
| test_avg/mae_surf_p (3-split, ex-cruise) | 103.364 | 100.831 | +2.5% |
| test_avg/mae_surf_p (**4-split, NaN-safe, finite ref**) | **94.846** | NaN | — |

- Student logged per-epoch gradient norms (min 10, mean 47, max 245) and clip%
  per epoch (100% in every epoch). `max_norm=1.0` is far below the natural
  pre-clip norm of 10-245, so every step was rescaled by 0.02-0.10× — the
  model effectively trained at 1-5% of the configured LR throughout, which
  is too slow to converge inside 14 epochs.
- The diagnosis is exemplary post-hoc analysis and exactly the kind of
  per-epoch instrumentation we want from every arm.
- **Lasting deliverable:** NaN-safe pre-filter in `evaluate_split` (identical
  to tanjiro's #1530 fix). 4-split test mean (94.846) is reproducible and
  finite. Workaround is now standard in all round-2 assignments.
- Suggested follow-ups (`max_norm ∈ {10, 25, 50}`, AGC) deferred — higher-EV
  round-2 hypotheses have priority. If architecture/loss levers stall, we
  will return to AGC.

## 2026-05-12 19:10 — PR #1423: Enable unified_pos=True with ref=8 — **CLOSED, worse than L1**

- Branch: `charliepai2g24h4-tanjiro/unified-pos`
- Hypothesis: Switch `unified_pos=False → True, ref=8` — add learned ref-grid
  positional features (Gaussian-RBF over an 8×8 grid in the (x, z) plane,
  repeat-interleaved to fill `ref**3 = 512`) before the preprocess MLP.
- Student noted real implementation concerns: `ref**3 = 512` packing inflates 2D
  features 8× (only 64 distinct grid cells); grid bounds were adjusted to
  `[-7, 7]` to match the actual data range. Proposed multi-scale RBFs and
  asymmetric per-axis grid bounds as round-2 follow-ups.

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best @ ep 14/14) | 118.605 |
| test_avg/mae_surf_p (4-split, NaN-safe) | 109.159 |
| L1 baseline (PR #1397) | 100.957 / 100.831 |
| Delta vs L1 baseline | +17.5% **worse** |

- Note: trained on MSE base (branched before L1 merge), so this is MSE+unified_pos
  rather than L1+unified_pos. Comparison is contaminated. Closed without rebase
  because (a) the absolute number is ~17% worse than L1, (b) re-running would
  consume 30 min for a hypothesis whose own author flagged implementation
  concerns, and (c) higher-EV ideas are queued. Multi-scale RBF variant may
  resurface later.
- Student also flagged the pre-existing `data/scoring.py` NaN-propagation bug
  (same one alphonse flagged in #1397) and committed a clean workaround in
  `evaluate_split`. We're propagating that workaround into all subsequent
  round-2 assignments.

## 2026-05-12 19:08 — PR #1403: Bump surf_weight 10 → 30 — **CLOSED, worse than L1**

- Branch: `charliepai2g24h4-askeladd/surf-weight-30`
- Hypothesis: Increase `surf_weight` from 10 → 30 to focus optimizer pressure on
  the surface field that drives the primary metric.

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best @ ep 12/14) | 133.386 |
| test_avg/mae_surf_p (4-split, NaN-safe re-eval) | 120.962 |
| L1 baseline (PR #1397) | 100.957 / 100.831 |
| Delta vs L1 baseline | +32.1% **worse** |

- Trained on MSE base (branched before L1 merge), so this is MSE+surf_weight=30.
  Under L1 (less outlier-sensitive than MSE) the optimal `surf_weight` is unlikely
  to be larger than the default 10. Closed without rebase: re-running would burn
  30 min on a single-value HP sweep when L1 already wins by 30%+. A proper
  L1+surf_weight sweep (10/15/25/50) is a small follow-up worth considering only
  if other levers stop moving.
- Student also flagged the pre-existing `data/scoring.py` NaN bug and produced an
  independent NaN-safe re-evaluation script. Confirmed root cause.

## 2026-05-12 19:05 — PR #1397: L1 (MAE) loss replaces MSE in normalized-space training — **MERGED, new baseline**

- Branch: `charliepai2g24h4-alphonse/l1-loss`
- Hypothesis: Align training loss with the eval metric (MAE). MSE
  over-weighted high-Re outlier nodes whose y range spans up to 29K with
  per-sample y std varying ~10× within a single split. Expected 2–8%
  improvement on `val_avg/mae_surf_p`.
- Implementation: `(pred - y_norm).abs()` replaces `(pred - y_norm)**2` in
  both the training inner loop and `evaluate_split`. Surface/volume
  decomposition and `surf_weight = 10.0` kept unchanged. All other HPs
  default.

| Metric | Value |
|---|---:|
| val_avg/mae_surf_p (best @ ep 13/14) | **100.9574** |
| test_avg/mae_surf_p (3-split, excl. cruise) | **100.8314** |
| test_avg/mae_surf_p (4-split, raw) | NaN (data bug) |
| val_single_in_dist / mae_surf_p | 127.371 |
| val_geom_camber_rc / mae_surf_p | 110.832 |
| val_geom_camber_cruise / mae_surf_p | 77.353 |
| val_re_rand / mae_surf_p | 88.273 |
| n_params | 0.66 M |
| peak GPU mem | 42.1 GB |
| wall time | 30.7 min (cut at SENPAI_TIMEOUT_MINUTES=30 after ep 14) |

- Metric artifacts (advisor branch): `models/model-charliepai2g24h4-alphonse-l1-loss-20260512-175404/metrics.jsonl`, `metrics.yaml`
- Training trajectory was monotone-descending: ep 1 223 → ep 13 101; ep 14
  bounced to 134 right before timeout. Cosine T_max=50 means LR only
  decayed ~16% from peak by ep 14 — schedule is mismatched to the 30-min
  wall-clock cap. Worth a follow-up arm.

### Conclusions and follow-ups

- L1 loss is a clear win and establishes the first numeric baseline on
  this advisor branch. Merged.
- Pre-existing data bug: `test_geom_camber_cruise/000020.pt` contains
  `inf` in y_p, propagating NaN through `data/scoring.py::accumulate_batch`
  even though the bad sample is correctly flagged. `data/scoring.py` is
  marked read-only, so we record the 3-split test mean and document the
  bug. Fix candidate for a later PR: in `train.py::evaluate_split`, pre-mask
  non-finite y samples by zeroing both the sample's `mask` and its y
  values before calling `accumulate_batch` (faithful trainer-side
  workaround that preserves the scoring contract).
- Round-2 candidate follow-ups suggested by student (in addition to the
  Round-2 idea file H1-H11): T_max=15 to align cosine with the 30-min
  wall-clock cap; small `surf_weight` sweep on top of L1 (10/15/25/50)
  since L1 is less outlier-dominated than MSE; Huber/SmoothL1 as a
  smooth alternative.


