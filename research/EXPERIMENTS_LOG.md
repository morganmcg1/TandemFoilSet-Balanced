# SENPAI Research Results — icml-appendix-charlie-pai2g-24h-r5

## 2026-05-13 16:41 — PR #2405: Lion β1 sweep β1=0.85 (WINNER — new baseline 43.73/41.86)

- Student branch: `charliepai2g24h5-askeladd/lion-beta1-sweep`
- Hypothesis: Lion's β1 controls how much the current gradient contributes to the sign update vs. the EMA momentum. Lower β1 → more reactive (current grad weighted heavier); higher β1 → more inertial (EMA dominated). Two-arm sweep: β1∈{0.85, 0.95} vs default 0.9.

### Results (vs GeGLU baseline #2287: val=45.92, test=44.35)

| Arm | β1 | best_epoch | val_avg/mae_surf_p | Δval | test_avg/mae_surf_p | Δtest |
|---|---:|---:|---:|---:|---:|---:|
| **A** | **0.85** | 15 | **43.73** | **−4.8%** | **41.86** | **−5.6%** |
| B | 0.95 | 15 | 54.97 | +19.7% | 52.21 | +17.7% |

**Arm A (β1=0.85) beats baseline on both val and test, evaluated at only 15/16 epochs (30-min cap hit at epoch 15).** Arm B (β1=0.95) catastrophically regresses.

### Per-split (Arm A β1=0.85, epoch 15)

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 48.34 | 41.42 |
| geom_camber_rc | 56.87 | 50.62 |
| geom_camber_cruise | 26.95 | 40.33 |
| re_rand | 42.77 | 35.09 |
| **avg** | **43.73** | **41.86** |

### grad_norm trace

| Epoch | Arm A (β1=0.85) | Arm B (β1=0.95) |
|---:|---:|---:|
| 1  | 28.83 | 18.99 |
| 4  |  8.43 |  5.16 |
| 8  |  5.50 |  3.62 |
| 12 |  3.84 |  2.48 |
| 15 |  3.16 |  1.98 |

Monotonically decaying grad_norm for both arms. No instability at β1=0.85.

### Metric artifacts
- `models/model-lion_beta1_0p85_20260513-152032-20260513-152034/metrics.jsonl` (Arm A)
- `models/model-charliepai2g24h5-askeladd-lion_beta1_0p95-20260513-155636/metrics.jsonl` (Arm B)

### Analysis

**Strong confirmation of the direction-smoothness axis.** β1=0.85 (more current-gradient weight inside Lion's sign) converges faster on the val MAE objective; β1=0.95 (sign dominated by stale EMA) significantly underperforms. With B=4 (noisy gradients), the conventional wisdom would favor a higher β1 to smooth noise — but empirically the opposite holds. The per-step Lion sign update at high β1 is dominated by an EMA that cannot track the rapidly changing gradient field of CFD with small batches. β1=0.85 balances freshness and smoothing.

The result is conservative: 15/16 epochs, so the 16th epoch would likely have improved Arm A further.

**New baseline: val=43.73, test=41.86** (pending merge of #2405).

### Follow-ups
- Try β1∈{0.80, 0.875} to determine whether optimum is even more reactive or 0.85 is near the sweet spot
- β1=0.85 + lion_lr=2.5e-4 (more reactive sign + larger step, potentially additive)

---

## 2026-05-13 16:15 — PR #2315: RMSNorm on GeGLU stack (CLOSED — pod stalled, no result)

- Student branch: `charliepai2g24h5-thorfinn/rmsnorm-vs-layernorm`
- Hypothesis: Replace all `nn.LayerNorm` with scale-only `RMSNorm` (LLaMA recipe, SwiGLU/GeGLU + RMSNorm co-change).

### Outcome

**No result — pod stalled.** Over 3.5h after assignment:
- 0 student commits, 0 student comments
- `M train.py` (uncommitted RMSNorm edits) at iteration 89 (`grep RMSNorm train.py` not verified)
- GPU dropped from 100% → 0% by iteration 93
- 2 advisor status checks (15:25, 15:51) received but not actioned

The pod is alive (heartbeating) but no training is running. Closing to free the queue and reassigning thorfinn to a simpler single-line config change (slice_num=48) to determine whether the pod is genuinely stuck or struggling with multi-step edits.

**Hypothesis remains untested.** RMSNorm + GeGLU is a real co-change worth revisiting. Re-assign in a future cycle if pod stability improves.

---

## 2026-05-13 16:15 — PR #2403: GeGLU mlp_ratio=2 swiglu_hidden 216→320 (SENT BACK — budget-bound, not architecture-bound)

- Student branch: `charliepai2g24h5-tanjiro/geglu-mlp-ratio-2`
- Hypothesis: Wider GeGLU MLP (swiglu_hidden 216→320, +48% MLP capacity) should compound on GeGLU win.

### Results (vs GeGLU baseline #2287: 45.92/44.35)

| Metric | swiglu_hidden=320 | Baseline | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 48.13 | 45.92 | +2.21 (+4.8%) |
| test_avg/mae_surf_p | 46.19 | 44.35 | +1.84 (+4.2%) |

**14/16 epochs completed** — 30-min cap hit. Per-epoch overhead was ~20% (vs ~5% expected). Val still descending at **−4.5/epoch** at termination.

### Mechanism analysis (student)

The wider GeGLU MLP has higher memory-bandwidth cost than estimated. The 20% per-epoch overhead truncated the run by 2 epochs. Linear extrapolation suggests val would have hit 39-43 range at epoch 16 — **potentially competitive with or beating baseline 45.92**. Result is inconclusive: budget-bound, not architecture-bound.

### Disposition

**SENT BACK** to test `swiglu_hidden=256` (mlp_ratio≈1.6, +20% capacity). Should fit ~10% per-epoch overhead → comfortable 16-epoch budget. Direct test of "is more MLP capacity helpful?" without the wall-clock confound.

- Metrics: `models/model-charliepai2g24h5-tanjiro-geglu_mlp_ratio2_20260513-152412-20260513-152415/metrics.jsonl`

---

## 2026-05-13 15:55 — PR #2005: surf_weight=15 on GeGLU stack (CLOSED — both axes regress)

- Student branch: `charliepai2g24h5-nezuko/surf-weight-sweep`
- Hypothesis: Raise surf_weight 10→15 to align loss with surface-p metric on the GeGLU+Lion+per-channel-δ+n160 stack.

### Results (vs GeGLU baseline #2287: 45.92/44.35)

| Metric | sw=15 | Baseline sw=10 | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **46.8959** | **45.92** | **+0.98 (+2.13%)** |
| **test_avg/mae_surf_p** | **44.5852** | **44.35** | **+0.24 (+0.54%)** |

Per-split val regressed on all 4 splits (+0.27 to +2.20). Per-split test: 3 of 4 splits regressed, only test_single_in_dist improved (−1.16). 15 of 16 epochs completed (30-min cap).

### Mechanism

Raising surf_weight 50% increased the surface term's gradient contribution but the *raw* per-element surface loss equilibrated similarly to baseline at epoch 15. With Lion's sign-quantized update (per-parameter ±lr regardless of gradient magnitude), reweighting two loss terms doesn't change the per-parameter step magnitude — it only changes which parameters get pushed each step. The model finds a similar minimum via a less efficient path.

**Principle:** Loss-balance reweighting is a weak knob with Lion's sign update; effective only if it shifts which parameters reach saturation in the sign quantization, not by reweighting per-step magnitudes.

### Disposition

**CLOSED.** Reassigned nezuko to n_layers=4 (cost-recovery probe complementing fern's #2349 n_layers=6 budget-cliff result).

- Metrics: `models/model-surf_weight_15_pcd_lr2e4-20260513-145644/metrics.jsonl`

---

## 2026-05-13 15:55 — PR #2352: Lion wd sweep on GeGLU stack (CLOSED — primary val not beaten, axis shallow)

- Student branch: `charliepai2g24h5-edward/wd-sweep-geglu-baseline`
- Hypothesis: Real wd above FP32 ulp floor may help on GeGLU stack (wd∈{2e-3, 5e-3}). #2177 showed wd<1.49e-4 at lr=2e-4 is FP32 ulp no-op.

### Results (vs GeGLU baseline #2287: 45.92/44.35)

| Arm | val_avg | Δ vs 45.92 | test_avg | Δ vs 44.35 |
|---|---:|---:|---:|---:|
| Baseline | 45.92 | — | 44.35 | — |
| Arm A wd=2e-3 | 46.49 | +1.21% (worse) | 44.28 | −0.15% (tied) |
| Arm B wd=5e-3 | 45.96 | +0.08% (noise) | 43.90 | **−1.01% (better)** |

### Mechanism

Param L2 grows ~58% from init at both wd=2e-3 and wd=5e-3 — the sign-update gradient growth dominates the wd shrink (theoretical per-step shrink at wd=5e-3, lr=2e-4 is ~0.000001 per step, vs sign-update magnitude of lr=2e-4 per parameter per step). The wd is "firing" but its magnitude is negligible compared to Lion's growth driver. Differences across arms are within seed noise.

The non-monotonicity (Arm A worse than Arm B on val) confirms noise dominates: a real wd-induced regularization curve would be smooth in wd.

**Test improvement on Arm B (−1.01%) is informative but unconfirmed:** without a seed pair this could be checkpoint-pick luck. The student suggested seed pair as follow-up.

### Disposition

**CLOSED.** Primary val not beaten on either arm. Student's own analysis says wd axis is shallow on this stack. GPU better spent on architectural axes (5 GeGLU follow-ups already in flight). Reassigned edward to n_head=8 sweep.

**Recorded principle:** wd ∈ {6e-5, 2e-3, 5e-3} at lr=2e-4 with Lion+GeGLU all within ±0.6 val of baseline — wd axis is shallow and noise-dominated. Do not assign further wd sweeps on this stack without architectural changes.

- Metrics: `models/model-charliepai2g24h5-edward-wd_2e3_geglu-20260513-141614/metrics.jsonl`, `models/model-charliepai2g24h5-edward-wd_5e3_geglu-20260513-145355/metrics.jsonl`

---

## 2026-05-13 15:03 — PR #1844: Lion β2=0.999 on GeGLU stack (CLOSED — warmup cost dominates)

- Student branch: `charliepai2g24h5-askeladd/lion-beta2-0_999`
- Hypothesis: β2=0.999 (vs default 0.99) gives ~10× longer momentum EMA timescale, potentially smoothing noisy B=4 sign updates.

### Results (vs GeGLU baseline #2287: 45.92/44.35)

| Metric | β2=0.999 | GeGLU baseline | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **48.83** | **45.92** | **+2.91 (+6.3%)** |
| **test_avg/mae_surf_p** | **46.36** | **44.35** | **+2.01 (+4.5%)** |

All 4 val and test splits regress uniformly (+1.5 to +3.7 MAE). Epochs completed: 15/16 (both hit 30-min cap).

### Per-epoch trajectory (β2=0.999 vs β2=0.99 baseline)

| Epoch | β2=0.999 val | β2=0.99 val | Δ |
|---:|---:|---:|---:|
| 1 | 217.62 | 181.18 | +36.44 |
| 5 | 102.11 | 111.66 | −9.55 |
| 10 | 65.33 | 71.03 | −5.70 |
| 13 | 53.76 | 51.37 | +2.39 |
| 15 (best) | 48.83 | 45.92 | +2.91 |

β2=0.999 is much slower in the first 4 epochs (~10× longer EMA timescale → ~1000 steps to settle vs ~100), briefly ahead at ep5-12, then falls behind in the cosine tail.

### Mechanism

β2=0.999 warmup cost is borne in epochs 1-4 (val loses 30+ MAE vs baseline). The model catches up mid-training but never fully recovers — the decisive convergence gain in the cosine tail (ep13-15) belongs to β2=0.99. Mechanistically sound hypothesis (smoother sign directions for noisy B=4 gradients) but the warm-up cost dominates within 30-min/16-epoch cap. With 25-30 epochs available, β2=0.999 could plausibly match baseline.

### Disposition

**CLOSED.** β2=0.99 confirmed optimal for this budget. Reassigned askeladd to Lion β1 sweep (instantaneous step direction weight, separate axis from EMA timescale).

- Metrics: `models/model-charliepai2g24h5-askeladd-lion_beta2_0999_geglu_stack-20260513-141846/metrics.jsonl`

---

## 2026-05-13 15:03 — PR #2332: SwiGLU preprocess entry projector — tanjiro (CLOSED — gating at low-dim input too lossy)

- Student branch: `charliepai2g24h5-tanjiro/swiglu-preprocess-mlp`
- Hypothesis: Replace GELU entry projector (24→320→160) with SwiGLU gating (hidden=280, param parity). If GeGLU gating generalises from block-MLP to all projectors, this stacks another selective routing pass.

### Results (vs SwiGLU baseline #2196: 47.43/45.01)

| Metric | SwiGLU preprocess | Baseline | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **52.54** | **47.43** | **+10.8%** |
| **test_avg/mae_surf_p** | **50.07** | **45.01** | **+11.2%** |

All 8 splits regress +9% to +17%. Worst hits: geom_camber_cruise (+16% val) and re_rand test (+17.4%). 15/16 epochs completed.

### Mechanism

Block-MLP gating works because gating operates on a normalized, 160-dim feature manifold — enough diversity for selective routing. The preprocess block runs on 24 heterogeneous raw physics dimensions (positions, AoA, Re, NACA codes, gap/stagger). At input dim=24, multiplicative gating **discards information** rather than routing it — there's no second chance to recover the lost channel interactions. The slowest-converging splits (camber_cruise, re_rand) rely most on fine-grained channel mixing from the dense GELU MLP; SwiGLU's sparsification is pure information loss here.

**Principle extracted:** Gating works at scale (input dim ≥ 160) — not as a universal architectural primitive at low input dims.

### Disposition

**CLOSED (dead end).** Gating below ~32 input dims discards more than it selects. Reassigned tanjiro to GeGLU mlp_ratio sweep (increase block-MLP hidden 216→320, giving the gate more capacity at scale).

- Metrics: `models/model-charliepai2g24h5-tanjiro-swiglu_preprocess_h280-20260513-140813/metrics.jsonl`

---

## 2026-05-13 15:03 — PR #2349: n_layers=6 on GeGLU stack — fern (CLOSED — budget-starved, not architecturally broken)

- Student branch: `charliepai2g24h5-fern/n-layers-6-geglu-stack`
- Hypothesis: An extra GeGLU gating pass (6th TransolverBlock) should compound on #2287 — especially for the hardest OOD splits.

### Results (vs GeGLU baseline #2287: 45.92/44.35)

| Metric | n_layers=6 | n_layers=5 (baseline) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **50.80** | **45.92** | **+10.6%** |
| **test_avg/mae_surf_p** | **47.96** | **44.35** | **+8.1%** |

12/13 epochs completed (hit 30-min wall cap mid-epoch 13). Val still descending at −4.0 per epoch at termination.

| Split | n_layers=6 val | Baseline val | Δ |
|---|---:|---:|---:|
| single_in_dist | 56.73 | 48.87 | +7.86 |
| geom_camber_rc | 62.39 | 58.78 | +3.61 |
| geom_camber_cruise | 32.63 | 29.99 | +2.64 |
| re_rand | 51.46 | 46.03 | +5.43 |
| **avg** | **50.80** | **45.92** | **+4.88** |

| Run characteristic | Value |
|---|---|
| n_params | 1,223,803 (+18% vs baseline ~1.04M) |
| Peak VRAM | 50.24 GB (+18% vs baseline 42.5 GB) |
| s/epoch | ~149 s (+18% vs baseline ~126 s) |
| Epochs completed | 12 of 13, hit 30-min cap |

### Mechanism

The extra TransolverBlock costs +18% per-epoch time. Under the 30-min hard cap:
- n_layers=5: ~15 epochs → well into cosine tail
- n_layers=6: ~12 epochs → still in steep-descent regime (slope −4.0 at ep12 vs baseline's ~−0.27 in the cosine tail)

The +4.88 val gap ≈ the slope at termination: 2-3 more epochs would likely close most of it. The depth hypothesis is **not architecturally broken — just budget-starved**.

### Disposition

**CLOSED.** +10.6% val regression at termination exceeds the 5% threshold. Reassigned fern to attention output gating (GeGLU-style gate in PhysicsAttention.to_out — directly extends the gating architecture to the attention output path, single axis, param-parity bottleneck at hidden=56).

Note: n_layers=4 is a viable opposite test — one fewer block → more epochs in budget.

- Metrics: `models/model-charliepai2g24h5-fern-n_layers6_geglu_20260513-141316-20260513-141319/metrics.jsonl`

---

## 2026-05-13 14:20 — PR #2287: GeGLU gate ablation — SiLU→GELU inside SwiGLU block-MLP gate (MERGED — all 8 splits improve, new best val=45.92/test=44.35)

- Student branch: `charliepai2g24h5-fern/geglu-gate-ablation`
- Hypothesis: SwiGLU (#2196) won with gate×SiLU. Does the choice of activation inside the gate matter? GeGLU (gate×GELU) tests whether GELU's gradient surface — which Lion is calibrated for — carries into the gate path.

### Results (vs SwiGLU baseline #2196: 47.43/45.01)

| Metric | SwiGLU (#2196) | GeGLU (this PR) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 47.43 | **45.92** | **−1.51 (−3.2%) ← NEW BEST** |
| **test_avg/mae_surf_p** | 45.01 | **44.35** | **−0.66 (−1.5%)** |

All 8 splits improve. Largest val gain: single_in_dist (−3.32). All test splits improve too.

### Per-split

| Split | val SwiGLU | **val GeGLU** | Δval | test SwiGLU | **test GeGLU** | Δtest |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 52.19 | **48.87** | −3.32 | 43.52 | **43.19** | −0.33 |
| geom_camber_rc | 59.75 | **58.78** | −0.97 | 53.81 | **52.54** | −1.27 |
| geom_camber_cruise | 30.87 | **29.99** | −0.88 | 43.91 | **43.55** | −0.36 |
| re_rand | 46.90 | **46.03** | −0.87 | 38.82 | **38.14** | −0.68 |

### Mechanism

1-character change: `F.silu` → `F.gelu` inside the SwiGLU gate input path. The gate form `out = w_out(F.gelu(w_in(x)) * w_gate(x))` is GeGLU. This resolves the apparent paradox:
- Bare GELU→SiLU regresses (#2176): Lion sign-updates tuned for GELU gradient surface
- SwiGLU (gate×SiLU) wins (#2196): gating is the architectural primitive, SiLU tolerated inside gate
- GeGLU (gate×GELU) wins MORE (#2287): gating + GELU's optimizer-aligned gradient = both benefits simultaneously

Fern's analysis: "the gate activation matters because Lion's sign quantization is sensitive to the per-element gradient magnitude distribution. GELU's characteristic shape (near-zero for negative inputs, approximately linear for large positive inputs) provides the smoothest gradient signal for Lion's sign computation."

### Disposition

**MERGED.** New baseline: val=45.92/test=44.35. Reassigned fern to **n_layers=6 on GeGLU stack** (#2349): depth test with gated MLPs.

- Metrics: `models/model-geglu_gate_vs_swiglu-20260513-130428/metrics.jsonl`

---

## 2026-05-13 14:15 — PR #2177 re-arm: Lion wd sweep at firing values wd∈{5e-4, 2e-3} (CLOSED — Arm D ties GELU baseline; regresses vs GeGLU baseline)

- Student branch: `charliepai2g24h5-edward/wd-sweep-lr2e4`
- Hypothesis (re-arm from FP32 ulp diagnostic): wd∈{5e-4, 2e-3} both exceed the FP32 ulp floor (wd > 1.49e-4 at lr=2e-4) so they genuinely apply per-step weight shrink. Will wd improve on the effective-wd=0 baseline?

### Results (vs GELU baseline #1656: 52.63/49.22; run on old GELU stack)

| Metric | GELU baseline | Arm C (wd=5e-4) | Arm D (wd=2e-3) | Best vs baseline |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 52.63 | 53.72 | **52.53** | −0.10 (−0.2%, essentially tied) |
| **test_avg/mae_surf_p** | 49.22 | 50.56 | **49.22** | ≈0 (tie) |

vs current GeGLU baseline (45.92): Arm D val=52.53 (+14.4% worse). Both arms are dead ends vs current best.

### Mechanism

Per edward's trajectory analysis: Arm D shows the expected param_l2 growth pattern (66.40 vs 65.19 baseline) and higher late-epoch grad_norm (2.67 vs 2.61), confirming real weight decay fires and keeps gradients slightly more active. But the *magnitude* of the effect is negligible on the GELU stack — wd is near-optimal at 0 for this problem. Split pattern: Arm D helps single_in_dist/cruise/re_rand, slightly hurts geom_camber_rc — net wash.

### Disposition

**CLOSED.** Ran on GELU stack which is now outdated. The wd axis is still worth probing on GeGLU stack — assigned to edward as PR #2352 with wd∈{2e-3, 5e-3}.

- Metrics: `models/model-wd_5e4_lr2e4-20260513-120642/metrics.jsonl`, `models/model-wd_2e3_lr2e4-20260513-125114/metrics.jsonl`

---

## 2026-05-13 14:05 — PR #2181: batch_size=8 Lion sign-vote test (CLOSED — epoch-budget cliff at fixed lr; step-count halved and undertrained)

- Student branch: `charliepai2g24h5-tanjiro/batch8-lion-sign-vote`
- Hypothesis: Larger batch reduces gradient noise before Lion's sign quantization, producing higher-quality sign votes and potentially lower MAE.

### Results (vs current baseline #2196: 47.43/45.01)

| Metric | Batch=8 (this run) | Baseline #2196 | Δ vs #2196 |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 64.91 | **47.43** | **+17.48 (+36.9% worse)** |
| **test_avg/mae_surf_p** | 60.29 | **45.01** | **+15.28 (+33.9% worse)** |

All 8 splits regress heavily (single_in_dist val +26.55, geom_camber_rc +15.80, cruise +12.98, re_rand +14.57).

### Mechanism

Tanjiro's analysis is definitive: B=8 halves the per-epoch step count (188 vs 376 at B=4). Lion's sign update has `step magnitude = lr × 1.0` — gradient magnitude does not enter, so the trajectory-per-epoch scales purely with step count. At the same LR and same epochs, B=8 moves the model only half as far through parameter space. The run got ~2820 optimizer steps vs baseline's ~6016. The per-epoch curve confirms severe under-training: val=64.91 at epoch 15, slope −1.8/epoch (vs baseline ~−0.27/epoch at end). Model is still in steep descent.

The `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` mitigation for VRAM fragmentation worked cleanly (77.5 GB peak, no OOM).

### Why the hypothesis is untestable at fixed lr=2e-4

The natural fix is linear LR scaling: `lr_B8 = 4e-4` (2× lr to compensate 2× fewer steps per epoch). But lr=3.5e-4 already proved unstable on the GELU stack (#2035). LR bowl on SwiGLU stack is being probed by frieren (#2288, lr∈{2.5e-4, 3e-4}) — let that land before committing to lr=4e-4 at B=8.

### Disposition

**CLOSED.** Dead end at fixed lr; requires LR compensation first. Confirmed dead end: "B=8 at lr=2e-4 is under-training-bound; untestable without LR scaling". Reassigned tanjiro to **SwiGLU preprocess MLP** (#2332): extend gating from block MLPs to the mesh-feature entry projector.

- Metrics: `models/model-batch8_lion_n160_pcd-20260513-125655/metrics.jsonl`

---

## 2026-05-13 13:45 — PR #2249: Lookahead wrapper around Lion (k=5, α=0.5 vs 0.8) (CLOSED — epoch-budget cliff; anchor lag costs too much convergence in ≤16 epochs)

- Student branch: `charliepai2g24h5-thorfinn/lookahead-lion-wrapper`
- Hypothesis: Lookahead (Zhang et al. 2019, k=5 inner steps, α∈{0.5,0.8} outer-loop interpolation) reduces Lion sign-update oscillation at batch=4 via outer-loop EMA snap-back. Two arms isolate lag vs stability tradeoff.

### Results (vs baseline #2196: 47.43/45.01)

| Run | val_avg/mae_surf_p | test_avg/mae_surf_p | Δval vs #2196 | Δtest vs #2196 |
|---|---:|---:|---:|---:|
| Baseline (#2196) | **47.43** | **45.01** | — | — |
| Arm A (k=5, α=0.5) | 57.6092 | 54.4829 | +10.18 (+21.5% worse) | +9.47 (+21.0% worse) |
| Arm B (k=5, α=0.8) | 53.4533 | 50.3188 | +6.02 (+12.7% worse) | +5.31 (+11.8% worse) |

Also worse than old #1656 baseline (52.63/49.22): Arm A +9.5%/+10.7%, Arm B +1.6%/+2.2%.

### Per-split (Arm B, best arm)

| Split | Arm B val | Baseline val | Arm B test | Baseline test |
|---|---:|---:|---:|---:|
| single_in_dist | 57.86 | 52.19 | 49.26 | 43.52 |
| geom_camber_rc | 67.42 | 59.75 | 59.97 | 53.81 |
| geom_camber_cruise | 35.36 | 30.87 | 47.80 | 43.91 |
| re_rand | 53.17 | 46.90 | 44.25 | 38.82 |

### Mechanism

Thorfinn's per-epoch curve is the smoking gun. At epoch 16:
- Arm A slope: −1.11 val/epoch; Arm B slope: −0.75 val/epoch; Baseline slope: ~−0.27 val/epoch.
Both Lookahead arms are **still falling steeply** at epoch 16 — the model is catching up, not converged. The slow-weight anchor (averaging fast weights back every k=5 steps) imposes a lag penalty proportional to how fast the fast weights are moving. Lion's aggressive sign-update moves weights quickly in early epochs; Lookahead drags these back toward the lagged interpolant, costing convergence speed. With 16-epoch hard cap the model never recoups the lost ground. α=0.8 (Arm B) lags less and predictably outperforms α=0.5 (Arm A).

### Disposition

**CLOSED.** Not a viable direction within the 16-epoch budget. This is a budget-cliff failure mode analogous to DropPath (#2044) and EMA (#1596) — all averaging-over-time mechanisms need more epochs than available to amortise the convergence cost. Add "Lookahead+Lion requires ≥30 epochs to amortise lag cost" to confirmed dead ends. Reassigned thorfinn to **RMSNorm vs LayerNorm** (#2315): scale-only normalisation, LLaMA-recipe co-change with SwiGLU.

- Metrics: `models/model-lookahead_k5_a05-20260513-121250/metrics.jsonl`, `models/model-lookahead_k5_a08-20260513-124822/metrics.jsonl`

---

## 2026-05-13 13:00 — PR #2182: Layer-wise LR decay (LLRD factor=0.85) (CLOSED — Lion + shallow stack + from-scratch training incompatible with LLRD)

- Student branch: `charliepai2g24h5-frieren/layerwise-lr-decay`
- Hypothesis: BERT-style LLRD (factor=0.85 per block inward) improves OOD generalization by applying higher LR to later blocks (output representations) and lower LR to earlier blocks (input features).

### Results (vs baseline #1656: 52.63/49.22)

| Metric | LLRD f=0.85 | Baseline | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 56.4927 | 52.6345 | **+3.86 (+7.3% worse)** |
| **test_avg/mae_surf_p** | 52.6110 | 49.2183 | **+3.39 (+6.9% worse)** |

All 8 splits regress uniformly by 2.5–4.4 MAE. Effective per-group LRs at run end: blocks[4]=2e-4, blocks[3]=1.7e-4, blocks[2]=1.445e-4, blocks[1]=1.228e-4, blocks[0]+preprocess=1.044e-4.

### Mechanism analysis

Frieren's own analysis is excellent and definitive:
1. **Lion's sign-step is linearly LR-sensitive**: no adaptive preconditioning; a 50% lr cut = 50% step cut, no recovery possible.
2. **Model is too shallow**: BERT-12-layers uses 0.85^11≈0.17 decay (gentle). Transolver-5 uses 0.85^4≈0.52, starving input-side blocks of gradient by 50%.
3. **Trained from scratch**: BERT/ViT LLRD protects pretrained early-layer representations. Transolver's preprocess + blocks[0] ARE the primary feature extractors — they need full lr, not protection.
4. **Already-regularized stack**: dropout + per-channel Huber + grad_clip are providing regularization. LR-side regularization on top just suppresses learning.

### Disposition

**CLOSED.** Do NOT use LLRD factor ≤ 0.85 on this stack. "Inverted LLRD" (higher LR on input-side) is theoretically interesting but lower priority. Reassigned frieren to **Lion lr sweep on SwiGLU baseline** (#2288): confirm whether the merged SwiGLU architecture shifts the lr optimum from 2e-4.

- Metrics: `models/model-llrd_factor085_n160_pcd-20260513-115539/metrics.jsonl`

---

## 2026-05-13 12:55 — PR #2196: SwiGLU gated MLP replacing GELU in block MLPs (MERGED — largest single-PR gain since Lion, val −9.9%)

- Student branch: `charliepai2g24h5-fern/swiglu-gated-mlp`
- Hypothesis: Bare SiLU activation swap (PR #2176) regressed every split. But gated GLU-family variants — where the activation is multiplied by a learned gate — are the actual source of "SiLU wins" in modern transformer papers. SwiGLU at parameter parity (hidden=216 = ceil(160×4/3)) tests whether multiplicative gating, not slope, is the load-bearing factor.

### Results (vs baseline #1656: 52.63/49.22)

| Metric | SwiGLU (this run) | Baseline | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **47.4287** | 52.6345 | **−5.20 (−9.9%) ← NEW BEST** |
| **test_avg/mae_surf_p** | **45.0147** | 49.2183 | **−4.21 (−8.6%)** |

Every single split improves, val and test. Broad gain — not concentrated on one regime.

### Per-split

| Split | val SwiGLU | val baseline | Δval | test SwiGLU | test baseline | Δtest |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 52.19 | 56.52 | −4.33 | 43.52 | 47.14 | −3.62 |
| geom_camber_rc | 59.75 | 67.35 | **−7.60** | 53.81 | 59.44 | −5.63 |
| geom_camber_cruise | 30.87 | 34.17 | −3.30 | 43.91 | 46.76 | −2.85 |
| re_rand | 46.90 | 52.50 | −5.60 | 38.82 | 43.54 | −4.72 |

Largest gain on val_geom_camber_rc (67.35→59.75, −7.60) — the hardest OOD split. Gate most valuable for uncertain input regimes where GELU over-weights ambiguous features.

### Implementation

`SwiGLU` class with `out = w_out(silu(w_in(x)) * w_gate(x))`. Hardcoded into `TransolverBlock.mlp` with `swiglu_hidden = ((n_hidden*4//3)+7)//8*8 = 216` for hidden_dim=160. Block-MLP only — preprocess MLP and mlp2 head remain GELU.

Config: n_params=1,034,183 (≈param-matched to GELU baseline). Peak VRAM=42.5 GB (+3.7 GB vs GELU). Training hit 30-min cap at epoch 15 (of 16 planned) — epoch 15 was best val.

One note: `test_geom_camber_cruise/loss = NaN` (pre-existing scoring quirk in data/scoring.py for non-finite GT volume nodes). MAE for that split is finite (43.91) and correct — scoring code skips non-finite GT correctly.

### Mechanism

GELU→SiLU slope swap (PR #2176) hurt because Lion is calibrated to GELU's gradient surface. SwiGLU doesn't change the slope — it adds a multiplicative gate that selectively suppresses low-confidence feature channels. The gate adds learned per-channel routing capacity without growing params (same mlp_ratio at 4/3 vs 2 for GELU). Hardest OOD split benefits most (geom_camber_rc −7.6): gate is most useful where uncertainty is highest.

### Disposition

**MERGED.** New baseline: val=47.43/test=45.01. Assigned follow-ups:
- fern → **GeGLU gate ablation** (#2287): SiLU→GELU inside the gate. Disentangles whether SiLU-in-gate is load-bearing or gating itself is the primitive.
- frieren → **Lion lr sweep on SwiGLU stack** (#2288): confirms whether lr=2e-4 remains optimal with SwiGLU's changed gradient surface.

- Metrics: `models/model-swiglu_mlp_ratio_4_3_n160-20260513-115449/metrics.jsonl`
- Result link: https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/2196#issuecomment-4441183779

---

## 2026-05-13 12:00 — PR #2177: Lion weight_decay sweep at lr=2e-4 (RE-ARMED — wd is FP32 ulp no-op for wd ≤ 1.49e-4)

- Student branch: `charliepai2g24h5-edward/wd-sweep-lr2e4`
- Hypothesis: Lion `wd` was set as wd=lr/5 at lr=3e-4 (giving wd=6e-5) and never re-tuned when lr moved to 2e-4. Probe wd ∈ {4e-5, 8e-5} for an optimum-shift signal.

### Results (vs baseline #1656: 52.63/49.22)

| Metric | Baseline (wd=6e-5) | Arm A (wd=4e-5) | Arm B (wd=8e-5) | Δ |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 52.63 | **52.6345** | **52.6345** | **0.0000** |
| **test_avg/mae_surf_p** | 49.22 | **49.2183** | **49.2183** | **0.0000** |

**Both arms produced bit-identical metrics to baseline** — every loss, every per-split MAE, every test result byte-for-byte the same. Edward diffed `metrics.jsonl` line-by-line: the only differences are wall-clock `seconds` fields.

### Mechanism (the actual finding)

The Lion update applies decoupled wd as `p.data.mul_(1. - lr * wd)`. The decay multiplier `(1 − lr·wd)` is computed in Python (FP64) then materialised as an FP32 scalar for the in-place `mul_`. Master weights `p` are FP32 (BF16 is compute-only).

| wd | lr·wd at lr=2e-4 | FP32 ulp = 2⁻²⁴ ≈ 5.96e-8 | (1 − lr·wd) in FP32 |
|---:|---:|---:|---:|
| 4e-5 (Arm A) | 8.0e-9 | « ulp | rounds to **1.0** → no-op |
| 6e-5 (baseline) | 1.2e-8 | « ulp | rounds to **1.0** → no-op |
| 8e-5 (Arm B) | 1.6e-8 | « ulp | rounds to **1.0** → no-op |

Empirical: edward measured 6000 mul_ steps at FP32 init N(0, 0.02²) and observed **zero** relative weight shrink for wd ∈ {4e-5, 6e-5, 8e-5}, vs analytical expectation 4.8e-5–9.6e-5. Threshold for the multiplier to land at ≤ 0.99999994 (one-ulp below 1.0) at lr=2e-4 is **wd > 1.49e-4**. The entire sweep + the merged baseline live in the no-op band.

### Retroactive reinterpretation of the Lion history

- PR #1641 (frieren) introduced wd=6e-5 at lr=3e-4. `lr·wd = 1.8e-8`, also « ulp. **No-op.**
- PR #2027 (merged Lion lr=2e-4). Inherited wd=6e-5. **No-op.**
- Every Lion experiment in this round has effectively trained with **wd = 0**. The "wd" field in BASELINE.md is currently load-bearing nominally only.

This is a single most-important mechanistic finding of the round so far: we have an entire unexplored axis of the optimizer.

### Disposition

**SEND BACK (re-armed).** Not merged (bit-identical to baseline, not an improvement). Not closed (the hypothesis is now actionable). Next arms target wd values that actually fire:

- **Arm C: wd=5e-4** (lr·wd ≈ 1.68× ulp, first clean signal above FP32 floor; 10× the on-paper wd).
- **Arm D: wd=2e-3** (Lion paper's sweet spot region ≈ 10× AdamW wd; lr·wd ≈ 6.7× ulp).

Predictions: modest gain (0.3–1.0% val) at C; D is higher-risk, could be a clear win or a regression if it over-shrinks the 1.03M params.

- Metrics: `models/model-wd_4e5_lr2e4-20260513-102424/metrics.jsonl`, `models/model-wd_8e5_lr2e4-20260513-110201/metrics.jsonl` (bit-identical-to-baseline arms; reference only).
- Result link: https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/2177#issuecomment-4440717183

---

## 2026-05-13 11:55 — PR #2161: MLP dropout + attention dropout rate sweep (CLOSED — dropout=0.1 attention-only is saturated)

- Student branch: `charliepai2g24h5-thorfinn/mlp-dropout-attn-dropout-sweep`
- Hypothesis: Test whether (a) moving regularization locus by adding MLP dropout on top of attention dropout (Arm A: attn=0.1+MLP=0.1) or (b) reducing the regularization rate (Arm B: attn=0.05+MLP=0.0) improves on PR #1656's attn=0.1 only.

### Results (vs baseline #1656: 52.63/49.22)

| Metric | Baseline | Arm A (attn=0.1+MLP=0.1) | Arm B (attn=0.05+MLP=0.0) |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 52.63 | 55.317 (+5.1%) | 53.657 (+2.0%) |
| **test_avg/mae_surf_p** | 49.22 | 51.951 (+5.5%) | 50.135 (+1.9%) |

Both arms regress, in both directions (more reg → worse; less reg → worse). Per-split: regression most pronounced on val_geom_camber_rc and val_re_rand across both arms (i.e. the harder OOD splits suffer the most).

### Mechanism analysis

PR #1656 already established that **attention-dropout=0.1 alone** beats no-dropout (val 53.10 → 52.63). This result completes the picture:

- **Arm A regresses (+5%)**: adding MLP dropout on top of attn dropout double-regularizes — the dropout=0.1 sweet spot is locus-specific, not magnitude-specific. The model's MLP blocks need *full* signal; dropping attention features is OK because the PhysicsAttention slice tokens are over-complete by construction.
- **Arm B regresses (+2%)**: lowering attention dropout to 0.05 with no MLP dropout under-regularizes — the attn=0.1 rate is at a thin-ridge local optimum where small perturbations in either direction hurt.

**Dropout axis SATURATED on this stack.** The local neighborhood of attn=0.1, MLP=0.0 has been thoroughly explored (PR #1656, PR #2074 δ_p sweep nearby, this PR). No further dropout-magnitude or dropout-locus sweeps will pay back the GPU time at the current stack.

### Disposition

**CLOSED**. Reassigned thorfinn to **Lookahead optimizer wrapper around Lion (k=5, α∈{0.5, 0.8})** via PR #2249 — orthogonal optimizer-side mechanism (different math, different effect), not yet another regularization sweep.

- Arm A metrics: `models/model-mlp_drop_01_attn_drop_01-20260513-*/metrics.jsonl`
- Arm B metrics: `models/model-attn_drop_005_mlp_drop_0-20260513-*/metrics.jsonl`

---

## 2026-05-13 11:00 — PR #2176: SiLU activation swap GELU → SiLU in MLP blocks (CLOSED — GELU locally optimal)

- Student branch: `charliepai2g24h5-fern/silu-activation-swap`
- Hypothesis: Test whether SiLU (smooth, no plateau near zero) outperforms GELU as the block-MLP activation; orthogonal to all regularization.

### Results (vs current baseline #1656: 52.63/49.22)

| Metric | GELU baseline | SiLU (this run) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 52.63 | **59.5337** | **+6.90 (worse)** |
| **test_avg/mae_surf_p** | 49.22 | **55.7072** | **+6.49 (worse)** |

All 4 splits regress uniformly by +3.5 to +9.7 MAE-p on both val and test. Worst single-split delta: val_single_in_dist (+9.67).

### Per-split detail

| Split | val SiLU | val GELU | Δ_val | test SiLU | test GELU | Δ_test |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 66.18 | 56.52 | +9.67 | 55.98 | 47.14 | +8.84 |
| geom_camber_rc | 74.37 | 67.35 | +7.02 | 66.65 | 59.44 | +7.21 |
| geom_camber_cruise | 39.13 | 34.17 | +4.96 | 50.28 | 46.76 | +3.52 |
| re_rand | 58.45 | 52.50 | +5.95 | 49.92 | 43.54 | +6.38 |

### Mechanism analysis

Training trajectory healthy: monotone descent, grad-norm 24.6→2.5 (similar to GELU baseline), no instability or divergence. So the regression is not an optimization failure — it's a genuinely worse local minimum within the 16-epoch budget. Mechanism: Lion's sign(m_t)·lr update was tuned around GELU's gradient surface. SiLU's smoother near-zero region shifts which channels accumulate signal per step → at lr=2e-4, GELU's slightly steeper near-zero region lets Lion build signal faster. Lion + activation interaction confirmed (consistent with Chen et al. 2024 on Lion-activation lr coupling). The +6.9 gap is too large to bridge with an lr-shift follow-up.

### Disposition

**CLOSED**. GELU confirmed locally optimal as bare activation. **However**, this rules out only the bare-slope effect of SiLU — not the gated SwiGLU/GeGLU variants where the activation is multiplied by a learned gate. The student's own suggested follow-up #3 ("GLU-family gated variants are sometimes the actual reason 'SiLU papers' win in transformer benchmarks — the gating is doing the work, not the bare activation") is the right next direction. Reassigning fern to test SwiGLU directly (#2196).

- Metrics: `models/model-silu_activation-20260513-102006/metrics.jsonl`
- Note: fern flagged a pre-existing minor bug in `evaluate_split` (surf_loss=NaN, vol_loss=Inf in metrics.yaml for test_geom_camber_cruise). Does not affect MAE-driven decisions (MAE filters via gt_finite_mask). Acknowledged; not fixed in this PR.

---

## 2026-05-13 10:28 — PR #2084: Cosine LR floor: eta_min=lr*0.05 (CLOSED — zero-LR tail is implicit regularizer for Lion)

- Student branch: `charliepai2g24h5-frieren/cosine-lr-floor`
- Hypothesis: LR decays to exactly 0 at epoch 16; setting eta_min=lr*0.05 prevents step-size collapse at the convergence tail. The "still descending at epoch 16" signal suggests more room to improve.

### Results (vs PR #2028 baseline 53.62/49.65 — student's reference; vs current #1656 baseline: 52.63/49.22)

| Metric | Baseline (#2028) | This run | Δ vs #2028 | vs current #1656 |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 53.62 | **54.05** | **+0.43 (worse)** | +1.42 (worse) |
| **test_avg/mae_surf_p** | 49.65 | **51.09** | **+1.44 (worse)** | +1.87 (worse) |

All 8 splits regress; test splits hit harder than val (Δtest=+1.44 vs Δval=+0.43). Per-split worst: test_geom_camber_rc (+1.87), test_single_in_dist (+1.82).

### Mechanism analysis

Lion's `sign(m_t) * lr` step has fixed magnitude scaled only by `lr`. With a 5% LR floor, the model keeps receiving perturbations of ~1.5e-5/step at the end of training instead of going to zero. This prevents the final settling phase where very small steps let Lion's signed update + decoupled WD settle into a tighter local minimum. The zero-LR cosine tail acts as an **implicit regularizer** in Lion's signed-update regime — rather than collapsing, it's providing a gentle weight-decay-only phase. The widened val/test gap under floor LR (+0.43 val vs +1.44 test) confirms: the zero-LR tail is primarily helping OOD generalization, not just convergence speed.

### Disposition

**CLOSED**. Do NOT lower eta_min on Lion-based runs. Zero-LR cosine tail is load-bearing regularization for Lion.

- Metrics: `models/model-cosine_lr_floor_005-20260513-085153/metrics.jsonl`

---

## 2026-05-13 10:28 — PR #2100: Lion lr=1.5e-4 bracket-from-below on per-channel δ+n160 stack (CLOSED — LR bowl bottomed at lr=2e-4)

- Student branch: `charliepai2g24h5-tanjiro/lr-bracket-from-below`
- Hypothesis: LR optimum has been moving downward (3e-4 → 2e-4); bracket-from-below to confirm whether it continues.

### Results (vs merged baseline PR #2027, lion_lr=2e-4: 52.78/49.42)

| Metric | lr=2e-4 baseline | lr=1.5e-4 (this run) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 52.78 | **53.156** | **+0.376 (worse)** |
| **test_avg/mae_surf_p** | 49.42 | **50.149** | **+0.729 (worse)** |

Per-split: in-dist + camber_rc val improved (−0.51, −0.49) but val_re_rand (+1.30) + val_geom_camber_cruise (+1.22) pulled val_avg up. All 4 test splits regressed.

### Mechanism analysis

Lower LR converges slower → 16-epoch ceiling penalizes lr=1.5e-4 more than lr=2e-4 on OOD splits where training isn't asymptoted. The LR bowl is confirmed bottomed at lr=2e-4 on the current per-channel δ + n_hidden=160 + dropout=0.1 stack. Pre-registered conclusion from the student: "LR optimum has been moving DOWN; 2e-4 is near-optimal on this stack." Both sides of the bowl now have positive data: lr=1.5e-4 loses; bracket-from-above (lr=3e-4) was confirmed by PR #2035 as already regressed. Do not probe lr<2e-4 or lr>2.5e-4 further on this stack.

### Disposition

**CLOSED**. LR bowl confirmed bottomed. No further LR sweeps in this direction on current stack.

- Metrics: `models/model-lion_lr1_5e4_pcd_n160-20260513-090539/metrics.jsonl`

---

## 2026-05-13 10:20 — PR #2044: DropPath / stochastic depth (rates 0.05, 0.1) on n_hidden=160 (CLOSED — wrong-shape regularization for budget)

- Student branch: `charliepai2g24h5-edward/droppath-stochastic-depth`
- Hypothesis: Add DropPath to Transolver block residuals to induce implicit ensemble regularization. Predicted OOD splits (geom_camber_rc, single_in_dist) would benefit most.

### Results (vs old baseline 55.92 / 51.92, PR #1755 — student used lion_lr=3e-4)

| Config | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline |
|---|---:|---:|---|
| Baseline (drop_path=0.0) | 55.92 | 51.92 | — |
| Arm A (drop_path=0.05) | **67.40** | **60.98** | **+11.5 val (+20.6%), +9.1 test (+17.5%)** |
| Arm B (drop_path=0.10) | **72.80** | **66.29** | **+16.9 val (+30.2%), +14.4 test (+27.7%)** |

Catastrophic regressions on both arms vs old baseline. Vs current new baseline (52.63), regressions are even larger.

### Mechanism analysis

Both arms still monotonically descending at epoch 16 (epoch 15→16: −0.58 for 0.05, −1.32 for 0.10) — the model is far from convergence within the budget. With 5 Transolver blocks × 2 residual paths = 10 paths, drop_path=0.05 means ~40% probability of dropping ≥1 path per forward pass. The model must learn redundant feature pathways across many epochs to recover — incompatible with 16-epoch cap.

Per-split val: all 4 splits regress uniformly by +9.4 to +13.6 in Arm A and +14.1 to +20.1 in Arm B. The predicted OOD-specific benefit does not materialize. This is the opposite of the dropout=0.1 (PR #1656) pattern, where within-layer feature masking preserves convergence speed.

### Disposition

**CLOSED** as wrong-shape regularization for this budget. DropPath only becomes viable at higher epoch budgets (40+ epochs) or extremely small rates (<0.02) on this architecture. Not a near-term direction for round 5.

- Metrics: `models/model-droppath_0_05-20260513-075433/metrics.jsonl`, `models/model-droppath_0_1-20260513-090422/metrics.jsonl`

---

## 2026-05-13 10:18 — PR #2074: Per-channel Huber δ_p sweep (0.15, 0.10) on n_hidden=160 stack (CLOSED — δ_p<0.20 over-regularizes pressure)

- Student branch: `charliepai2g24h5-fern/per-channel-delta-refinement`
- Hypothesis: Test whether δ_p=0.15 or δ_p=0.10 beats the merged δ_p=0.20 from PR #2028. (Note: ran with stale lion_lr=1.5e-4 default; results compared against PR #2028 baseline 53.62/49.65.)

### Results (vs PR #2028 baseline 53.62 / 49.65 with stale lion_lr=1.5e-4)

| Config | val_avg | test_avg | Δ val | Δ test | val/test gap |
|---|---:|---:|---:|---:|---:|
| Baseline (δ_p=0.20, lion_lr=3e-4) | 53.62 | 49.65 | — | — | −3.97 |
| Arm A (δ_p=0.15, lion_lr=1.5e-4) | 53.54 | 50.14 | −0.14% | **+0.98%** | −3.40 |
| Arm B (δ_p=0.10, lion_lr=1.5e-4) | 53.19 | 50.35 | −0.81% | **+1.41%** | −2.84 |

Both arms underperform the current baseline (52.63). Val gain is small but test regresses substantially — a classic overfitting signal.

### Mechanism analysis

The val/test gap shrinks monotonically as δ_p decreases (−3.97 → −3.40 → −2.84). This is **leading indicator of over-regularization**: the loss-shape change is increasing the training signal/noise ratio in a way that doesn't transfer to held-out distributions. Per-split test: test_geom_camber_rc goes 58.75 → 60.80 → 60.84 — the hardest OOD split regresses most under tighter pressure capping.

Lower δ_p over-saturates pressure gradients into the linear regime, training on capped (and thus implicitly cherry-picked) residuals. δ_p=0.20 is the optimum on this stack.

### Disposition

**CLOSED** as informative negative result. δ_p=0.20 (from PR #2028) is confirmed optimum; lower values over-regularize. Removed from queue.

- Metrics: `models/model-pcd_p015-20260513-081930/metrics.jsonl`, `models/model-pcd_p010-20260513-085541/metrics.jsonl`

---

## 2026-05-13 09:58 — PR #1656: Dropout=0.1 on Lion lr=2e-4 + per-channel δ + n_hidden=160 (MERGED — new baseline 52.63/49.22)

- Student branch: `charliepai2g24h5-thorfinn/dropout-0_1`
- Hypothesis (final rerun): Add `dropout=0.1` to PhysicsAttention on the fully-aligned current-best stack (lion_lr=2e-4 + per-channel δ + n_hidden=160). Final arm after two earlier runs confirmed the direction but with stale optimizer configs.

### Results (vs baseline 52.78 / 49.42, PR #2027 — Lion lr=2e-4 + per-channel δ + n_hidden=160)

| Metric | Baseline (no dropout) | **This PR (dropout=0.1)** | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 52.78 | **52.6345** | **−0.15 (−0.27%)** ✅ |
| **test_avg/mae_surf_p** | 49.42 | **49.2183** | **−0.20 (−0.41%)** ✅ |

### Per-split val/test (epoch 16, best checkpoint)

| Split | val baseline | val this PR | test baseline | test this PR |
|---|---:|---:|---:|---:|
| single_in_dist | 56.24 | 56.52 | 46.75 | 47.14 |
| geom_camber_rc | 67.45 | 67.35 | 59.92 | 59.44 |
| geom_camber_cruise | 34.25 | 34.17 | 47.47 | 46.76 |
| re_rand | 53.17 | 52.50 | 43.52 | 43.54 |
| **avg** | **52.78** | **52.6345** | **49.42** | **49.2183** |

### Val curve (strictly monotone-decreasing)

All 16 epochs descending — final best at epoch 16 (53.78 → 52.63). No dropout-induced instability.

### Cross-stack ablation summary (thorfinn's 3 runs)

| Stack | Val | Test | Dropout gain (val) |
|---|---:|---:|---:|
| Lion δ=0.5 + n128 + lr=1.5e-4 baseline | 66.32 | — | — |
| + dropout=0.1 | 62.52 | 57.85 | −5.7% |
| Lion per-ch δ + n160 + lr=3e-4 baseline | 53.62 | 49.65 | — |
| + dropout=0.1 | 53.087 | 49.215 | −1.0% |
| **Lion per-ch δ + n160 + lr=2e-4 baseline** | **52.78** | **49.42** | — |
| **+ dropout=0.1 (MERGED)** | **52.63** | **49.22** | **−0.27%** |

Diminishing returns consistent with saturating regularization budget: each additional regularization axis captures a narrower slice of the remaining variance. The direction is strictly additive; feature masking and gradient/loss/width regularization remain non-redundant.

### Disposition

**MERGED.** Both primary metrics improve. New baseline val=52.63/test=49.22.

- Metrics: `models/model-charliepai2g24h5-thorfinn-dropout_0_1_n160_pcd_lr2e4-20260513-091653/metrics.jsonl`

---

## 2026-05-13 09:15 — PR #1656: Dropout=0.1 on n_hidden=160 + per-channel δ stack (SENT BACK ×2 — close, test wins but val falls 0.6% short)

- Student branch: `charliepai2g24h5-thorfinn/dropout-0_1`
- Hypothesis (rerun): Add `dropout=0.1` to PhysicsAttention to introduce feature-level stochastic noise on top of the upgraded baseline (n_hidden=160 + per-channel Huber δ). Dropout's regularization axis (feature masking) is orthogonal to width (n_hidden) and loss-shape (δ); compose-or-absorb test.

### Results (vs merged baseline 52.78 / 49.42, PR #2027 — Lion lr=2e-4 + per-channel δ + n_hidden=160)

**IMPORTANT caveat:** thorfinn's rerun used `--lion_lr 3e-4` (per advisor's previous instructions, before PR #2027 merged), NOT the new merged baseline's `--lion_lr 2e-4`. The PR #2027 baseline (lion_lr=2e-4) merged after thorfinn launched, so this run isolates dropout on the OLD optimizer config.

| Metric | Baseline (lion_lr=2e-4, no dropout) | This PR (lion_lr=3e-4 + dropout=0.1) | Δ vs new baseline |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | **52.78** | 53.087 | +0.31 (+0.6%) ✗ |
| **test_avg/mae_surf_p** | **49.42** | **49.215** | **−0.20 (−0.4%)** ✓ |

| Comparison | Your val | Baseline val | Δ | Your test | Baseline test | Δ |
|---|---:|---:|---:|---:|---:|---:|
| vs new merged baseline (lion_lr=2e-4) | 53.087 | **52.78** | +0.31 (+0.6%) | 49.215 | 49.42 | **−0.20 (−0.4%)** ✓ |
| vs old baseline (lion_lr=3e-4, no dropout) | 53.087 | 53.62 | **−0.53 (−1.0%)** ✓ | 49.215 | 49.65 | **−0.44 (−0.9%)** ✓ |

### Per-split val at epoch 16 (best checkpoint)

| Split | mae_surf_p |
|---|---:|
| val_single_in_dist | 56.66 |
| val_geom_camber_rc | 66.73 |
| val_geom_camber_cruise | 35.72 |
| val_re_rand | 53.24 |
| **val_avg** | **53.087** |

### Per-split test at epoch 16

| Split | mae_surf_p |
|---|---:|
| test_single_in_dist | 47.10 |
| test_geom_camber_rc | 58.65 |
| test_geom_camber_cruise | 47.48 |
| test_re_rand | 43.63 |
| **test_avg** | **49.215** |

### Analysis

**Dropout's gain shrinks but doesn't vanish.** Two perspectives:

1. **Vs old lr=3e-4 baseline:** Dropout reduces val by −1.0% and test by −0.9% — gain has shrunk from −5.7% (on n_hidden=128 + Huber δ=0.5 + lr=1.5e-4) to −1.0% on this stronger stack. This is consistent with a *saturating regularization budget*: width (n_hidden=160) and loss-shape (per-channel δ) already absorb some of the variance that dropout was previously soaking up. Dropout still helps because feature-level stochastic masking attacks a different axis (representational redundancy).

2. **Vs new lr=2e-4 baseline:** Test BEATS baseline by 0.4%, val falls 0.6% short. The val gap is small enough that the difference between thorfinn's lr=3e-4 and the new baseline's lr=2e-4 likely explains it — lr=2e-4 alone gave −0.84 val gain; if dropout adds ~−0.3 to ~−0.5 on top of lr=2e-4, the result would be 52.3–52.5 val, comfortably below baseline.

**Test-side signal is the most interesting result.** Even with the "wrong" lr=3e-4 optimizer config, dropout still pulls test down by 0.4% vs the new baseline. This is real evidence that dropout adds something the optimizer-config improvement of #2027 can't capture.

### Disposition

**SENT BACK** for a single re-run on the current best optimizer config (lion_lr=2e-4 + lion_weight_decay=6e-5). This isolates dropout's marginal effect on the new baseline and answers the key open question:

- **Best case (additive):** dropout's −0.53 val gain stacks on top of lr=2e-4's −0.84 → val~52.2 (clean merge).
- **Middle case (diminishing returns):** dropout gives ~−0.3 → val~52.45 (still merge).
- **Worst case (fully absorbed):** dropout no longer helps → val flat at ~52.78 (close as "regularization budget saturated").

Test-side signal suggests middle case is most likely.

- Metrics: `models/model-charliepai2g24h5-thorfinn-dropout_0_1_pchan_huber-20260513-082622/metrics.jsonl`

---

## 2026-05-13 09:00 — PR #2027: Lion lr=2e-4 on per-channel δ + n_hidden=160 stack (MERGED — new baseline 52.78)

- Student branch: `charliepai2g24h5-tanjiro/lion-lr-sweep-n160`
- Hypothesis (rerun): Lion lr=2e-4 should compound with the per-channel Huber δ change (#2028) since both narrow the loss landscape. Original run on δ=0.3 stack already showed signal; this confirms on current per-channel δ + n_hidden=160 stack.

### Results (vs baseline 53.62 / 49.65, per-channel δ + n_hidden=160)

| Metric | Baseline (lr=3e-4) | **This PR (lr=2e-4)** | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 53.62 | **52.7778** | **−0.84 (−1.6%)** ✅ |
| **test_avg/mae_surf_p** | 49.65 | **49.4184** | **−0.23 (−0.5%)** ✅ |
| Peak VRAM | 37.99 GB | 37.99 GB | 0 |
| s/epoch | ~115 s | ~109 s | −5% |

### Per-split val/test (lr=2e-4 + per-channel δ + n_hidden=160, epoch 16)

| Split | val baseline | val this PR | Δ val | test baseline | test this PR | Δ test |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 58.46 | **56.24** | **−2.22** | 48.40 | **46.75** | **−1.65** |
| geom_camber_rc | 67.34 | 67.45 | +0.11 | 58.75 | 59.92 | +1.17 |
| geom_camber_cruise | 35.10 | **34.25** | **−0.85** | 47.64 | **47.47** | −0.17 |
| re_rand | 53.58 | **53.17** | **−0.41** | 43.83 | **43.52** | −0.31 |
| **avg** | **53.62** | **52.78** | **−0.84** | **49.65** | **49.42** | **−0.23** |

### Analysis

**Three of four val splits improve, one flat.** The strongest gain is on `single_in_dist` (−2.22 val, −1.65 test), consistent with smaller LR producing tighter in-distribution fits. `geom_camber_rc` is essentially flat on val (+0.11) but +1.17 on test — the hardest OOD split is mildly worse under smaller LR (the same pattern observed in frieren's #2035 upward probe, but in reverse direction). Both directions away from lr=3e-4 hurt OOD slightly.

**Per-epoch trajectory analysis:** Divergence between lr=2e-4 and lr=3e-4 only appears AFTER epoch 10 (once cosine decay and sign-voting settle). Early epochs (1–7) are near-identical. The lr=2e-4 advantage compounds in the cosine tail.

**Mechanism:** Wider model (n_hidden=160, 1.6× params) has larger per-parameter gradient contributions. Lion's sign-quantized step size at lr=3e-4 over-shoots in the back half of training when the model is fine-tuning the final basin. Reducing to lr=2e-4 produces tighter convergence.

**Compound stack analysis:**
- baseline before #1755 (δ=0.3, lr=3e-4, n_hidden=128):  val=56.90 / test=53.20
- n_hidden=160 alone (#1755, lr=3e-4):                     val=55.92 / test=51.92
- per-channel δ alone (#2028, lr=3e-4):                    val=53.62 / test=49.65
- **per-channel δ + lr=2e-4 (#2027):                       val=52.78 / test=49.42**

Per-channel δ does most of the work (−2.30 val); lr=2e-4 adds another −0.84 on top. The Lion LR optimum continues moving down as the loss landscape tightens — consistent with the original Chen et al. paper recommending Lion lr~1.5e-4 for tight loss regimes.

### Disposition

**MERGED.** Both primary metrics improve. Updates baseline to val=52.78/test=49.42.

**Note on train.py defaults:** train.py still has stale `lion_lr=1.5e-4 / lion_weight_decay=3e-5` defaults from #1641 sweep — these are NOT the merged baseline config. All future experiments MUST pass `--lion_lr 2e-4 --lion_weight_decay 6e-5` explicitly. Pending follow-up to update defaults in a separate bug-fix PR.

**Follow-up assigned (PR #2100):** Lion lr=1.5e-4 bracket-from-below — test whether LR optimum continues falling.

- Metrics: `models/model-lion_lr2e4_n160_pcd-20260513-081331/metrics.jsonl`

---

## 2026-05-13 08:30 — PR #2035: Lion lr=3.5e-4 on n_hidden=160 + δ=0.3 stack (CLOSED — plateau confirmed)

- Student branch: `charliepai2g24h5-frieren/lion-lr-upward-n160`
- Hypothesis: LR optimum continues moving upward on δ=0.3 stack (mechanism: tighter δ puts more residuals in quadratic regime, needing larger LR). Testing lr=3.5e-4 vs baseline lr=3.0e-4.

### Results (vs old baseline 55.92 / 51.92, n_hidden=160 + δ=0.3)

| Metric | Baseline (lr=3e-4) | This PR (lr=3.5e-4) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 55.92 | **55.90** | −0.02 (within noise) |
| test_avg/mae_surf_p | 51.92 | **52.24** | +0.32 (worse) |

*Note: New baseline is 53.62 (PR #2028). Neither this result nor the old baseline beats the new baseline.*

### Per-split val (lr=3.5e-4, epoch 16)

| Split | lr=3.5e-4 | lr=3.0e-4 | Δ |
|---|---:|---:|---:|
| single_in_dist | 59.05 | 61.14 | **−2.09** |
| geom_camber_rc | 70.00 | 69.82 | +0.18 |
| geom_camber_cruise | 38.06 | 37.23 | +0.83 |
| re_rand | 56.51 | 55.51 | +1.00 |
| **avg** | **55.90** | **55.92** | **−0.02** |

### Key finding: Split-pattern diagnostic

Higher LR helps the easiest split (`single_in_dist`, −2.09) but hurts all three OOD splits by ~+1 MAE each. This is mild over-stepping — the optimizer converges faster on easy samples (in-distribution) while regressing on harder OOD domains that prefer slower, more careful convergence. Average comes out flat.

### Frieren's mechanism revision (valuable paper analysis)

The upward-LR-with-δ prediction held for narrow models (n_hidden=128) but breaks down for n_hidden=160. Wider models are typically MORE LR-sensitive (larger per-step gradient contributions), which over-rides the δ-driven step-magnitude argument. The LR response bowl is WIDE-AND-FLAT in the 3.0–3.5e-4 region.

### LR response curve (cumulative, n_hidden=160 + δ=0.3 stack)

| lion_lr | val | test | Source |
|---:|---:|---:|---|
| 2.0e-4 | TBD | TBD | #2027 tanjiro (rerun in flight) |
| **3.0e-4** | **55.92** | **51.92** | #1755 (merged baseline) |
| **3.5e-4** | **55.90** | **52.24** | #2035 this PR |

### Disposition

**Closed.** Clear negative result against old baseline; not competitive against new baseline (53.62). The bowl-flat finding ends the upward LR probe. Frieren's analysis quality is paper-worthy.

**Reassigned to PR #2084:** Cosine LR floor (eta_min=lr×0.05). Directly motivated by frieren's observation that epoch 16 is always best and the curve is still descending.

- Metrics: `models/model-lion_lr3_5e4_n160-20260513-072359/metrics.jsonl`

---

## 2026-05-13 08:05 — PR #2027: Lion lr=2e-4 on n_hidden=160 baseline (SENT BACK — rerun needed on current stack)

- Student branch: `charliepai2g24h5-tanjiro/lion-lr-sweep-n160`
- Hypothesis: Lion LR optimum shifts down as model widens (n_hidden=160 is 1.6× params); lr=2e-4 may beat lr=3e-4 on the n160 stack.

### Results (vs OLD baseline 55.92, uniform δ=0.3 code)

| Metric | Old baseline (lr=3e-4, δ=0.3) | This PR (lr=2e-4, δ=0.3) | New baseline (lr=3e-4, per-ch δ) |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 55.92 | **52.795** | **53.62** |
| test_avg/mae_surf_p | 51.92 | 49.856 | **49.65** |

**Per-split (tanjiro, lr=2e-4, uniform δ=0.3):**

| Split | val MAE_p | test MAE_p |
|---|---:|---:|
| single_in_dist | 56.24 | 48.75 |
| geom_camber_rc | 67.43 | 59.72 |
| geom_camber_cruise | 34.51 | 46.87 |
| re_rand | 53.01 | 44.09 |

### Analysis

The directional signal is clear: lr=2e-4 beats lr=3e-4 on the old uniform-δ=0.3 stack (val 52.795 < 55.92). Every val split improves. Test is close but slightly above the new baseline (49.856 vs 49.65). The mechanism: wider model (1.6× params) has larger gradient-norm contribution per step → smaller LR prevents overshoot in Lion's sign-quantized regime. Tanjiro's trajectory shows learning curve still descending at epoch 16 (monotonic from epoch 1, best at final epoch).

**BUT:** This run used the old uniform-δ=0.3 code. While this PR was in flight, PR #2028 (per-channel Huber δ) merged and became the new baseline (53.62/49.65) using DIFFERENT code. Direct comparison is not valid — two different hyperparameter points in different code variants. We can't claim a win over the current baseline with code from the old codebase.

**Decision:** Sent back. Tanjiro to rebase onto current advisor branch (picks up per-channel δ from #2028) and re-run with `--lion_lr 2e-4` to confirm the improvement holds on the combined stack. The lr=2e-4 signal is strong enough to expect confirmation.

- Metrics: `models/model-lion_lr2e4_n160-20260513-071233/metrics.jsonl`

---

## 2026-05-13 08:00 — PR #2028: Per-channel Huber δ=[Ux=0.5, Uy=0.5, p=0.2] on n_hidden=160 (MERGED — new baseline 53.62)

- Student branch: `charliepai2g24h5-fern/per-channel-huber-delta`
- Hypothesis: Pressure and velocity residuals have different distributions; per-channel δ outperforms the uniform scalar δ=0.3 baseline.

### Results (vs baseline 55.92 / 51.92)

| Metric | Baseline (uniform δ=0.3) | **This PR (per-ch δ=[0.5,0.5,0.2])** | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 55.92 | **53.62** | **−2.30 (−4.1%)** ✅ |
| **test_avg/mae_surf_p** | 51.92 | **49.65** | **−2.27 (−4.4%)** ✅ |
| Peak VRAM | 37.99 GB | 37.99 GB | 0 |
| s/epoch | ~115 s | ~115 s | 0 |

### Per-split val/test (per-channel δ=[0.5, 0.5, 0.2], epoch 16)

| Split | val baseline | val per-ch | Δ val | test baseline | test per-ch | Δ test |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 61.14 | **58.46** | −2.68 | 51.41 | **48.40** | −3.01 |
| geom_camber_rc | 69.82 | **67.34** | −2.48 | 60.85 | **58.75** | −2.10 |
| geom_camber_cruise | 37.23 | **35.10** | −2.13 | 48.82 | **47.64** | −1.18 |
| re_rand | 55.51 | **53.58** | −1.93 | 46.61 | **43.83** | −2.78 |
| **avg** | **55.92** | **53.62** | **−2.30** | **51.92** | **49.65** | **−2.27** |

### Analysis

**Clean uniform win across all 8 splits.** All 4 val splits improve (−1.93 to −2.68), all 4 test splits improve (−1.18 to −3.01). No regressions anywhere. Code change: single tensor literal in the Huber loss computation (`abs_err.new_tensor([0.5, 0.5, 0.2])`).

Mechanism: Pressure (p) residuals are high-variance in this CFD dataset — large outlier gradients dominate training when δ is applied uniformly. Keeping δ_p=0.2 (tight outlier cap on the dominant high-variance channel) while expanding velocity δ to 0.5 (restores more quadratic gradient signal for the lower-variance Ux/Uy channels) decouples the two regimes optimally. The response surface matches the uniform-δ monotone trend (0.3 was optimal for pressure when coupled with velocity), but now velocity and pressure are independently tuned.

Training still improving at epoch 16 (val trajectory: 64.72 → 58.03 → 54.55 → 53.62 at epochs 13→14→15→16). Per-channel δ curve not bottomed out — refinement of pressure δ (0.2 → 0.15/0.10) is the natural next experiment (#2074, assigned to fern).

- Metrics: `models/model-per_channel_huber_delta-20260513-071528/metrics.jsonl`

---

## 2026-05-13 07:40 — PR #1470: Per-sample instance-norm loss (CLOSED — dead end with valuable root-cause)

- Student branch: `charliepai2g24h5-edward/instance-norm-loss`
- Hypothesis: Per-sample normalization `huber_err * (1/y_std_s)` equalises gradient magnitudes across Re domains.

### Results (vs baseline 56.90 / 53.20 — n_hidden=128 reference; new baseline 55.92/51.92 on n160)

| Metric | This PR (n128 + δ=0.3 + inst-norm) | Baseline n128 | Baseline n160 | Δ vs n128 | Δ vs n160 |
|---|---:|---:|---:|---:|---:|
| val_avg/mae_surf_p | **59.02** | 56.90 | 55.92 | +2.12 (+3.7%) | +3.10 (+5.5%) ❌ |
| test_avg/mae_surf_p | **56.14** | 53.20 | 51.92 | +2.94 (+5.5%) | +4.22 (+8.1%) ❌ |

### Edward's root-cause analysis (paper-quality)

The 1e-6 clamp permitted catastrophic amplification of degenerate samples:

| Epoch | mean inst_scale | min | max |
|---|---:|---:|---:|
| 1 | 7.35 | 0.142 | **1342** |
| 5 | 11.19 | 0.142 | 2230 |
| 10 | 7.88 | 0.142 | 1299 |
| 16 | 6.90 | 0.142 | 1271 |

Max amplification was 1271-2230× (predicted ~12×). Root cause: nearly-uniform-field samples (very low-Re cruise) with per-sample y_std ≈ 4.5e-4 to 7.9e-4 in normalised space pass the 1e-6 clamp. Lion's sign-update mitigates magnitude but momentum still accumulates from scaled gradients → noisy convergence to worse optimum.

### Disposition

**Closed.** Literal hypothesis is broken at the data-distribution level. The principled fix (RevIN-style pre-residual normalization with `clamp(min=0.05)`) is queued for future exploration but unlikely to clear 55.92 since Huber already handles outliers at the per-element level.

**Reassigned to PR #2044:** DropPath / stochastic depth on n_hidden=160 baseline. Diversifies the experiment portfolio (3 active PRs were Lion-LR variants).

- Metrics: `models/model-charliepai2g24h5-edward-instance_norm_loss-20260513-065754/metrics.jsonl`

---

## 2026-05-13 07:25 — PR #1782 (3rd iteration): Lion lr=2e-4 on Huber δ=0.3+n128 stack (CLOSED — negative; valuable mechanism insight)

- Student branch: `charliepai2g24h5-frieren/lion-lr-scan`
- Hypothesis: lion_lr=2e-4 may continue beating 3e-4 as the loss landscape tightens further (δ=0.5 → δ=0.3).

### Results (vs baseline 56.90 / 53.20)

| Metric | Baseline (lr=3e-4) | This PR (lr=2e-4) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 56.90 | **58.82** | **+1.92 (worse)** ❌ |
| test_avg/mae_surf_p | 53.20 | **54.56** | **+1.36 (worse)** ❌ |

### Cumulative Lion-LR response curve (across stacks)

| lion_lr | 13ep+MSE | 16ep+δ=0.5 | 16ep+δ=0.3 |
|---:|---:|---:|---:|
| 2.0e-4 | 72.08 | **58.00** ← opt | 58.82 |
| 2.5e-4 | **71.54** ← opt | 58.99 | — |
| 3.0e-4 | 73.15 | — | **56.90** ← opt (baseline) |
| 4.0e-4 | 74.40 | — | — |

### Frieren's mechanism analysis (key insight, paper-worthy)

The LR optimum is **non-monotone in δ**:
- MSE → δ=0.5: optimum moved DOWN (2.5e-4 → 2e-4)
- δ=0.5 → δ=0.3: optimum moved UP (2e-4 → ≥3e-4) — reversal!

Proposed mechanism: As δ decreases past the residual-mass median (~0.4-0.5 for our normalised pressure residuals), MORE residuals fall in the *quadratic* regime, producing *smaller* per-step magnitudes → the optimizer needs *larger* step sizes to compensate. This predicts the LR optimum will continue moving upward on more aggressive δ regimes.

### Disposition

**Closed (not merged):** Below-baseline result. But the mechanism insight is the highest-value output: it predicts a specific direction (higher LR) for the next probe, and explains the surprising non-monotone curve.

**Reassigned to PR #2035:** lion_lr=3.5e-4 on n_hidden=160 + δ=0.3 (the new merged baseline). Directly tests frieren's mechanism prediction. Tanjiro (#2027) is testing lr=2e-4 on the same baseline in parallel, defining the lower half of the LR curve.

- Metrics: `models/model-charliepai2g24h5-frieren-lion_lr2e4_huber_d03-20260513-061016/metrics.jsonl`

---

## 2026-05-13 07:15 — PR #1879: Compound Huber δ=0.5+epochs=16 (CLOSED — hypothesis absorbed)

- Student branch: `charliepai2g24h5-tanjiro/huber-plus-epochs16`
- Hypothesis: Huber δ=0.5 + epochs=16 should compound two independent wins (#1639 + #1780).

### Results

| Metric | This PR | Baseline (#1880) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 56.8955 | 56.8955 | 0 (bit-identical) |
| test_avg/mae_surf_p | 53.2015 | 53.2015 | 0 (bit-identical) |

Tanjiro correctly rebased onto the new advisor branch (after baseline notification), which had δ=0.3 as default. Running with `--epochs 16` and the rebased code reproduced the #1880 winning arm exactly to 4+ decimal places (same seed=42, same code, deterministic).

### Analysis

The compound hypothesis was already realized by PR #1880, which tested δ=0.3 on the epochs=16 stack and merged as the new baseline. After rebasing, tanjiro's run was code-identical to the #1880 winner arm.

**Closed (not merged):** No new information beyond confirming seed-locked reproducibility. Tanjiro's analysis was scientifically honest and correct — he did not attempt to claim a tie as a win.

**Tanjiro's suggested follow-ups (noted for queue):** (1) Cosine LR floor (η_min = lr×0.1 instead of 0), (2) curriculum δ schedule (0.3→0.2 in last 3 epochs), (3) epochs=17 push. All three added to queued ideas.

- Metrics: `models/model-charliepai2g24h5-tanjiro-huber_epochs16-20260513-062410/metrics.jsonl`

---

## 2026-05-13 07:10 — PR #1755: n_hidden=160 on Huber δ=0.3+epochs=16 stack (MERGED — new baseline 55.92)

- Student branch: `charliepai2g24h5-fern/wider-model-nhidden192-bf16`
- Hypothesis: n_hidden=160 + δ=0.3 compound — width gain should be orthogonal to loss-shape regularization.

### Results (vs new baseline 56.90 / 53.20, PR #1880)

| Metric | Baseline n128+δ=0.3 | **This PR n160+δ=0.3** | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 56.90 | **55.92** | **−0.98 (−1.7%)** ✅ |
| **test_avg/mae_surf_p** | 53.20 | **51.92** | **−1.28 (−2.4%)** ✅ |
| Peak VRAM | 32.95 GB | 37.99 GB | +15% |
| s/epoch | ~102 s | ~115 s | +13% |

### Per-split val/test (n_hidden=160 + δ=0.3, epoch 16)

| Split | val n128 | val n160 | Δ val | test n128 | test n160 | Δ test |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 60.26 | 61.14 | +0.88 | 52.32 | 51.41 | −0.91 |
| geom_camber_rc | 75.20 | **69.82** | **−5.38** | 64.24 | **60.85** | **−3.39** |
| geom_camber_cruise | 37.01 | 37.23 | +0.22 | 49.15 | 48.82 | −0.33 |
| re_rand | 55.11 | 55.51 | +0.40 | 47.10 | 46.61 | −0.49 |
| **avg** | **56.90** | **55.92** | **−0.98** | **53.20** | **51.92** | **−1.28** |

- Metrics: `models/model-nhidden160_huber_d03_final-20260513-061938/metrics.jsonl`

### Analysis

**Width and loss-shape are orthogonal levers.** Val gain is concentrated in val_geom_camber_rc (racecar-camber OOD, hardest split) which drops −5.38. The other 3 val splits are within ±1 (tied, slight regressions likely within run-to-run noise). Test is cleaner: all 4 test splits improve.

Generalization gap preserved (test−val: −4.00 vs −3.70 baseline). Both models converge flat at epoch 16 (n160 final epoch delta: −0.03 val, gradient_norm ~2.3 — asymptote reached). Peak VRAM 37.99 GB on H100; wall-clock 30.7 min (fits).

**This is the 5th submission of this hypothesis** (4 send-backs due to moving baseline, final-gate framing). The width gain is real and compounds with δ=0.3, as predicted.

---

## 2026-05-13 06:55 — PR #1481: Double physics-attention slices: slice_num 64→128 (CLOSED — dead end)

- Student branch: `charliepai2g24h5-nezuko/double-attention-slices`
- Hypothesis: Increasing slice_num from 64→128 doubles the number of physics-attention basis tokens, potentially capturing finer-scale flow features. Orthogonal to optimizer/loss changes.

### Results (vs baseline 56.90 / 53.20)

| Metric | Baseline (slice_num=64) | **This PR (slice_num=128)** | Δ | % |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p | 56.90 | **69.69** | +12.79 | **+22.5%** |
| test_avg/mae_surf_p | 53.20 | **64.90** | +11.70 | **+22.0%** |

Both val and test regress heavily. All 4 val splits worse.

### Per-split val (slice_num=128)

| Split | Baseline | slice_num=128 | Δ |
|---|---:|---:|---:|
| single_in_dist | 60.26 | 72.55 | +12.29 |
| geom_camber_rc | 75.20 | 88.47 | +13.27 |
| geom_camber_cruise | 37.01 | 47.88 | +10.87 |
| re_rand | 55.11 | 69.87 | +14.76 |

- Metrics: `models/model-charliepai2g24h5-nezuko-slice128_ep16-<timestamp>/metrics.jsonl`

### Analysis & disposition

**Root cause: budget cliff.** Doubling the slice tokens increases per-epoch compute from ~102 s to ~144 s (+41%). With a 30-min hard wall-clock cap the model completed only **13 epochs** instead of 16 — losing the critical cosine-tail epochs (14–16) where val typically falls by ~3–5 points. This is the same failure pattern observed when n_hidden=192 cost 12 epochs vs 13.

**Matched-epoch analysis confirms no per-step improvement either:** even at matched epoch 13, slice_num=128 regresses vs the baseline. The additional attention resolution does not compensate for the increased compute cost — more slice tokens appear to make optimization harder, not easier, consistent with the Transolver design intent (64 slices already covers the mesh resolution adequately).

**Closed (not sent back):** Two independent failure signals (regression + budget cliff). No evidence of value at any epoch count. `n_layers` or `n_head` width changes are cheaper architectural levers to try next.

**Nezuko reassigned:** PR #2005, surf_weight sweep (15 vs 5) on the current δ=0.3 + Lion + BF16 + epochs=16 stack.

---

## 2026-05-13 06:30 — PR #1755: n_hidden=160 single-arm on Huber δ=0.5+epochs=16 stack (SENT BACK — baseline moved 3rd time)

- Student branch: `charliepai2g24h5-fern/wider-model-nhidden192-bf16`
- Hypothesis: n_hidden=160 on the new Lion+Huber δ=0.5+epochs=16 stack — does the −1.71 val width gain from old Lion-only stack compose with epochs=16 + Huber δ=0.5?

### Results vs OLD δ=0.5 baseline 66.32 / 61.14 (baseline moved to 56.90 during run)

| Metric | Baseline n128+δ=0.5 | **This PR n160+δ=0.5** | Δ vs OLD | Δ vs NEW 56.90 |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p | 66.32 | **57.34** | **−8.97 (−13.5%)** | +0.44 (above new baseline) |
| test_avg/mae_surf_p | 61.14 | **53.69** | **−7.45 (−12.2%)** | +0.49 |

### Per-split val/test (n160 winner)

| Split | val_n160 | test_n160 |
|---|---:|---:|
| single_in_dist | 60.74 | 52.56 |
| geom_camber_rc | 72.74 | 63.32 |
| geom_camber_cruise | 38.72 | 50.59 |
| re_rand | 57.18 | 48.30 |
| **avg** | **57.34** | **53.69** |

### Analysis

**Strong architectural signal:** every val and test split improved over the old 66.32 baseline. Gen gap narrowed from −5.18 → −3.65 (test better generalizing). Per-epoch trajectory still descending at epoch 16 (slope −1.7/epoch). Peak VRAM 38 GB, s/epoch 115, 30.7 min wall-clock.

**Why sent back (3rd time):** PR #1880 (Huber δ=0.3) merged during fern's run, dropping the baseline from 66.32 → 56.90. Fern's n_hidden=160 + δ=0.5 result (57.34) is now +0.44 above the new baseline. Need one more single-arm run with n_hidden=160 + δ=0.3 (current default) to test compound. If width's gain is orthogonal to loss shape, expected val ~52-55. Final gate framing: this is the last re-run for this hypothesis — either it beats 56.90 and merges, or we close the n_hidden direction.

- Metrics: `models/model-nhidden160_huber_ep16-20260513-051534/metrics.jsonl`

---

## 2026-05-13 06:01 — PR #1782: Lion LR re-scan on Huber+epochs=16 stack (SENT BACK — baseline moved again)

- Student branch: `charliepai2g24h5-frieren/lion-lr-scan`
- Hypothesis: Re-test LR scan (lr=2e-4, 2.5e-4) on the merged Huber δ=0.5 + epochs=16 stack. Previous scan on 13-epoch MSE stack found lr=2.5e-4 was optimal (vs default 3e-4, val=71.54).

### Results (2-arm, vs OLD baseline 66.32 / 61.14 — baseline moved to 56.90 during run)

| lion_lr | val_avg | test_avg | vs OLD 66.32 | vs NEW 56.90 |
|---:|---:|---:|---:|---:|
| **2.0e-4 (winner)** | **58.00** | **53.91** | −12.55% | **+1.10 (above new baseline)** |
| 2.5e-4 | 58.98 | 54.35 | −11.07% | +2.08 |

Both arms beat OLD baseline 66.32. Neither beats NEW baseline 56.90 (from δ=0.3 merge).

### Key findings

- **Optimum shifted from 2.5e-4 → 2e-4** when moving from 13ep+MSE to 16ep+δ=0.5 stack. The "softer loss landscape" (Huber caps large-residual gradient contribution) combined with longer cosine tail favors slightly smaller step size.
- LR response curve: at 13ep+MSE: min at 2.5e-4; at 16ep+δ=0.5: min at 2e-4.
- Both arms still descending at epoch 16 — headroom remains.
- Timing/VRAM unchanged: 101.6s/epoch, 32.95 GB, 27.1 min per arm.

- Arm A metrics: `models/model-charliepai2g24h5-frieren-lion_lr2e4_e16_huber-20260513-035740/metrics.jsonl`
- Arm B metrics: `models/model-charliepai2g24h5-frieren-lion_lr2_5e4_e16_huber-20260513-050329/metrics.jsonl`

### Why sent back (2nd time)

PR #1880 (Huber δ=0.5 → δ=0.3, val=56.90) merged while frieren was running. Sent back to re-run **single arm lr=2e-4 on δ=0.3 stack**. If the LR-down shift continues (from 2.5e-4 to 2e-4 to possibly 1.75e-4), the lr=2e-4 arm on δ=0.3 should beat 56.90. Worth verifying before merging a LR change that may no longer be optimal.

---

## 2026-05-13 06:00 — PR #1880: Huber δ=0.3 scan (MERGED — new baseline 56.90)

- Student branch: `charliepai2g24h5-alphonse/huber-delta-scan`
- Hypothesis: Huber δ curve hasn't bottomed out at 0.5 (monotonic improvement from 1.0→0.5 in PR #1639). Test δ=0.3 and δ=0.2 on the epochs=16 stack.

### Results (2-arm, vs baseline 66.32 / 61.14)

| δ | best epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | Δval | Δtest |
|---:|---:|---:|---:|---:|---:|
| Baseline δ=0.5 (13ep) | 13 | 66.32 | 61.14 | — | — |
| **δ=0.3 (winner)** | 16 | **56.90** | **53.20** | **−9.42 (−14.2%)** | **−7.94 (−13.0%)** |
| δ=0.2 | 16 | 56.94 | 53.23 | −9.38 | −7.91 |

δ=0.3 and δ=0.2 essentially tied (Δ=0.04 val / 0.03 test). Curve flattened — optimal δ is at or near 0.3. NOTE: both arms ran epochs=16 (the merged code default), so the 13ep→16ep contribution is included.

### Per-split val / test (δ=0.3 winner, epoch 16)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 60.26 | 52.32 |
| geom_camber_rc | 75.20 | 64.24 |
| geom_camber_cruise | 37.01 | 49.15 |
| re_rand | 55.11 | 47.10 |
| **avg** | **56.90** | **53.20** |

- Winner metrics: `models/model-huber_delta0_3-20260513-035824/metrics.jsonl`
- Runner-up: `models/model-huber_delta0_2-20260513-050217/metrics.jsonl`

### Analysis

Split-level picture: δ=0.2 slightly wins tail-heavy single/raceCar splits; δ=0.3 wins cruise/re_rand (lower y-std domains where aggressive δ=0.2 over-saturates into linear regime). Overall avg: δ=0.3 wins both val and test.

Huber δ response curve (full): 1.0→67.41, 0.5→66.32, 0.3→56.90, 0.2→56.94. Jump from 0.5 to 0.3 is the largest gain; curve then flattens. Hypothesis confirmed: δ=0.3 is the optimal floor; smaller δ (0.1) is unlikely to improve further.

Code change merged: `torch.where(abs_err < 0.3, 0.5*abs_err**2, 0.3*abs_err - 0.045)` — 1-line change in train.py.

Peak VRAM 32.95 GB; wall-clock ~27 min per arm.

---

## 2026-05-13 05:55 — PR #1656: Dropout=0.1 in PhysicsAttention (SENT BACK — baseline moved)

- Student branch: `charliepai2g24h5-thorfinn/dropout-0_1`
- Hypothesis: Dropout=0.1 in PhysicsAttention (attention output projection + attention mechanism) adds feature-level stochastic regularization, complementing Huber+grad_clip's gradient-level regularization.

### Results (vs baseline 66.32 / 61.14 — baseline since moved to 56.90)

| Config | val_avg | test_avg | Δval vs 66.32 |
|---|---:|---:|---:|
| Baseline (no dropout) | 66.32 | 61.14 | — |
| **Dropout=0.1** | **62.52** | **57.85** | **−3.80 (−5.7%)** |

### Per-split val (dropout=0.1, epoch 16)

| Split | mae_surf_p |
|---|---:|
| val_single_in_dist | 70.20 |
| val_geom_camber_rc | 79.58 |
| val_geom_camber_cruise | 41.02 |
| val_re_rand | 59.27 |
| val_avg | 62.52 |

| Split | test mae_surf_p |
|---|---:|
| test_single_in_dist | 59.35 |
| test_geom_camber_rc | 69.43 |
| test_geom_camber_cruise | 51.82 |
| test_re_rand | 50.77 |
| test_avg | 57.85 |

- Metrics: `target/models/model-charliepai2g24h5-thorfinn-dropout_0_1_clean-20260513-052522/metrics.jsonl`

### Analysis & disposition

Clean 5.7% val improvement, monotone curve still descending at epoch 16. Implementation verified: dropout fires only during training (model.train/eval flip confirmed). Only attention dropout applied (to_out projection + SDPA); MLP dropout not yet tested.

**Sent back (not closed):** PR #1880 merged during the run. Dropout's 62.52 is above new 56.90 baseline. Regularization mechanisms are orthogonal (Huber operates at loss level, dropout at activation level), so the gain should compose. Sent back for single-arm re-run: `--epochs 16` on δ=0.3 stack with dropout=0.1 still in train.py.

---

## 2026-05-13 04:35 — PR #1755: Width sweep 2-arm follow-up — n_hidden=160 / n_hidden=192+lr4e-4 (SENT BACK — baseline moved)

- Student branch: `charliepai2g24h5-fern/wider-model-nhidden192-bf16`
- Hypothesis: Original PR found n_hidden=192 had better per-epoch trajectory but lost the wall-clock race (12 vs 13 epochs). Two-arm follow-up: (A) intermediate width n_hidden=160 + Lion lr=3e-4 (apples-to-apples 13-epoch budget) vs (B) wider n_hidden=192 + Lion lr=4e-4 (scaled LR to recover lost epoch via faster per-step progress).

### Results (vs OLD Lion baseline 73.15 / 66.76 — baseline since moved to 66.32 / 61.14)

| Config | val_avg | test_avg | s/epoch | Epochs | n_params | vs OLD 73.15 | vs NEW 66.32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLD baseline (n128 lr3e-4) | 73.15 | 66.76 | 100.87 | 13 | 656k | — | +6.83 |
| **Arm A (n160 lr3e-4)** | **71.44** | **66.25** | 115.96 | 13 | 1.03M | **−1.71 val / −0.51 test** | **+5.12 val / +5.11 test** |
| Arm B (n192 lr4e-4) | 73.90 | 68.91 | 127.81 | 12 | 1.47M | +0.75 val / +2.15 test (regress) | +7.58 val / +7.77 test |

### Per-epoch val_avg trajectory (Arm A clearly beats n128 baseline at matched steps)

| Epoch | n128 (baseline) | **Arm A n160** | Arm B n192 |
|---:|---:|---:|---:|
| 8 | 96.93 | 96.23 | 104.04 |
| 9 | 90.45 | 88.52 | 96.37 |
| 10 | 83.76 | 86.25 | 81.04 |
| 11 | 80.47 | 78.59 | 77.51 |
| 12 | 76.10 | 73.12 | 73.90 |
| 13 | 73.15 | **71.44** | (no budget) |

- Arm A metrics: `models/model-nhidden160_bf16_lion-20260513-030220/metrics.jsonl`
- Arm B metrics: `models/model-nhidden192_lr4e4-20260513-035213/metrics.jsonl`

### Analysis & disposition

**Arm A was a clean, real signal:** −1.71 val / −0.51 test against the (then-current) Lion baseline 73.15, all val splits improved, no widened gen gap, trajectory still falling at epoch 13. The 1.6× param model used 38 GB VRAM with 116 s/epoch and consumed the full 13-epoch budget.

**Arm B confirmed n_hidden=192 dead end:** Higher LR (4e-4) didn't recover the lost epoch — val regresses +0.75, test regresses +2.15. Grad_norm spiked to 94 at epoch 6 (vs ~37 at same epoch in Arm A), indicating instability. Two PRs now (the original #1755 and this Arm B) show n_hidden=192 regresses on test; direction is closed.

**Why sent back, not merged:** PR ran before #1780 (Lion+epochs=16) and #1639 (Huber δ=0.5) merged, which moved baseline to 66.32 val / 61.14 test. Arm A is +5 on both vs new baseline. The width gain was small (−1.71 val on a 73.15 base = ~2.3% relative) and we don't know if it composes with the new Huber+epochs=16 stack. Sent fern back to re-run **Arm A only** with `--epochs 16` on the merged Huber stack (n_hidden=160). Dropped Arm B.

**Expected outcome of re-run:** If the width gain composes with epochs=16+Huber, val should be ~63-65 (∼−1.7 vs 66.32). If gain was specific to the old Lion-only stack, val will be flat/slightly worse and the n_hidden direction is closed.

---

## 2026-05-13 03:51 — PR #1639: Huber δ=0.5 loss on Lion stack (MERGED — new baseline 66.32)

- Student branch: `charliepai2g24h5-alphonse/huber-loss`
- Hypothesis: Huber (Smooth-L1) loss with δ=0.5 replaces MSE. Outlier residuals (high-Re tandem near-surface samples) dominate MSE gradients; Huber caps per-element gradient at δ pre-aggregation, complementing grad_clip which caps the global gradient norm post-aggregation. Two arms: δ=1.0 and δ=0.5. Stack: Lion lr=3e-4 + BF16 + grad_clip + warmup3+cosine13, epochs=13.

### Results

| Config | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline (73.15) |
|---|---:|---:|---:|
| Baseline (Lion lr=3e-4, MSE) | 73.15 | 66.76 | — |
| Huber δ=1.0 (Arm 1) | 67.41 | 62.65 | −7.85% val |
| **Huber δ=0.5 (winner)** | **66.32** | **61.14** | **−9.34% val** |

### Per-split val (δ=0.5 winner, epoch 13)

| Split | Baseline | Huber δ=0.5 | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 80.78 | 71.66 | −11.3% |
| val_geom_camber_rc | 90.86 | 82.99 | −8.7% |
| val_geom_camber_cruise | 51.56 | 46.06 | −10.7% |
| val_re_rand | 69.39 | 64.56 | −7.0% |
| **val_avg** | **73.15** | **66.32** | **−9.3%** |

### Per-split test (δ=0.5 winner)

| Split | Baseline | Huber δ=0.5 | Δ |
|---|---:|---:|---:|
| test_single_in_dist | 69.02 | 62.73 | −9.1% |
| test_geom_camber_rc | 77.38 | 69.80 | −9.8% |
| test_geom_camber_cruise | 59.49 | 56.26 | −5.4% |
| test_re_rand | 61.14 | 55.79 | −8.8% |
| **test_avg** | **66.76** | **61.14** | **−8.4%** |

- Metrics (δ=0.5 winner): `models/model-charliepai2g24h5-alphonse-huber_delta0_5_lion-20260513-025216/metrics.jsonl`
- Metrics (δ=1.0 arm): `models/model-charliepai2g24h5-alphonse-huber_delta1_lion-20260513-021619/metrics.jsonl`

### Analysis

**Outstanding across-the-board result.** δ=0.5 uniformly beats δ=1.0 on ALL 8 splits (4 val + 4 test). No tradeoff — smaller δ is better everywhere. This confirms the outlier-capping hypothesis and critically suggests the **response curve hasn't bottomed out** (monotonic improvement from 1.0 → 0.5 → smaller?).

The orthogonality with grad_clip is confirmed: Huber caps outliers at the per-element level (before mean reduction), while grad_clip normalizes the full parameter gradient (after backprop aggregation). They stack cleanly.

Key implication: **the optimal δ is below 0.5**. alphonse's next assignment is a δ scan at 0.3 and 0.2.

Also notable: this result (66.32) slightly beats #1780's epochs=16 result (66.44) using only 13 epochs. The combination of Huber+epochs=16 should compound both improvements (tanjiro's #1879).

---

## 2026-05-13 03:50 — PR #1780: Lion + epochs 13→16 (MERGED — new baseline 66.44)

- Student branch: `charliepai2g24h5-tanjiro/longer-cosine-lion-epochs16`
- Hypothesis: Lion's training was non-converged at epoch 13 (trajectory still monotonically descending). With BF16 reducing s/epoch to ~101s, 16 epochs = 27.1 min — within the 30-min cap. Extended cosine schedule (T_max = 16−3 = 13) fully decays LR to ~0 at epoch 16. No code change needed — runtime flag only.

### Results

| Epoch | val_avg/mae_surf_p | Δ vs prev |
|---:|---:|---:|
| 13 | 73.81 | (matches old baseline 73.15 within noise) |
| 14 | 69.97 | −3.84 |
| 15 | 68.38 | −1.59 |
| **16 (best)** | **66.44** | **−1.94** |

| Metric | Value | vs baseline (73.15) |
|---|---:|---:|
| val_avg/mae_surf_p | **66.44** | **−9.2%** |
| test_avg/mae_surf_p | **61.78** | **−7.5%** |
| Wall-clock | 27.1 min | within 30-min cap |

### Per-split val (epoch 16)

| Split | val_avg/mae_surf_p | Δ |
|---|---:|---:|
| val_single_in_dist | 71.11 | −12.0% |
| val_geom_camber_rc | 81.78 | −10.0% |
| val_geom_camber_cruise | 48.92 | −5.1% |
| val_re_rand | 63.96 | −7.8% |
| **val_avg** | **66.44** | **−9.2%** |

- Metrics: `models/model-lion_epochs16-20260513-015116/metrics.jsonl`

### Analysis

Clean confirmation that Lion was non-converged at epoch 13. The per-epoch improvement sequence (−3.84, −1.59, −1.94) shows the model still making meaningful progress through the final epoch. Cosine LR reached ≈0 exactly at epoch 16 — fully decayed as expected.

This is a structural improvement: the `--epochs 16` flag becomes the new standard for all future experiments on this stack (BF16 budget allows it). All in-flight WIP students notified to re-run with `--epochs 16`.

---

## 2026-05-13 03:26 — PR #1782: Lion LR scan (2e-4, 2.5e-4, 4e-4) (SENT BACK — below new baseline)

- Student branch: `charliepai2g24h5-frieren/lion-lr-scan`
- Hypothesis: Scan the LR gap between winning 3e-4 and arm-1 1.5e-4. Three arms: lr=2e-4, 2.5e-4, 4e-4 (and the existing 1.5e-4/3e-4 data from #1641).
- All ran epochs=13 (old schedule) on Lion+BF16 stack.

### Results

| lion_lr | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline (73.15) |
|---:|---:|---:|---:|
| 2.0e-4 | 72.08 | 66.31 | −1.47% |
| **2.5e-4 (best)** | **71.54** | **65.95** | **−2.21%** |
| 3.0e-4 (baseline) | 73.15 | 66.76 | — |
| 4.0e-4 | 74.40 | 67.96 | +1.72% |

### Analysis

Clear minimum at lr≈2.5e-4. Both 2e-4 and 2.5e-4 beat old baseline; 4e-4 worse. The finding: **2.5e-4 is marginally better than 3e-4 on 13 epochs**. Difference is small (71.54 vs 73.15).

However, after merging #1780 (66.44) and #1639 (66.32), the new baseline is **66.32**. Frieren's best (val=71.54) doesn't beat it.

Sent back with request to re-run both lr=2.5e-4 and lr=2e-4 on the new combined stack (Huber δ=0.5 + epochs=16). If lr=2.5e-4 holds its ~1.6-point advantage on the new stack, expected outcome is ~64.

---

## 2026-05-13 03:10 — PR #1755: n_hidden=192 + BF16 + Lion (SENT BACK — budget-cliff regression)

- Student branch: `charliepai2g24h5-fern/wider-model-nhidden192-bf16`
- Hypothesis: Wider model (n_hidden=192) on BF16+Lion stack — VRAM headroom from BF16 (32.94 GB → ~43 GB) unlocks the wider model that was previously infeasible.
- Single change: `n_hidden=128 → 192` in Transolver config. 12 epochs (one less than Lion baseline's 13 due to 27% slower per-epoch at wider width).

### Results

| Metric | Lion baseline (#1641, 13 epochs) | n_hidden=192 (this PR, 12 epochs) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 73.15 | 73.11 | −0.04 (tie, within noise) |
| **test_avg/mae_surf_p** | **66.76** | **68.76** | **+2.00 (REGRESSION)** |
| Peak VRAM (GB) | 32.94 | 43.01 | +30% |
| s/epoch | 100.87 | 127.74 | +27% |
| Epochs completed | 13 | 12 | −1 (budget cliff) |
| n_params | 656k | 1.47M | +2.2× |

### Per-epoch trajectory (wider model systematically ahead at matched steps)

| Epoch | n_hidden=128 (Lion) | n_hidden=192 |
|---:|---:|---:|
| 10 | 83.76 | 81.54 |
| 11 | 80.47 | 76.26 |
| 12 | 76.10 | 73.11 |
| 13 | 73.15 | (out of budget) |

- Metrics: `models/model-nhidden192_bf16-20260513-021849/metrics.jsonl`

### Analysis

Tie on val (−0.04, within noise) but **test regresses by 2.00 points**. Cannot merge per criteria (test is paper-facing metric, must not regress).

However, the per-epoch trajectory is clean: at matched epoch counts, n_hidden=192 is systematically ahead of n_hidden=128 by 3–4 points. The wider model has the better learning dynamics; it just lost the race because of the **budget cliff**: n_hidden=192 fits only 12 epochs in 30 min (vs baseline's 13), and Lion's last-epoch jump (76→73 in baseline) is significant.

The fix: either (a) reduce width to n_hidden=160 to fit 13 epochs, or (b) keep n_hidden=192 but scale Lion LR up (4e-4) to make 12 epochs deliver baseline's 13-epoch progress.

### Decision

**Sent back to fern with 2-arm follow-up:**
- Arm A: n_hidden=160 + Lion lr=3e-4 (intermediate width, full 13-epoch budget)
- Arm B: n_hidden=192 + Lion lr=4e-4 (wider with scaled LR to recover lost epoch)

---

## 2026-05-13 03:01 — PR #1463: SWA from epoch 25 on Lion stack (CLOSED — averages bad early checkpoints)

- Student branch: `charliepai2g24h5-askeladd/swa-final-three-warmup-grad-clip-3`
- Hypothesis: SWA (Stochastic Weight Averaging, Izmailov 2018) finds a flatter, more generalizable minimum by averaging recent checkpoints late in training. SWA from epoch 25 onward, paired with SWALR (constant LR phase after the cosine schedule), should compose with Lion stack.
- Stack: Lion lr=3e-4 + warmup3+cosine13 + grad_clip(1.0) + BF16. SWA start_epoch=25 (turned out infeasible — training capped at 13 epochs in 30-min budget), so effective SWA window was different.

### Results

| Metric | Lion baseline (#1641) | SWA (this PR) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 73.15 | 76.14 | **+2.99 (+4.1% REGRESSION)** |
| test_avg/mae_surf_p | 66.76 | 70.29 | **+3.53 (+5.3% REGRESSION)** |

### Per-split breakdown

| Split | Lion baseline | SWA | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 80.78 | 84.12 | +3.34 (regress) |
| **val_geom_camber_rc** | **90.86** | **87.19** | **−3.67 (improve)** |
| val_geom_camber_cruise | 51.56 | 56.31 | +4.75 (regress) |
| val_re_rand | 69.39 | 76.92 | +7.53 (regress) |

### Analysis (mechanistic, valuable negative result)

**Core failure modes:**
1. **Averaging in pre-convergence checkpoints.** SWA-start was nominally epoch 25 but training only ran 13 epochs (30-min cap). SWALR likely kicked in well before convergence, averaging weights that still had significant per-epoch progress.
2. **SWALR perturbs Lion's cosine schedule.** Lion's cosine-annealed sign-quantized steps are tuned to the warmup3+cosine13 trajectory. Imposing a SWALR constant-LR phase on top fights the underlying optimizer's own schedule.

**Interesting partial signal:** val_geom_camber_rc IMPROVES (−3.67 val, −0.94 test). This is exactly the split where SWA's flat-minima story should help most (worst OOD split, where over-fitting val_avg's mode collapses generalization). The cost on the other 3 splits dominates the average, but the camber_rc improvement is real and consistent.

**Conclusion:** SWA needs (a) much later start to avoid averaging in pre-convergence checkpoints, and (b) decoupled averaging that doesn't perturb the underlying optimizer's LR schedule. In the 13-epoch budget regime, vanilla SWA from any epoch is dominated by Lion's own monotonic improvement.

### Decision

Closed. The improvement on camber_rc is interesting enough to revisit if/when we have a longer training budget (24-30 epochs), where checkpoint averaging late in training could outperform single-epoch picks. Right now in the 13-epoch monotonic-improvement regime, every form of mid-training averaging will regress.

askeladd reassigned to PR #1844 (Lion β2=0.99 → 0.999 single-knob sweep).

---

## 2026-05-13 01:20 — PR #1641: Lion optimizer (MERGED — new baseline 73.15)

- Student branch: `charliepai2g24h5-frieren/lion-optimizer`
- Hypothesis: Lion (sign-based optimizer, Chen et al. 2023) is the logical endpoint of gradient renormalization. Where grad_clip(max_norm=1.0) renormalizes to unit L2 norm globally, Lion per-parameter sign-quantizes every gradient to ±lr. With our existing renorm stack, testing Lion tests whether per-parameter uniformity outperforms global L2 renorm.
- Two arms: Lion lr=1.5e-4 (Arm 1) and Lion lr=3e-4 (Arm 2, winner). Both ran 13 epochs FP32 (pre-BF16 merge) on warmup3+cosine13+grad_clip stack.

### Results

| Arm | optimizer | lion_lr | lion_wd | val_avg/mae_surf_p | Δ vs baseline (94.22) | test_avg/mae_surf_p | Δ vs baseline (87.10) |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | AdamW (BF16) | — | — | 94.22 | — | 87.10 | — |
| Lion Arm 1 | Lion | 1.5e-4 | 3e-5 | 75.17 | **−19.05 (−20.2%)** | 70.13 | **−16.97 (−19.5%)** |
| **Lion Arm 2 (winner)** | Lion | **3e-4** | **6e-5** | **73.15** | **−21.07 (−22.4%)** | **66.76** | **−20.34 (−23.4%)** |

### Per-split val at best epoch (epoch 13, Arm 2 winner)

| Split | Baseline (94.22) | Lion lr=3e-4 | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 107.86 | 80.78 | −24.9% |
| val_geom_camber_rc | 105.04 | 90.86 | −13.5% |
| val_geom_camber_cruise | 73.65 | 51.56 | −30.0% |
| val_re_rand | 90.33 | 69.39 | −23.2% |
| **val_avg** | **94.22** | **73.15** | **−22.4%** |

### Per-split test (Arm 2 winner)

| Split | Lion lr=3e-4 |
|---|---:|
| test_single_in_dist | 69.02 |
| test_geom_camber_rc | 77.38 |
| test_geom_camber_cruise | 59.49 |
| test_re_rand | 61.14 |
| **test_avg** | **66.76** |

### Training trajectory (both arms monotonically improving at epoch 13)

| Epoch | Lion lr=1.5e-4 | Lion lr=3e-4 |
|---:|---:|---:|
| 1 | 210.83 | 192.42 |
| 5 | 131.30 | 127.88 |
| 10 | 87.29 | 83.76 |
| 13 | **75.17** | **73.15** |

- Metrics (winner): `models/model-charliepai2g24h5-frieren-lion_lr3e4-20260512-225827/metrics.jsonl`
- Metrics (arm 1): `models/model-charliepai2g24h5-frieren-lion_lr1_5e4-20260512-235646/metrics.jsonl`

### Analysis

**Outstanding result** — largest single-PR gain of the round. Lion outperforms AdamW by >22% on both val and test, with consistent gains across all 4 splits (val improvements range from −13.5% to −30.0%).

Why it works: Lion's per-parameter sign update produces uniform ±lr steps for each parameter regardless of gradient magnitude. This is strictly stronger than grad_clip(max_norm=1.0)'s global L2 renorm. For Transolver's heterogeneous parameter space (PhysicsAttention slices, MLP projections, layer norms have very different gradient scales), uniform per-parameter steps appear dramatically more beneficial than globally-normalized steps.

Critical observation: **Both arms are still improving monotonically at epoch 13.** This means Lion has NOT converged in the 13-epoch budget. More epochs could yield further gains — key hypothesis for follow-up.

The LR relationship holds: lr=3e-4 (= AdamW lr/3.3) beats lr=1.5e-4 (= AdamW lr/6.7). The Lion paper's guideline of lr = AdamW_lr / 3 to / 10 is validated here.

### Suggested follow-ups (from frieren + advisor)

1. **Lion + longer cosine (epochs=16–18 with BF16)** — both arms non-converged at epoch 13, more epochs almost certainly help.
2. **Lion + BF16 (now merged)** — the merged stack has both BF16 and Lion. First BF16+Lion run to establish the new true baseline.
3. **Lion lr mid-point (2e-4, 2.5e-4)** — narrow the LR scan between the two arms (gap is small at 73.15 vs 75.17).
4. **Lion β2 = 0.999** — lion-pytorch default is (0.9, 0.99); at batch=4 gradient noise is high per step, slower momentum might help.
5. **Lion + n_hidden=192 (fern's current experiment)** — architecture width × sign optimizer composition.

---

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
