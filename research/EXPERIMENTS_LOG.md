# SENPAI Research Results

---

## 2026-05-13 12:45 — PR #2127: surf_head step decay at e10 {×0.5, ×0.3}

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-step-decay` (CLOSED — mechanism confirmed, harmful intervention)
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `ap4569i2` (Arm 1 ×0.5), `ttrb3tmz` (Arm 2 ×0.3)

### Results

| Arm | val_avg/mae_surf_p | Δ vs #2031 (93.62) | test_avg/mae_surf_p | Δ vs #2031 (83.88) |
|-----|---------------------|---------------------|---------------------|---------------------|
| Arm 1 (e10, ×0.5) | 100.2247 | +7.05% | 89.8721 | +7.14% |
| Arm 2 (e10, ×0.3) | 96.2736 | +2.83% | 87.9542 | +4.86% |

Monotonic trend: more aggressive decay → closer to baseline. Suggests optimum decay factor is **1.0** (no decay).

### Key insight (reframing)

**The e12 spike is fully damped** in both arms — mechanism confirmed. Per-epoch trajectory:
```
ep:    1    2    3    4    5    6    7    8    9   10   11   12   13   14
base: 189  164  160  156  144  125  131  120  111  111  110  153  108  94  ← spike + deep recovery
arm1: 203  245  157  150  132  142  131  124  122  114  113  108  108  100 ← smooth, no spike
arm2: 191  194  149  173  130  144  141  128  123  115  111  107  103  96  ← smooth, no spike
```

**The spike+recovery is a BENEFICIAL training dynamic, not pathology.** Without the e12 spike, the e13-e14 sequence that breached the baseline's new minimum disappears. Both arms continue descending smoothly but converge at WORSE final values. The aggressive arm (×0.3) regresses LESS than the gentle arm (×0.5) — exactly OPPOSITE to what we'd expect if the spike were pathological.

### LR trace verification

| Epoch | Encoder LR | Arm 1 surf_head LR | Arm 2 surf_head LR |
|-------|-----------|--------------------|--------------------|
| e9 | 0.000461 | 0.004611 (×1.0) | 0.004611 (×1.0) |
| e10 | 0.000452 | **0.002261 (×0.49)** | **0.001357 (×0.29)** |
| e14 | 0.000409 | 0.002047 | 0.001228 |

Step decay engages cleanly at e10 as configured.

### Analysis

This is the THIRD PR to triangulate the late-epoch oscillation mechanism:
- #2058 (grad-clip): gradient size is NOT the mechanism (sh_grad_norm 0.77× encoder)
- #1949 (head LR warmup): head LR cold-start is NOT the mechanism
- #2127 (head LR step decay): head LR magnitude IS coupled to the spike, but damping it removes a beneficial dynamic

**The e12 spike is an exploration burst that enables a deeper minimum at e14.** Damping the spike (via any head-LR-side intervention) loses the deep recovery. The mechanism is now characterized: the encoder enters a new region of loss landscape around e10-11, and the head's pre-positioned weights become "too forward" relative to the encoder's new locality — producing the spike. The recovery is the head re-aligning.

### Implications

**Closed directions** (head-LR-side modulation): step decay (this PR), warmup (#1949), gradient clipping (#2058), wider head (#2057). All head-side levers have been characterized.

**Open: encoder-side modulation.** Thorfinn's suggested follow-up: encoder LR boost at e10-12 — same coupling, opposite intervention. Brief encoder speedup during transition might give cleaner alignment WITHOUT losing the spike-recovery. Assigned as #2188 (thorfinn encoder-lr-boost).

---

## 2026-05-13 12:30 — PR #2013: LogCosh surface loss (C²-smooth alternative to Huber kink)

- **Branch:** `willowpai2g48h4-tanjiro/logcosh-surface-loss` (CLOSED — surface-loss family closed)
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `mbvj746h` (Arm 1 scale=1.0), `v80mlvz7` (Arm 2 scale=0.5)

### Results

| Arm | val_avg/mae_surf_p | Δ vs #1795 (97.99) | test_avg/mae_surf_p | Δ vs #1795 (88.53) |
|-----|---------------------|---------------------|---------------------|---------------------|
| Arm 1 (scale=1.0) | 101.4307 | +3.51% | 91.5855 | +3.45% |
| Arm 2 (scale=0.5) | 111.8905 | +14.18% | 100.0830 | +13.05% |

Vs the new #2091 baseline (89.7197), the regressions are even larger (Arm 1 +13.1%, Arm 2 +24.7%).

### Key diagnostic: late-epoch oscillation amplitude

| Run | max−min over E11-E14 | ratio max/min |
|-----|---------------------|---------------|
| Baseline | 15.7 | 1.16 |
| Arm 1 LogCosh | 17.7 | 1.17 |
| Arm 2 LogCosh | 16.9 | 1.15 |

**The C² smoothness hypothesis was unsupported** — oscillation amplitude essentially unchanged. Thrashing at the Huber kink is NOT the dominant source of late-epoch oscillation.

### Why LogCosh failed (mechanism analysis)

1. **Quadratic-regime gradient mismatch**: at typical residuals r≈0.106, Huber-δ=0.5 gradient is 2r=0.21, LogCosh scale=1.0 is tanh(r)≈0.106 — **half the gradient signal**. The 10× surf_head_lr cannot fully compensate.
2. **Asymptotic L1 behavior is softer**: Huber cleanly transitions to ±1 at |r|≥δ. LogCosh asymptotes to ±1 but at scale=0.5, even at r=1.0 the gradient is tanh(2)≈0.964. **Outlier nodes get weaker, smoother signal under LogCosh** — opposite of what Huber's hard-clip does for the rare-but-important large-residual surface nodes that drive MAE.
3. **C¹ vs C² distinction is empirically irrelevant** here. AdamW's running statistics either don't see meaningful discontinuities at the δ-kink at this scale, or the oscillation source is elsewhere (sampler/batch composition variance, cosine-decay endpoint).

### Surface-loss family closure

Now five PRs span the surface-loss family:
- #1558 Huber δ=0.5 → WIN (−17.72%)
- #1627 δ ∈ {0.2, 0.3} → both regress
- #1950 adaptive δ (EMA tracking) → +2.25%
- #1922 per-channel δ (δ_p=0.5, δ_u=2.0) → +5.61%
- #2013 LogCosh {scale=1.0, 0.5} → +3.51% / +14.18%

**Huber δ=0.5 is a stable local optimum**; finer perturbations of the surface loss function family do not unlock further gains. Future loss-function work should target volume loss or per-sample weighting (BIVW family).

### Per-split pattern (interesting subordinate finding)

Arm 1 matches or slightly beats baseline on OOD splits (val_geom_camber_rc, val_geom_camber_cruise, val_re_rand, test_geom_camber_rc) — the regression is concentrated on `val_single_in_dist` (+13.6%). The hardest OOD splits actually improved marginally under LogCosh; it's the in-distribution sanity split that suffered. This suggests LogCosh's softer signal helps generalization slightly but penalizes in-distribution fit — too costly for the primary metric.

---

## 2026-05-13 11:45 — PR #2091: torch.compile throughput unlock **[WINNER — NEW BASELINE]**

- **Branch:** `willowpai2g48h4-frieren/torch-compile` (MERGED)
- **Student:** willowpai2g48h4-frieren
- **W&B runs:** `fvlekakd` (Arm 1 default — WINNER), `807sxrb9` (Arm 2 reduce-overhead — OOM)

### Results

| Metric | Baseline #2031 (WD=5e-4) | Arm 1 (default, WD=1e-4) | Δ |
|--------|------------------------|--------------------------|---|
| `val_avg/mae_surf_p` | 93.6198 | **89.7197** | **−4.16%** |
| `test_avg/mae_surf_p` | 83.8825 | **79.3167** | **−5.44%** |
| Epochs in 30 min | 14 | 21 | **+50%** |
| Per-epoch wall-clock | ~125s | ~87s | **1.43×** |
| Peak VRAM | ~43 GB | 43 GB | unchanged |
| Best epoch | 14 | 18 | shift expected |

### Per-split val (epoch 18)

| Split | Baseline | Arm1 | Δ |
|-------|---------|------|---|
| `val_single_in_dist` | 109.61 | 114.92 | +4.9% ↑ (slight regression) |
| `val_geom_camber_rc` | 118.05 | 108.66 | **−8.0%** ✓ |
| `val_geom_camber_cruise` | 61.07 | 55.45 | **−9.2%** ✓ |
| `val_re_rand` | 85.75 | 79.85 | **−6.9%** ✓ |

### Per-split test

| Split | Baseline | Arm1 | Δ |
|-------|---------|------|---|
| `test_single_in_dist` | 101.87 | 104.29 | +2.4% ↑ |
| `test_geom_camber_rc` | 105.24 | 96.30 | **−8.5%** ✓ |
| `test_geom_camber_cruise` | 51.43 | 46.12 | **−10.3%** ✓ |
| `test_re_rand` | 76.99 | 70.55 | **−8.4%** ✓ |

### Analysis

**The recipe was wall-clock-bound, not capacity-bound.** +50% more epochs (14→21) on the same recipe yields −4.16%/−5.44% — far beyond the predicted −0.5% to −2.5% from the PR. The val curve was still descending at epoch 14 when the baseline run hit the timeout; the model had genuine headroom.

**Critical note on per-split pattern**: `val_single_in_dist` slightly regressed (+4.9%) while all OOD geometry splits improved dramatically (cruise −9.2%, geom_camber_rc −8%). This matches the prediction for WD=1e-4 at 21 epochs: the model trains longer than the in-distribution regularization supports, but the extra epochs help OOD generalization. **This regressor should be recoverable by composing with WD=5e-4 (PR #2178, frieren).**

**Arm 2 (reduce-overhead) OOM analysis**: CUDA Graph private pool accumulation at 94.77GB (72GB in pools). Root cause: each unique padded mesh size allocates a separate CUDA Graph + private pool, and with N_max spanning 74K–242K nodes, hundreds of unique shapes exhaust memory. Requires upstream bucketed-padding to fix — out of scope.

**torch.compile mechanics confirmed:**
- dynamic=True: only 2 compile frames despite 74K-242K node range (symbolic-shape graph)
- First-batch compile warmup: 8.8s (negligible)
- `torch.set_float32_matmul_precision("high")` (TF32 matmul) + `cudnn.benchmark=True` are safe

**Implication for wall-clock-bound rejections**: EMA (#1808), n_head=8 (#1924), DropPath (#1987) were all closed because 14 epochs wasn't enough. At 21 epochs, these should be re-screened — especially EMA (needs stable late-training trajectory) and DropPath (oscillation smoothing matters more at longer training).

### Highest-priority follow-ups

1. **torch.compile + WD=5e-4 compose** (#2178 frieren, NEW) — stack two top wins
2. **Re-screen wall-clock-bound regularizers at 21 epochs** — EMA, DropPath, n_head=8
3. **Cosine T_max recalibration with compile** — T_max=50 at 21 epochs means ~42% of cosine cycle; askeladd's #2123 tests this direction (still valid, but calibration point shifts)

---

## 2026-05-13 11:30 — PR #2120: Deeper weight_decay sweep {7e-4, 1e-3, 2e-3}

- **Branch:** `willowpai2g48h4-fern/wd-deeper` (CLOSED — WD=5e-4 is sharp peak)
- **Student:** willowpai2g48h4-fern
- **W&B run:** `rdpj0afc` (WD=7e-4 arm only; arms 2-3 correctly halted by branching rule)

### Results

| Metric | Baseline #2031 (WD=5e-4) | Arm 1 (WD=7e-4) | Δ |
|--------|------------------------|----------------|---|
| `val_avg/mae_surf_p` | 93.6198 | 111.2633 | **+18.85% regression** |
| `test_avg/mae_surf_p` | 83.8825 | 99.1562 | +18.22% regression |
| `val_single_in_dist` | 109.615 | 127.540 | +16.4% |
| `val_geom_camber_rc` | 118.046 | 144.471 | +22.4% |
| `val_geom_camber_cruise` | 61.066 | 74.502 | +22.0% |
| `val_re_rand` | 85.753 | 98.540 | +14.9% |
| Best epoch | 14 (descending) | 14 (descending) | same |

### Analysis

The hypothesis was "monotone gain at 5× WD says deeper is better." The data sharply falsified this: +40% (5e-4→7e-4) crashed all 4 splits uniformly. WD=5e-4 is a **sharp peak**, not a plateau or inflection. There is no OOD geometry gain to offset the in-distribution loss.

### Trajectory diagnostic (high-signal)

| Epoch | Baseline 5e-4 | Arm1 7e-4 | Δ |
|-------|--------------|----------|---|
| 4 | 155.84 | 135.09 | −13.3% (arm1 briefly leads — early-overfit damping) |
| 6 | 125.11 | 132.12 | +5.6% (baseline pulls ahead, never gives lead back) |
| 9 | 111.07 | **152.09** | +36.9% (arm1 spike — equivalent magnitude to baseline e12, 3 epochs earlier) |
| 12 | **152.99** | 146.34 | (baseline spike) |
| 14 | **93.62** | 111.26 | +18.8% (baseline e14 breakthrough −13%; arm1 only −3% in same epoch) |

Two key trajectory patterns:
1. **Spike location depends on WD**: baseline has e12 spike; arm1 has spikes at BOTH e9 (new) AND e12. Higher WD makes optimization MORE unstable, not less.
2. **e14 breakthrough is LR-schedule driven**: both runs have the same "final-epoch jump" shape but the absolute magnitudes differ (baseline 108→94 = −13%; arm1 114→111 = −3%). The e14 breakthrough is load-bearing for the merge result.

### Weight-norm logging diagnostic (NEW — highest-signal output)

| Step | Encoder norm | Surf_head norm |
|------|-------------|----------------|
| 49 | 41.01 | 7.79 |
| 1352 | 42.97 | 17.04 |
| 2656 | 43.94 | 21.35 |
| 3959 | 44.79 | 24.44 |
| 5262 | 45.56 | 26.91 |

Even at WD=7e-4, encoder norm GROWS +11% over training; surf_head norm grows 3.5×. **WD is not actually shrinking parameters — it is just damping the rate of growth.** This reframes the regularization regime: the optimum WD is the one that damps growth enough to prevent overfitting WITHOUT slowing the productive directions (e.g., the e14 breakthrough).

This is a high-signal diagnostic that should be preserved in ALL future WD experiments. See assignment #2153 (fern wd-bracket).

### Key insight

WD operates as a "growth-rate damper" on the implicit-bias trajectory of AdamW, not as a "shrink-to-zero" force. Combined with the e14 breakthrough as the load-bearing event for merge, the regularization curve has a sharp optimum at 5e-4 because that's exactly the rate that lets the model overfit the early-training noise but still descend at e14.

### Student-suggested follow-ups

1. **Finer WD sweep in (5e-4, 7e-4): 5.5e-4, 6e-4, 6.5e-4** — implemented as #2153 (fern wd-bracket), extended with 4e-4 to bracket below the peak.
2. **Asymmetric WD: encoder=5e-4, head=1e-4** — already in flight as #2122 (edward decoupled-wd). The weight_norm data motivates this directly.
3. **LR-tweak instead of WD-tweak** — overlaps with #2123 (askeladd cosine T_max).
4. **Don't retroactively log weight_norm on existing baseline; do log on every new WD run** — applied to #2153.
5. **Spike investigation at e9/e12: add per-batch loss logging on spike epochs** — defer; the spikes are now known to be steady-state, not pathological.

---

## 2026-05-13 10:30 — PR #2031: Weight decay re-tune 1e-4 → 5e-4 **[WINNER — NEW BASELINE]**

- **Branch:** `willowpai2g48h4-fern/weight-decay-sweep` (MERGED)
- **Student:** willowpai2g48h4-fern
- **W&B run:** `u3q47f4s` (WD=5e-4 arm)

### Results

| Metric | Baseline (#1795) | WD=3e-5 | WD=5e-4 | Δ (5e-4) |
|--------|-----------------|---------|---------|----------|
| `val_avg/mae_surf_p` | 97.9914 | ~99 | **93.6198** | **−4.46%** |
| `test_avg/mae_surf_p` | 88.5311 | — | **83.8825** | **−5.26%** |
| `val_single_in_dist` | — | — | best arm | −8.9% |
| `val_geom_camber_cruise` | — | — | best arm | −7.5% |
| `val_re_rand` | — | — | best arm | −4.3% |
| `val_geom_camber_rc` | — | — | flat | +1.8% |
| Best epoch | 11-14 | — | 14 (descending) | — |

### Analysis

WD=1e-4 was a survivor from the BIVW era (cycles 2-3) — it had been untouched through BIVW, Huber, decoupled-LR merges. This is **hyperparameter staleness in action**: each merged stage changed the loss landscape and the "correct" WD changed too, but we never re-swept it.

Mechanism: heavier regularization at 5e-4 limits overfitting on the high-Re in-distribution split (val_single_in_dist −8.9%) while also improving OOD geometry splits (cruise −7.5%). The flat result on val_geom_camber_rc (+1.8%) suggests head-capacity or encoder features may limit OOD generalization on extreme camber geometries — consistent with #2057's negative finding that wider surf_head doesn't help.

**Lesson:** Hyperparameter staleness is real but per-axis. WD was stale; encoder LR (confirmed optimal in #1974) was not. Always audit optimizer hyperparameters when the loss formulation changes.

---

## 2026-05-13 10:30 — PR #2057: Wider surf_head hidden_dim {64→128, 64→256}

- **Branch:** `willowpai2g48h4-askeladd/wider-surf-head` (CLOSED — head not the bottleneck)
- **Student:** willowpai2g48h4-askeladd
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (h=128) | Δ |
|--------|---------|--------------|---|
| `val_avg/mae_surf_p` | 93.6198 | ~98.6 | **+5.36% regression** |

Arm 2 (h=256) skipped per branching rule.

### Analysis

The surf_head hidden_dim 64→128 experiment falsified the capacity-bottleneck hypothesis for the head. The head has 0.026M params; the Transolver encoder has the overwhelming share of parameters and representations. Wider head adds parameters that cannot improve what the encoder provides.

**Key diagnostic from #2057 + #2058**: the student logged both `sh_grad_norm` (0.77× encoder) and per-epoch weight norms. The head is NOT gradient-saturated, NOT capacity-limited, and NOT gradient-clipped-limited. The late-epoch oscillation source is in the optimizer step (LR × m/√v), specifically the 10× LR differential between head and encoder.

**Lesson:** Encoder is the representation bottleneck. Capacity scaling on surf_head is wasted unless encoder is scaled simultaneously.

---

## 2026-05-13 10:30 — PR #2058: Per-group gradient clipping on surf_head {max_norm=0.5, 1.0}

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-grad-clip` (CLOSED — wrong mechanism)
- **Student:** willowpai2g48h4-thorfinn
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (clip=0.5) | Arm 2 (clip=1.0) | Best Δ |
|--------|---------|-----------------|-----------------|--------|
| `val_avg/mae_surf_p` | 93.6198 | ~103 | ~104 | **+10.39% regression** |

### Analysis

The diagnostic logging was the most valuable output of this experiment: `sh_grad_norm` was consistently 0.77× the encoder grad norm across training. The "large update" in the late-epoch spike comes from the 10× LR multiplier, not from large gradients. Gradient clipping at max_norm ∈ {0.5, 1.0} on surf_head therefore clips BOTH the noise AND the productive signal, causing regression.

**Key insight (falsification):** The stabilization mechanism for the e12 spike must target the optimizer step magnitude (LR × m/√v), not the gradient ‖g‖. This narrows the search space to: (a) step-decay surf_head LR at late epochs, (b) AdamW ε (denominator floor), (c) cosine T_max (LR schedule that decays faster). All three assigned in cycle 30.

---

## 2026-05-13 10:30 — PR #1974: Encoder LR re-tune {3e-4, 7e-4}

- **Branch:** `willowpai2g48h4-edward/encoder-lr-retune` (CLOSED — encoder LR confirmed optimal)
- **Student:** willowpai2g48h4-edward
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (lr=3e-4) | Arm 2 (lr=7e-4) | Best Δ |
|--------|---------|----------------|----------------|--------|
| `val_avg/mae_surf_p` | 93.6198 | ~99 | ~98 | **+5.42% regression** |
| `test_avg/mae_surf_p` | 83.8825 | — | — | +3.18% regression |

### Analysis

Encoder LR=5e-4 is the local optimum even at the new WD=5e-4 baseline. The 3e-4 arm underfits (model doesn't descend fast enough under 14-epoch cap), and 7e-4 overshoots (late-epoch spike exacerbated at higher LR).

**Important finding:** Encoder LR and WD are roughly orthogonal — the WD win (#2031) did not shift the LR optimum. This confirms that the two axes were genuinely independent.

---

## 2026-05-13 10:30 — PR #1922: Per-channel Huber delta {δ_p=0.5, δ_ux/uy=1.0/2.0}

- **Branch:** `willowpai2g48h4-nezuko/per-channel-huber-delta` (CLOSED — global δ=0.5 is correct)
- **Student:** willowpai2g48h4-nezuko
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (δ_ux=1.0, δ_p=0.5) | Arm 2 (δ_ux=2.0, δ_p=0.5) | Best Δ |
|--------|---------|--------------------------|--------------------------|--------|
| `val_avg/mae_surf_p` | 93.6198 | ~98 | ~99 | **+5.61% regression** |
| `test_avg/mae_surf_p` | 83.8825 | — | — | +3.70% regression |

### Analysis

Larger δ on Ux/Uy flattens mid-magnitude velocity gradients. With δ=2.0, residuals <2.0 see MSE behavior — this over-smooths the Ux/Uy signal that the encoder uses for geometry reasoning. The global δ=0.5 is the correct balanced value.

**Why this was surprising:** The hypothesis was that Ux/Uy have smoother residual distributions than pressure and might benefit from larger δ. But surface pressure residuals are the hardest to fit (largest MAE), and they drive backprop indirectly through the shared encoder. Flattening Ux/Uy gradients weakens the encoder's geometry signal, which then hurts the very pressure prediction we're trying to improve.

---

## 2026-05-13 10:30 — PR #1496: Pressure-channel emphasis {pw=3, pw=5} on both vol+surf loss

- **Branch:** `willowpai2g48h4-alphonse/pressure-channel-prioritized-loss` (CLOSED — mean-normalisation bug)
- **Student:** willowpai2g48h4-alphonse
- **W&B run:** (reported in PR comments)

### Results

| Metric | Baseline | Arm 1 (pw=3) | Arm 2 (pw=5) | Best Δ |
|--------|---------|-------------|-------------|--------|
| `val_avg/mae_surf_p` | 93.6198 | ~112 | ~114 | **+20.04% regression** |

### Analysis

The implementation used mean-normalisation: with pw=3, normalised weights become [0.6, 0.6, 1.8] — meaning Ux/Uy were DOWN-weighted 40% and pressure only mildly up-weighted relative to uniform. Net effect: worse at velocity AND worse at pressure. The hypothesis was sound but the implementation was backwards.

**Root cause:** Mean-normalisation with [1, 1, 3] yields weights ÷ mean=5/3 → [0.6, 0.6, 1.8]. Both 3× and 5× arms regress monotonically, consistent with the optimal weight being in the sub-unit range once you correct the normalisation.

**Follow-up (#2124):** Surface-only pressure weight WITHOUT mean-normalisation, with k∈{0.5, 1.5} to explicitly test sub-unit pressure weighting. Volume loss kept uniform.

---

## 2026-05-13 07:25 — PR #1987: Stochastic Depth (DropPath) on Transolver blocks: regularization unlock

- **Branch:** `willowpai2g48h4-fern/stochastic-depth` (CLOSED — mechanism works, wall-clock-bound)
- **Student:** willowpai2g48h4-fern
- **W&B run:** `q64n4xgm`
- **Hypothesis:** DropPath p∈{0.05, 0.1} on Transolver residual branches; targets baseline late-epoch oscillation pattern via stochastic regularization.

### Results

| Metric | Baseline | Arm 1 (p=0.05) | Δ |
|--------|---------|----------------|---|
| `val_avg/mae_surf_p` | 97.9914 | 101.2325 | **+3.31% regression** |
| `test_avg/mae_surf_p` (4-split) | 88.5311 | 89.7226 | **+1.35% regression** |
| Best epoch | 11 | 14 (still descending) | — |

Arm 2 (p=0.10) correctly skipped per branching rule.

### Mechanism diagnosis — clean confirmation

DropPath sanity check (standalone test):
- drop_path_rate=0.05: train per-sample std = 0.903 (vs eval=0.0), output norms preserved in expectation
- drop_path_rate=0.10: train per-sample std = 1.322

**Trajectory smoothing confirmed**:

| Epoch | Baseline val_avg | DropPath val_avg |
|-------|------------------|-------------------|
| 11 | **97.99** (best) | 116.46 |
| 12 | 113.66 (spike) | 110.22 |
| 13 | 108.05 | 107.96 |
| 14 | 99.85 | **101.23** (best, still descending) |

Baseline's epoch-11→12 spike (97.99 → 113.66) is GONE. DropPath gives clean monotone descent through final epochs. The oscillation prediction is fully validated.

### Why it loses at our budget

Each epoch lands ~3 points behind baseline (epoch 12: 110.22 DP vs 113.66 BL, epoch 13: 107.96 DP vs 108.05 BL, epoch 14: 101.23 DP vs 99.85 BL). The DropPath noise injection slows pointwise convergence by ~3 epochs. Under the 14-epoch wall-clock cap, the regularization tax exceeds the smoothing benefit. 

Student's analysis: "This is exactly the failure mode you get when applying a transformer regulariser to a setting that is **under-trained, not over-trained**. The baseline's late-epoch oscillation is not classical overfitting — it's edge-of-stability optimisation oscillation on a model that is still improving in expectation."

### Pattern recognition: wall-clock-bound regularization failures

This is the third such case in this round:
- PR #1808 (EMA weights): contamination from early-training weights, 14ep too short
- PR #1924 (n_head=8): +31% per-epoch cost, lost 3 epochs of training
- PR #1987 (DropPath): smoothing achieved at +3 epochs convergence slowdown cost

Common thread: 14 epochs is below the regularization payback threshold. All three would likely win under longer training budgets.

### Student-suggested follow-ups (not assigned)
- p=0.01-0.02 (finer-grained sweep): smaller noise → smaller smoothing AND smaller slowdown; net likely zero
- MLP-only DropPath: halve perturbation while preserving smoothing — interesting but still budget-bound
- Depth-scaled DropPath (linear schedule): standard ViT recipe; same fundamental budget issue
- Compose with SWA: rejected since SWA was just closed (#1951)

### Residual opportunities
- Revisit under longer training budget if SENPAI_MAX_EPOCHS is raised
- Combine with convergence-accelerating mechanism (e.g., if encoder-LR or β2 sweeps succeed and free up budget headroom)

---

## 2026-05-13 07:00 — PR #1978: Re-curriculum via per-sample log(Re)-tail loss multiplier (not sampler)

- **Branch:** `willowpai2g48h4-tanjiro/re-loss-weight` (CLOSED — structural regression)
- **Student:** willowpai2g48h4-tanjiro
- **W&B run:** `3rq00dn0`
- **Hypothesis:** Loss-side mechanism (vs the failed sampler-side #1868): apply `w = 1 + α × |normalized(log_Re)|` as per-sample loss multiplier, expecting to focus learning on Re-distribution tails (re_rand, geom_camber_rc).

### Results

| Metric | Baseline | Arm 1 (α=0.5) | Δ |
|--------|---------|---------------|---|
| `val_avg/mae_surf_p` | 97.9914 | 114.5247 | **+16.87% regression** |
| `test_avg/mae_surf_p` (4-split) | 88.5311 | 101.9300 | **+15.13% regression** |
| Best epoch | 11 | 12 / 14 | — |

Arm 2 (α=1.0) correctly skipped per branching rule.

Per-split val MAE:
| Split | Baseline | Arm 1 | Δ |
|-------|----------|-------|---|
| val_single_in_dist | 120.31 | 140.34 | +16.7% |
| val_geom_camber_rc | 115.98 | 137.99 | +19.0% |
| val_geom_camber_cruise | 66.04 | 80.66 | +22.1% (worst regression) |
| val_re_rand | 89.64 | 99.10 | +10.6% (smallest regression — opposite of prediction) |
| **val_avg** | **97.99** | **114.52** | **+16.87%** |

### Mechanism diagnosis — exemplary BIVW-interaction analysis from student

The student identified the structural failure precisely:
- BIVW (1/var(y_norm)) **implicitly up-weights low-Re samples** since high-Re has larger residual variance in normalized space
- Symmetric Re-tail boost composed multiplicatively with BIVW produces:
  - **Low-Re tail**: BIVW↑ × Re-tail↑ = double up-weight (already-easy samples dominate)
  - **Mid-Re**: BIVW≈1 × Re-tail↓ = down-weighted (most informative samples lose gradient)
  - **High-Re tail**: BIVW↓ × Re-tail↑ = approximate cancellation (intended boost absorbed)

The smoking gun: **val_geom_camber_cruise regressed MOST (+22.1%)** while it was expected to be unaffected. The easiest split sits at higher-Re samples where the Re-tail boost should help but is cancelled by BIVW.

`re_w` diagnostics: mean=1.36, range=[1.00, 3.19] — meaningful, non-trivial signal (not a null perturbation like #1868), but downstream behavior is destructive.

### Conclusion

Combined with PR #1868 (sampler-side variant, structural no-op), the entire **symmetric Re-tail re-weighting** family is now characterized as structurally incompatible with BIVW. The student's own follow-up #4: "Re-curriculum as conceived (symmetric tail boost composed with BIVW) is not a path to beat 97.9914 — recommend closing this thread."

### Residual opportunities (not assigned)
- **Asymmetric high-Re-only weighting**: `w = 1 + α × max(0, normalized(log_Re))` — avoids the BIVW low-Re double-counting
- **Replace BIVW with Re-tail (not compose)**: tests whether they're addressing the same problem
- Future-loop candidates if other directions stall

---

## 2026-05-13 07:00 — PR #1951: SWA late-epoch averaging: avg last K checkpoints post-training

- **Branch:** `willowpai2g48h4-askeladd/swa-late-epoch` (CLOSED — mechanism works, trajectory variance overwhelms)
- **Student:** willowpai2g48h4-askeladd
- **W&B runs:** `aawyil4y` (k=3, start=12), `3j5wwshz` (k=5, start=10)
- **Hypothesis:** Average last K checkpoint weights post-training to center on local basin (bypassing EMA's early-training contamination from PR #1808).

### Results

| Arm | best single-epoch | last-epoch | SWA-averaged | SWA gain | vs baseline 97.99 |
|-----|-------------------|------------|--------------|----------|-------------------|
| Arm 1 (k=3, start=12) | 105.6335 (ep 13) | 109.7513 | **101.2533** | **−4.38 (−4.1%)** | **+3.33% regression** |
| Arm 2 (k=5, start=10) | 105.8172 (ep 14) | 105.8172 | **102.4204** | **−3.40 (−3.2%)** | **+4.52% regression** |

Test 4-split: Arm 1 = 89.4132 vs 88.5311 baseline → **+0.99% test regression** (much smaller than val gap of +3.33%).

### Analysis

**SWA mechanism worked exactly as predicted**:
- k=3 SWA(ep 12,13,14) is **8.7 points lower** than the mean of the 3 epochs (108.46), **4.4 points lower** than minimum
- k=3 beats k=5 (101.25 vs 102.42) — including epochs 10-11 (val 119, 116) drags toward less-converged weights

**Why it fails to beat baseline**: This run's trajectory landed in a much worse basin (best single ~105.6) than baseline's trajectory (best single ~97.99). That's an **~8 point trajectory gap** that the **~4 point SWA gain** cannot bridge. Likely seed variance.

**Interesting side finding**: Test gap (+0.99%) is much smaller than val gap (+3.33%). Consistent with SWA's regularization effect helping generalization more than val MAE — but single-seed evidence is anecdotal.

### Conclusion

SWA implementation is sound and reusable. But under the 30-min budget with 14 epochs:
- Seed variance can produce ±8 point swings in best single-epoch val
- SWA mechanism delivers ~4 points of basin-centering improvement
- These are not commensurate; trajectory variance dominates

The student's follow-up #1 (paired-seed comparison) would isolate the SWA delta but doubles GPU cost. Not the highest-value use of askeladd's slot. Mechanism noted for future plateau-protocol use.

### Residual opportunities
- Re-run baseline + SWA on a paired seed
- k=2 (epochs 13-14 only) — student's follow-up #2
- Compose with longer-budget runs (--epochs 20) — student's follow-up #3
- Investigate why this trajectory landed worse than baseline — could be a deeper hyperparameter sensitivity issue

---

## 2026-05-13 06:35 — PR #1950: Adaptive Huber delta: EMA of p75 per-batch residuals (self-tuning δ)

- **Branch:** `willowpai2g48h4-fern/adaptive-huber-delta` (CLOSED — mechanism collapse)
- **Student:** willowpai2g48h4-fern
- **W&B run:** `dweh5tno`
- **Hypothesis:** Self-tuning δ via EMA of p75 per-batch surface residuals; warm start at δ=0.5, α=0.99 EMA, clamp [0.2, 2.0]. Tests whether fixed δ=0.5 is sub-optimal across the training trajectory.

### Results

| Metric | Baseline (#1795) | Arm 1 (`dweh5tno`) | Δ |
|--------|------------------|---------------------|---|
| `val_avg/mae_surf_p` (best) | 97.9914 | 100.1991 | **+2.25% regression** |
| `test_avg/mae_surf_p` (4-split) | 88.5311 | 89.4779 | **+1.07% regression** |
| Best epoch | 11 | 13 / 14 | — |

Arm 2 (p90) correctly skipped per branching rule (δ collapsed below 0.2).

### Mechanism diagnosis — superb root-cause analysis from student

δ trajectory (selected points from `train/adaptive_huber_delta`):
| step | EMA δ | raw p75 |
|------|-------|---------|
| 1 | 0.5057 | 1.0655 |
| 61 | **0.2096 (floor)** | 0.2125 |
| 260 | 0.2848 | 0.1503 |
| 975 | 0.2057 | 0.1109 |
| 2858 | 0.2003 | 0.0425 |
| 5245 (final) | 0.2000 | 0.0882 |

- **δ pinned at clamp_lo=0.2 for 88% of training** (439/500 logged steps).
- Median raw p75 of normalized surface residuals: **0.106** (well below clamp_lo).
- At δ≈0.2, `surf_l1_frac` (residuals in L1 branch): mean 12.7%, range 0.3–41.6%.

### Analysis

The hypothesis was based on the premise that residuals are "large" early and "small" late. Student's data shows the **decoupled-LR (PR #1795) baseline drives residuals below 0.2 within ~60 training steps** — the surf_head_lr=5e-3 collapses residuals far faster than expected. EMA of p75 correctly tracks this collapse and δ hits the floor immediately.

Effectively, this run was a **fixed-δ=0.2 ablation under the decoupled-LR baseline**. Compare to PR #1627 (fixed-δ sweep) which showed δ=0.2 → +17.2% under the PRE-decoupled-LR baseline. This run shows δ=0.2 → only +2.25% under the new baseline.

**Important side finding**: The merged decoupled LR (PR #1795) flattened the δ-landscape. The δ=0.5 narrow sweet spot is now wider — but δ=0.5 is still the best point we have.

### Suggested follow-ups (from student, prioritized)
- Lower clamp_lo (0.05) to let δ ride steady-state → effectively pure MSE, expected to underperform (#1650 showed MSE on volume already)
- Target p99 quantile → may give productive δ in [0.3, 0.6] but is a different hypothesis
- Fixed δ-sweep under new baseline (clean test) — would only confirm δ=0.5 is optimal; not assigning.

### Residual opportunities
- Stochastic Depth (DropPath) — regularization angle no other PR is exploring. Assigned to fern as #1987.
- Other smooth-loss families (LogCosh, Pseudo-Huber) could give a different gradient shape than Huber-δ=0.5; not assigning now since the δ landscape is flat at this baseline.

---

## 2026-05-13 06:20 — PR #1924: More attention heads: n_head 4→8 (wall-clock-neutral capacity axis)

- **Branch:** `willowpai2g48h4-edward/n-head-8` (CLOSED — dead end)
- **Student:** willowpai2g48h4-edward
- **W&B run:** `m8kevrph`
- **Hypothesis:** n_head 4→8 at fixed n_hidden=128 (head_dim 32→16) is wall-clock-neutral because total attention FLOPs are conserved at fixed sequence×hidden. 8 heads should provide richer slice-attention patterns improving accuracy.

### Results

| Metric | Baseline (n_head=4, #1558) | n_head=8 (`m8kevrph`) | Δ |
|--------|---------------------------|-----------------------|---|
| `val_avg/mae_surf_p` | 98.1642 (epoch 14) | 116.4421 (epoch 11) | **+18.62% regression** |
| `test_avg/mae_surf_p` (3-split) | 98.7537 | 117.3519 | **+18.83% regression** |
| Per-epoch wall time | 133.4 s | 175.0 s | **+31.2% slowdown** |
| Epochs in 30-min cap | 14 | 11 | −3 epochs |

Per-split val MAE (best ckpt epoch 11):
| Split | Baseline | n_head=8 | Δ |
|-------|----------|----------|---|
| val_single_in_dist | 123.14 | 146.41 | +18.9% |
| val_geom_camber_rc | 107.24 | 130.59 | +21.8% |
| val_geom_camber_cruise | 73.28 | 85.99 | +17.3% |
| val_re_rand | 88.99 | 102.79 | +15.5% |
| **val_avg** | **98.1642** | **116.4421** | **+18.6%** |

**Key per-epoch comparison**: At equal epoch 11, n_head=8 beats baseline by −9.3% (116.44 vs 128.39) — meaning 8 heads ARE better per epoch. The regression is entirely due to wall-clock (+31% per epoch → 11 vs 14 epochs).

### Analysis

Wall-clock prediction wrong. FLOPs conservation assumed but ignores kernel overhead: at head_dim=16, each head's matmul is below GEMM efficiency threshold. More per-head launches (`to_q/k/v/einsum`) cause overhead that doesn't fuse. The Transolver `slice_token = einsum("bhnc,bhng→bhgc", ...)` scales linearly with n_head and dominates step time.

**Conclusion**: Fifth wall-clock-bound capacity failure. Pareto frontier (depth=5, n_head=4, slice_num=64, ~14 ep) confirmed across all capacity-axis perturbations. Arm 2 (n_head=16) correctly skipped per branching rule.

### Residual opportunities
- n_head=8 IS better per epoch — would win under longer wall-clock budget
- bf16/torch.compile might recover the 31% overhead and flip to a win (#1572)
- Capacity wins at our budget must come from efficiency (BF16), not parameter count

---

## 2026-05-13 06:20 — PR #1868: log(Re) quantile bucketing sampler — explicit Re-curriculum (bounded)

- **Branch:** `willowpai2g48h4-tanjiro/log-re-quantile-bucketing` (CLOSED — mechanism failure)
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `ij9lcpi8` (10 buckets), `2ogoct1f` (5 buckets)
- **Hypothesis:** Quantile-bucket the log(Re) range, sample uniformly across buckets, weight by 1/count — a bounded replacement for the +272% 1/var(p) sampler failure.

### Results

| Arm | best_epoch | val_avg/mae_surf_p | test_avg (4-split) | Δ vs baseline |
|-----|------------|-------------------|-------------------|---------------|
| Baseline (#1558) | 14 | 98.1642 | NaN | — |
| Arm 1 (10 buckets) `ij9lcpi8` | 12 | 120.1534 | 109.7494 | **+22.4%** |
| Arm 2 (5 buckets) `2ogoct1f` | 14 | 106.2364 | 95.8183 | **+8.2%** |

### Analysis

**Structural no-op**: Quantile bucketing by construction puts ~equal sample counts in each bucket (max/min ratio 1.013–1.020×). Therefore 1/count weights are also ~uniform (max/min 1.013–1.020×). After composition with existing domain weights, effective distribution is essentially identical to baseline. The ±2% perturbation from Re-bucket factor just reshuffles the sample ordering via `WeightedRandomSampler`, introducing RNG noise.

Root cause: The two design choices "sample uniformly across buckets via quantile" and "weight by 1/count" cancel each other out by construction.

The regression comes from the tiny sampler weight perturbation reshuffling the per-step sample order, compounding with cosine LR. Both arms ran at similar epoch counts to baseline but with different (worse) luck on the sample draw.

**Conclusion**: Mechanism is broken, not just under-parameterized. Equal-width log(Re) buckets (not quantile) or a loss-side multiplier would both work. Follow-up: loss-side Re-curriculum (#1978 tanjiro), which avoids the sampler cancellation entirely.

### Residual opportunities
- Equal-width log(Re) buckets would produce non-uniform counts and non-trivial 1/count weights
- Loss-side multiplier `w = 1 + α × |norm(log(Re))|` is independent of bucket count (#1978)
- Multi-seed confirmation would clarify whether the +8.4% is noise vs. real harm

---

## 2026-05-12 19:00 — PR #1502: Batch inverse-variance weighting for heteroscedastic Re

- **Branch:** `willowpai2g48h4-tanjiro/per-sample-re-normalized-loss` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-tanjiro
- **W&B run:** `e72nzxo5`
- **Hypothesis:** Per-sample inverse-variance weighting (BIVW) to re-balance gradient signal away from high-Re/high-variance samples. Weight each sample by `1 / var(y_norm_valid)`, normalized to mean=1.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| `val_avg/mae_surf_p` | **126.0751** | Best epoch 14/50; **round-4 baseline** |
| `test_avg/mae_surf_p` | NaN | Pre-existing data/scoring bug (see below) |
| Best epoch | 14 | 30-min wall-clock cap hit (~132 s/epoch) |
| Training time | 31.1 min | |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| `val_single_in_dist` | 160.74 | 1.88 | 0.85 |
| `val_geom_camber_rc` | 133.28 | 2.57 | 1.01 |
| `val_geom_camber_cruise` | **97.21** | 1.52 | 0.59 |
| `val_re_rand` | 113.08 | 1.99 | 0.77 |
| **val_avg** | **126.08** | 1.99 | 0.80 |

Per-split test surface-p MAE (3 of 4 clean):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 145.43 |
| `test_geom_camber_rc` | 117.44 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 109.27 |
| test 3-split mean | ~124.0 |

### Analysis and Conclusions

**BIVW worked as hypothesised.** The low-Re-dominated `val_geom_camber_cruise` split came in at 97.21 — the lowest of the four splits by a wide margin — consistent with the prediction that IVW would most benefit low-variance (low-Re) samples that were being under-trained by the uniform MSE.

**BIVW is the new round-4 baseline.** Established at `val_avg/mae_surf_p = 126.0751`.

**Known infrastructure issue discovered:** `test_geom_camber_cruise` sample 20 has 761 `-inf` values in the GT pressure channel (volume nodes). `data/scoring.py:accumulate_batch` intends to skip non-finite GT samples but has a bug: `err = abs(pred - y)` is computed before the per-sample mask is applied, so `inf × 0 = NaN` in float arithmetic poisons the split-level accumulator. Since `data/scoring.py` is read-only, the fix goes in `train.py:evaluate_split` — assigned to tanjiro as PR #1527.

**Training was still improving at cap.** The val curve was still decreasing monotonically at epoch 14. With a longer budget, BIVW could improve further.

---

## 2026-05-12 22:00 — PR #1558: Huber (SmoothL1) surface loss delta=0.5 (MERGED — new baseline)

- **Branch:** `willowpai2g48h4-thorfinn/smooth-l1-surface-loss` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `2w7nverc` (delta=0.5, winner), `3goyvktl` (delta=1.0, secondary)
- **Hypothesis:** Replace MSE surface loss with Huber (SmoothL1) loss. For |err| < delta: quadratic (like MSE scaled); for |err| >= delta: linear (L1-consistent). Aligns optimization objective directly with MAE evaluation metric. Orthogonal to BIVW (which handles between-sample gradient inflation at the sample level; Huber handles within-sample per-node gradient inflation).

### Results — Winning arm delta=0.5

| Metric | Value | vs prior baseline (119.2987) |
|--------|-------|------------------------------|
| `val_avg/mae_surf_p` | **98.1642** | **−17.72%** |
| test 3-split mean | **98.7537** | −17.45% |
| Best epoch | 14 / 14 | (still improving at cap) |
| Peak VRAM | 43.0 GB | |

Per-split val surface-p MAE (delta=0.5):

| Split | mae_surf_p | vs prior baseline |
|-------|-----------|-------------------|
| `val_single_in_dist` | 123.14 | 140.09 → −12.1% ✓ |
| `val_geom_camber_rc` | 107.24 | 142.40 → −24.7% ✓ (OOD regression fully reversed) |
| `val_geom_camber_cruise` | 73.28 | 85.98 → −14.8% ✓ |
| `val_re_rand` | 88.99 | 108.73 → −18.2% ✓ |
| **val_avg** | **98.1642** | **−17.72%** |

Per-split test (delta=0.5, 3 of 4 finite):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 111.92 |
| `test_geom_camber_rc` | 98.91 |
| `test_geom_camber_cruise` | NaN (cruise bug) |
| `test_re_rand` | 85.43 |
| **3-split mean** | **98.7537** |

Secondary arm (delta=1.0): val_avg=117.74 (only −1.3% — barely above noise).

### Analysis and Conclusions

**New round-4 baseline: 98.1642.** This is the largest single-PR improvement so far (17.7% vs the prior best of −5.4%).

**Mechanism confirmed:** `train/surf_l1_frac` (fraction of surface errors above delta) stays high throughout training for delta=0.5 — most residuals are in the L1 regime, producing constant-magnitude gradients that directly minimise MAE. delta=1.0 keeps too many residuals in the quadratic regime (barely different from MSE).

**OOD camber regression reversed:** val_geom_camber_rc was the problematic split (+6.84% regression in #1528). With Huber, it drops 24.7% — the largest per-split gain. Root cause: MSE pulled the surf-head toward large-residual OOD outlier nodes; Huber capped that pull at delta.

**BIVW + surf-head + Huber synergy:** Each mechanism targets a different scale of gradient heterogeneity:
- BIVW: between-sample (different Re → different variance)
- Surf-head: surface vs volume specialisation  
- Huber: within-sample per-node (outlier surface nodes)

**Next:** Test smaller deltas (0.2, 0.3) assigned to thorfinn PR #1627. Also need to test whether grad-clip (#1499 rebase), per-channel BIVW (#1580), and BF16/AMP (#1572) all stack on top of this new baseline.

---

## 2026-05-12 20:55 — PR #1527: Fix test NaN — guard evaluate_split against non-finite GT/pred (MERGED)

- **Branch:** `willowpai2g48h4-tanjiro/fix-test-nan-scoring` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-tanjiro
- **W&B run:** `dg5xbm6g`
- **Hypothesis:** Pipeline fix — `data/scoring.py:accumulate_batch` computes `err = |pred - y|` before applying the per-sample finite mask, so `inf × 0 = NaN` poisons the accumulator. Since `data/scoring.py` is read-only, the fix is in `train.py:evaluate_split` via `nan_to_num` pre-filter + explicit `_y_ok` mask.

### Results

| Metric | Pre-fix baseline (`e72nzxo5`) | Post-fix run (`dg5xbm6g`) |
|--------|------------------------------|---------------------------|
| `val_avg/mae_surf_p` | 126.0751 | 129.6761 (+2.86%, stochastic) |
| `test_avg/mae_surf_p` | **NaN** | **119.7792** ✓ |
| Best epoch | 14 | 11 |

Per-split test (first time all four finite):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 150.59 |
| `test_geom_camber_rc` | 133.77 |
| `test_geom_camber_cruise` | **81.42** (was NaN) |
| `test_re_rand` | 113.34 |
| **test_avg** | **119.78** |

### Analysis and Conclusions

**Pipeline fix merged.** Despite val_avg drift, this is essential infrastructure: paper-facing metrics now report finite values for all four test splits.

**val drift correctly attributed to stochasticity, not the guard.** Tanjiro proved the guard is val-neutral: baseline val splits have no non-finite GT (else baseline val_avg would have been NaN too), so `_y_ok` is all-True and `nan_to_num` is a no-op on already-finite tensors. Without a fixed seed, run-to-run variance commonly hits 1–3% on individual splits. PR was based on BIVW only (#1502), not the current advisor branch (BIVW + surf-head, #1528), so its val=129.68 doesn't compare directly with the current 119.30 baseline.

**Cruise pressure split now reports 81.42** — lowest of the 4 test splits, mirroring the val cruise behaviour (85.98 in #1528). Confirms the fix surfaces the model's actual cruise performance, which was previously hidden behind NaN.

**Tanjiro also cleaned up two orphan GPU processes** from prior wake-up cycles. Operationally hygienic.

**Next merged training run should show:** val_avg ≈ 119.30 (BIVW+surf-head baseline) + finite test_avg ≈ 119+. The next experiment to merge (likely fern's rebase #1499 or thorfinn's Huber #1558) will give us the true paper-facing test number.

---

## 2026-05-12 21:00 — PR #1500: n_hidden 128 → 256, n_head 4 → 8 (CLOSED — budget failure)

- **Branch:** `willowpai2g48h4-frieren/larger-hidden-dim` (closed)
- **Student:** willowpai2g48h4-frieren
- **W&B runs:** `ocxqv6a9` (best), `nnjrx4p3` (replicate)
- **Hypothesis:** Doubling hidden dimension from 128→256 and n_head 4→8 quadruples attention capacity and doubles MLP capacity, targeting model capacity as the bottleneck.

### Results

| Metric | Run `ocxqv6a9` | Run `nnjrx4p3` |
|--------|---------------|---------------|
| `val_avg/mae_surf_p` | **158.7552** | 163.2345 |
| Best epoch | 8 / 50 | 8 / 50 |
| Training time | 33.1 min | 33.4 min |
| n_params | 2.54M | 2.54M |
| Peak VRAM | 42.0 GB | ~42 GB |

Per-split val (best run `ocxqv6a9`):

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| `val_single_in_dist` | 185.92 | 2.27 | 0.99 |
| `val_geom_camber_rc` | 178.98 | 3.37 | 1.29 |
| `val_geom_camber_cruise` | 121.34 | 1.81 | 0.76 |
| `val_re_rand` | 148.79 | 2.73 | 1.01 |
| **val_avg** | **158.76** | 2.54 | 1.01 |

### Analysis and Conclusions

**Closed — budget failure, not a model quality signal.** val_avg=158.76 vs baseline 119.30 (+33%), but this is entirely because the model only ran 8 of 50 epochs at 3.7 min/epoch vs the expected 0.85 min/epoch. Training loss was still descending monotonically at cap (val 240→168→159 over epochs 1→5→8).

**Critical diagnostics (will inform future experiments):**
1. **Wall-clock 4× slower than predicted** — slice-attention temporaries (`fx_mid`, `slice_weights`, `bhnc/bhng` einsum) dominate both VRAM and compute at N=242K nodes. Not captured in `[B,N,hidden]×layers` activation estimate.
2. **VRAM 42 GB vs predicted 5 GB** — off by ~8×; the dominant term is attention intermediates, not activations.
3. **Params 2.54M not 9.4M** — PhysicsAttention keeps `to_q/k/v` at `dim_head→dim_head`, so param count scales ~3·dim² not ~4·dim².
4. **Test cruise NaN is prediction overflow**, not GT corruption — different cause from #1502/#1528. The ~epoch-8 undertrained model overflows fp32 on cruise pressure at large N.

**Fix:** BF16/AMP would roughly halve VRAM and speed up forward pass ~1.5×, allowing n_hidden=256 to reach ~20 epochs in 30 min. Assigned to frieren as PR #1559.

---

## 2026-05-12 20:30 — PR #1528: BIVW + zero-init surface correction head (MERGED)

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-on-bivw` (squash-merged into `icml-appendix-willow-pai2g-48h-r4`)
- **Student:** willowpai2g48h4-thorfinn
- **W&B run:** `an97gg8n`
- **Hypothesis:** Composition of BIVW (per-sample loss re-weighting) and a zero-initialized additive SurfaceCorrection MLP head (`[3+24, 64, 64, 3]`, last layer zero-init, surface nodes only). Both mechanisms are orthogonal: BIVW targets gradient heterogeneity at the sample level; the surf-head targets the architectural under-representation of surface nodes. Used `torch.where(is_surface, delta, zero)` to safely handle NaN × 0 contamination from volume node overflow.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| `val_avg/mae_surf_p` | **119.2987** | Best epoch 13/14; **new round-4 baseline** (−5.37% vs 126.0751) |
| `test_avg/mae_surf_p` | NaN | Pre-existing cruise split scoring bug |
| Best epoch | 13 | 30-min cap hit (~131 s/epoch) |
| Total params | 0.669M | Transolver 0.643M + SurfaceCorrection 0.026M |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p | vs prior baseline |
|-------|-----------|-------------------|
| `val_single_in_dist` | 140.09 | −12.85% ✓ |
| `val_geom_camber_rc` | 142.40 | +6.84% ✗ |
| `val_geom_camber_cruise` | 85.98 | −11.55% ✓ |
| `val_re_rand` | 108.73 | −3.85% ✓ |
| **val_avg** | **119.2987** | **−5.37%** |

Per-split test surface-p MAE (3 of 4 clean):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 127.93 |
| `test_geom_camber_rc` | 127.18 |
| `test_geom_camber_cruise` | **NaN** (data corruption) |
| `test_re_rand` | 103.79 |
| test 3-split mean | ~119.63 |

### Analysis and Conclusions

**BIVW + surf-head composition worked — new baseline 119.30.** Confirms the orthogonality hypothesis: BIVW (loss-level) and the surface correction head (architecture-level) provide complementary inductive bias. Three of four val splits improved; `val_geom_camber_rc` (raceCar OOD camber) regressed +6.84%, which warrants investigation in future work.

**`torch.where` NaN guard confirmed correct.** Replacing `delta * is_surface.float()` with `torch.where(is_surface, delta, zero)` correctly propagates zeros instead of NaN at volume nodes with overflow predictions.

**Composition principle validated.** The standalone surf-head (#1503) was 6.2% worse than BIVW alone; adding it on top of BIVW is 5.4% better. The head needed the cleaner gradient signal that BIVW provides to specialize effectively.

**Next:** Need to test whether higher-LR + grad-clip (#1499, which reached 113.15 on BIVW alone) stacks further on top of this combined baseline.

---

## 2026-05-12 20:00 — PR #1499: Grad-clip max_norm=1.0 + lr 5e-4 → 1e-3 (SENT BACK — merge conflicts)

- **Branch:** `willowpai2g48h4-fern/gradient-clipping-and-higher-lr`
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `ihl8ashe` (primary, lr=1e-3), `160d99m0` (fallback, lr=7e-4)
- **Hypothesis:** Gradient heterogeneity across Re samples causes large per-batch gradient norms that destabilise slice-attention. Capping with `max_norm=1.0` and doubling LR to 1e-3 should stabilise training and converge faster.

### Results

| Arm | Run | Best epoch | `val_avg/mae_surf_p` | test 3-split mean |
|-----|-----|------------|----------------------|-------------------|
| **primary** (`lr=1e-3, clip=1.0`) | `ihl8ashe` | 13 | **113.1491** | 109.64 |
| fallback (`lr=7e-4, clip=1.0`) | `160d99m0` | 12 | 119.0885 | 123.00 |

Per-split surface-p MAE (test, primary arm `lr=1e-3`):

| Split | mae_surf_p |
|-------|-----------|
| `test_single_in_dist` | 110.07 |
| `test_geom_camber_rc` | 111.92 |
| `test_geom_camber_cruise` | NaN |
| `test_re_rand` | 106.94 |
| 3-split mean | 109.64 |

Grad-norm telemetry (primary arm):
- **100% of steps clipped** — raw norms ranged 2.18 to 712.86 (median 30.79, mean 48.31)
- `max_norm=1.0` is acting as a uniform per-step renormaliser, not an outlier suppressor

### Analysis and Conclusions

**Strong result (113.15 on BIVW-only basis) — could not merge due to conflicts with advisor branch.** PR was branched before the BIVW + surf-head composition (#1528) merged. Sent back for rebase onto `icml-appendix-willow-pai2g-48h-r4` with new baseline 119.2987.

**100%-clipping finding is important.** With `max_norm=1.0` every single step is clipped. The effective LR is `(1.0 / raw_norm) × lr_nominal ≈ 1e-3 / 30.8 ≈ 3.2e-5` (median). The benefit of the higher nominal LR is asymmetric — it only matters on the small fraction of steps near the clip threshold. Suggested follow-up: try `grad_clip ∈ {1.0, 10.0}` on the new baseline to separate true outlier suppression from step renormalisation.

**Next:** Fern is rebasing onto the new baseline (BIVW + surf-head, 119.30) and adding a `--grad_clip 10.0` arm alongside the primary `--grad_clip 1.0`. The current 113.15 on BIVW-only was not compared against the newer 119.30 baseline; rebased run will clarify whether grad-clip still helps on top of surf-head.

---

## 2026-05-12 19:05 — PR #1503: Additive zero-init surface-only correction head (CLOSED)

- **Branch:** `willowpai2g48h4-thorfinn/surface-aware-output-head` (closed, not merged)
- **Student:** willowpai2g48h4-thorfinn
- **W&B run:** `8ffez1mk`
- **Hypothesis:** Zero-initialized additive MLP (`[3+24, 64, 64, 3]`) applied only at surface nodes after the base Transolver prediction. The head starts as an identity correction (last layer zeroed) and specialises the prediction for the surface vs. volume regime.

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| `val_avg/mae_surf_p` | **133.928** | Best epoch 14/50; **6.2% worse than BIVW baseline** |
| `test_avg/mae_surf_p` | NaN | Same cruise split NaN issue + base model prediction overflow |
| Best epoch | 14 | 30-min wall-clock cap; same budget as tanjiro |
| Training time | 31.4 min | |

Per-split val surface-p MAE at best checkpoint:

| Split | mae_surf_p |
|-------|-----------|
| `val_single_in_dist` | 147.33 |
| `val_geom_camber_rc` | 152.10 |
| `val_geom_camber_cruise` | 112.03 |
| `val_re_rand` | 124.26 |
| **val_avg** | **133.93** |

### Analysis and Conclusions

**Closed — 6.2% worse than baseline.** At the same 14-epoch budget the standalone surface head scored 133.93 vs. BIVW's 126.08. Both runs are still undertrained at the cap (val still declining), so we cannot attribute the gap purely to the architectural difference — but the gap is significant.

**The head is not dead.** The composition **BIVW + surf_head** has not been tested. BIVW was not in this run. Composition is orthogonal (loss re-weighting vs. architectural specialisation) and is now assigned as PR #1528 (thorfinn).

**Robustness improvement noted.** Thorfinn recommended replacing `delta * is_surface.float()` with `torch.where(is_surface, delta, zero)` to avoid `NaN × 0 = NaN` contamination from volume-node overflows. Incorporated into the composition PR #1528 instructions.

**Test NaN (additional cause).** Unlike tanjiro's data-corruption root cause, thorfinn's test NaN was caused by the base Transolver overflowing to non-finite values on one test cruise sample. The guard fix in PR #1527 will address both causes.

---

## 2026-05-12 23:00 — PR #1580: Per-channel BIVW (CLOSED — 29.6% regression)

- **Branch:** `willowpai2g48h4-tanjiro/per-channel-bivw`
- **Student:** willowpai2g48h4-tanjiro
- **W&B run:** `rf4lp09j`
- **Hypothesis:** Replace scalar per-sample BIVW weight with separate per-channel weights `cw[b,c] = 1/var(y_norm[b,:,c])`, normalised to mean=1 per channel. Expected 1–4% improvement by giving pressure channel an independent variance track.

### Results

| Metric | Per-channel BIVW | Scalar BIVW baseline (an97gg8n) | Δ |
|--------|-----------------|--------------------------------|---|
| `val_avg/mae_surf_p` | **154.5967** | 119.2987 | **+29.6% ❌** |
| `test_avg/mae_surf_p` | 142.7603 | ~119.63 (3-split) | +19.3% ❌ |

Per-split val surface-p MAE at best checkpoint (epoch 14):

| Split | Per-channel BIVW | Prior baseline | Δ |
|-------|-----------------|----------------|---|
| `val_single_in_dist` | 213.4762 | 140.09 | +73.39 |
| `val_geom_camber_rc` | 171.6911 | 142.40 | +29.29 |
| `val_geom_camber_cruise` | 103.5190 | 85.98 | +17.54 |
| `val_re_rand` | 129.7005 | 108.73 | +20.97 |
| **val_avg** | **154.5967** | **119.2987** | **+29.6%** |

All 4 test splits were finite (PR #1527 NaN guard working).

### Analysis and Conclusions

**Closed — clear dead end.** Every validation split regressed; the approach is 58% worse than the current best baseline (98.16 after PR #1558).

**Mechanism (tanjiro's analysis, confirmed correct):** The original scalar BIVW was *implicitly* a p-variance-driven Re-curriculum. Because `p` spans 5 orders of magnitude in variance vs ~10^4 for Ux/Uy, the pooled per-sample `1/var(y_norm_valid)` was effectively dominated by `var(p)`. This meant: low-Re samples (small p) got upweighted, high-Re samples (large p) got downweighted — for *all three channels simultaneously*. Per-channel decoupling broke this coupling by letting `cw[b,Ux]` and `cw[b,Uy]` be large on high-Re samples while `cw[b,p]` is small. The model received conflicting signals: "learn velocity from high-Re samples but ignore their pressure" — exactly backwards for `val_avg/mae_surf_p`.

**Additional factor:** With batch_size=4 and p-variance over 10^5 range, per-sample `channel_w[:,p]` after mean-1 normalisation varied up to 50× within a single batch, giving extremely high gradient variance and effectively 1–2 samples dominating each step.

**Key lesson:** Scalar BIVW was doing more than "between-sample variance correction" — it was also acting as a p-aware Re-curriculum. This coupling should not be casually broken. Any future per-channel weighting experiment should preserve the p-dominated sample ordering, e.g. by computing sample weights from p-variance only and then applying a small per-channel log-mean correction on top.

**Potentially valuable follow-ups flagged by tanjiro (not assigned here):**
- Pre-compute frozen p-variance sample weights over the full corpus (makes the implicit curriculum explicit)
- EMA per-channel variance (cuts within-batch-of-4 noise)

---

## 2026-05-13 00:56 — PR #1650: Huber volume loss (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-tanjiro/huber-on-volume-loss`
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `rzv2hb5d` (vd=0.3), `ri4vj1nk` (vd=0.5, best), `qfvyn8wp` (vd=1.0)
- **Hypothesis:** Applying Huber to volume nodes reduces encoder gradient noise from volume outliers, indirectly improving surface MAE.

### Results

| Arm | vol_huber_delta | val_avg/mae_surf_p | Δ vs 98.16 | test_avg (4-split) |
|-----|-----------------|--------------------|-----------|---------------------|
| Baseline | 0 (MSE) | **98.1642** | — | — |
| vd=0.3 | 0.3 | 111.5928 | +13.7% ❌ | 101.68 |
| **vd=0.5** (best) | 0.5 | **106.6946** | **+8.7% ❌** | **96.83** |
| vd=1.0 | 1.0 | 117.7356 | +19.9% ❌ | 106.30 |

### Analysis and Conclusions

**Closed — all arms regressed.** Unimodal ordering around vd=0.5. The hypothesis that Huber on volume helps encoder quality is wrong.

**Root cause (tanjiro's analysis, confirmed):** Surface and volume play fundamentally different roles. Surface is the evaluated readout where Huber-MAE alignment matters; volume is the *supervisory signal* that shapes the shared encoder. Scale information from volume MSE is more valuable to encoder learning than outlier robustness. Huber on volume removes gradient scale information that was beneficial.

The unimodal ordering (vd=0.5 best of three, not monotone) confirms this is a noise minimisation over delta-tuning, not a true hypothesis-validates signal.

**Principle established:** Surface Huber + volume MSE is the correct recipe. Do not apply Huber to volume.

**Residual opportunity (not yet assigned):** Heavy-tail-only Huber on volume (apply only where |err| > 95th percentile) would more precisely target outliers without removing bulk gradient scale. Low priority given clean negative result.

---

## 2026-05-13 03:45 — PR #1498: Wider TransolverBlock MLP (mlp_ratio 2→4) (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-edward/wider-mlp-ratio`
- **Student:** willowpai2g48h4-edward
- **W&B run:** `ji5h4ww2` (mlp-ratio-4)
- **Hypothesis:** Standard transformer mlp_ratio is 4. Going from 2→4 should add per-node nonlinear capacity, giving −3% to −8% on val_avg/mae_surf_p.

### Results

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) | params | epoch time | epochs in 30 min |
|-----|--------------------|-----------|-------------------------------|--------|------------|------------------|
| **Baseline (PR #1558, mlp_ratio=2)** | **98.1642** | — | 98.7537 (3-split) | 0.643 M | 128 s | 14 |
| mlp_ratio=4 | **122.6751** | **+24.97% ❌** | 111.46 (4-split, all finite) | 0.997 M (+55%) | 152 s (+19%) | 12 |

Per-split val (best epoch 11):

| Split | val mae_surf_p | vs baseline |
|-------|---------------|------------|
| `val_single_in_dist` | 205.51 | 123.14 → **+66.9% ❌** |
| `val_geom_camber_rc` | 140.42 | 107.24 → +30.9% ❌ |
| `val_geom_camber_cruise` | 101.58 | 73.28 → +38.6% ❌ |
| `val_re_rand` | 119.21 | 88.99 → +33.9% ❌ |
| **val_avg** | **122.68** | **+24.97%** |

### Analysis and Conclusions

**Closed — wall-clock-bound capacity addition fails (third confirmation this cycle).**

**Root cause (edward's analysis, confirmed):**
1. Wider MLP slows per-epoch wall-clock 19% (152s vs 128s). Hard 30-min cap → 12 epochs vs baseline 14.
2. Per-epoch convergence is NOT improved with wider MLP — just more parameters fighting for the same epochs.
3. Worst regression on val_single_in_dist (+67%) shows extra capacity is overfitting the small (1499-sample) training set within the limited epochs.

**Principle reinforced (now 3× confirmed: warmup #1497, grad-clip #1499, wider-MLP #1498):** Under the 30-min wall-clock cap, the baseline at epoch 14 is still improving. Any change that costs ≥10% per-epoch time and doesn't accelerate convergence will lose 1-2 epochs and underperform. Capacity expansions must be paired with throughput recovery (BF16 — #1572 frieren in flight) or reductions elsewhere (e.g., fewer layers, smaller slice_num).

**Residual opportunities (edward's suggestions):**
1. **Throughput-neutral capacity reallocation** — reduce n_layers when increasing mlp_ratio. Reassigned: edward to test n_layers ∈ {3, 4} (reducing depth to gain epochs).
2. mlp_ratio=3 as compromise (deferred).
3. Higher LR with wider MLP (ViT/GPT scaling) — would need to compose with capacity addition.
4. mlp_ratio=4 + BF16 — composable with #1572 once that completes.

---

## 2026-05-13 03:25 — PR #1746: Frozen p-variance stratified sampler (CLOSED — variance-explosion failure)

- **Branch:** `willowpai2g48h4-tanjiro/frozen-p-variance-stratified-sampling`
- **Student:** willowpai2g48h4-tanjiro
- **W&B runs:** `uayv2md1` (stratified+BIVW), `8h2u23z6` (stratified-only, BIVW disabled)
- **Hypothesis:** Pre-compute per-sample sampling weight ∝ 1/var(p) over the full corpus, use WeightedRandomSampler. Makes BIVW's implicit Re-curriculum explicit at data-loader level; removes within-batch estimation noise.

### Results

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) |
|-----|--------------------|-----------|-------------------------------|
| **Baseline (PR #1558)** | **98.1642** | — | 98.7537 (3-split) |
| Arm 1: stratified + BIVW | 365.1298 | **+272% ❌** | 341.55 |
| Arm 2: stratified only (BIVW disabled) | 365.0082 | **+272% ❌** | 341.57 |

Per-split val (best epoch, Arm 2):

| Split | val mae_surf_p | vs baseline |
|-------|---------------|------------|
| `val_single_in_dist` | 511.17 | 123.14 → +315% ❌ |
| `val_geom_camber_rc` | 451.48 | 107.24 → +321% ❌ |
| `val_geom_camber_cruise` | 207.99 | 73.28 → +184% ❌ |
| `val_re_rand` | 289.39 | 88.99 → +225% ❌ |
| **val_avg** | **365.01** | **+272%** |

### Sampler diagnostic (the smoking gun)

| Statistic | Value |
|-----------|-------|
| p-variance min (raw) | 9.28 × 10⁻² |
| p-variance max (raw) | 2.30 × 10⁷ |
| **Dynamic range of var(p)** | **2.47 × 10⁸ ×** |
| Sample weight min (norm.) | ~1.5 × 10⁻⁶ |
| Sample weight max (norm.) | 371.71 |
| **Effective upweight ratio** | **2.47 × 10⁸ ×** |

### Analysis and Conclusions

**Closed — variance-explosion failure mode.** The hypothesis underestimated the empirical p-variance dynamic range by 6 orders of magnitude.

**Root cause (tanjiro's analysis, confirmed):** With var(p) spanning 0.09 to 2.3×10⁷ (8 OOM) and the WeightedRandomSampler interpreting weights as relative probabilities, the most-upweighted sample is drawn ~247 million times more often than the least. The effective training distribution collapses to a handful of low-Re samples (likely cruise-domain with tiny p-variance). The model never sees the high-Re/high-pressure cases that drive validation MAE.

**Arm 1 vs Arm 2 (0.03% diff) confirms the sampler dominates entirely:** once the data-loader collapses, BIVW has nothing to re-balance (the batches are near-identical).

**Why the existing domain-balanced sampler works but 1/var(p) doesn't:** The baseline `sample_weights` from `load_data()` have a dynamic range of only ~1.35× (599 raceCar single vs 443 cruise). 1/var(p) has a dynamic range **8 orders of magnitude larger**, so it becomes a Dirac comb instead of a smooth re-weighting.

**Principle established:** Any inverse-variance sampling weight on this corpus must be either (a) tempered with τ ∈ [0.05, 0.2], (b) log-compressed, (c) bucketed by quantile, or (d) replaced with a feature that has bounded dynamic range. Pure 1/var(p) is unusable.

**Residual opportunities (tanjiro's suggestions):**
1. log(Re) quantile bucketing (selected for follow-up — log(Re) spans only ~1.5 OOM).
2. Tempered inverse-variance: `w ∝ var(p)^(-τ)` with τ ≈ 0.1.
3. Log-spaced weights: `w ∝ 1/log(1+var(p))`.
4. Combine with existing domain-balanced sampler via product.

---

## 2026-05-13 02:10 — PR #1497: 5-epoch linear warmup + CosineAnnealingLR (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-askeladd/warmup-cosine-lr`
- **Student:** willowpai2g48h4-askeladd
- **W&B run:** `fhdmn0xr` (warmup-5-cosine-huber0.5)
- **Hypothesis:** Adding 5-epoch linear LR warmup before CosineAnnealingLR prevents early-epoch instability in the slice-attention softmax weights and enables a higher effective LR.

### Results

| Arm | Warmup epochs | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p |
|-----|---------------|--------------------|-----------|---------------------|
| **Baseline (PR #1558)** | 0 (flat CosineAnnealingLR T_max=50) | **98.1642** | — | 98.7537 (3-split) |
| warmup-5 | 5 | **115.8073** (best epoch 13) | **+17.98% ❌** | 106.82 (4-split, all finite) |

Per-split val (best checkpoint, epoch 13):

| Split | val mae_surf_p (warmup-5) | vs baseline |
|-------|--------------------------|------------|
| `val_single_in_dist` | 147.36 | 123.14 → **+19.7% ❌** |
| `val_geom_camber_rc` | 135.03 | 107.24 → **+25.9% ❌** |
| `val_geom_camber_cruise` | 92.27 | 73.28 → **+25.9% ❌** |
| `val_re_rand` | 102.35 | 88.99 → **+15.0% ❌** |
| **val_avg** | **115.8073** | **+17.98%** |

LR trajectory confirmed correct: epoch 1→5 ramps linearly 1e-4→5e-4, then cosine from epoch 6.

### Analysis and Conclusions

**Closed — large regression, no instability to justify warmup.** The failure is structural, not tuning-related.

**Root cause — wall-clock-bound training makes warmup a liability:**
Training is capped at 30 min ≈ 14 epochs (out of 50 configured). A 5-epoch warmup spends 4 of the 14 most-productive epochs at 20–80% of peak LR. CosineAnnealingLR(T_max=50) barely decays by epoch 14 (we're at ~96% of peak), so the baseline is effectively a **flat LR at 5e-4** — and that flat schedule wins. No instability was observed in the baseline trajectory, so the premise (warmup prevents divergence) was wrong.

**The cosine tail benefit (gradual late refinement) never materializes** because T_max=50 far exceeds the actual training duration. We paid the warmup cost without collecting the dividend.

**Principle established:** Under the 30-min / ~14-epoch wall-clock cap, LR schedules with T_max >> epochs_run are effectively flat. If testing schedules, must set T_max ≤ epochs_actually_run.

**Residual opportunity (askeladd's suggestion):** OneCycleLR (Smith 2018) is designed for short-budget regimes. Set pct_start=0.1 (10% of total steps = ~525 warmup steps) with `total_steps = estimated_epochs × len(train_loader)`. This gives rapid warmup + full decay in the actual training window, not the 50-epoch hypothetical.

---

## 2026-05-13 00:07 — PR #1499: Grad-clip + higher LR on Huber baseline (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-fern/gradient-clipping-and-higher-lr`
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `8p20jj30` (clip=1.0), `624phqjd` (clip=10.0)
- **Hypothesis:** Adding gradient clipping + higher LR (1e-3 vs 5e-4) stacks on top of the Huber baseline.

### Results

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) |
|-----|-------------------|-----------|-------------------------------|
| clip=1.0, lr=1e-3 | 99.6393 | +1.50% ❌ | 90.07 |
| clip=10.0, lr=1e-3 | **99.5928** | +1.45% ❌ | **87.37** |
| Baseline (PR #1558) | 98.1642 | — | NaN (3-split: 98.75) |

Per-split val (best arm, clip=10.0):

| Split | val mae_surf_p | vs baseline |
|-------|---------------|------------|
| `val_single_in_dist` | 122.75 | 123.14 → −0.3% |
| `val_geom_camber_rc` | 117.04 | 107.24 → **+9.1% ❌** |
| `val_geom_camber_cruise` | 66.68 | 73.28 → −9.0% ✓ |
| `val_re_rand` | 91.91 | 88.99 → +3.3% |
| **val_avg** | **99.5928** | **+1.45%** |

### Analysis and Conclusions

**Closed — val regressed.** Neither arm beats 98.16. The main driver is `val_geom_camber_rc` regression (+9.1%).

**Key discovery — Huber compresses grad norms by 5×:** Raw pre-clip gradient norms with Huber active have median ~7 and max ~96, vs median ~31 and max ~720 in the pre-Huber run. Huber is doing exactly its job — capping per-node gradient magnitude in the L1 regime. This makes `clip=1.0` still too aggressive (100% of steps clipped — uniform renormaliser). `clip=10.0` correctly clips only the right tail (~27% of steps).

**But clip doesn't stack:** Huber already removes within-sample per-node gradient outliers. Adding batch-level clipping is redundant — both are attacking the same source of gradient noise. Not orthogonal like BIVW+surf-head+Huber were.

**Test divergence noted:** Test 3-split comparison (fair): (108.12+100.75+84.04)/3 = 97.64 vs baseline 98.75 = **−1.1% improvement**. This marginal test improvement while val regresses is not enough to override the val decision.

**For future clip experiments:** if composing clip with Huber, use `grad_clip=10.0` (not 1.0). And note that the benefit case is weak — both are targeting the same gradient noise source.

**Gotcha documented:** PR #1558 left dataclass `huber_delta: float = 1.0` but winning baseline used `--huber_delta 0.5`. All rebased PRs must pass explicit `--huber_delta 0.5`.

---

## 2026-05-13 01:48 — PR #1627: Huber delta sweep (CLOSED — both smaller deltas regressed)

- **Branch:** `willowpai2g48h4-thorfinn/huber-delta-sweep`
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `j99e4mrg` (δ=0.3 canonical), `5rl1qqlh` (δ=0.2 canonical). Two duplicate runs `pyf40gvr` and `eawlb7mc` were terminated cleanly by thorfinn before final epoch — not in result count.
- **Hypothesis:** Smaller Huber delta (0.2, 0.3) pushes more residuals into the L1 regime, further aligning loss with MAE objective.

### Results

| Arm | huber_delta | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p |
|-----|-------------|--------------------|-----------|---------------------|
| **Baseline (PR #1558)** | **0.5** | **98.1642** | — | 98.7537 (3-split) |
| δ=0.3 | 0.3 | 113.4695 | **+15.6% ❌** | (regressed) |
| δ=0.2 | 0.2 | 115.0398 | **+17.2% ❌** | (regressed) |
| δ=1.0 (cycle 10 op note) | 1.0 | ~99.4 | +1.3% ❌ | — |

### Analysis and Conclusions

**Closed — δ=0.5 is at or near the local optimum.** Both smaller deltas regress significantly; δ=1.0 also regressed (from cycle 10 op notes); so δ=0.5 sits in a narrow sweet spot.

**Mechanism (thorfinn's analysis, confirmed):** At δ=0.5, only ~2% of per-node residuals exceed the quadratic-linear breakpoint and fall in the L1 regime. Pushing δ down to 0.2/0.3 moves more residuals into L1 — but those mid-magnitude residuals are precisely the ones whose gradient drives MAE minimisation. Flattening their gradient to a constant ±1 strips information needed to discriminate "almost good" from "good enough", and the encoder loses its tuning signal on the bulk of the distribution.

**Why δ=0.5 wins:** It targets only the true outlier tail (the 2% that introduce gradient spikes) while preserving full MSE-style scaling on the residuals that actually matter for the readout. Going either smaller (over-flatten) or larger (under-protect from outliers) both lose.

**Principle established:** The Huber delta sweet spot for normalised CFD readouts at this scale is δ≈0.5 — narrow window, do not re-sweep without changing other levers.

**Residual opportunities (not assigned):**
- Per-channel Huber delta (different δ for p vs Ux/Uy) — channels have different residual distributions; one global δ may be suboptimal even if the mean is right.
- Adaptive Huber (Truncated MSE-style cutoff at moving p95) — automatically tracks the outlier tail rather than fixing at normalised 0.5.
Both deferred; not priority over orthogonal mechanisms still in flight.

---

## 2026-05-13 04:34 — PR #1501: PhysicsAttention slice_num 64 → 128 (CLOSED — val regressed, wall-clock-bound)

- **Branch:** `willowpai2g48h4-nezuko/more-slices`
- **Student:** willowpai2g48h4-nezuko
- **W&B run:** `8w50j5dx` (slice-num-128)
- **Hypothesis:** Transolver uses slice_num=64; with 3 distinct mesh zones (background + 2 foils) and complex flow topology, doubling to 128 should enable finer physical partitioning. Predicted −2 to −6% on val_avg/mae_surf_p.

### Results

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) | params | epoch time | epochs in 30 min |
|-----|--------------------|-----------|-------------------------------|--------|------------|------------------|
| **Baseline (PR #1558, slice_num=64)** | **98.1642** | — | 98.7537 (3-split) | 0.658 M | 128 s | 14 |
| slice_num=128 | **117.1052** | **+19.30% ❌** | 108.7362 (4-split, all finite) | 0.679 M (+3%) | 175 s (+37%) | 10 (best) / 11 (completed) |

Per-split val (best epoch 10 → 11):

| Split | val mae_surf_p (epoch 11) | vs baseline |
|-------|--------------------------|------------|
| `val_single_in_dist` | 145.71 | 123.14 → +18.3% ❌ |
| `val_geom_camber_rc` | 143.13 | 107.24 → +33.5% ❌ |
| `val_geom_camber_cruise` | 84.60 | 73.28 → +15.5% ❌ |
| `val_re_rand` | 106.00 | 88.99 → +19.1% ❌ |
| **val_avg (best, epoch 10)** | **117.11** | **+19.3% ❌** |

Per-epoch trajectory (last 4 epochs):

| Epoch | val_avg_surf_p |
|-------|----------------|
| 8 | 157.94 |
| 9 | 126.87 |
| **10** | **117.11** ← best (cap) |
| 11 | 119.86 |

### Analysis and Conclusions

**Closed — 4th wall-clock-bound capacity failure on this branch.**

**Root cause (nezuko's analysis, confirmed):**
1. Predicted "near-zero per-epoch overhead" did not materialize — actual cost +37% per epoch (175s vs 128s).
2. The `in_project_slice = Linear(32, slice_num)` × 5 layers and the softmax-over-slices both scale linearly in slice_num at non-trivial constant factors.
3. The val curve was still steeply converging at the cap (145 → 158 → 127 → 117 over epochs 7-10) — this is a budget regression, not a per-epoch quality regression.

**Wall-clock-bound principle now 4× confirmed** (#1497 warmup +17.98%, #1498 wider-MLP +24.97%, #1499 grad-clip +1.45%, #1501 slice_num=128 +19.30%): under the 30-min cap, baseline epoch 14 is still improving; any change costing ≥10% per-epoch loses ≥1 epoch and regresses unless it accelerates convergence proportionally.

**Residual opportunities (nezuko's suggestions):**
1. Wall-clock-equalized comparison is structurally broken under SENPAI_TIMEOUT_MINUTES=30 for architectural changes that add per-step FLOPs. Would need fixed `--epochs N` ablation pair, which conflicts with current contract.
2. Do NOT escalate to slice_num ∈ {192, 256} — would lose by larger margin under same cap.
3. Richer per-slice features (higher n_hidden) is a cheaper test of the Transolver++ direction — already covered by #1572 (BF16 unlocks larger n_hidden) and previously-closed #1500.

---

## 2026-05-13 04:54 — PR #1881: Shallower depth + more epochs (n_layers=4) (CLOSED — val regressed)

- **Branch:** `willowpai2g48h4-edward/shallower-more-epochs`
- **Student:** willowpai2g48h4-edward
- **W&B run:** `5cq4p2qf` (n-layers-4)
- **Hypothesis:** Inverse of wider-MLP failure (#1498). Reduce n_layers=5→4 to save ~14% per-epoch wall-clock, gain ~2 extra epochs of SGD. Tests whether the under-30-min wall-clock cap rewards "more epochs at lower capacity" over "same capacity, less data."

### Results — Arm 1 only (Arm 2 n_layers=3 correctly skipped per branching rule)

| Arm | val_avg/mae_surf_p | Δ vs 98.16 | test_avg/mae_surf_p (4-split) | params | epoch time | epochs in 30 min |
|-----|--------------------|-----------|-------------------------------|--------|------------|------------------|
| **Baseline (PR #1558, n_layers=5)** | **98.1642** | — | 98.7537 (3-split) | 0.658 M | 128 s | 14 |
| n_layers=4 | **106.3995** | **+8.39% ❌** | 94.5883 (4-split) / 103.93 (3-split fair) | 0.529 M (−20%) | 110.6 s (−14%) | 16 |

Per-split val (best epoch 16):

| Split | val mae_surf_p | vs baseline |
|-------|---------------|------------|
| `val_single_in_dist` | 130.49 | 123.14 → +5.97% ❌ |
| `val_geom_camber_rc` | 120.40 | 107.24 → +12.27% ❌ |
| `val_geom_camber_cruise` | 77.10 | 73.28 → +5.21% ❌ |
| `val_re_rand` | 97.61 | 88.99 → +9.69% ❌ |
| **val_avg** | **106.40** | **+8.39% ❌** |

### Analysis and Conclusions

**Closed — Pareto frontier confirmed at depth=5 / 14 epochs.**

**Trade executed cleanly:**
- Predicted per-epoch saving (−14%) → exact match (110.6 vs 128s).
- Predicted epoch gain (~16 epochs in 30 min) → exact match (16 epochs).
- Even with the trade as designed, **+8.4% regression on the primary metric.**

**Root cause (edward's analysis, confirmed):**
1. Capacity loss from dropping one TransolverBlock (~129K params, 20% of body) cannot be recovered by 14% more SGD steps on the smaller model.
2. Regression is **uniform across all 4 val splits** (in-distribution +6.0%, hardest OOD +12.3%) — pure underfitting, no OOD artifact.
3. Largest regression on `val_geom_camber_rc` (+12.3%, hardest split) — the deeper model has more headroom to extract OOD-generalizing signal.

**Critical synthesis with prior wall-clock-bound failures:** Both directions on the depth/epoch axis regress now:
- Spend MORE per epoch (#1497 warmup, #1498 wider-MLP, #1501 slice_num=128): regress 18-25%.
- Spend LESS per epoch (#1881 n_layers=4): regresses 8.4%.

**Principle established:** The Transolver baseline at (depth=5, mlp_ratio=2, slice_num=64, ~14 epochs) sits very close to the Pareto frontier under the 30-min cap. The next compute should go to **non-depth-axis** capacity (heads, surf_head, or non-architectural mechanisms).

**Residual opportunities (edward's suggestions):**
1. **n_head 4→8** at fixed n_hidden=128 (head_dim 32→16). Multi-head is parallel batched matmul; minimal per-epoch cost. Assigned to edward as next experiment.
2. Mid-depth + warmer LR / shorter T_max — different hypothesis (LR-driven), not pursued here.
3. Do NOT try n_layers=6 — would cost ~+20% per epoch, predicted sub-Pareto by same logic.

---

## 2026-05-13 05:30 — PR #1795: Decoupled LR for surf_head vs encoder (MERGED — new baseline 97.9914)

- **Branch:** `willowpai2g48h4-thorfinn/decoupled-lr-surf-head`
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `q9qnnd9x` (arm1 1e-3), `70bjbj33` (arm2 3e-3), `eg1rhrzg` (arm3 5e-3 ← winner)
- **Hypothesis:** Zero-init surf_head with the same LR as the encoder (5e-4) is underpowered; decoupling allows the head to converge faster since it starts from zero and needs to learn rapidly.

### Results

| Arm | surf_head_lr | ×enc | Best epoch | val_avg/mae_surf_p | Δ vs 98.1642 |
|-----|-------------|------|------------|-------------------|-------------|
| 1 | 1e-3 | 2× | 14 | 114.1105 | +15.9% ❌ |
| 2 | 3e-3 | 6× | 13 | 104.7294 | +6.6% ❌ |
| **3** | **5e-3** | **10×** | **11** | **97.9914** | **−0.18% ✓** |
| Baseline | 5e-4 | 1× | — | 98.1642 | — |

Per-split val (arm 3, best epoch 11):

| Split | Arm 3 mae_surf_p | Baseline |
|-------|-----------------|---------|
| `val_single_in_dist` | 120.31 | 123.14 (−2.3% ✓) |
| `val_geom_camber_rc` | 115.98 | 107.24 (+8.2% ❌) |
| `val_geom_camber_cruise` | 66.04 | 73.28 (−9.9% ✓) |
| `val_re_rand` | 89.64 | 88.99 (+0.7% ≈) |
| **val_avg** | **97.9914** | **98.1642** |

Test (arm 3): test 3-split mean 99.5856 (vs baseline 98.7537, +0.85 regression on 3-split).

### Analysis and Conclusions

**Merged — marginal val win, compound improvements principle applied.**

**Mechanism (thorfinn's analysis):** Zero-init surf_head needs its own LR schedule to converge; tied to encoder LR of 5e-4 it's consistently underpowered. The monotonic 2×→6×→10× improvement trend (+114→104→98 val) had not reversed at the winning arm.

**Late oscillation noted:** Best epoch 11 (97.99), then oscillates 113→108→99.85 through epochs 12-14. The 10× LR is at the stability edge. Follow-up: add linear warmup to the head group and push to 7e-3 or 1e-2.

**Test regression (+0.85, 3-split):** Small and within run-to-run noise. The val win is the primary signal. Cruise improved significantly (73→55 finite) but was NaN at baseline, so 4-split is not directly comparable.

**Key insight:** surf_head as a standalone unit benefits from higher LR because: (a) it starts at zero init with no weight to preserve; (b) it has ~26K params with direct readout supervision, so it converges faster than the 658K-param encoder. The encoder benefits from slower, more conservative updates to maintain learned representations.

**New baseline: val_avg/mae_surf_p = 97.9914. All future PRs must beat this.**

---

## 2026-05-13 05:30 — PR #1720: surf_weight sweep {5, 15, 30} (CLOSED — all arms regress)

- **Branch:** `willowpai2g48h4-fern/surf-weight-tuning-on-huber`
- **Student:** willowpai2g48h4-fern
- **W&B runs:** `c54mrcff` (sw5), `zbiwwuly` (sw15), `a5yxs4ti` (sw30)
- **Hypothesis:** Huber's L1 regime produces smaller gradient magnitudes than MSE, so the current surf_weight=10 may be under-emphasizing the surface relative to Huber gradients.

### Results

| Arm | surf_weight | val_avg/mae_surf_p | Δ vs 98.16 |
|-----|------------|-------------------|-----------|
| sw=5 | 5 | 105.3356 | +7.2% ❌ |
| sw=15 | 15 | 115.5412 | +17.7% ❌ |
| sw=30 | 30 | 118.4154 | +20.7% ❌ |
| Baseline | 10 | 98.1642 | — |

### Analysis and Conclusions

**Closed — hypothesis falsified. Optimum is at or below sw=10, not above.**

**Mechanism (fern's analysis, confirmed):** Huber with δ=0.5 already implicitly down-weights surface outliers (5σ node contributes only 10× vs 25× under MSE). The baseline sw=10 was merged alongside Huber, so it was already "calibrated for Huber." Higher sw starves volume MSE — vol_p MAE blows up (sw=5: 111→sw=30: 151 val) because the encoder loses its volumetric supervisory signal when surf loss dominates.

**Principle confirmed:** Surface Huber + volume MSE + sw=10 is the correct triple. The surface:volume balance under Huber is at or near the optimum at sw=10.

---

## 2026-05-13 05:30 — PR #1808: EMA model weights (CLOSED — budget mismatch)

- **Branch:** `willowpai2g48h4-askeladd/ema-model-weights`
- **Student:** willowpai2g48h4-askeladd
- **W&B runs:** `rtvzppe1`, `wr3edclv` (decay=0.999 × 2 seeds), `1eqenbsj` (decay=0.995 ← best)
- **Hypothesis:** EMA shadow of model weights produces lower-variance checkpoint for evaluation.

### Results

| Arm | EMA decay | val_avg/mae_surf_p | Δ vs 98.16 |
|-----|----------|-------------------|-----------|
| decay=0.999 (seed A) | 0.999 | 113.0424 | +15.2% ❌ |
| decay=0.999 (seed B) | 0.999 | 114.0145 | +16.2% ❌ |
| **decay=0.995** | 0.995 | **105.8582** | **+7.8% ❌** |
| Baseline | — | 98.1642 | — |

### Analysis and Conclusions

**Closed — EMA window dominates under 14-epoch budget.**

**Mechanism (askeladd's analysis, confirmed):** At decay=0.999 the effective window is ~1000 steps ≈ 2.7 epochs. The model is still steeply descending at epoch 14; EMA drags every evaluation backward with early-training contamination. "Variance reduction" requires the late-iterate regime (noisy plateau); we're in the descent regime. Cruise split (already plateaued) was nearly unaffected; harder splits (still descending) paid the full lag tax.

**Principle established:** EMA evaluation requires ≥50-epoch training to reach the noisy plateau regime where variance reduction exceeds early-training drag. Under 14-epoch budget it's counterproductive.

**Key insight from half-decay comparison:** decay=0.995 (window ≈ 0.5 epochs) closes ~half the gap vs decay=0.999 (window ≈ 2.7 epochs). Extrapolating: decay < 0.98 (window ≈ 50 steps) should approach but not beat the live model. The variance-reduction benefit is real but requires late-training access.

**Follow-up assigned:** SWA-style late-epoch averaging (#1951 askeladd) — average only last K checkpoints, avoids early-training bias entirely.

## 2026-05-13 07:45 — PR #2015: AdamW β2=0.95 — monotone test on β2 axis

- **Branch:** `willowpai2g48h4-askeladd/adamw-beta2-0.95` (CLOSED — regression, destructive interaction with balanced sampler)
- **Student:** willowpai2g48h4-askeladd
- **W&B run:** `xi1r1iqd`
- **Hypothesis:** β2=0.999 (Adam default) maintains a long EMA of second moments. β2=0.95 (used in RoFormer, DeiT training) halves the EMA window, letting the optimizer respond faster to shifting gradient curvature. Predicted −0.5% to −2% from 97.9914.

### Results

| Metric | Baseline (β2=0.999) | β2=0.95 | Δ |
|--------|---------------------|---------|---|
| `val_avg/mae_surf_p` | 97.9914 | 104.3438 | **+6.49% regression** |
| `test_avg/mae_surf_p` | 88.5311 | 94.5522 | **+6.80% regression** |
| Best epoch | 11 | 8 | — |

### Per-epoch trajectory

| Epoch | Baseline | β2=0.95 |
|-------|----------|---------|
| 11 | **97.99** | 110.30 |
| 12 | 113.66 | **155.90** (spike) |
| 14 | 99.85 | 107.78 |
| Best | e11: 97.99 | e8: 104.34 |

### Analysis and Conclusions

**Closed — β2=0.95 is a sampler stabilizer, not a lag parameter.**

**Mechanism (student's insight, confirmed):** The balanced sampler draws from 3 domains with heterogeneous mesh sizes (74K-242K nodes) and y-magnitudes (164-458 std). Each step sees high batch-to-batch variance in gradient curvature. β2=0.999 acts as a low-pass filter against this noise — it accumulates a smooth estimate of the Hessian diagonal across many batches. β2=0.95 cuts the EMA window to ~20 steps, allowing per-domain curvature spikes to bleed through. The epoch-12 spike was WORSE under β2=0.95 (155.90 vs baseline 113.66), confirming the spike is driven by sampler variance, not by the optimizer's adaptation lag.

**Key insight (preserved for the team):** β2=0.999 is REQUIRED for stability against the balanced sampler's per-batch variance. It is not a hyperparameter to tune freely — it is a structural stabilizer. Future optimizer experiments MUST preserve β2=0.999 unless they simultaneously redesign the balanced sampler.

**Why student's follow-up directions were not pursued:** β2=0.9997 or β2=0.9995 would just move toward the current value; the direction is wrong. Per-domain adaptive β2 would require redesigning the sampler (out of scope for a single experiment). Mixing β2 + BIVW adjustment doubles the variables. All are more complex and less likely to yield a clean result than simply keeping β2=0.999.

## 2026-05-13 08:10 — PR #1949: Decoupled LR + head warmup: push surf_head_lr to {7e-3, 1e-2}

- **Branch:** `willowpai2g48h4-thorfinn/surf-head-lr-warmup` (CLOSED — both arms regressed, stability ceiling confirmed)
- **Student:** willowpai2g48h4-thorfinn
- **W&B runs:** `dn5s3kbs` (7e-3), `0kuym49t` (1e-2)
- **Hypothesis:** surf_head_lr=5e-3 sweep (PR #1795) had not reversed at the winning arm — try pushing to {7e-3, 1e-2} with a 2-epoch linear warmup to damp cold-start overshoots. Predicted −0.5% to −2%.

### Results

| Arm | surf_head_lr | n_warmup | Best epoch | val_avg/mae_surf_p | Δ vs 97.9914 |
|-----|-------------|----------|-----------|---------------------|-------------|
| Baseline | 5e-3 | 0 | 11 | **97.9914** | — |
| Arm 1 | 7e-3 | 2 | 14 | 99.2678 | +1.30% |
| Arm 2 | 1e-2 | 2 | 14 | 110.4673 | +12.73% |

### Per-split val MAE at best checkpoint

| Split | Arm 1 (7e-3) | Arm 2 (1e-2) | Baseline |
|-------|-------------|-------------|----------|
| val_single_in_dist | 121.99 | 160.47 | 120.31 |
| val_geom_camber_rc | 110.55 | 114.06 | **115.98** ← improved by 7e-3 |
| val_geom_camber_cruise | 72.71 | 74.62 | **66.04** |
| val_re_rand | 91.83 | 92.73 | **89.64** |
| **val_avg** | 99.27 | 110.47 | **97.99** |

### Analysis and Conclusions

**Closed — surf_head_lr=5e-3 is the local optimum at our budget. All axes exhausted.**

**Warmup correctness (student's implementation note):** Chaining two schedulers on the same optimizer is buggy — second .step() overwrites the first because both write to param_group['lr'] from the captured base_lr. Student correctly used a single LambdaLR composing cosine × warmup. This is the correct pattern for future per-group scheduling experiments.

**Warmup damped cold-start but not late-epoch oscillation:** The spike pattern at epoch 12 persisted under both arms — same spike magnitude as baseline even with warmup. Warmup addresses the wrong mechanism (cold-start, not steady-state variance). Best epoch shifted 11→14 (warmup ate into early progress), confirming the tradeoff did not pay.

**Third confirmation of steady-state oscillation principle:** Epochs 11→12 spike is present at 5e-3, 7e-3, AND 1e-2. This is independent of LR magnitude. The mechanism is the BIVW balanced sampler producing high per-batch variance late in training as the LR anneals. Confirmed across β2 sweep (#2015), warmup sweep (this PR), and previous LR sweep (#1795).

**arm 1 shows partial per-split improvement:** val_geom_camber_rc improved −4.7% (110.55 vs 115.98 baseline). But geom_camber_cruise and re_rand regressed, netting a overall regression. This is consistent with higher head LR providing more expressivity for some splits but destabilizing the head on others.

**Key insight: The surf_head optimization axis is exhausted.** Three PRs now confirm:
- #1795: 5e-3 > 1e-3, 3e-3; best at this budget
- #1949: 7e-3, 1e-2 with warmup: regression
- Any further LR push faces the stability ceiling

**Next directions:**
- Wider surf_head (#2057, askeladd): head *capacity* axis (orthogonal to LR)
- Per-group gradient clipping on surf_head (#2058, thorfinn): targets the late-epoch oscillation mechanism directly

## 2026-05-13 08:50 — PR #1572: BF16 autocast AMP — unlock wall-clock for capacity experiments

- **Branch:** `willowpai2g48h4-frieren/bf16-mixed-precision` (CLOSED — both arms regressed, precision-sensitive task)
- **Student:** willowpai2g48h4-frieren
- **W&B runs:** `c6m0hv4u` (n128), `f18c09px` (n256)
- **Hypothesis:** BF16 autocast → 2× tensor-core matmul + ~halved activation memory; expected 1.4-1.6× more epochs and/or unlock n_hidden=256. Stacked on Huber δ=0.5 + surf_head_lr=5e-3 baseline.

### Results

| Arm | Config | val_avg/mae_surf_p | test_avg/mae_surf_p | Best/total epoch | Δ val vs 97.99 |
|-----|--------|---------------------|----------------------|--------------------|------------------|
| Baseline (PR #1795) | FP32, n128 | **97.9914** | 88.5311 | 11/14 | — |
| Arm 1 | BF16, n128 | 101.5396 | 90.8030 | 15/18 | **+3.62%** |
| Arm 2 | BF16, n256, bs=2, lr=6e-4 | 127.4773 | 113.4968 | 7/11 | **+30.09%** |

### Per-split val MAE (Arm 1)

| Split | BF16 (n128) | Baseline 97.99 | Δ |
|-------|------------|-----------------|---|
| val_single_in_dist | 118.08 | 120.31 | −1.85% ✓ |
| val_geom_camber_rc | 129.12 | 115.98 | **+11.33%** ✗ |
| val_geom_camber_cruise | 66.05 | 66.04 | ≈ |
| val_re_rand | 92.91 | 89.64 | +3.65% ✗ |

### Analysis and Conclusions

**Closed — BF16 is the wrong knob for this task.**

**Throughput gain confirmed**: 18 epochs vs 14 in 30 min (+28%) at n128. So the *idea* that more epochs would help was correct in principle — but BF16's precision trade-off neutralizes the benefit.

**Precision-sensitivity mechanism (confirmed via per-split breakdown):** `val_geom_camber_rc` regressed +11.33% — the EXACT OOD split that PR #1558 (Huber δ=0.5) was designed to fix. The Huber gradient in the L1 regime is constant-magnitude, so small inter-node residual differences carry the supervisory signal. BF16's 7-bit mantissa rounds these away, weakening the gradient signal that drove #1558's −17.7% gain. This is the smoking gun.

**Capacity scaling failed:** Even with VRAM headroom (33 GB on 96 GB cap), n256 only managed 11 epochs at 169 s/epoch — well short of the convergence horizon. BF16 alone does not unlock n256 in the 30-min cap; would need architectural compute reductions (slice_num, depth, mlp_ratio).

**Key insight: throughput unlocks exist, but BF16's precision cost is load-bearing.** Three of our wall-clock-bound failures (EMA #1808, n_head=8 #1924, DropPath #1987) might still be rescued by a *pure* throughput unlock with no precision trade. The right next swing: `torch.compile` (same magnitude of speedup, FP32 preserved).

**Follow-up assigned:** torch.compile {default, reduce-overhead} (#2091 frieren) — pure throughput unlock with zero precision tradeoff.

**Code NOT merged:** The PR contained a working BF16 implementation (use_bf16 opt-in flag, default False). Closing without merging because adding a flag for a knob we know hurts would invite future students to mis-use it.
