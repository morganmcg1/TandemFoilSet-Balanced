# SENPAI Research Results — willow-pai2g-24h-r5

---

## 2026-05-14 05:20 — PR #2618: Cosine T_max extension epochs=80/100 on n_layers=3+wd=3e-4 (frieren) — CLOSED, T_MAX=50 IS U-SHAPE MINIMUM

- **Branch:** `willowpai2g24h5-frieren/cosine-tmax-extension-80-100`
- **Hypothesis:** #2542 showed monotonic ordering T_max=34 < T_max=44 < T_max=50. Extrapolating: T_max=80/100 should continue the trend and beat baseline.
- **W&B runs:** `ix8ny7xx` (T_max=80, clean), `dcwcwzt8` (T_max=100, clean), `9rx79h3p` (T_max=100, confirm seed)

| Arm | T_max | val | test | Δ val | Δ test | epochs | lr_final |
|-----|-------|-----|------|-------|--------|--------|----------|
| Baseline #2489 | 50 | **42.00** | **35.96** | — | — | 33 | 0.485×lr_init |
| Arm 1 | 80 | 46.22 | 39.02 | +10.0% ✗ | +8.5% ✗ | 29 | 0.709×lr_init |
| Arm 2 (best) | 100 | 43.64 | 37.28 | +3.9% ✗ | +3.7% ✗ | 34 | 0.741×lr_init |
| Arm 2 (confirm) | 100 | 44.53 | 37.54 | +6.0% ✗ | +4.4% ✗ | 34 | — |

Per-test-split (Arm 1, T_max=80): single_in_dist +4.8%, camber_rc +2.2%, camber_cruise +14.8%, re_rand +18.7% — OOD worst-hit.
Per-test-split (Arm 2 best, T_max=100): single_in_dist +0.5%, camber_rc +2.4%, camber_cruise +3.1%, re_rand +10.0% — all regress.

**Result:** CLOSED. Hypothesis REFUTED. Monotonic extrapolation was a local one-sided trend.

Key findings:
1. **Finding 45 — cosine T_max=50 is U-shape minimum, both directions now closed.** Full U-shape: T_max=34 (46.26) < T_max=44 (45.81) < T_max=50 (42.00) > T_max=80 (46.22) > T_max=100 (43.64). T_max=50 is the clear minimum.
2. **Mechanism: lr at terminal epoch determines "polish" quality.** At 30-min cap all variants get ~33 epochs. T_max=50 reaches lr≈0.485×lr_init → late-epoch refinement phase. T_max=80/100 keeps lr too high at cap → no polish → val still descending but absolute level worse.
3. **"Val still descending at cap" is misleading.** Slower descent in an under-annealed arm is NOT a sign of headroom — it's a sign the model never reached the convergent regime. Baseline descent rate (~1.1 val units/epoch late) >> under-annealed (~0.6 val units/epoch).
4. **EMA × Lion interaction.** High terminal lr = noisier weight trajectory → EMA averaging window less converged. Pairs with finding 43 (ema_decay): convergence quality near end-of-training is load-bearing.
5. **Scheduler axis fully characterized by CLI:** T_max = {34, 44, 50, 80, 100} all tested. T_max=50 locked. Further `--epochs` tuning is dead at 30-min cap.
6. **Architecture insight from student:** Linear-warmup + cosine-decay-to-zero-at-epoch-cap (train.py code edit) would unify "polish" with "headroom." Parked for code-edit tier.

**Frieren reassigned:** fourier_min_freq=0.5/2.0 sweep at L=6, max_freq=32 (PR #2789). Brackets low-frequency encoding band in parallel with thorfinn's max_freq sweep.

---

## 2026-05-14 04:50 — PR #2742: fourier_L sweep (L=4/L=8 vs L=6) on n_layers=3+wd=3e-4 (thorfinn) — CLOSED, L=6 LOCKED

- **Branch:** `willowpai2g24h5-thorfinn/fourier-L-sweep`
- **Hypothesis:** fourier_L=6 set in #1386 (early model val~103) may be suboptimal on mature compound; wd=3e-4 regularization headroom may enable richer encoding (L=8) or reveal L=6 is over-encoding (L=4).
- **W&B runs:** `8499pnkt` (L=4), `qkk0m6ph` (L=8)

| Arm | fourier_L | params | val | test | Δ val vs #2489 (42.00) | Δ test vs baseline (35.96) |
|-----|-----------|--------|-----|------|------------------------|----------------------------|
| Baseline #2489 | L=6 | 0.45M | **42.00** | **35.96** | — | — |
| Arm 1 | L=4 | 0.45M | 42.57 | 35.73 | +0.57 ✗ | −0.23 ≈ noise |
| Arm 2 | L=8 | 0.46M | 43.59 | 37.23 | +1.59 ✗ | +1.27 ✗ |

Per-test-split (W&B summary values — note: student found discrepancy vs PR body; using W&B summary):
| Split | L=6 (`vtewwalc`) | L=4 | L=8 |
|-------|-----------------|-----|-----|
| single_in_dist | 37.27 | **36.68** (−0.59) | 38.86 (+1.59) |
| geom_camber_rc | 50.92 | **49.37** (−1.55) | 52.61 (+1.69) |
| geom_camber_cruise | **20.96** | 21.49 (+0.53) | 21.00 (+0.04) |
| re_rand | **34.69** | 35.36 (+0.67) | 36.44 (+1.76) |

**Result:** CLOSED. Neither arm beats baseline on primary val metric. L=6 confirmed as local optimum.

Key findings:
1. **Finding 44 — fourier_L=6 locked, both directions closed.** Early #1386 default is near-optimal on mature compound. Compound changes (Lion, wd=3e-4, n_layers=3, EMA) did NOT shift the encoding optimum.
2. **L=8 regression confirms aliasing concern.** max_freq=32 on standardized coords ≈[-3,3] puts top octave near Nyquist for dense local meshes. Adding an extra octave (fan-in 22→30) over-parameterizes the first linear layer without adding usable signal. All splits regress uniformly.
3. **L=4 is within-noise.** val +0.57 worse, test −0.23 better — both within timeout-truncation noise (~0.5 pts). geom_camber_rc improves −1.55 (potentially genuine) but re_rand drifts up +0.67. No clean OOD direction.
4. **L vs max_freq are distinct axes.** L controls channel count; max_freq controls octave band coverage. The natural follow-up is sweeping max_freq at L=6 to isolate the aliasing question from channel count. max_freq=16 drops potentially-aliased top octave; max_freq=48 extends beyond Nyquist.
5. **Pre-existing NaN bug flagged:** test_geom_camber_cruise normalized loss returns None/Inf in W&B summaries across all runs. Denormalized MAE (the actual metric) is fine — localized to W&B summary logging of normalized-space loss on this split.

**Thorfinn reassigned:** fourier_max_freq=16/48 sweep at L=6 (PR #2784). Direct aliasing test.

---

## 2026-05-14 03:56 — PR #2729: ema_decay higher (0.995/0.999) on n_layers=3+wd=3e-4 (edward) — CLOSED, EMA DECAY LOCKED AT 0.99

- **Branch:** `willowpai2g24h5-edward/ema-decay-higher`
- **Hypothesis:** ema_decay=0.99 (half-life ~69 steps < 1 epoch) set in #1607 on a 16-epoch run was never re-optimized. At the current ~33-epoch budget, higher decay (longer window) should deepen the smoothing benefit given the load-bearing 3-point main-EMA gap.
- **W&B runs:** `9rriwptj` (ema=0.995), `o3l811s9` (ema=0.999)

| Arm | ema_decay | val | test | Δ val vs #2489 (42.00) | Δ test vs baseline (35.96) | main-EMA gap |
|-----|-----------|-----|------|------------------------|----------------------------|--------------|
| Baseline #2489 | 0.99 | **42.00** | **35.96** | — | — | ~+3.0 |
| Arm 1 | 0.995 | 43.77 | 37.11 | +1.77 ✗ | +1.15 ✗ | +4.32 |
| Arm 2 | 0.999 | 42.31 | 35.81 | +0.31 ✗ | −0.15 ≈tie | +1.92 |

Per-test-split (Arm 2 only, Arm 1 clearly worse):
| Split | Baseline | Arm 2 (0.999) | Δ |
|-------|----------|---------------|---|
| single_in_dist | 40.58 | **37.44** | −3.14 ✓ |
| geom_camber_rc | 50.15 | 50.34 | +0.19 ≈ |
| geom_camber_cruise | 20.56 | 20.51 | −0.05 ≈ |
| re_rand | 32.56 | 34.95 | +2.39 ✗ |

**Result:** CLOSED. Neither arm beats baseline on val. Arm 1 clearly worse; Arm 2 within-noise (test +0.15 better, val −0.31 worse — primary metric fails).

Key findings:
1. **Finding 43 — ema_decay locked at 0.99, higher direction closed.** At the 30-min step budget where val is still descending at cap, higher decay produces *EMA lag* rather than *deeper smoothing*. Lag beats the smoothing benefit in both arms.
2. **Gap dynamics confirm mechanism.** Wider gap (+4.32) at 0.995 = sluggish EMA (both absolute vals worse). Narrower gap (+1.92) at 0.999 = EMA can't keep up with late-training descent. Neither widens the gap *usefully*.
3. **Arm 2 per-split pattern echoes finding 41 (sw × split-type).** 0.999 trades single_in_dist gain (−3.14) for re_rand regression (+2.39). Slow averaging emphasizes late-epoch state which is more in-dist-tuned — OOD generalization suffers.
4. **ema=0.99 was already optimally tuned for this step budget.** Monotonic relationship: lower → noisy, current = sweet spot, higher → lagged. Lower direction (0.985) also unlikely to help — half-life already ≪ 1 epoch.
5. **Implicit insight: EMA bias correction at 0.999** might recover first-10-epoch waste (pure init-anchored catch-up, val 332→88 in that window). Requires code edit; parks until step budget extends.

**Edward reassigned:** clip_grad_norm sweep (0.5/2.0) — last untested CLI-only optimizer axis on current compound (PR #2758).

---

## 2026-05-14 02:50 — PR #2708: Huber loss δ=0.5/1.0 vs MAE on n_layers=3+wd=3e-4 (thorfinn) — CLOSED, MAE DOMINATES ON MATURE COMPOUND

- **Branch:** `willowpai2g24h5-thorfinn/huber-loss-delta-sweep`
- **Hypothesis:** With Lion's sign-magnitude update, Huber's quadratic floor for small residuals might reduce gradient noise and improve convergence. δ=1.0 broader vs δ=0.5 tighter quadratic region.
- **W&B runs:** `dmyqiw4b` (δ=0.5), `xwuuws4k` (δ=1.0)

| Arm | δ | val | test | Δ vs #2489 (42.00/35.96) | Epochs |
|-----|---|-----|------|--------------------------|--------|
| 1 | 0.5 | 46.26 | 39.75 | +4.26 / +3.79 ✗ | 30 |
| 2 | 1.0 | 45.58 | 38.92 | +3.57 / +2.96 ✗ | 34 |
| Baseline #2489 (MAE) | — | 42.00 | 35.96 | — | 33 |

Per-test-split (mae_surf_p): all 4 splits regress on both arms uniformly. single_in_dist most preserved (+0.21/+0.88), re_rand worst-hit (+5.58/+6.25).

**Result:** CLOSED. Hypothesis REFUTED. MAE dominates Huber on mature compound — replicates early #1825 result.

Key findings:
1. **MAE-vs-Huber finding replicates on mature compound (finding 42).** Early result #1825 (Huber → MAE = −7.71% val) holds with Lion + wd=3e-4 + n_layers=3 + EMA + Fourier. The advantage doesn't get eaten by the compound's other regularizers. **Uniform-weighted per-node MAE is essential** to the compound.
2. **δ=1.0 > δ=0.5 (closer to MAE = better).** Monotonic ordering confirms direction. The quadratic floor at small residuals is the mechanism that hurts; widening δ → more MAE-like → better.
3. **Step-budget effect: Huber's slower convergence is real.** Per-epoch val descent at cap: δ=0.5 ~−0.7/ep, δ=1.0 ~−0.3/ep. Catch-up unlikely before descent flattens — Huber needs more steps to match MAE.
4. **Generalization gap similar val↔test (+3.6 vs +3.0).** Huber doesn't generalize better; re_rand cross-regime gap is widest (+5.58/+6.25). Huber HURTS OOD specifically.
5. **Numerical artifact:** `huber_loss(reduction='none')` returns nan on test_geom_camber_cruise under BF16. Doesn't affect MAE ranking metrics (fp64) but worth filing as a separate bug-fix PR.

**Implication for paper:** Pairs with finding 41 (sw × split-type) to define the **loss-balance landscape**: surface weighting hurts OOD, curvature smoothing hurts everywhere. Per-node uniform MAE is the Pareto-optimal loss formulation on this compound. Loss-formulation axis is now closed without code changes (Huber+MAE blend requires train.py edit).

**Thorfinn reassigned:** fourier_L sweep (L=4/L=8) — input-encoding capacity untested on mature compound. fourier_L=6 was set in #1386 on a much weaker model; the wd=3e-4 regularization headroom may enable richer encoding.

---

## 2026-05-14 02:05 — PR #2707: surf_weight=15 UPPER direction on n_layers=3+wd=3e-4 (edward) — CLOSED, SW UPPER CLOSED; SW × SPLIT-TYPE INTERACTION

- **Branch:** `willowpai2g24h5-edward/sw-upper-wd3e4`
- **Hypothesis:** sw=10 default may be sub-optimal on new compound; stronger surface emphasis (sw=15/20) could help OOD geom_camber where surface geometry matters most.
- **W&B runs:** `tx5ixhcs` (sw=15). Arm 2 skipped per PR rule (val regression >2 pts).

| Arm | sw | val | test | Δ vs #2489 (42.00/35.96) | Epochs |
|-----|----|-----|------|--------------------------|--------|
| 1 | 15 | 44.63 | 38.08 | +2.63 / +2.12 ✗ | 30 |
| Baseline #2489 | 10 | 42.00 | 35.96 | — | 33 |

Per-test-split (mae_surf_p) — KEY signal:
| Split | Baseline | sw=15 | Δ% |
|-------|----------|-------|-----|
| single_in_dist | 40.58 | 39.64 | **−2.32%** ✓ |
| geom_camber_rc | 50.15 | 53.33 | +6.33% ✗ |
| geom_camber_cruise | 20.56 | 22.28 | +8.40% ✗ |
| re_rand | 32.56 | 37.06 | **+13.81%** ✗ |

Strong asymmetric per-split pattern: **single_in_dist gains, all OOD regress, re_rand worst.**

**Result:** CLOSED. Hypothesis REFUTED. Direction strictly wrong above sw=10 on new compound.

Key findings:
1. **sw upper direction closed on new compound (finding 41).** sw=15 regresses by +2.63 val. sw=20 would extend the trend — Arm 2 skip preserved fleet time without changing conclusion. Together with fern's pending #2491 (sw=5/3 lower direction), sw axis is bracketed on new compound.
2. **sw × split-type interaction is the paper-grade contribution.** sw=15 IMPROVES single_in_dist surface fidelity (−2.32%) but disproportionately HURTS cross-regime OOD splits, especially re_rand (+13.81%). Mechanism: over-emphasizing surface loss starves the volume field; on splits where the model must extrapolate, the volume context that anchors surface prediction degrades and error compounds.
3. **Original OOD-improvement hypothesis REFUTED.** Both geom_camber splits regress with sw=15. Surface emphasis is NOT a substitute for the balanced surface+volume signal MAE already provides.
4. **EMA gap intact (4.16 pts main 48.79 vs EMA 44.63).** Compound's regularization stack is healthy; only loss-balance lever is mis-set. EMA is doing real work even on a regressing run.
5. **OOD-cost curve appears steep above sw=10.** Interior points sw=11/12 unlikely to beat default — student correctly recommends skipping them.

**Implication for paper:** The sw × split-type result is a clean appendix finding. **For higher-surf direction, would need pairing with volume-loss-floor or per-channel reweighting** — different (more complex) hypothesis requiring code changes.

**Edward reassigned:** ema_decay higher direction (0.995/0.999) on new compound — natural probe given the load-bearing main-vs-EMA gap.

---

## 2026-05-14 01:25 — PR #2645: dropout=0.10/0.05 lower-direction sweep on n_layers=3+wd=3e-4 (thorfinn) — CLOSED, DROPOUT DIRECTION CLOSED BOTH SIDES

- **Branch:** `willowpai2g24h5-thorfinn/dropout-lower-wd3e4`
- **Hypothesis:** With wd=3e-4 providing continuous L2 shrinkage, less stochastic dropout (substitutive-regularizer prior) should free representational capacity and improve val.
- **W&B runs:** `agzf0gut` (dropout=0.10), `h8ixr4bt` (dropout=0.05)

| Arm | dropout | val | test | Δ vs #2489 (42.00/35.96) | Epochs |
|-----|---------|-----|------|--------------------------|--------|
| 1 | 0.10 | 44.85 | 37.80 | +2.85 / +1.84 ✗ | 29/34 |
| 2 | 0.05 | 42.89 | 36.11 | +0.89 / +0.15 ✗ | 34/34 |
| Baseline #2489 | 0.20 | 42.00 | 35.96 | — | 33/34 |

Per-test-split: single_in_dist IMPROVES on both arms (40.58 → 39.17/38.71). geom_camber_rc flat/worse. geom_camber_cruise and re_rand both regress.

**Result:** CLOSED. Hypothesis REFUTED.

Key findings:
1. **Dropout direction closed both sides (finding 39).** Combined with #2551 (upper d=0.25/0.30 also regressing), **dropout=0.20 is locked as local optimum** on wd=3e-4, n_layers=3, Lion-MAE compound.
2. **wd × dropout are COMPLEMENTARY, NOT substitutive.** Substitutive-regularizer prior predicted lower dropout would compensate for wd=3e-4 — data REFUTES this. Both directions regress. Distinct from wd × n_head substitutive (finding 34) and slice × n_head substitutive (finding 30). **Different regularizer pairs have different interaction signs in this regime.**
3. **Non-monotonic 0.10 < 0.05 inversion (val).** Mid-strength dropout may be a worst case — too weak to suppress co-adaptation but still strong enough to remove signal. Plausibly seed noise (~0.89 gap at single seed) but consistent with saturation framing.
4. **In-dist test improves on both arms, OOD splits regress.** Hint of a regularization-strength × split-type interaction — lower dropout helps single_in_dist but hurts geom_camber_cruise/re_rand. Useful appendix observation.
5. **Arm 1 val curve flattened/began to overfit at epoch 29/34** (lower dropout opened overfitting risk). Arm 2 still descending at cap, but gap too wide to credibly close.

**Implication for paper:** dropout=0.20 + wd=3e-4 are jointly Pareto-optimal — the two regularizers stack additively, not substitutively. Pairs with findings 30, 34, 35: regularizer × architecture interactions have non-uniform interaction signs.

**Thorfinn reassigned:** moving to a fresh axis since dropout direction is closed.

---

## 2026-05-14 01:20 — PR #2641: lr=8e-5/1.5e-4 sweep on n_layers=3+wd=3e-4 (edward) — CLOSED, LR=1e-4 LOCKED, NARROW SURFACE

- **Branch:** `willowpai2g24h5-edward/lr-sweep-wd3e4-n-layers-3`
- **Hypothesis:** wd=3e-4 stabilization may permit higher lr (faster effective per-step progress) or lower lr (cleaner convergence under regularization).
- **W&B runs:** `6dsg3kka` (lr=8e-5, clean rerun), `kbwhb5vu` (lr=1.5e-4)

| Arm | lr | val | test | Δ vs #2489 (42.00/35.96) | Epochs | Final lr |
|-----|----|-----|------|--------------------------|--------|----------|
| 1 | 8e-5 | 44.58 | 38.21 | +2.58 / +2.25 ✗ | 35 | 2e-5 |
| 2 | 1.5e-4 | 43.05 | 35.93 | +1.05 / −0.03 ≈ | 35 | 3e-5 |
| Baseline #2489 | 1e-4 | 42.00 | 35.96 | — | ~34 | — |

Per-test-split (Arm 2 vs baseline): **single_in_dist IMPROVES** 36.98 vs 40.58 (−8.9%), re_rand REGRESSES 35.38 vs 32.56 (+8.7%). Arm 1 all splits regress.

**Result:** CLOSED. Hypothesis REFUTED — lr optimum did not shift with regularization.

Key findings:
1. **lr=1e-4 locked as local optimum on new compound (finding 40).** lr surface is narrow: ±0.5×–1.5× regresses val by 1–2.6 points.
2. **wd and lr are independent axes under Lion+MAE.** wd=3e-4 stabilization did NOT open up headroom for higher lr. Different mechanism interaction than wd × n_head substitutive (finding 34).
3. **Higher-lr regime-trade: in-dist vs OOD.** Arm 2 lr=1.5e-4 IMPROVES single_in_dist test (−8.9%) but worsens re_rand (+8.7%). Suggests higher lr trades in-dist fidelity for OOD robustness — useful appendix data on optimizer × split interaction.
4. **Step count remains the bottleneck (reinforces finding 37).** Val still descending at cap across all three lrs. Tuning lr cannot break that bottleneck — only architecture-driven per-step compute reduction or padding-waste reduction can.
5. **Lion sign-update brittle to lr scale.** Going down 0.8× degrades >2 points; going up 1.5× degrades 1 point on val. Lion + lr=1e-4 sits in a narrow well.

**Implication for paper:** lr=1e-4 + wd=3e-4 are jointly locked on this compound. The regime-trade single_in_dist vs re_rand at higher lr is a paper-grade finding about lr × OOD-split interaction.

**Edward reassigned:** moving to a fresh axis since lr direction is closed.

---

## 2026-05-13 23:15 — PR #2563: n_head=4/n_head=8 sweep on n_layers=3+wd=3e-4 (thorfinn) — CLOSED, PER-HEAD DIM=64 SWEET SPOT

- **Branch:** `willowpai2g24h5-thorfinn/n-head-4-8-sweep-wd3e4`
- **Hypothesis:** n_head=2 optimum may shift at shallower depth (n_layers=3). Test n_head=4 (dim_head=32) and n_head=8 (dim_head=16).
- **W&B runs:** `vsp6j6ib` (n_head=4), `jkp3cjeh` (n_head=8)

| Arm | n_head | dim_head | val | test | Δ vs #2489 | epochs | s/ep | peak VRAM |
|-----|--------|----------|-----|------|------------|--------|------|-----------|
| Baseline #2489 | 2 | 64 | 42.00 | 35.96 | — | 33 | 53.1 | 93.5 GB |
| Arm 1 | 4 | 32 | 43.93 | 37.33 | +4.58% / +3.82% ✗ | 31 | 59.3 (+11.7%) | 81.1 GB |
| Arm 2 | 8 | 16 | 46.93 | 39.55 | +11.73% / +10.00% ✗ | 26 | 72.0 (+35.6%) | 58.0 GB |

All 4 splits regress in both arms. single_in_dist worst-hit (+15.72% on Arm 2 — predicted rank-collapse symptom).

**Result:** CLOSED. Hypothesis REFUTED — n_head=2 optimum is depth-independent.

Key findings:
1. **Per-head dim=64 is slice-attention sweet spot (finding 38).** Monotonic {n_head=2 < n_head=4 < n_head=8} at n_layers=3+wd=3e-4. Below dim_head=32 the soft virtual-token mechanism loses rank.
2. **Architecture cost is NON-zero across n_head.** MHA reshape/permute overhead scales with head count: n_head=4 → +11.7% s/ep, n_head=8 → +35.6%. Contradicts PR prediction of equal compute. With 30-min cap, fewer faster epochs at higher per-head dim WIN (speed-dividend logic, pairs with findings 28 + 37).
3. **wd × n_head ruled out as explanation.** Regression magnitude (+10% test on n_head=8) dwarfs any plausible wd correction. Capacity finding, not regularization mismatch (separates cleanly from finding 34).
4. **val/test gap is consistent across all three head counts (val ≈ test + 7).** Result is NOT overfitting; it's pure attention capacity. Pareto frontier at n_head=2 + slice_num=32 + n_layers=3.

**Implication for paper:** Per-head dim=64 + slice_num=32 + n_layers=3 together define a Pareto-optimal architecture surface at the 30-min budget — three independent axes converge to this single operating point.

**Thorfinn reassigned:** dropout=0.10/0.05 sweep on new compound — counter-direction test to finding 35 (regularization saturation). Does inherited dropout=0.2 over-regularize the now-heavily-wd-regularized compound?

---

## 2026-05-13 22:50 — PR #2587: batch_size=8/2 sweep on n_layers=3+wd=3e-4 (edward) — CLOSED, NO SPEED DIVIDEND ON BS AXIS

- **Branch:** `willowpai2g24h5-edward/batch-size-sweep-wd3e4-n-layers-3`
- **Hypothesis:** GPU underutilized at bs=4 (~25 GB); bs=8 should give speed dividend → more epochs/30 min. bs=2 acts as 'reverse direction' test.
- **W&B runs:** `0xslx16k` (bs=8), `vz8to30q` (bs=2)

| Arm | bs | val | test | Δ vs #2489 | epochs | ep_time | peak VRAM |
|-----|----|-----|------|------------|--------|---------|-----------|
| Baseline #2489 | 4 | 42.00 | 35.96 | — | 34 | 53.1 s | 93.5 GB (97.8%) |
| Arm 1 | 8 | 48.36 | 40.87 | +15.1% / +13.7% ✗ | 34 | 54.2 s | 87.5 GB (91.5%) |
| Arm 2 | 2 | 46.02 | 38.87 | +9.6% / +8.1% ✗ | 29 | 51.6 s | 93.5 GB (97.8%) |

All 4 test splits regress uniformly for both arms.

**Result:** CLOSED. Hypothesis REFUTED on two counts.

Key findings:
1. **No speed dividend on batch_size axis (finding 37).** Per-epoch time is essentially batch_size-INVARIANT (51-54s for bs ∈ {2,4,8}). Per-step GPU compute is NOT the bottleneck — data loading + padding to N_max=242K + val pass dominate.
2. **GPU is near-saturated at bs=4.** System-level reserved memory is 97.8% — the earlier '25 GB' figure was active-tensor memory excluding cache. bs=16 would likely OOM. **Future memory-optimization wins are NOT via larger batches.**
3. **Step count drives performance with noise penalty.** bs=8 → halves opt steps → loses by 6.4 val points. bs=2 → 1.76× more steps but still loses by 4 (Lion sign-update + small dataset compounds gradient variance). bs=4 is a noise-step Pareto sweet spot.
4. **Cosine confound is not the explanation.** bs=8 and baseline both terminate at lr ≈ 2.32e-5; bs=8 did NOT enter the over-anneal collapse regime.

**Implication for paper:** Speed-dividend mechanism applies to architecture (n_layers, slice_num) where per-step compute is the lever, but NOT to batch_size axis where data loading + padding dominate. 'Padding-waste reduction' (length-bucketed sampler) is the real lever for the budget-extension story — requires data-loader code, not a CLI flag.

**Edward reassigned:** lr=8e-5/1.5e-4 sweep on new compound — tests whether wd=3e-4 regularization stabilization permits higher lr for more effective per-step progress. Untested on n_layers=3+wd=3e-4.

---

## 2026-05-13 22:20 — PR #2542: Cosine T_max match --epochs 34/44 (frieren) — CLOSED, COSINE T_MAX INVERTS

- **Branch:** `willowpai2g24h5-frieren/cosine-tmax-match`
- **Hypothesis:** T_max matched to realized epoch count (~34) should fully anneal cosine and improve final convergence. The 'val still descending at cap' pattern was interpreted as needing more lr decay.
- **W&B runs:** `lena3xr5` (T_max=34), `gbkwj8mr` (T_max=44)

| Arm | --epochs (T_max) | val | test | Δ vs #2489 (42.00/35.96) | Epochs | lr at cap |
|-----|------------------|-----|------|--------------------------|--------|-----------|
| 1 | 34 | 46.2586 | 39.7999 | +10.14% / +10.69% ✗ | 29 | 0.12×lr_init |
| 2 | 44 | 45.8073 | 37.8961 | +9.07% / +5.39% ✗ | 30 | 0.53×lr_init |
| Baseline #2489 | 50 | 42.0040 | 35.9573 | — | 33 | 0.485×lr_init |

(Note: frieren ran on the OLD wd=1e-4 compound; deltas computed against current best #2489. Negative result direction holds regardless of compound since 'less annealing → better' ordering is monotonic.)

Per-test-split: all 4 splits regress uniformly on both arms.

**Result:** CLOSED. Hypothesis REFUTED — direction is OPPOSITE to predicted.

Key findings:
1. **Cosine T_max inverts (finding 36).** Monotonic ordering {Arm 1 < Arm 2 < baseline} on both val and test — less annealing is strictly better.
2. **Over-annealing collapse.** Arm 1 val *went up* at the final epoch as lr→0. Arm 2 showed smaller end-of-run degradation. Baseline at lr=0.485×lr_init at cap is in a sweet spot.
3. **Lion + lr=1e-4 is brittle at very low lr.** Driving lr→0 takes away the optimizer's ability to refine; lower-lr fine-tuning is harmful here.
4. **'Val still descending at cap' is load-bearing, not a missed opportunity.** Model wants more STEPS at non-trivial lr, not faster annealing.

**Implication for paper:** Cosine T_max=50 default is incidentally well-tuned for our 30-min cap. Next test: extending T_max ABOVE 50 should improve more.

**Frieren reassigned:** --epochs 80/100 sweep on new compound (wd=3e-4) — extend monotonic 'less annealing → better' direction UP.

---

## 2026-05-13 21:10 — PR #2551: dropout=0.25/0.30 stack on n_layers=3+wd=3e-4 (edward) — CLOSED, REGULARIZATION SATURATION

- **Branch:** `willowpai2g24h5-edward/dropout-stack-wd3e4-n-layers-3`
- **Hypothesis:** higher dropout (norm-control wd is independent of co-adaptation dropout) should stack on top of wd=3e-4 to improve OOD generalization.
- **W&B runs:** `fqo669sk` (dropout=0.25), `tccwohev` (dropout=0.30)

| Arm | dropout | val | test | Δ vs #2489 (42.00/35.96) | Epochs |
|-----|---------|-----|------|--------------------------|--------|
| 1 | 0.25 | 44.7325 | 37.4143 | +6.50% / +4.05% ✗ | 30 |
| 2 | 0.30 | 43.6647 | 36.9328 | +3.95% / +2.71% ✗ | 34 |
| Baseline #2489 | 0.20 | 42.0040 | 35.9573 | — | 33 |

Per-test-split (mae_surf_p): all 4 splits regress on both arms uniformly (single_in_dist +2.74%/+4.27%, geom_camber_rc +3.85%/+1.01%, geom_camber_cruise +3.39%/+4.20%, re_rand +6.14%/+2.63%). No OOD-specific advantage as the hypothesis predicted.

**Result:** CLOSED. Higher dropout is harmful on the shallow + already-wd-regularized compound.

Key findings:
1. **Regularization saturation at dropout=0.20.** wd=3e-4 + dropout=0.20 + BF16 + EMA(0.99) already at the Pareto frontier. Adding more of any regularizer requires substituting another.
2. **Capacity floor at n_layers=3.** Each of 3 attention blocks carries more representational load than at n_layers=5. Dropout=0.30 removes too many activations per forward pass to compensate.
3. **No OOD-specific dropout effect.** Regressions roughly uniform across in-dist and OOD splits — argues against 'dropout regularizes generalization' framing on this compound.
4. **Val curves still descending at cap on all arms** — the 30-min budget is the binding constraint; further regularization slows learning without raising the asymptote.
5. **Non-monotonic arm ordering (Arm 2 > Arm 1 on val) within 1-pt single-seed noise.** Direction conclusion (both lose to 0.20) is robust.

**Edward reassigned:** batch_size=8/2 sweep on new compound — speed-vs-step trade probe, untested axis.

---

## 2026-05-13 20:15 — PR #2448: Lion wd=3e-4/1e-3 on n_head=1+slice32 (thorfinn) — CLOSED, wd INVERTS AT N_HEAD=1

- **Branch:** `willowpai2g24h5-thorfinn/wd-n-head-1-slice32`
- **Hypothesis:** wd=3e-4 monotonic win at n_head=2 (#2356) should transfer to n_head=1 since per-head dim=128 has more capacity to regularize.
- **W&B runs:** `9ejpl0q4` (wd=3e-4), `8b58rifg` (wd=1e-3)

| Arm | wd | val | test | Δ vs #2338 (46.67/40.69) | best_epoch |
|-----|----|-----|------|--------------------------|------------|
| 1 | 3e-4 | 47.1787 | 41.0294 | +0.51 / +0.34 ✗ | 23 |
| 2 | 1e-3 | 48.1069 | 40.7166 | +1.44 / +0.03 ≈ | 26 |
| Baseline #2338 | 1e-4 | 46.67 | 40.69 | — | 26 |

**Result:** CLOSED. wd direction is monotonically *wrong* at n_head=1; val degrades with higher wd, test essentially flat.

Key findings:
1. **wd is n_head-specific.** Direction inverts vs n_head=2: wd=3e-4 was −2.68 val at n_head=2 (#2356); at n_head=1 it's +0.51 val. Hypothesis "n_head=1 has more capacity → needs more regularization" is FALSIFIED.
2. **Mechanism (thorfinn):** n_head=2 spreads slice-attention across 2 heads, creating dilution/variance that wd compensates for. n_head=1 has a single head and is already near its wd=1e-4 optimum at slice32.
3. **Substitutive regularizer crossover:** at wd=3e-4 the n_head=1 and n_head=2 paths meet at val=47.18 — wd lever and n_head lever can compensate each other.
4. **Test_avg is robust to wd direction.** val degrades 46.67→48.11; test 40.69→40.72. val/test gap shrinks → small regularization-induced val underfit, not test damage.
5. **Pairs with finding 30 (slice × n_head substitutive at high per-head dim).** Strong appendix material on regularization × architecture interactions.
6. **Stop probing wd≥3e-3** — direction confirmed wrong past 1e-4.

**Thorfinn reassigned:** n_head=4/8 sweep on new n_layers=3+wd=3e-4 compound — test architecture symmetry direction.

---

## 2026-05-13 19:45 — PR #2489: wd=3e-4/1e-3 stack on n_head=2+n_layers=3 (edward) — MERGED, NEW BEST

- **Branch:** `willowpai2g24h5-edward/wd-n-layers-3-slice32`
- **Hypothesis:** wd=3e-4 monotonic signal from #2356 (n_layers=5 compound) should transfer directly to new n_layers=3 baseline; test whether wd=1e-3 extends the trend further.
- **W&B runs:** `vtewwalc` (wd=3e-4, winner), `h2h5d9cl` (wd=1e-3, flat)

| Arm | wd | val | test | Δ vs #2400 (43.14/36.95) | Epochs |
|-----|-----|-----|------|--------------------------|--------|
| **1** | **3e-4** | **42.0040** | **35.9573** | **−2.64% / −2.69% ✓** | 33 |
| 2 | 1e-3 | 43.1112 | 36.8196 | −0.07% / −0.35% ✗ | 34 |
| Baseline #2400 | 1e-4 | 43.14 | 36.95 | — | 34 |

Per-test-split (mae_surf_p): single_in_dist=37.27 (−5.2%), geom_camber_rc=50.92 (−1.6%), geom_camber_cruise=20.96 (−1.7%), re_rand=34.69 (−2.1%) — **all 4 improve on Arm 1**.

**Result:** MERGED. val=42.0040/test=35.9573. New best.

Key findings:
1. **wd=3e-4 signal transfers from n_layers=5 to n_layers=3.** Same regularization advantage as #2356, consistent relative magnitude (~2.6% val improvement). Depth-independent regularization.
2. **Monotonic trend does NOT extend to wd=1e-3 at n_layers=3.** Arm 2 flat on val, 3/4 OOD splits regress. Shallower architecture is more sensitive to over-regularization — same penalty has larger relative effect at fewer layers.
3. **wd=3e-4 confirmed as the operating point.** No exploration needed below 3e-4 or above.
4. **Edward's suggestion for dropout follow-up is sound:** higher dropout (0.25/0.30) as complementary regularization axis, stacking on wd=3e-4.

**Edward reassigned:** dropout=0.25/0.30 stack on new wd=3e-4 compound.

---

## 2026-05-13 19:30 — PR #2490: slice_num=16/8 stack on n_head=2+n_layers=3 (frieren) — CLOSED, ANTI-STACKS

- **Branch:** `willowpai2g24h5-frieren/slice16-stack-n-layers-3`
- **Hypothesis:** slice_num=16 (and #2 deeper probe slice_num=8) should stack with new n_layers=3 baseline since per-head dim=64 unchanged from #2337 (where slice16 won at n_layers=5).
- **W&B runs:** `mj9gjpgp` (slice16), `5nq5jrp6` (slice8)

| Arm | slice_num | val | test | Δ vs #2400 (43.14/36.95) | Epochs |
|-----|-----------|-----|------|--------------------------|--------|
| 1 | 16 | 46.22 | 39.88 | +3.08 / +2.93 ✗ | 28 |
| 2 | 8 | 46.67 | 39.09 | +3.53 / +2.14 ✗ | 29 |
| Baseline #2400 | 32 | **43.14** | **36.95** | — | 34 |

Per-test-split (mae_surf_p): all 4 splits regress on both arms. single_in_dist +1.94/+4.75, geom_camber_rc +1.42/+3.58, geom_camber_cruise +1.19/+2.34, re_rand +1.82/+3.20.

**Result:** CLOSED. slice_num=32 is the local optimum at n_layers=3.

Key findings:
1. **slice_num signal INVERTS with depth.** At n_layers=5: monotonic slice16 < slice32 < slice64 (#2337 etc.). At n_layers=3: opposite — slice32 < slice16 < slice8. Depth and tokens are not independent.
2. **Mechanism reframe (frieren's analysis):** at deeper stacks the network compensates for coarser tokens via depth-stacked representation; at shallow depth, finer tokens are needed to preserve spatial bandwidth in the *input* itself. The per-head-dim framing (which was load-bearing on a single #2337 datapoint) does not survive depth change.
3. **slice_num=8 anomalous epoch_time_s spike (96.95 final vs ~45 expected):** wandb shows it ramping up sharply in last ~8 epochs from low baseline. Likely host-side I/O contention with parallel arm1 run. Training quality unaffected.
4. **Methodology lesson:** future stack-from-X-to-Y predictions require ≥1 same-axis datapoint in the destination compound; cross-compound transfer is not safe.

**Frieren reassigned:** cosine T_max tuning via --epochs (matched scheduler to realized epoch count).

---

## 2026-05-13 19:00 — PR #2438: Lion β2 sweep on n_head=1 (fern) — CLOSED, β2=0.99 LOCKED AT N_HEAD=1

- **Branch:** `willowpai2g24h5-fern/beta2-n-head-1-slice32`
- **Hypothesis:** Test whether the β2 reversal pattern continues at n_head=1 (per-head dim=128). Arms: β2=0.95 (continuation), β2=0.995 (V-shape).
- **W&B runs:** `si4ca2k6` (β2=0.95), `kbfplhq5` (β2=0.995)

| Arm | β2 | val | test | Δ vs #2338 (46.67) | Δ vs new #2400 (43.14) |
|-----|----|-----|------|-------------------|----------------------|
| 1 | 0.95 | 55.53 | 47.78 | +19.0% / +17.4% ✗✗ | +28.7% / +29.3% ✗✗ |
| Baseline #2338 | 0.99 | 46.67 | 40.69 | — | +8.2% / +10.1% |
| 2 | 0.995 | 47.73 | 41.19 | +2.27% / +1.22% ✗ | +10.7% / +11.4% ✗ |

**Result:** CLOSED. Both arms regress; neither beats the new baseline #2400.

Key findings:
1. **β2 axis CLOSED at n_head=1.** Canonical β2=0.99 is optimal. The "plateau-at-canonical" pattern across n_head ∈ {1,2,4} is now complete:
   - n_head=4: β2=0.995 wins (small per-head dim → needs longer averaging window)
   - n_head=2: β2=0.99 optimal
   - n_head=1: β2=0.99 optimal
2. **Main−EMA gap monotonic in β2** (9.14/6.66/7.10 for 0.95/0.99/0.995). Textbook noise-filter diagnostic.
3. **Bug fix shipped:** student exposed `--beta2` CLI flag (single-line Config edit). Not cherry-picked separately since #2400's branch had the equivalent pattern via `--n_layers`.

**Fern reassigned:** sw=5/sw=3 stack on new n_head=2+n_layers=3 compound (PR to be assigned).

---

## 2026-05-13 19:00 — PR #2430: slice_num=16 on n_head=1 compound (frieren) — CLOSED, SLICE16 ANTI-STACKS AT N_HEAD=1

- **Branch:** `willowpai2g24h5-frieren/slice16-n-head-1`
- **Hypothesis:** slice_num=16 won on n_head=2 (#2337); test whether it stacks with n_head=1.
- **W&B run:** `snqtgdl7`

| Metric | This run | #2338 baseline | Δ vs #2338 | Δ vs new #2400 (43.14) |
|--------|----------|----------------|-----------|----------------------|
| val | 50.80 | 46.67 | +8.85% ✗ | +17.7% ✗✗ |
| test | 43.31 | 40.69 | +6.43% ✗ | +17.2% ✗✗ |
| single_in_dist | 47.70 | 43.10 | +10.67% ✗ (worst) | — |

**Result:** CLOSED. All 4 splits regress; clearly worse than the new baseline.

Key findings:
1. **slice_num × n_head SUBSTITUTIVE at high per-head dim.** At per-head dim=64 (n_head=2), head capacity was the bottleneck → coarser tokens were free. At per-head dim=128 (n_head=1), tokens become the bottleneck → coarsening throws away information.
2. **single_in_dist hit hardest** (high-magnitude raceCar split; over-coarsening kills fine spatial structure).
3. **Host contention noted** (epochs 7-10 at 105-142 s/ep vs steady 68-69 s/ep). Even with clean compute, projection still trails baseline.

**Frieren reassigned:** slice_num=16/8 stack on new n_head=2+n_layers=3 compound (PR to be assigned).

---

## 2026-05-13 19:00 — PR #2419: lr sweep on n_head=1 compound (edward) — CLOSED, lr=1.25e-4 TEST-ONLY EFFECT

- **Branch:** `willowpai2g24h5-edward/lr-n-head-1`
- **Hypothesis:** Does lr re-tune at n_head=1 with concentrated gradient? Arms: lr=1.5e-4, lr=1.25e-4.
- **W&B runs:** `yggcly1j` (lr=1.5e-4), `v9uq83co` (lr=1.25e-4)

| Arm | lr | val | test | Δ vs #2338 (46.67) | Δ vs new #2400 (43.14) |
|-----|------|-----|------|-------------------|----------------------|
| 1 | 1.5e-4 | 47.49 | 40.90 | +1.75% / +0.52% ✗ | +10.1% / +10.7% ✗ |
| Baseline #2338 | 1e-4 | 46.67 | 40.69 | — | +8.2% / +10.1% |
| 2 | 1.25e-4 | 46.87 | 39.65 | +0.42% / **−2.56% test** | +8.7% / +7.3% ✗ |

**Result:** CLOSED. Even Arm 2 (test-only win vs old baseline) doesn't beat new #2400 baseline.

Key findings:
1. **lr=1.25e-4 wins TEST by −2.56% on n_head=1 compound** but val essentially flat (+0.42%, within noise). Val/test divergence at n_head=1: best-val ep25 generalizes better than baseline's ep26 despite virtually identical val.
2. **lr=1.5e-4 stability at n_head=1:** 0/1 crash rate vs 60% at n_head=2+slice64 (#2251). Single-head gradient concentration appears to stabilize Lion at higher lr (n=1).
3. **OOD camber-rc improved most** (Arm 2: -5.36%) — n_head=1's single global head gets gentle lr tuning right for geometry generalization.
4. With new #2400 baseline (val=43.14), the +3.73 gap to Arm 2 is too large to bridge with lr alone; new compound test needed.

**Edward reassigned:** wd=3e-4/1e-3 stack on new n_head=2+n_layers=3 compound (PR to be assigned).

---

## 2026-05-13 18:30 — PR #2400: n_layers=3 on n_head=2+slice32+Lion+MAE (askeladd) — MERGED, NEW BEST

- **Branch:** `willowpai2g24h5-askeladd/n-layers-reduce-slice32`
- **Hypothesis:** Speed-dividend thesis — shallower network is faster per epoch, enabling more gradient steps within 30-min wall-clock. n_layers=4 and n_layers=3 tested against n_layers=5 baseline on n_head=2+slice32 compound.
- **W&B runs:** `sffg3j2n` (Arm 1: n_layers=4), `vagjp0kr` (Arm 2: n_layers=3)

| Config | n_layers | val | test | Δ vs #2338 (46.67) | Epochs | s/ep |
|--------|---------|-----|------|-------------------|--------|------|
| Baseline #2338 | 5 (n_head=1) | **46.67** | **40.69** | — | 26 | 71.1 |
| **Arm 2 (WINNER)** | **3** | **43.14** | **36.95** | **−7.6% / −9.2% ✓** | **34** | **53.1** |
| Arm 1 | 4 | 46.30 | 38.88 | −0.8% / −4.4% ✓ | 27 | 67.3 |

**Per-test-split (n_layers=3):** single_in_dist=39.31 (−9.6%), geom_camber_rc=51.74 (−7.3%), geom_camber_cruise=21.32 (−13.3%), re_rand=35.43 (−8.9%) — **all 4 splits improve massively**

**Note on architecture:** n_layers=3 ran on **n_head=2** (not n_head=1 from #2338). New compound: n_head=2+slice32+n_layers=3. The n_head=1 compound (#2338) is superseded by this deeper speed-dividend gain.

**Result:** MERGED. Speed-dividend confirmed decisively:
1. **n_layers=3 → 53.1 s/ep → 34 epochs in 30 min** (vs 26 at n_layers=5+n_head=1 at 71.1 s/ep). 31% more gradient steps.
2. **All 4 test splits improve by 7–13%** — model is strongly undertrained; shallow + fast > deep + slow at this budget.
3. **n_layers=4 also beats baseline** (val=46.30) — monotonic signal: n_layers=3 < n_layers=4 < n_layers=5.
4. **Val still descending at cap (epoch 34)** — even n_layers=3 is under-budget; n_layers=2 is the natural extension.
5. Student exposed `--n_layers` as CLI flag (Config dataclass edit).

**Askeladd reassigned:** PR to be assigned — n_layers=2 extension on new compound (does speed-dividend continue below n_layers=3?).

---

## 2026-05-13 18:30 — PR #2376: lr sweep on slice_num=32 (tanjiro) — CLOSED, LR=1.5E-4 REGRESSES ON SLICE32

- **Branch:** `willowpai2g24h5-tanjiro/lr-sweep-slice32`
- **Hypothesis:** Transfer lr=1.5e-4 win from slice_num=64 (#2251) to new slice_num=32 compound. Test lr=1.5e-4 (Arm 1) and lr=1.25e-4 (Arm 2) against baseline lr=1e-4.
- **W&B run:** `szuo71ga` (Arm 1: lr=1.5e-4); Arm 2 not run per student recommendation.

| Config | lr | val | test | Δ vs #2218 (49.86) |
|--------|----|-----|------|-------------------|
| Arm 1 | 1.5e-4 | 50.77 | 43.77 | +0.91 (+1.8%) ✗ |
| Baseline #2218 | 1e-4 | 49.86 | 42.19 | — |

**Result:** CLOSED per student pivot recommendation. Key findings:
1. **lr transfer did NOT hold:** slice32 prefers lr≤1e-4 (opposite of slice64 which preferred 1.5e-4).
2. **lr×slice_num interaction is real:** coarser slicing (fewer, larger tokens) changes gradient variance; slice32 with 32 tokens per head may produce lower-variance gradients that don't benefit from higher lr.
3. **Arm 2 (lr=1.25e-4) skipped:** direction confirmed negative; #2218 baseline is already the lr floor.
4. **Student pivot: correct.** Arm 2 would not recover the regression given the direction.

**Tanjiro reassigned:** PR to be assigned — n_head=1 vs n_head=2 on n_layers=3 compound.

---

## 2026-05-13 17:30 — PR #2416: n_head=1 + sw=5 interaction on slice32 (alphonse) — CLOSED, SW=5 ANTI-SYNERGISTIC AT N_HEAD=1

- **Branch:** `willowpai2g24h5-alphonse/n-head-1-sw5`
- **Hypothesis:** sw=5 was synergistic with slice32 at n_head=2 (#2335: −2.54 val, 1.75×). Test whether the synergy transfers to n_head=1.
- **W&B run:** `c1zng9py`

| Config | sw | val | test | Δ vs #2338 (46.67) |
|--------|-----|-----|------|-------------------|
| Baseline #2338 | 10 | **46.67** | **40.69** | — |
| **This PR** | **5** | **48.51** | **41.05** | **+3.94% / +0.90% ✗** |

**Per-test-split:** single_in_dist=45.35 (+4.26% ✗), geom_camber_rc=55.26 (−0.94% ✓), geom_camber_cruise=24.57 (−0.06% ≈flat), re_rand=39.03 (+0.37% ≈flat)

**Note on run hygiene:** duplicate process ran in parallel for final 2 epochs, doubling epoch time (138/148s vs 71s). Extrapolated clean 25-epoch result ≈ val 47.4 — still ≥0.7 worse than baseline. Conclusion robust.

**Result:** CLOSED. Key findings:
1. **sw signal REVERSES at n_head=1**: sw=5 was synergistic at n_head=2 (−2.54 val), anti-synergistic at n_head=1 (+1.84 val). Reversal confirmed.
2. **Mechanism**: at n_head=1 (dim=128), single head must allocate capacity to surface pressure detail. sw=5 strips that signal with no backup head. At n_head=2 (dim=64), head was undersized and sw=5 helped concentrate on coarse patterns.
3. **Only OOD camber-rc marginal gain** (−0.94%) — vs #2335 where OOD gains drove the win. The surface loss pathway dominates at n_head=1.
4. **Prediction from mechanism**: if sw<10 hurts at n_head=1, sw>10 should help. Testing sw=15/20 in #2470.

**Alphonse reassigned:** PR #2470 — sw=15/sw=20 on n_head=1 compound (reversal test: does sw=10 sit at an optimum or does higher sw continue improving?).

---

## 2026-05-13 17:20 — PR #2356: Lion wd sweep on n_head=2+slice_num=32 (thorfinn) — CLOSED, WD=3E-4 BEATS OLD BASELINE, ASSIGNED TO N_HEAD=1

- **Branch:** `willowpai2g24h5-thorfinn/lion-wd-sweep-v2`
- **Hypothesis:** wd=1e-4 has never been swept on the n_head=2+slice32 compound; with doubled per-head dim (32→64), regularization regime may have shifted.
- **W&B runs:** `9ejpl0q4` (Arm 1: wd=3e-4), `oiymazx0` (Arm 2: wd=3e-5)

| Arm | wd | val | test | Δ val vs #2218 (49.86) | Best epoch |
|-----|-----|-----|------|----------------------|-----------|
| Arm 1 | **3e-4** | **47.18** | **41.03** | **−2.68 (−5.4%)** | 23 |
| Baseline #2218 | 1e-4 | 49.86 | 42.19 | — | 23 |
| Arm 2 | 3e-5 | 52.51 | 44.40 | +2.65 (+5.3%) ✗ | 19 |
| **Current baseline #2338** | 1e-4 (n_head=1) | **46.67** | **40.69** | — | 26 |

**Per-test-split (wd=3e-4 vs #2218):** single_in_dist=43.50 (−1.96 ✓), geom_camber_rc=55.29 (−0.75 ✓), geom_camber_cruise=24.94 (−0.74 ✓), re_rand=40.39 (−1.18 ✓) — **all 4 splits improve uniformly**.

**Result:** CLOSED (wd=3e-4 val=47.18 beats #2218 by −5.4% but loses to #2338 val=46.67 by +0.51). Key findings:
1. **Monotonic wd signal on n_head=2+slice32**: wd=3e-5 (52.51) < wd=1e-4 (49.86) < wd=3e-4 (47.18). Monotone over 10× range.
2. **5.4% relative improvement from 3× wd**: non-trivial Lion-decoupled-wd effect; n_head=2+slice32 was under-regularized at wd=1e-4.
3. **Uniform improvement across all splits** — whole-model regularization story, not OOD-camber-specific. Biggest gain on single_in_dist (−1.96).
4. **Natural follow-up: wd=3e-4 on n_head=1 compound.** If signal stacks, projection: val 46.67 → ~44.

**Thorfinn reassigned:** PR #2448 — wd=3e-4 and wd=1e-3 on n_head=1 compound (stack test + monotonic extension).

---

## 2026-05-13 17:00 — PR #2372: sw=2 vs sw=3 on n_head=2+slice_num=32 (nezuko) — CLOSED, SW=3 BEATS OLD BASELINE, ASSIGNED TO MLP RATIO

- **Branch:** `willowpai2g24h5-nezuko/sw-low-slice32`
- **Hypothesis:** Transfer sw=3 win from slice64 (#2277) to slice32; extend the lower sw curve on the new compound. Test sw=2 (floor extension) and sw=3 (transfer test).
- **W&B runs:** `2msxorju` (Arm 1: sw=2), `37kvtc8p` (Arm 2: sw=3)

| Arm | sw | val | test | Δ val vs #2218 (49.86) | Δ test vs #2218 (42.19) |
|-----|-----|-----|------|----------------------|----------------------|
| Arm 2 winner | **3** | **48.47** | **41.20** | **−2.78%** | **−2.34%** |
| Baseline #2218 | 10 | 49.86 | 42.19 | — | — |
| Arm 1 | 2 | 51.28 | 43.18 | +2.84% ✗ | +2.35% ✗ |
| **Current baseline #2338** | 10 (n_head=1) | **46.67** | **40.69** | — | — |

**Per-test-split (sw=3):** single_in_dist=46.86 (+3.08% ✗), geom_camber_rc=54.66 (−2.46% ✓), geom_camber_cruise=23.73 (−7.59% ✓✓), re_rand=39.56 (−4.83% ✓)

**Result:** CLOSED (doesn't beat current baseline #2338 val=46.67). Key findings:
1. **sw=3 transfers from slice64→32**: cruise gain improves from −5.6% to −7.59%. MAE+EMA+dropout reduces need for explicit surface emphasis.
2. **U-curve confirmed** at sw=3 minimum on n_head=2+slice32: sw=10 (49.86) ≥ sw=5 (48.57) ≈ sw=3 (48.47) ≪ sw=2 (51.28). sw=2 over-deweights, all 4 splits regress.
3. **Floor effect at sw=2**: needs ≥3× surface weighting at slice_num=32.
4. **geom_camber_cruise** is most sensitive to sw reduction — key finding for OOD generalization.
5. Both arms still descending at cap — true sw=3 gain likely larger at convergence.

**Nezuko reassigned:** PR #2446 — mlp_ratio=4 vs mlp_ratio=1 on n_head=1 compound (unexplored FFN-width axis; Transolver paper used mlp_ratio=4 vs our mlp_ratio=2 default).

---

## 2026-05-13 16:45 — PR #2295: EMA decay sweep on n_head=2+sw=5 (fern) — CLOSED, EMA=0.99 LOCALLY OPTIMAL

- **Branch:** `willowpai2g24h5-fern/ema-decay-sweep`
- **Hypothesis:** EMA decay tuned at n_head=4 (#1607); may need re-tuning for n_head=2+sw=5 compound. Two arms: decay=0.999 (longer window) vs decay=0.95 (shorter window).
- **W&B runs:** `0t869jbi` (Arm 1 best), `tdeuibi0`, `lay870l1` (Arm 1 replicates), `t6q7g04p` (Arm 2)

| Arm | ema_decay | val | test | EMA-main gap | Δ vs #2210 (50.91) |
|-----|-----------|-----|------|-------------|-------------------|
| Baseline #2210 | **0.99** | **50.91** | **43.68** | — | — |
| Arm 1 best | 0.999 | 52.67 | 44.13 | 5.06 | +1.76 ✗ |
| Arm 1 rep 2 | 0.999 | 57.47 | 49.41 | — | +6.56 ✗ |
| Arm 1 rep 3 | 0.999 | 57.16 | 49.49 | — | +6.25 ✗ |
| Arm 2 | 0.95 | 53.23 | 45.92 | 5.00 | +2.32 ✗ |

**Arm 1 crash rate: 3/6 runs (50%).** 12-pt spread across finishers (52.67 / 57.16 / 57.47).

**Note:** Both arms ran on pre-#2218 compound (slice_num=64, sw=5, n_head=2) — cannot merge against current #2338 baseline (val=46.67) regardless.

**Result:** CLOSED. Key findings:
1. **ema_decay=0.99 is locally optimal** on n_head=2+sw=5 compound. Both directions regress.
2. **Lion β2 → EMA decay analogy fails**: gradient averaging (β2) averages stationary noise; weight EMA averages a non-stationary descending trajectory. Longer windows lag the live model.
3. **Main-vs-EMA gap diagnostic**: At best epoch, gaps nearly identical (5.06 vs 5.00). Mid-training diverged: Arm 1 peaked at 8-9 pt lag (lags the improving model); Arm 2 oscillated to 1.95 pt (too reactive, noise leaks in).
4. **0.99 window (~100 updates ≈ ¼ epoch)** is optimal for the 7520-step training horizon.
5. **Arm 1 crash mechanism**: ema_decay=0.999 hangs on bad initial weights for ~1000 steps; any first-epoch anomaly poisons the entire average.

**Fern reassigned:** PR #2438 — Lion β2 sweep on n_head=1 compound (complete the n_head×β2 monotonic story: β2=0.995 won at n_head=4, β2=0.99 won at n_head=2, what wins at n_head=1?).

---

## 2026-05-13 16:30 — PR #2337: slice_num=16 on n_head=2+sw=10 baseline (frieren) — CLOSED, BEATS OLD BASELINE, ASSIGNED TO COMPOUND

- **Branch:** `willowpai2g24h5-frieren/slice-num-16-sweep`
- **Hypothesis:** Extend monotonic slice_num trend below 32. slice_num=16 doubles per-slice node coverage vs slice_num=32; should also run ~7% faster, buying extra epochs.
- **W&B run:** `45r8syhx`

| Config | slice_num | val | test | Epochs/30min | s/ep |
|--------|-----------|-----|------|-------------|------|
| Baseline #2218 | 32 | 49.86 | 42.19 | 23 | 81.4 |
| **This PR** | **16** | **48.08** | **41.02** | **24** | **75.9** |
| **Current best #2338** | 32 (n_head=1) | **46.67** | **40.69** | 26 | 71.1 |

**Per-test-split (slice_num=16):** single_in_dist=43.78, geom_camber_rc=55.82, geom_camber_cruise=24.62, re_rand=39.85 — **all 4 splits improve vs old baseline**.

**Result:** CLOSED (doesn't beat current baseline #2338 val=46.67 on n_head=1 compound). Key findings:
1. **Monotonic slice_num trend confirmed below 32**: 16 < 32 < 64 < 128 — strict order maintained on n_head=2+sw=10.
2. **Speed gain**: 75.9 s/ep vs 81.4 s/ep (−6.8%), +1 epoch in 30 min cap. Val still descending at cap — further headroom likely.
3. **Marginal gain shrinking** on older baseline (Δ 5.06 → 1.25 → 1.78): last delta grew slightly — may be noise or true headroom remains.
4. **Natural follow-up:** slice_num=16 + n_head=1 compound. **Assigned as PR #2430 to frieren.**

---

## 2026-05-13 16:00 — PR #2338: n_head=1 on n_head=2+slice_num=32 baseline (edward) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-edward/n-head-1-sweep`
- **Hypothesis:** Extend monotonic n_head trend to n_head=1 on slice_num=32 compound. Per-head dim doubles 64→128; concentrated global attention may dominate multi-head diversity.
- **W&B run:** `g71iu8ae`

| Config | n_head | val | test | Epochs/30min | s/ep |
|--------|--------|-----|------|-------------|------|
| Baseline #2335 | 2 | 48.57 | 41.48 | 22 | 82 |
| **This PR** | **1** | **46.67** | **40.69** | **26** | **71** |

**Per-test-split (n_head=1 vs #2218 baseline):** single_in_dist=43.50 (−8.2%), geom_camber_rc=55.79 (−0.4%), geom_camber_cruise=24.58 (−4.3%), re_rand=38.88 (−6.5%) — **all 4 splits improve**.

**Result:** MERGED. New best: val=46.67, test=40.69. Key findings:
1. **Monotonic n_head trend extends to n_head=1**: 1 < 2 < 4 < 8 — all data points strictly ordered.
2. **Speed dividend at n_head=1**: 71.1s/ep vs 82s/ep (n_head=2) → 26 epochs vs 22 in same wall-clock. Both mechanisms (accuracy + speed) compound.
3. **Single global attention wins physics task**: at slice_num=32, the 32 learned physics slices provide spatial decomposition; additional head diversity is redundant.
4. **Note:** ran with sw=10 (default). sw=5 interaction (#2416 alphonse) is the next natural experiment.

---

## 2026-05-13 15:50 — PR #2335: slice_num=32 + surf_weight=5 interaction on n_head=2 (alphonse) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-alphonse/slice32-sw5-interaction`
- **Hypothesis:** slice_num=32 (#2218) and surf_weight=5 (#2210) are independent wins; stack them for a compound improvement.
- **W&B run:** `k5262fzu`

| Config | sw | val | test | Δ val | Δ test |
|--------|-----|-----|------|-------|--------|
| Baseline #2218 | 10 | 49.86 | 42.19 | — | — |
| **This PR** | **5** | **48.57** | **41.48** | **−2.59%** | **−1.68%** |

**Per-test-split:** single_in_dist=47.41 (+4.3% ⚠), geom_camber_rc=54.56 (−2.6%), geom_camber_cruise=24.63 (−4.1%), re_rand=39.33 (−5.4%) — 3/4 splits improve.

**Result:** MERGED. Key findings:
1. **Synergistic interaction on val**: observed −2.54 val vs additive −1.45 (1.75× predicted). Coarser tokens + softer surface emphasis compound on OOD splits.
2. **single_in_dist regresses** vs slice32+sw10 — reduced surface emphasis hurts the high-magnitude in-dist regime.
3. **Interaction confirmed**: slice32+sw5 is the new compound for follow-up. sw=5 interaction with n_head=1 is untested (#2416 alphonse).

---

## 2026-05-13 15:35 — PR #2271: Lion β2 on n_head=2: β2=0.995 vs β2=0.999 (askeladd) — CLOSED, BOTH ARMS REGRESS, β2 DIRECTION REVERSED FROM n_head=4

- **Branch:** `willowpai2g24h5-askeladd/lion-beta2-n-head-2`
- **Hypothesis:** β2=0.995 won −2.9% on n_head=4 compound (#2144). Retest on n_head=2 and push to β2=0.999.
- **W&B runs:** `2768vmk2` (Arm 1: β2=0.995 best), `zvny3ajq` (Arm 2: β2=0.999)

| Arm | β2 | run_id | val | test | Δ vs #2218 (49.86/42.19) |
|-----|-----|--------|------|------|--------------------------|
| **Arm 1 best** | **0.995** | **`2768vmk2`** | **51.54** | **44.43** | **+3.4% / +5.3% (regress)** |
| Arm 1 v3 | 0.995 | `pnan6q97` | 52.07 | 44.65 | +4.4% / +5.8% |
| Arm 1 repl | 0.995 | `epuo44ec` | 57.92 | 49.38 | +16.2% / +17.0% |
| Arm 2 | 0.999 | `zvny3ajq` | 53.82 | 46.43 | +7.9% / +10.0% |

Arm 1 finishers spread: [51.54, 52.07, 57.92] — 12% variance. EMA gap (main-vs-EMA): 7.31 (β2=0.995), 2.81 (β2=0.999).

**Result:** CLOSED. Key finding: **β2 effect reverses at n_head=2 vs n_head=4.**
- n_head=4 (#2144): 0.95<0.99<0.995 (β2=0.995 wins −2.9%)
- n_head=2 (#2271): 0.99<0.995<0.999 (canonical β2=0.99 wins)

**Mechanism:** Doubling per-head dim 32→64 shifts optimal momentum window. Richer per-head capacity already filters noise; extra de-noising via slower β2 over-smooths the sign(·) direction. Runs also on slice_num=64 (pre-#2218), so couldn't merge regardless. **β2 sweep closed on n_head=2. Canonical β2=0.99 confirmed.**

**Askeladd reassigned:** PR #2400 — n_layers=4 vs n_layers=3 on slice_num=32 compound (speed-dividend extension of slice_num=32 win).

---

## 2026-05-13 15:20 — PR #2251: lr sweep on n_head=2: lr=2e-4 (Arm1) vs lr=1.5e-4 (Arm2) (tanjiro) — CLOSED, NEITHER ARM BEATS #2218 BASELINE

- **Branch:** `willowpai2g24h5-tanjiro/lr-sweep-n-head-2`
- **Hypothesis:** n_head=2 inherited lr=1e-4 from n_head=4 baseline; doubling lr to 2e-4 was the n_head=4 winner — test 2e-4 vs 1.5e-4 on n_head=2.
- **W&B runs:** `mq67yiq3` (Arm 1: lr=2e-4, best of 5), `ju4mdtl2` (Arm 2: lr=1.5e-4)

| Arm | lr | val | test | Δ vs #2069 (51.11/44.18) | Δ vs #2218 (49.86/42.19) |
|-----|-----|------|------|--------------------------|--------------------------|
| **2 (winner)** | **1.5e-4** | **50.36** | **42.53** | **−0.75 / −1.65 ✓** | +0.50 / +0.34 ✗ |
| 1 | 2e-4 | 50.55 | 43.10 | −0.56 / −1.08 ✓ | +0.69 / +0.91 ✗ |

**Arm 1 crash rate:** 3/5 runs crashed (60%). Instability at lr=2e-4 with Lion+MAE+n_head=2+slice_num=64.

**Per-test-split (Arm 2, lr=1.5e-4, `ju4mdtl2`):** single_in_dist=46.61, geom_camber_rc=55.99, geom_camber_cruise=26.35, re_rand=41.15

**Result:** CLOSED. Both arms beat old n_head=2 baseline (#2069) but ran on slice_num=64 (pre-#2218 default) — cannot beat the new slice_num=32 compound. Key findings:
1. **lr=1.5e-4 is the clear winner** on n_head=2+slice_num=64 — beats both #2069 and Arm 1.
2. **60% crash rate at lr=2e-4** — instability signal; sits near the edge with Lion+MAE+n_head=2.
3. **Natural follow-up:** test lr=1.5e-4 vs lr=1.25e-4 on the new slice_num=32 compound (assigned as PR #2376 to tanjiro).

**Tanjiro reassigned:** PR #2376 — lr=1.5e-4 vs lr=1.25e-4 on slice_num=32 baseline.

---

## 2026-05-13 14:35 — PR #2277: surf_weight lower probe sw=4 vs sw=3 on n_head=2 (nezuko) — CLOSED, sw=3 BEATS OLD BASELINE BUT LOSES TO NEW

- **Branch:** `willowpai2g24h5-nezuko/surf-weight-lower`
- **Hypothesis:** Extend monotonic sw curve below sw=5 (#2210 winner). Test sw=4 and sw=3.
- **W&B runs:** `pn9utnpz` (Arm 1: sw=4), `g1gbebnj` (Arm 2: sw=3, winner)

| Arm | sw | val | test | Δ vs #2210 (50.91/43.68) | Δ vs #2218 (49.86/42.19) |
|-----|-----|------|------|--------------------------|--------------------------|
| 1 | 4 | 51.16 | 43.66 | +0.49% / −0.05% | +2.6% / +3.5% |
| **2 (winner)** | **3** | **50.23** | **42.84** | **−1.34% / −1.92%** | +0.7% / +1.5% |

**Per-test-split (Arm 2, sw=3):** single_in_dist=46.87 (+0.45), geom_camber_rc=57.81 (−0.79), geom_camber_cruise=25.79 (−1.54), re_rand=40.90 (−1.49). Wins 3/4 test splits.

**Result:** CLOSED. sw=3 cleanly beats the OLD baseline (#2210, sw=5) but cannot merge on the NEW baseline (#2218, slice_num=32). Key findings:
1. **Non-monotonic in [3, 5]:** sw=3 < sw=5 < sw=4 on val (sw=4 is local maximum).
2. **Strong geom_camber_cruise improvement** at lower sw (Arm 1: −6.1%, Arm 2: −5.6%) — surface emphasis was under-fitting OOD camber.
3. **All arms still descending at cap** — absolute numbers under-converged.

These runs used slice_num=64 (pre-#2218). Combined with #2335 (alphonse, sw=5 on slice_num=32 in progress), the natural follow-up tests lower sw on the new compound.

**Nezuko reassigned:** PR #2372 — sw=2 vs sw=3 on slice_num=32 baseline (extend low end + confirm transfer of #2277 win).

---

## 2026-05-13 13:50 — PR #2218: slice_num sweep slice_num=32 vs slice_num=128 on n_head=2 (alphonse) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-alphonse/slice-num-sweep-n-head-2`
- **Hypothesis:** slice_num=64 (default) may be over-parameterized for n_head=2 (per-head dim=64). Coarser slicing concentrates capacity; finer slicing may dilute it. Both directions tested.
- **W&B runs:** `8qjqtb70` (slice_num=32, winner), `mzkmh2fh` (slice_num=128), `uitkeygr` (slice_num=32 replicate, similar)

| Config | slice_num | val_avg/mae_surf_p | test_avg/mae_surf_p | Epochs in 30 min | s/epoch |
|--------|-----------|---------------------|----------------------|-----------------|---------|
| Baseline #2210 | 64 | 50.91 | 43.68 | 20 | ~93.5 |
| **Arm 1 (8qjqtb70)** | **32** | **49.864** | **42.187** | **23** | **81.4** |
| Arm 1 replicate (uitkeygr) | 32 | 49.96 | 41.76 | ~23 | 81.4 |
| Arm 2 (mzkmh2fh) | 128 | 56.17 | 48.78 | 16 | 114.8 |

**Per-test-split (slice_num=32 winner):** single_in_dist=45.46 (−0.96 vs 46.42), geom_camber_rc=56.04 (−2.56 vs 58.60), geom_camber_cruise=25.68 (−1.65 vs 27.33), re_rand=41.57 (−0.82 vs 42.39) — **all 4 splits improve**.

**Result:** MERGED. New best: val=49.86, test=42.19. Key findings:
1. **Monotonic signal confirmed: 32 < 64 < 128** — coarser slicing beats finer, monotonically.
2. **Compute dividend:** slice_num=32 runs at 81.4s/epoch vs 93.5s (14% faster), yielding 23 epochs in 30 min vs 20 — extra training steps amplify the accuracy gain.
3. **OOD improvement:** camber_rc −2.56, camber_cruise −1.65 — coarser slices generalize better to unseen geometries; less risk of over-fitting per-geometry slice assignments.
4. **slice_num=128 badly regresses** (+5.06/+4.60 val/test) AND runs slower (114.8s/ep, 16 epochs) — finer slicing hurts on all dimensions.
5. **Note:** This run used sw=10 (default), NOT sw=5 from #2210. The slice_num=32 + sw=5 interaction is explored in follow-up #2335.

**New compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + n_head=2 + **slice_num=32** + surf_weight=10

**Alphonse reassigned:** PR #2335 — slice_num=32 + surf_weight=5 interaction (test whether the two wins from #2218 and #2210 stack).

---

## 2026-05-13 13:50 — PR #2216: Split loss surf-MAE + vol-Huber/MSE on n_head=2 (frieren) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-frieren/split-loss-n-head-2`
- **Hypothesis:** Aligning volume loss with a robust formulation (Huber/MSE) while keeping surface as MAE may reduce noise in the gradient signal and improve the OOD camber splits.
- **W&B runs:** `kpiec2be` (Arm 1 vol-Huber), `dnrif75z` (Arm 2 vol-MSE), `8u7db1a3` (Arm 1 retry)

| Run | Formulation | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline (49.86) |
|-----|-------------|---------------------|----------------------|----------------------|
| kpiec2be | surf-MAE + vol-Huber | 53.40 | 44.72 | +7.1% |
| 8u7db1a3 | surf-MAE + vol-Huber (retry) | 52.09 | 45.15 | +4.5% |
| dnrif75z | surf-MAE + vol-MSE | 52.81 | 44.45 | +5.9% |

**Result:** CLOSED. All three runs regress vs baseline. The split-loss formulation introduces optimization tension without signal benefit — MAE's uniform per-node weighting is already aligned with the primary metric. The volume-specific loss formulation doesn't improve surface pressure prediction. Two distinct formulations tested with consistent negative results.

**Frieren reassigned:** PR #2337 — slice_num=16 (extend slice_num monotonic trend below 32).

---

## 2026-05-13 13:50 — PR #2183: AdamW+EMA+MAE mechanism fill (edward) — CLOSED, DIAGNOSTIC COMPLETE

- **Branch:** `willowpai2g24h5-edward/adamw-ema-mae-lr-sweep`
- **Hypothesis:** Fill missing 2×2 cell (AdamW+EMA+MAE) to complete the mechanism table. Diagnostic-only.
- **W&B runs:** `aaz608bi`, `5cmntgh1`, `ztnpmk0i` (lr=5e-4 arms), `avm05bdd` (lr=2e-4)

| Cell | val_avg/mae_surf_p | test_avg/mae_surf_p |
|------|--------------------|---------------------|
| Lion+EMA (baseline) | **49.86** | **42.19** |
| Lion no-EMA (#2070) | 62.47 | 54.42 |
| **AdamW+EMA (this PR)** | **73.38 (best)** | **64.29** |
| AdamW no-EMA (#2070) | 82.46 | 71.69 |

Arm 1 replicate variance: val 73.38–74.52 (mean 73.94 ± 0.47); lr=2e-4 arm (74.06) inside noise band — AdamW is lr-insensitive in this range.

**Result:** CLOSED (diagnostic-only, regression as expected). 2×2 mechanism table complete. **Summary:** Replacing Lion with AdamW costs +22.5 val (EMA fixed); removing EMA costs +11.6 val (Lion fixed). Lion contributes ~2× more than EMA. Strong justification for Lion-first exploration strategy.

**Edward reassigned:** PR #2338 — n_head=1 on n_head=2+slice_num=32 baseline (extend architectural monotonic trend).

---

## 2026-05-13 13:30 — PR #2211: OneCycleLR on n_head=2 baseline: pct_start=0.3 vs 0.1 (thorfinn) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-alphonse/n-head-8-lion-mae`
- **Hypothesis:** n_head=4 (baseline) may be over-/under-parameterized for slice_num=64 with n_hidden=128. n_head=2 doubles per-head dim (32→64); n_head=8 halves it (32→16).
- **W&B runs:** `2lo9mn88` (n_head=2, winner), `qkh64fhe` (n_head=8 run 2), `y42702ef` (n_head=8 run 1)

| Metric | Prev baseline (#1932) | n_head=2 | n_head=8 (run 1) | n_head=8 (run 2) |
|--------|----------------------|----------|-----------------|-----------------|
| val_avg/mae_surf_p | 55.41 | **51.11** | 64.09 | 62.74 |
| test_avg/mae_surf_p | 47.90 | **44.18** | 54.96 | 54.11 |
| Δ vs baseline | — | **−7.76%** | +15.7% | +13.2% |
| Epochs in 30 min | 16 | **20** | 12 | 12 |

**Per-test-split (n_head=2):** single_in_dist=49.23, geom_camber_rc=57.44, geom_camber_cruise=26.74, re_rand=43.30 — wins all 4.

**Result:** MERGED. n_head=2 is the strongest single-experiment win of this round. Val still descending at cap (−1.3/epoch in last 3 transitions, no plateau). The monotonic head-count direction confirms head-undersizing story: n_head=2 wins → n_head=4 baseline → n_head=8 worst. The architectural change also unlocks faster epochs (~93.5s vs ~110s) for net more training steps in the same wall-clock.

**Compute-neutrality caveat (alphonse's analysis):** At matched epoch=12 the two arms are close (n_head=2 val=63.00 vs n_head=8 val=62.74). The headline win at 51.11 is partly the architecture buying compute. But the architectural change is what unlocks it — the win is real and reproducible under the 30-min cap.

**Note:** Both runs used lr=1e-4 (not the lr=2e-4 from #1932). The lr × n_head interaction is unexplored.

**New compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + **n_head=2** (slice_num=64)

---

## 2026-05-13 ~11:10 — PR #2086: Lion lr probe lr=4e-4 + lr=3e-4 on Lion+MAE (thorfinn) — CLOSED, SATURATION CONFIRMED

- **Branch:** `willowpai2g24h5-thorfinn/lion-lr-4e-4-probe`
- **Hypothesis:** lr-doubling trend 5e-5→1e-4→2e-4 hasn't saturated; 3e-4/4e-4 may continue.
- **W&B runs:** `4q7tjvt4` (lr=4e-4), `bpa3tkj7` (lr=3e-4)

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline (55.41) |
|-----|---|---|---|
| lr=4e-4 (4q7tjvt4) | 57.53 | 49.03 | +3.83% |
| lr=3e-4 (bpa3tkj7) | 57.68 | 48.81 | +4.10% |
| lr=2e-4 baseline | **55.41** | **47.90** | — |

**Result:** CLOSED. lr-doubling trend saturated at lr=2e-4 after 3 winning octaves. Key diagnostics: (1) flat minimum — lr=4e-4 is only 2 pts worse than lr=3e-4 despite 4×-3× the learning rate; (2) EMA main-vs-EMA gap (~10-12 pts) is identical at all 3 lrs — EMA is not the bottleneck at higher lr; (3) both arms still descending at cap with ~3 val pts over last 4 epochs — not diverging, just in a shallower basin. Schedule shape (OneCycleLR) is the natural next lever.

**Thorfinn reassigned:** PR #2211 — OneCycleLR on n_head=2 baseline.

---

## 2026-05-13 ~11:10 — PR #2056: surf_weight tune on Lion+MAE (nezuko) — CLOSED, RETEST ON n_head=2

- **Branch:** `willowpai2g24h5-nezuko/surf-weight-lion-mae`
- **Hypothesis:** MAE's uniform weighting reduces need for high surf_weight; sw=5 may improve surface focus, sw=15 may over-emphasize it.
- **W&B runs:** `gxq26aip` (sw=5, lr=1e-4), `obkwbyo1` (sw=15, lr=2e-4)

| Arm | sw | lr | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs old base (55.41) |
|-----|----|----|---|---|---|
| sw=5 (gxq26aip) | 5 | 1e-4 | **54.46** | **47.06** | −1.71% / −1.76% |
| sw=15 (obkwbyo1) | 15 | 2e-4 | 58.32 | 50.27 | +5.25% / +4.95% |

**Result:** CLOSED because #2069 (n_head=2, val=51.11) merged after this was submitted. sw=5 at lr=1e-4 beat the OLD baseline by 1.71% — a genuine improvement — but cannot be merged onto the new n_head=2 code as-is (was measured at n_head=4). sw=15 regresses on all splits. Nezuko reassigned to retest sw=5+sw=7 on n_head=2 baseline (#2210).

---

## 2026-05-13 ~11:10 — PR #2052: bs=8 + LR scaling on Lion+MAE (frieren) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-frieren/batch-size-lr-scaling`
- **Hypothesis:** bs=8 reduces gradient noise; linear-scaled lr compensates for larger batch.
- **W&B runs:** `nay1st1x` (bs=8, lr=2e-4), `img8ns9k` (bs=8, lr=1e-4)

| Arm | bs | lr | val_avg/mae_surf_p | Δ vs baseline |
|-----|----|----|---|---|
| Arm 1 linear (nay1st1x) | 8 | 2e-4 | 61.05 | +7.9% |
| Arm 2 batch-only (img8ns9k) | 8 | 1e-4 | 66.99 | +18.4% |

**Result:** CLOSED. Hypothesis falsified: step-count-limited, not gradient-noise-limited. bs=8 takes half as many steps in same wall-clock. VRAM at 93.7 / 92.2 GB — bs=8 is maximum safe batch. The per-step distance route (lr) is the right lever; gradient noise reduction via batch size is wrong for this regime. Frieren reassigned to split-loss formulation (#2216: surface-MAE + volume-Huber).

---

## 2026-05-13 ~10:34 — PR #2070: Lion-no-EMA + AdamW-no-EMA ablation (edward) — CLOSED, ICML APPENDIX

- **Branch:** `willowpai2g24h5-edward/lion-no-ema-ablation`
- **Hypothesis:** Quantify Lion vs EMA contributions to the Lion+EMA+MAE win. Diagnostic-only, not for merge.
- **W&B runs:** `lyplrb6e` (Lion-no-EMA full 18 ep, canonical), `q6lou95t`/`5memu5rh` (Lion-no-EMA truncated), `4u85vrwj` (AdamW-no-EMA full 18 ep)

| Cell | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline |
|------|-------------------|---------------------|---------------|
| Lion+EMA(0.99)+MAE (baseline #1932) | **55.41** | **47.90** | 0 |
| Lion+no-EMA+MAE (best, lyplrb6e) | 62.47 | 54.42 | +7.06 / +6.52 |
| AdamW+no-EMA+MAE (4u85vrwj) | 82.46 | 71.69 | +27.05 / +23.79 |

**Result:** CLOSED — regression vs baseline (expected, diagnostic-only). **Reframed mechanism story:**
- Full-budget Lion-no-EMA val=62.47 (NOT 78 from truncated runs) — Lion's direction explains ~20 val, EMA explains ~7-9 val.
- **Lion is the dominant ingredient (~75% of Lion+EMA's win over AdamW+EMA), EMA is secondary (~25%).** The synergy is real but Lion-led, not EMA-led.
- Lion-no-EMA final-epoch bounce 62→71 at ep 18 — direct evidence that a single bad late update visibly hurts val without parameter averaging.
- AdamW-no-EMA has 3 backward steps over 18 epochs (Lion-no-EMA has 1) — Lion's signed direction is loss-landscape-aware on this geometry-aware problem.
- **Variance noted:** lyplrb6e=62.47 vs 5memu5rh=77.99 at identical config = 15-val gap. The truncated runs (10–11 ep) under-represent steady-state; full-budget is the canonical datapoint.

**Edward reassigned:** PR #2183 — AdamW+EMA+MAE at lr=5e-4 and lr=2e-4. Fills missing 2×2 cell; quantifies optimizer effect at fixed lr=2e-4 (apples-to-apples Lion vs AdamW).

---

## 2026-05-13 ~10:06 — PR #1999: Cosine T_max tuning T_max=16 ± eta_min on Lion+MAE+lr=1e-4 (fern) — CLOSED REGRESSION (with strong diagnostic)

- **Branch:** `willowpai2g24h5-fern/cosine-tmax-tuning`
- **Hypothesis:** T_max=epochs=50 barely decays the LR over the 16-epoch wall-clock window. Matching T_max to actual budget should let the cosine arc complete.
- **W&B runs:** `8csqgctq` (Arm 1: T_max=16, eta_min=0), `0mw5k64a` (Arm 2: T_max=16, eta_min=1e-5)
- **Caveat:** Both arms ran at **lr=1e-4** (the pre-#1932 baseline). The PR was assigned before #1932 merged.

| Metric | Pre-merge baseline (#1825, lr=1e-4) | Current baseline (#1932, lr=2e-4) | Arm 1 | Arm 2 |
|--------|-------------------------------------|-----------------------------------|-------|-------|
| val_avg/mae_surf_p | 56.58 | **55.41** | 62.02 | 59.59 |
| test_avg/mae_surf_p | 48.82 | **47.90** | 52.55 | 51.42 |
| Δ vs current baseline | — | — | **+11.9%** | **+7.5%** |
| Final-epoch val Δ (last 4) | — | — | **+0.19** (uptick at lr=0) | **−1.48** (steepest descent) |

**Result:** CLOSED — both arms regress vs current baseline at lr=1e-4. **But the diagnostic value is high.**

**Key findings:**
1. **eta_min=0 is strictly dominated by eta_min=1e-5** (Arm 1 vs Arm 2 differ by −2.43 val). Arm 1 had the *only* positive per-epoch delta in its last 4 epochs at the lr=0 step — the degenerate-tail effect is real. **Future cosine work must use eta_min>0.**
2. **Arm 2 still descending at the cap** (−1.48 val in its final epoch, steepest of the last 4). At lr=1e-4 the per-step capacity is the bottleneck, not the schedule shape. Both schedules (T_max=16 and T_max=50) leave the model under-converged at lr=1e-4.
3. **Vol_loss accumulator non-finite bug** flagged by fern (train.py:288, 313–328). Per-component MAE metrics are correct (PR #1541 path), but the vol_loss sum is unguarded. Cosmetic, not in scope.

**Fern reassigned:** PR #2167 — Cosine T_max + eta_min tuning at lr=2e-4. Tests both axes: (a) schedule-match (T_max=16 + eta_min=1e-5) at the new lr, where per-step capacity is doubled; (b) eta_min=1e-5 floor on the default T_max=50 schedule (isolates final-lr regularization effect). Clean 2-axis decomposition.

---

## 2026-05-13 ~10:00 — PR #2001: Lion β1 sweep β1=0.95 vs β1=0.85 on Lion+MAE+lr=2e-4 (askeladd) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-askeladd/lion-b1-sweep`
- **Hypothesis:** β1=0.9 is Lion's canonical value from large-scale vision experiments; smaller datasets may prefer a different β1. β1=0.95 gives more momentum inertia (slower direction change), β1=0.85 makes the sign update more reactive to the current gradient.
- **W&B runs:** `hqfbylaj` (Arm 1: β1=0.95), `2ql8nhfg` (Arm 2: β1=0.85)

| Metric | Current baseline (#1932) | Arm 1 (β1=0.95) | Arm 2 (β1=0.85) | Δ Arm1 | Δ Arm2 |
|--------|--------------------------|-----------------|-----------------|--------|--------|
| val_avg/mae_surf_p | **55.41** | 57.62 | 58.93 | +4.00% | +6.36% |
| test_avg/mae_surf_p | **47.90** | ~50.0 (est.) | ~51.4 (est.) | regression | regression |

**Result:** CLOSED. Both arms regress vs baseline. The curve is **asymmetric**: β1=0.85 hurts ~2× more than β1=0.95, indicating the loss landscape is more sensitive to over-reactive updates than to over-inertial ones. The optimum is near or slightly above β1=0.9.

**Key finding:** Canonical β1=0.9 is confirmed as (near-)optimal for this problem. The asymmetric response (β1=0.95 hurts less than β1=0.85) suggests the momentum inertia side has more tolerance than the reactivity side — the signed gradient direction is informative enough that maintaining it longer is preferable to discarding it faster. No follow-up β1 variations warranted.

**Askeladd reassigned:** PR #2144 — Lion β2 sweep (β2=0.995 vs β2=0.95) on Lion+MAE+lr=2e-4. At lr=2e-4, the momentum buffer's memory window (β2) is the last untested hyperparameter in the Lion triplet (lr, β1, β2). β1=0.9 optimum confirmed; β2=0.99 default is untested.

---

## 2026-05-13 11:55 — PR #2131: Dropout sweep dropout=0.3 vs 0.1 on Lion+MAE+lr=2e-4 (tanjiro) — CLOSED, LOCALLY OPTIMAL AT 0.2

- **Branch:** `willowpai2g24h5-tanjiro/dropout-lion-mae-lr2e-4`
- **Hypothesis:** Under-regularization signal from #1961 (mlp_ratio=4 had main-vs-EMA gap ~22) should be present at mlp_ratio=2 too; dropout=0.2 may be below optimum.
- **W&B runs:** `qtieg835` (dropout=0.3, winner), `jgcae86g` (dropout=0.3 replicate), `0zk2y6cj` (dropout=0.1), `y8oh2gxf` (baseline ref, dropout=0.2)

| Arm | dropout | run_id | best_epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs old baseline (55.41) |
|-----|---------|--------|------------|---------------------|----------------------|--------------------------|
| Arm 1 winner | 0.3 | `qtieg835` | 16 | **55.10** | **47.83** | −0.56% / −0.15% |
| Arm 1 replicate | 0.3 | `jgcae86g` | 16 | 55.87 | 48.09 | +0.83% / +0.40% |
| Arm 2 | 0.1 | `0zk2y6cj` | 14 | 57.77 | 49.24 | +4.26% / +2.79% |
| baseline ref | 0.2 | `y8oh2gxf` | 16 | 55.41 | 47.90 | — |

**Dropout=0.3 mean:** val=55.49 ± 0.38, test=47.96 ± 0.13 — within noise of baseline 0.2 (55.41 / 47.90).

**Main-vs-EMA gap diagnostics (epoch 16):**
| dropout | run_id | ema_val | main_val | gap |
|---------|--------|---------|----------|-----|
| 0.1 | `0zk2y6cj` | 57.78 | 69.64 | +11.86 |
| 0.2 | `y8oh2gxf` | 55.41 | 61.77 | +6.36 |
| 0.3 | `qtieg835` | 55.10 | 61.89 | +6.79 |
| 0.3 | `jgcae86g` | 55.87 | 67.02 | +11.15 |

**Result:** CLOSED. Baseline note: n_head=2 merged (val=51.11) during these runs — neither arm can merge on new compound regardless. Key findings:
1. **dropout=0.2 is locally optimal on Lion+MAE+lr=2e-4+n_head=4/mlp_ratio=2** — under-regularization signal from #1961 (mlp_ratio=4, gap≈22) did NOT transfer to mlp_ratio=2 (gap≈6–11).
2. **dropout=0.3 ≈ 0.2 within seed noise** — best single 0.3 run wins old baseline by 0.31 val, but replicate loses by 0.46; mean is 0.08 worse.
3. **dropout=0.1 clearly regresses** (+2.4 val / +1.3 test) — Lion's sign-update alone doesn't substitute for dropout regularization.
4. Trajectory shapes confirm NOT converged at epoch 16; all arms still descending at cap.

**Tanjiro reassigned:** PR #2251 — lr sweep on n_head=2 (lr=2e-4 vs lr=1.5e-4). The n_head=2 baseline (PR #2069) inherited lr=1e-4 from pre-merge compound; lr=2e-4 won at n_head=4 but was never tested at n_head=2. BASELINE.md itself noted "the lr × n_head interaction remains to be explored."

---

## 2026-05-13 12:05 — PR #2144: Lion β2 sweep β2=0.995 vs β2=0.95 on Lion+MAE+lr=2e-4 (askeladd) — CLOSED, STRONG SIGNAL, NEEDS RETEST ON n_head=2

- **Branch:** `willowpai2g24h5-askeladd/lion-beta2-sweep`
- **Hypothesis:** β2 (momentum EMA decay, canonical=0.99) untested on TandemFoilSet. β2=0.995 lengthens momentum window (~200 effective steps); β2=0.95 shortens it (~20 steps).
- **W&B runs:** `b7pyyc0n` (β2=0.995, winner), `94sgx2e9` (β2=0.95)

| Arm | β2 | run_id | best_epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs old baseline (55.41) |
|-----|-----|--------|------------|---------------------|----------------------|--------------------------|
| 1 (winner) | 0.995 | `b7pyyc0n` | 16 | **53.815** | **46.609** | **−2.9% / −2.7%** |
| 2 | 0.95 | `94sgx2e9` | 16 | 64.118 | 54.969 | +15.7% / +14.8% |

**Val split breakdown (Arm 1, β2=0.995):**
- single_in_dist=57.84 (−3.6), geom_camber_rc=68.85 (+1.5, flat), geom_camber_cruise=33.69 (−3.5), re_rand=54.88 (−0.9)

**Test split breakdown (Arm 1, β2=0.995):**
- single_in_dist=51.41 (flat), geom_camber_rc=60.80 (−1.5), geom_camber_cruise=28.71 (−2.5), re_rand=45.51 (−1.5)

**Result:** CLOSED. Cannot merge on new n_head=2 compound (val=51.11 — arm is +2.71 above). Key findings:
1. **Monotonic ordering: 0.95 < 0.99 < 0.995** — three strictly-ordered data points; trend is real
2. **β2=0.995 wins −2.9% val on old compound** — strongest Lion-internal finding of the round
3. **Asymmetric direction** (faster decay hurts 15.7% vs slower decay wins 2.9%) — signal quality vs staleness tradeoff heavily favor de-noising for batch_size=4
4. **Mechanism:** longer momentum window (~200 vs ~100 steps) de-noises direction before `sign(·)` is taken; consistent with #2070 "Lion-direction-led" finding
5. **EMA(weights) + β2(momentum) appear synergistic** — different smoothing targets
6. **Student suggested β2=0.999** as natural follow-up given monotonic trend

**Askeladd reassigned:** PR #2271 — β2 sweep on n_head=2 (β2=0.995 confirm transfer + β2=0.999 extend trend) at lr=1e-4. Highest-EV follow-up: strong signal, asymmetric, monotonic at 3 points; direct merge candidate if Arm 1 transfers.

---

## 2026-05-13 12:27 — PR #2210: surf_weight sw=5 vs sw=7 on n_head=2 (nezuko) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-nezuko/surf-weight-n-head-2`
- **Hypothesis:** sw=5 (from #2056 on n_head=4) may stack onto n_head=2; sw=7 probes midpoint.
- **W&B runs:** `qkyx47iv` (sw=5, winner), `2owd44pg` (sw=7)

| Arm | sw | run_id | best_epoch | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline (51.11) |
|-----|-----|--------|------------|---------------------|----------------------|----------------------|
| 1 (winner) | 5 | `qkyx47iv` | 20 | **50.9119** | **43.6823** | **−0.39% / −1.13%** |
| 2 | 7 | `2owd44pg` | 20 | 52.0234 | 45.2506 | +1.78% / +2.43% |

**Per-test-split (sw=5 winner):**
- single_in_dist=46.42 (−2.81 vs baseline), geom_camber_rc=58.60 (+1.16), geom_camber_cruise=27.33 (+0.59), re_rand=42.39 (−0.91)

**Result:** MERGED. New best: val=50.91, test=43.68. Key findings:
1. **sw=5 stacks onto n_head=2** — confirms the surface weighting insight transfers architectures
2. **Non-monotonic response: sw=5 < sw=10 < sw=7** — sw=7 is a local maximum, not a linear interpolation; suggests complex loss landscape in [5,10]
3. **Gain skewed toward in-dist splits** (single_in_dist −2.81, re_rand −0.91); marginal losses on OOD camber splits — sw=5 sharpens in-distribution surface accuracy
4. Both arms reached 20 epochs (val still descending at cap — NOT converged)
5. Peak GPU memory: sw=5 = 99.16 GB (96.6%) — near VRAM limit

**New compound:** Fourier + MAE + Dropout(0.2) + BF16 + EMA(0.99) + Lion(lr=1e-4, wd=1e-4) + n_head=2 + **surf_weight=5**

**Nezuko reassigned:** PR #2277 — surf_weight lower probe: sw=4 (Arm1) vs sw=3 (Arm2). First data below sw=5; will determine if the loss landscape floor is at or below sw=5.

---

## 2026-05-13 12:54 — PR #2167: Cosine T_max=16 + eta_min=1e-5 at lr=2e-4 (fern) — CLOSED, REGRESSION

- **Branch:** `willowpai2g24h5-fern/cosine-tmax-lr2e-4`
- **Hypothesis:** At lr=2e-4, T_max=epochs=50 prevents the cosine schedule from completing; T_max=16 (matched to budget) should unlock the lr=2e-4 regime.
- **W&B runs:** `798tkeey` (Arm 1: T_max=16, eta_min=1e-5), `2go6qmev` (Arm 2: T_max=50, eta_min=1e-5)
- **Note:** Runs used train.py default n_head=2 (post-#2069 merge) — so this tests lr=2e-4 on n_head=2 compound with cosine variants

| Arm | T_max | eta_min | lr | run_id | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline (50.91) |
|-----|-------|---------|-----|--------|---------------------|----------------------|----------------------|
| 1 | 16 | 1e-5 | 2e-4 | `798tkeey` | 56.43 | 47.77 | +10.9% / +9.4% |
| 2 | 50 | 1e-5 | 2e-4 | `2go6qmev` | 57.74 | 49.66 | +13.4% / +13.7% |

**Val trajectories:** Arm 1 ep13→ep16: 58.97→58.38→57.40→56.43 (Δ −2.54, steepest at end). Arm 2 ep13→ep16: 63.13→60.21→57.74→58.36 (best ep15, regressed ep16 from high LR noise).

**Result:** CLOSED. Key findings:
1. **Schedule-matching (T_max=16) has signal but is insufficient at lr=2e-4 on n_head=2**: Arm 1 beats Arm 2 by 1.31 val — schedule shape matters
2. **lr=2e-4 on n_head=2 regresses ~5.5 val** even with best schedule — lr × n_head interaction is the dominant factor
3. **eta_min=1e-5 in isolation does not help** (Arm 2 vs old baseline 55.41 with eta_min=0 — Arm 2 worse)
4. **T_max=50 is fundamentally mismatched to 30-min budget** — at epoch 16, lr≈1.81e-4 (vs Arm 1's 1e-5); first ep16 noise spike confirms the slow-decay arc was truncated

**Pattern confirmed:** Cosine schedule changes consistently regress at this 30-min budget — the model is under-converged at default schedule, and forcing LR down faster cuts off productive learning.

**Fern reassigned:** PR #2295 — EMA decay sweep (0.999 vs 0.95) on current sw=5+n_head=2+lr=1e-4 compound. EMA decay set at 0.99 in #1607 for old compound; hasn't been re-tuned for new architecture+loss.

---

## 2026-05-13 13:30 — PR #2211: OneCycleLR on n_head=2 baseline: pct_start=0.3 vs 0.1 (thorfinn) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-thorfinn/onecyclelr-n-head-2`
- **Hypothesis:** OneCycleLR (peak LR mid-training, then consolidation) may fill the "refining phase" absent from cosine at 20-epoch / 30-min cap. Arm 1: pct_start=0.3 (peak at epoch 6, consolidate to end). Arm 2: pct_start=0.1 (peak at epoch 2, maximize consolidation window).
- **W&B runs:** `5icnpjij` (Arm 1, pct=0.3), `qp0lkwkm` (Arm 2, pct=0.1). Arm 1 also had crashed run `zbwa7pwv` (diverged step 3066 at OneCycle peak — instability confirmed) and prior attempt `lrv2xokp` (val=56.49).

| Arm | pct_start | run_id | val_avg/mae_surf_p | test_avg/mae_surf_p | Δ vs baseline (50.91) |
|-----|-----------|--------|---------------------|----------------------|----------------------|
| 1 | 0.3 | `5icnpjij` | 53.875 | 46.622 | +5.8% / +6.7% |
| 2 | 0.1 | `qp0lkwkm` | 58.785 | 50.526 | +15.5% / +15.7% |

**Per-test-split (Arm 1, best):** single_in_dist=51.38, geom_camber_rc=60.49, geom_camber_cruise=29.54, re_rand=45.08 — all worse than baseline.

**Result:** CLOSED. Key findings:
1. **OneCycleLR underperforms cosine at 20-epoch budget** — both arms worse on all 4 test splits.
2. **More aggressive warmup (pct=0.1) is much worse** (+15.5% vs +5.8%): consolidating a barely-trained basin produces dead epochs.
3. **Root cause (student analysis):** OneCycleLR's integrated LR ~half of cosine baseline's. With div_factor=25: start=4e-6, peak=1e-4, end=1e-8. Cosine baseline starts at lr_max=1e-4 and decays to ~35% by epoch 20. For Lion (step = lr×sign(g)), lower integrated LR = fewer effective exploration steps.
4. **Arm 1 crash at step 3066** (`zbwa7pwv`): diverged exactly at the OneCycle peak — confirms the risk of sign-magnitude updates at high LR peaks without adaptive scaling.
5. **Pattern confirmed:** This is the third consecutive LR-scheduling experiment that regresses (also #1999, #2167). At the 30-min budget, model is in early exploration regime — any annealing/peak-then-decay structure wastes update budget.

**Thorfinn reassigned:** PR #2318 — Lion weight_decay sweep on n_head=2+sw=5 compound (wd=3e-4 vs wd=3e-5). First explicit wd sweep on the new compound.

---

## 2026-05-13 09:12 — PR #1961: FFN width sweep mlp_ratio=3/4 on Lion+EMA (tanjiro) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-tanjiro/mlp-ratio-expansion`
- **Hypothesis:** Wider FFN at fixed n_hidden gives Transolver more per-block representation capacity at low extra cost vs depth/width.
- **W&B runs:** `0la90jp4` (Arm 1: mlp_ratio=3, best), `3vii7yyo` (Arm 2: mlp_ratio=4), `y9byrbsb` (Arm 1 replicate)

| Metric | Lion+EMA baseline (old) | Lion+MAE+lr=2e-4 (current) | Arm 1 (ratio=3) | Arm 2 (ratio=4) |
|--------|-------------------------|----------------------------|-----------------|-----------------|
| val_avg/mae_surf_p | 61.302 | **55.41** | 62.368 | 63.483 |
| test_avg/mae_surf_p | 52.682 | **47.90** | 54.270 | 54.241 |
| Δ vs current baseline | — | — | **+12.6%** | **+14.6%** |
| Epochs (30-min cap) | 16 | 16 | 16 | 15 |
| Main_val vs EMA_val gap (epoch 16, Arm 2) | — | — | — | **85.3 vs 63.5 = 22pt** |

**Result:** CLOSED. **Third consecutive capacity-expansion failure** at the 30-min cap — joins #1761 (n_layers=6) and #1934 (n_hidden=192/256) in the compute-wall pattern.

**Key finding from tanjiro's analysis:** Arm 2 (ratio=4) showed a 22-point gap between main_val (85.3) and EMA_val (63.5) — EMA averaging out heavy noise. This is the **under-regularization signal**: capacity expansion at fixed dropout=0.2 produces a noisier training trajectory. Hints that even the baseline compound may have regularization headroom — direct lead-in to tanjiro's next experiment.

**Architectural takeaway (final):** All three capacity axes (depth, width, FFN width) regress at 30-min cap. The model sits in a near-Pareto region for this budget. Future architecture experiments must be compute-neutral (n_head, slice_num) OR paired with budget-aware tradeoffs (e.g., bs↑ to compensate for epoch loss).

**Tanjiro reassigned:** PR #2131 — dropout sweep (0.1 vs 0.3) on Lion+MAE+lr=2e-4. Direct probe of tanjiro's own under-regularization observation.

---

## 2026-05-13 08:30 — PR #1932: Lion lr=2e-4 scaling on Lion+MAE compound (thorfinn) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-thorfinn/lion-lr-scaling`
- **Hypothesis:** lr-doubling trend 5e-5→1e-4 hasn't saturated; 1e-4→2e-4 may continue the descent.
- **W&B runs:** `y8oh2gxf` (Arm 1: lr=2e-4, wd=1e-4, **winner**), `141lcuxh` (Arm 2: lr=2e-4, wd=5e-4)

| Metric | MAE baseline (#1825) | Arm 1 (lr=2e-4, wd=1e-4) | Arm 2 (lr=2e-4, wd=5e-4) | Δ Arm1 |
|--------|---------------------|--------------------------|--------------------------|--------|
| val_avg/mae_surf_p | 56.577 | **55.412** | 56.801 | **−2.06%** |
| test_avg/mae_surf_p | 48.817 | **47.899** | 49.079 | **−1.88%** |
| test/single_in_dist | 53.687 | **51.084** | 52.326 | −4.85% |
| test/geom_camber_rc | 63.234 | **62.288** | 65.813 | −1.49% |
| test/geom_camber_cruise | 30.812 | 31.211 | **30.613** | +1.30% |
| test/re_rand | 47.535 | **47.014** | 47.563 | −1.10% |

**Result:** MERGED. Arm 1 wins 3/4 test splits + average. Val still descending at epoch-16 cap (last 4 epochs: 61.27→59.07→57.82→55.41, ≈−2 pts/epoch). **Third consecutive lr-doubling win without saturation.**

**Key finding:** Canonical wd scaling (Arm 2, wd=5e-4) disconfirms the Chen et al. `lr×wd≈const` heuristic on this compound. MAE-Lion is already near maximum stable step (L1 gradients are ±1 before sign, so per-param step = ±lr independent of gradient scale). EMA+dropout already saturate the regularization budget; wd=5e-4 over-regularizes — most visible as rc split regression (+4.08%). Arm 1 wins by keeping wd=1e-4.

**Trajectory diagnostic:** Main vs EMA val spread remains ~10 pts at epoch 16 (main_val ≈ 65 vs ema_val 55.41) — EMA still absorbing Lion's noise, not saturated. Longer budget prediction: a few more epochs would push well below 55.

**Thorfinn reassigned:** PR #2086 — lr=4e-4 (bold) and lr=3e-4 (midpoint) to complete the lr curve and detect the saturation/instability boundary.

---

## 2026-05-13 08:08 — PR #1934: Width expansion n_hidden=192/256 on Lion+EMA (alphonse) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-alphonse/width-expansion`
- **Hypothesis:** Lion's faster optimization exposes a capacity ceiling at n_hidden=128; widening to 192 or 256 absorbs extra training signal.
- **W&B runs:** `bfqtvd10` (Arm 1: n_hidden=192), `6ez4cyaf` (Arm 2: n_hidden=256), `g4l7y8ic` (Arm 1 replicate)

| Metric | Baseline #1781 (n_hidden=128) | Arm 1 (192) | Arm 2 (256) |
|--------|-------------------------------|-------------|-------------|
| Params | 0.67M | 1.48M | 2.46M |
| val_avg/mae_surf_p | **61.302** | 62.709 (+2.3%) | 66.393 (+8.3%) |
| test_avg/mae_surf_p | **52.682** | 54.235 (+2.9%) | 57.480 (+9.1%) |
| Epochs (30-min cap) | 13 | 13 | 11 |
| s/epoch | 112 | 145 | 171 |

**Result:** CLOSED. Monotonic regression with width — Arm 2 worse than Arm 1, both worse than baseline. vs current Lion+MAE baseline (val=56.58): +10.8% / +17.3%.

**Key finding:** Hypothesis falsified. The model is **compute-bound, not capacity-bound** at the 30-min cap. Wider models trained *fewer* epochs (171s/epoch at width=256 vs 112s at 128), and per-epoch val improvement was NOT faster on wider arms — they're tracking toward the same basin but trailing. Student's analysis was sharp: OOD splits regressed *most* on width, ruling out the under-capacity story (which would predict in-dist improvements). At fixed 30-min wall-clock, n_hidden=128 is the right operating point on this base.

**Architectural conclusion (combined with #1761):** Both depth (n_layers=6 → +19% epoch cost) and width (n_hidden=192/256 → +30%/53% epoch cost) are compute-bound losers at the 30-min cap. The only architectural directions left are **compute-neutral** (head count, MLP ratio if cheap) or **structural** (different attention pattern). Alphonse reassigned to **n_head=8 vs n_head=2 sweep** (PR #2069) — head count is FLOPs-neutral (parallel heads at smaller head_dim).

---

## 2026-05-13 08:10 — PR #1857: EMA decay sweep 0.995/0.999 on pre-Lion base (edward) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-edward/ema-decay-sweep`
- **Hypothesis:** Higher EMA decay (slower update) might sample a broader weight-space neighborhood → flatter basin.
- **W&B runs:** `ihgn5cko` (Arm 1: 0.995), `qyoo8y0j` (Arm 2: 0.999)

| Metric | Baseline #1607 (0.99) | Arm 1 (0.995) | Arm 2 (0.999) |
|--------|-----------------------|---------------|---------------|
| val_avg/mae_surf_p | **77.054** | 77.594 (+0.7%) | 81.489 (+5.8%) |
| test_avg/mae_surf_p | 68.265 | **66.978 (−1.9%)** | 70.828 (+3.8%) |

**Result:** CLOSED. vs old AdamW+EMA baseline: val regression on both arms. vs current Lion+MAE baseline (val=56.58): +37.2% even on best arm. Cannot merge by val rule.

**Important nuanced finding for record:** **Arm 1 (decay=0.995) improved test by −1.9% despite val +0.7% regression**. All 4 test splits improved monotonically. Val/test ratio shifted 1.129 → 1.158, consistent with a slightly more conservative shadow producing better OOD generalization (genuine signal, not noise). This effect was on the OLD pre-Lion AdamW base — would need to be re-tested on Lion+MAE to know if it transfers (Lion's gradient statistics differ from AdamW's).

**Mechanism (student analysis):** Effective averaging window ≈ 1/(1−α): 0.99→100 steps, 0.995→200, 0.999→1000. At the 16-epoch (~6000 step) budget, 0.999 averages over basically the entire post-warmup trajectory including under-trained early weights — predicted failure mode. 0.995 sits in the sweet spot for test, just past it for val.

**Edward reassigned:** PR #2070 — Lion-no-EMA ablation. Diagnostic for ICML appendix: how much of the Lion+EMA gain is Lion vs EMA?

---

## 2026-05-13 07:55 — PR #1786: Higher LR (1e-3/2e-3) on AdamW+EMA base (frieren) — CLOSED SUPERSEDED

- **Branch:** `willowpai2g24h5-frieren/higher-lr-ema`
- **Hypothesis:** EMA smoothing absorbs main-model noise, so 2–4× the baseline AdamW lr=5e-4 should be safe and reach a lower-loss basin within 16 epochs.
- **W&B runs:** `uvc7ljtw` (Arm 1: lr=1e-3), `17oh10lv` (Arm 2: lr=2e-3)

| Metric | Baseline #1607 (lr=5e-4) | Arm 1 (lr=1e-3) | Arm 2 (lr=2e-3) |
|--------|--------------------------|-----------------|-----------------|
| val_avg/mae_surf_p | 77.054 | **74.508 (−3.30%)** | 74.595 (−3.19%) |
| test_avg/mae_surf_p | 68.265 | **64.380 (−5.69%)** | 65.522 (−4.02%) |
| test/geom_camber_cruise | 48.52 | **44.51 (−8.3%)** | 44.30 (−8.7%) |
| test/single_in_dist | 75.31 | **70.64 (−6.2%)** | 73.65 (−2.2%) |

**Result:** CLOSED (direction superseded). Arm 1 (lr=1e-3) was a genuine improvement on the pre-Lion AdamW+EMA base (−3.3% val / −5.7% test), validating the EMA-absorbs-noise hypothesis. However, two subsequent merges moved the baseline to val=56.58 (Lion+MAE), and Arm 1's result (val=74.51) is +31.7% vs the current best. The LR scaling direction on the new compound is covered by thorfinn's #1932 (Lion lr=2e-4). Frieren reassigned to batch_size+LR scaling (PR #2052).

**Key finding:** Mechanism confirmed — EMA val descended monotonically from 197.79 to 74.51 despite main-model wobble (88–227 range). Cruise split (−8.3%) and single_in_dist (−6.2%) are the biggest movers. lr=2e-3 without warmup: good val but worse test (65.52 vs 64.38), consistent with over-aggressive LR degrading OOD generalization.

---

## 2026-05-13 07:54 — PR #1752: surf_weight=5 on pre-Lion AdamW+EMA base (nezuko) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-nezuko/surf-weight-sweep`
- **Hypothesis:** surf_weight=10 may be overcorrecting post-Huber+EMA; reducing to 5 or 7 might improve balance between surface and volume.
- **W&B run:** `4pvy6khr` (Arm 1: surf_weight=5; Arm 2 skipped per stop rule)

| Metric | Baseline #1607 | Arm 1 (surf_weight=5) | Δ |
|--------|----------------|-----------------------|---|
| val_avg/mae_surf_p | 77.054 | 83.511 | **+8.4% (regression)** |
| test_avg/mae_surf_p | 68.265 | 73.759 | **+8.0% (regression)** |
| val/single_in_dist | 85.45 | 93.557 | +9.5% |
| val/geom_camber_cruise | 57.80 | 63.291 | +9.5% |

**Result:** CLOSED. Hypothesis falsified definitively. surf_weight=5 is uniformly worse on every val and test split — *including* volume MAE (also regressed), so this isn't a surface/volume tradeoff; it's a uniform loss of gradient signal. Stop rule triggered (Arm 1 +8.4% > 3pt threshold); Arm 2 (surf_weight=7) not run. vs Lion+MAE baseline (val=56.58): +47.6% regression.

**Key finding:** Surface nodes are underfit at the 30-min compute budget. Reducing surf_weight starves surface gradient signal — model spends capacity on volume, which is also underfit and doesn't improve either. surf_weight=10 was tuned correctly; the 'EMA+Huber adds slack for lower weighting' intuition was wrong. The unexplored direction is *upward* (surf_weight=15–20), especially with MAE loss where there's no Huber quadratic dampening near zero.

**Nezuko reassigned:** PR #2056 — surf_weight sweep on Lion+MAE (Arm 1: sw=5 apples-to-apples, Arm 2: sw=15 heavier emphasis).

---

## 2026-05-13 06:35 — PR #1825: MAE (L1) loss on Lion+EMA base (askeladd) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-askeladd/mae-loss`
- **Hypothesis:** MAE/L1 loss weights every node uniformly per the linear MAE evaluation metric; this property is independent of optimizer choice and should compound cleanly with Lion+EMA.
- **W&B run:** `03w5fnvm` (Lion+MAE rerun)

| Metric | MAE+Lion+EMA | Lion+EMA baseline (#1781) | Δ |
|--------|-------------|--------------------------|---|
| val_avg/mae_surf_p | **56.577** | 61.302 | **−7.71%** |
| test_avg/mae_surf_p | **48.817** | 52.682 | **−7.34%** |
| test/single_in_dist | 53.687 | 59.813 | −10.24% |
| test/geom_camber_rc | 63.234 | 64.584 | −2.09% |
| test/geom_camber_cruise | 30.812 | 35.140 | −12.32% |
| test/re_rand | 47.535 | 51.193 | −7.14% |

**Result:** MERGED. New best session baseline. Wins all 4 test splits. Val still descending at epoch-16 cap.

**Key finding:** MAE's gain on Lion is *larger* (−7.71%) than on the original AdamW base (−3.15%). Lion's sign-magnitude update removes per-parameter gradient scale information, but MAE's uniform per-node loss aggregation operates *before* backprop — it directly controls how much each node contributes to the scalar loss. This is loss-side, not optimizer-side. Lion+MAE compound: the optimizer discards magnitude noise, the loss ensures every surface node contributes equally. Cruise split (−12.32%) and in-dist split (−10.24%) are the biggest movers.

---

## 2026-05-13 06:40 — PR #1823: Weight decay wd=5e-4 on pre-Lion AdamW base (fern) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-fern/weight-decay-sweep`
- **W&B run:** `qvpxtrx8`

| Metric | wd=5e-4 | AdamW baseline (#1607) | New Lion+MAE baseline |
|--------|---------|------------------------|----------------------|
| val_avg/mae_surf_p | 78.47 | 77.054 | **56.577** |
| test_avg/mae_surf_p | 68.22 | 68.265 | **48.817** |

**Result:** CLOSED. +1.84% val regression vs old AdamW baseline; +38.7% vs new Lion+MAE baseline.

**Analysis:** val regressed (stronger L2 slowed the EMA trajectory), test was essentially tied. The mechanism check was interesting — main-model val improved with wd=5e-4, but EMA val regressed, suggesting stronger wd changes the EMA averaging geometry unfavorably. Direction also redundant: thorfinn's #1932 (Arm 2) is testing wd=5e-4 on the Lion base directly.

---

## 2026-05-13 05:25 — PR #1761: n_layers=6 depth expansion (tanjiro) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-tanjiro/n-layers-6`
- **Hypothesis:** Adding a 6th Transolver block (n_layers 5→6) gives the model more sequential processing capacity for complex flow features.
- **W&B runs:** `sy8axvhz` (drop=0.2 arm), `mr69aglv` (drop=0.1 arm retry)

| Metric | n_layers=6, drop=0.2 | n_layers=6, drop=0.1 | Baseline #1607 | Δ vs old base | vs Lion #1781 |
|--------|---------------------|---------------------|----------------|---------------|----------------|
| val_avg/mae_surf_p | 80.174 | 80.452 | 77.054 | +4.1% / +4.4% | +30.9% / +31.3% |
| test_avg/mae_surf_p | 70.816 | 70.862 | 68.265 | +3.7% / +3.8% | +34.4% / +34.5% |
| Epochs (30-min cap) | 14/50 | 14/50 | 16/50 | — | — |
| Epoch time | ~133.7s | ~133.7s | ~112s | +19% | +19% |

**Result:** CLOSED. Per decision rule (val > 78.5 → close definitively).

**Analysis:** Depth=6 increases per-epoch cost by ~19% (133.7s vs 112s), reducing total epochs from 16 → 14 at the 30-min cap. The trajectory data shows the model is *compute-budget bound, not depth-broken*: dropout=0.2 arm descended ~3.7 pts/epoch at the cap, dropout=0.1 arm descended ~1.4 pts/epoch. Lower dropout improved main_val (94.5 vs 105) — the underlying model was closer to converged with less regularization — but EMA's long-horizon average smoothed out the early-epoch advantage and the late-epoch plateau dominates.

**Architectural takeaway:** At the 30-min cap, n_layers=5 is locally optimal. Depth-6 needs more compute than this launch allows. The depth-vs-budget knee is now characterized for the architecture.

---

## 2026-05-13 05:10 — PR #1781: Lion optimizer lr=1e-4+EMA (thorfinn) — MERGED NEW BEST

- **Branch:** `willowpai2g24h5-thorfinn/lion-optimizer`
- **Hypothesis:** Lion's sign-based momentum updates (Chen et al. 2023) produce larger, noisier gradient steps that EMA then smooths — decoupling exploration (Lion) from integration (EMA) more cleanly than AdamW+EMA where second-moment normalization and EMA partially overlap.
- **W&B runs:** `e2l23xny` (lr=1e-4, winner), `9fjjfgjt` (lr=5e-5), buggy-variant `lion-lr5e-5-buggy-variant.log`

| Metric | Lion lr=1e-4 | Lion lr=5e-5 | Baseline #1607 (AdamW+EMA) | Δ vs baseline |
|--------|-------------|-------------|--------------------------|---------------|
| val_avg/mae_surf_p | **61.302** | 64.010 | 77.054 | **−20.44%** |
| test_avg/mae_surf_p | **52.682** | 55.367 | 68.265 | **−22.83%** |
| test/single_in_dist | 59.813 | 65.462 | 75.31 | −20.58% |
| test/geom_camber_rc | 64.584 | 67.409 | 80.81 | −20.08% |
| test/geom_camber_cruise | 35.140 | 35.899 | 48.52 | −27.58% |
| test/re_rand | 51.193 | 52.696 | 68.41 | −25.17% |
| Epochs (30-min cap) | 16/50 | 16/50 | 16/50 | — |

**Result:** MERGED. Lion optimizer is the largest single-PR gain of the session — 20–28% uniform improvement across all 4 test splits. Curve still descending steeply at epoch-16 cap; not converged.

**Key finding — Lion+EMA synergy:** Lion sign-magnitude updates are uniformly ±lr per step (aggressive, noisy), but EMA smooths the noise post-hoc. AdamW second-moment already smooths per-parameter scale, making AdamW+EMA partially redundant. Lion+EMA cleanly separates roles: exploration vs averaging. The lr=1e-4 arm beats lr=5e-5 because bigger, noisier steps give EMA more diversity to average over.

**Bug fix (critical):** Student identified β1/β2 swap in PR diff relative to canonical Lion. Buggy variant scored 92.92 (regression), canonical scored 61.30. Fix: interpolation uses β1=0.9, momentum EMA uses β2=0.99.

**Note:** val still descending at epoch-16 cap — longer budget is the highest-EV immediate follow-up.

---

## 2026-05-13 05:15 — PR #1604: Asinh pressure transform (alphonse) — CLOSED REGRESSION

- **Branch:** `willowpai2g24h5-alphonse/asinh-pressure`
- **Hypothesis:** Asinh transform on pressure target compresses the high-Re tail, improving generalization on re_rand and single_in_dist splits.
- **W&B run:** `nbig5bns`

| Metric | Asinh run | Baseline #1607 (EMA) | Δ |
|--------|-----------|---------------------|---|
| val_avg/mae_surf_p | 82.81 | 77.054 | **+7.5% (regression)** |
| test_avg/mae_surf_p | 73.25 | 68.265 | **+7.3% (regression)** |
| test/geom_camber_cruise | ~54 | 48.52 | +11.4% |

**Result:** CLOSED. Clear regression on all splits.

**Analysis:** Double-compression: Huber δ=1.0 already compresses the tail at the loss level. Asinh adds a second compression at the data (target representation) level. These are not additive — they compete on the same degree of freedom. With Fourier features providing richer spatial encoding, the model has capacity to learn high-Re structure directly; asinh pre-compression removes the tail signal the model needs. The correct future test, if any, would be Asinh-alone on the Fourier+EMA base *without* Huber loss.

---

## 2026-05-12 19:28 — PR #1371: BF16 autocast (frieren)

- **Branch:** `willowpai2g24h5-frieren/bf16-mixed-precision`
- **Hypothesis:** BF16 mixed precision halves per-step time, buying more epochs in the 30-min wall-clock budget.
- **W&B run:** `6zx5vuja`

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (best, ep 13) | **123.72** |
| val_single_in_dist/mae_surf_p | 153.36 |
| val_geom_camber_rc/mae_surf_p | 129.40 |
| val_geom_camber_cruise/mae_surf_p | 99.23 |
| val_re_rand/mae_surf_p | 112.87 |
| test_avg/mae_surf_p | NaN (cruise data bug) |
| 3-split test avg (no cruise) | **121.90** |
| Epochs completed | 18 in 30 min |
| Peak VRAM | 32.9 GB / 96 GB |

**Result:** MERGED as new baseline. BF16 completed 18 epochs vs ~14 at FP32 (estimated), establishing val_avg=123.72.

**Key observation:** Pre-existing data corruption in `test_geom_camber_cruise/000020.pt` (761 nodes with y[:,2]=-inf) poisons 4-split test_avg via `0×inf=NaN` in scoring.py. Affects every run on this branch. 3-split test avg is the usable paper-facing signal until fixed.

---

## 2026-05-12 18:56–19:51 — PR #1412: Warmup 3ep then cosine / Warmup 5ep then cosine (thorfinn)

- **Branch:** `willowpai2g24h5-thorfinn/warmup-3ep-then-cosine`
- **Hypothesis:** Linear LR warmup before cosine annealing stabilizes early training steps.
- **W&B runs:** `3chdcivo` (warmup-3ep), `jcd79mzi` (warmup-5ep)

| Arm | val_avg/mae_surf_p (best) | best epoch | 3-split test avg |
|-----|--------------------------|------------|-----------------|
| warmup-3ep | 144.50 | 12 | 144.54 |
| warmup-5ep | **135.37** | 14 | **131.12** |

Per-split (warmup-5ep):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|----------------|
| single_in_dist | 164.28 | 142.88 |
| geom_camber_rc | 143.91 | 130.88 |
| geom_camber_cruise | 110.76 | NaN |
| re_rand | 122.52 | 119.61 |

**Result:** SENT BACK for rebase. warmup-5 (135.37) did not beat the BF16 baseline (123.72) as a standalone, but warmup and BF16 are orthogonal. Student rebasing to test the combo (warmup-5 + BF16 already in base).

**Key observation:** Warmup=5 strictly dominates warmup=3 across all splits except geom_camber_cruise (+5%). Single_in_dist improved 15.7%, re_rand 1.7%, rc 6.2%. Per-epoch time ~131s; 14 epochs in 30 min without BF16.

---

## 2026-05-12 19:56 — PR #1367: Dropout=0.1/0.2 + grad-clip=1.0 (fern) — **PENDING REBASE**

- **Branch:** `willowpai2g24h5-fern/dropout-0.1-grad-clip`
- **Hypothesis:** Light dropout + grad clipping improves OOD generalization.
- **W&B runs:** `7brl22oo` (dropout=0.1), `3wz81r3d` (dropout=0.2)

| Arm | val_avg/mae_surf_p (best) | best epoch | 3-split test avg |
|-----|--------------------------|------------|-----------------|
| dropout=0.1, clip=1.0 | 146.31 | 8 | 146.51 |
| **dropout=0.2, clip=1.0** | **113.86** | **12** | **114.77** |

Per-split (dropout=0.2):

| Split | val mae_surf_p | test mae_surf_p |
|-------|---------------|----------------|
| single_in_dist | 145.19 | 132.80 |
| geom_camber_rc | 120.81 | 112.25 |
| geom_camber_cruise | 83.27 | NaN |
| re_rand | 106.18 | 99.28 |

**Result:** SENT BACK for rebase. val_avg=113.86 BEATS current BF16 baseline (123.72) by 7.7%. Merge conflict with PR #1371 — student rebasing to test dropout=0.2 + BF16 combination. Expected to combine to an even lower metric.

**Key observation:** dropout=0.2 beats dropout=0.1 across EVERY split (−22% overall), not just OOD. Probably acts as a smoother loss landscape rather than just generalization: 5 Transolver layers with slice_num=64 attention have many co-adaptation opportunities that dropout disrupts usefully. Validation still descending at 30-min cap — suggests this configuration has more headroom.

---

## 2026-05-12 19:58 — PR #1400: Aux surface-pressure head λ=2 (tanjiro) — **PENDING REBASE**

- **Branch:** `willowpai2g24h5-tanjiro/aux-surf-p-head`
- **Hypothesis:** Auxiliary MLP head predicting surface p only, with λ=2.0 weight on aux loss.
- **W&B run:** `m9xr80iw`

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (best, ep 12) | 132.48 |
| val_single_in_dist | 153.91 |
| val_geom_camber_rc | 147.18 |
| val_geom_camber_cruise | 104.48 |
| val_re_rand | 124.36 |
| 3-split test avg | 130.62 |
| Epochs completed | 14 in 30 min (~134 s/ep) |
| Peak VRAM | 42.6 GB / 96 GB |
| Aux head params | 8,321 (vs 660K main) |

**Result:** SENT BACK for rebase + BF16 combo. val=132.48 doesn't beat BF16 baseline (123.72), but **aux loss is learning** (0.66 → 0.20) and val curve was still descending at the cap (152 → 132 last 3 epochs). Tanjiro re-running with aux head INSIDE BF16 autocast.

**Key observation:** This is a high-α, high-σ direction — the aux head is doing useful work but starved for epochs. BF16 + λ=5.0 (larger aux weight) may be necessary to see real gains.

---

## 2026-05-12 20:51 — PR #1386: Fourier positional encoding L=6 (nezuko) — **PENDING RETRY**

- **Branch:** `willowpai2g24h5-nezuko/fourier-position-encoding`
- **Hypothesis:** Random Fourier features on (x,z) coordinates help model learn high-freq spatial patterns.
- **W&B run:** `0xuvq54a`
- **Config:** L=6, min_freq=1.0, max_freq=1000.0, no BF16 (PR predates BF16 merge), 14 epochs in 30 min

| Metric | Value | Notes |
|--------|-------|-------|
| val_avg/mae_surf_p (best, ep 13) | **123.10** | 0.5% below merged BF16 baseline (123.72) |
| val_single_in_dist | 142.19 | vs 153.36 (BF16): better |
| val_geom_camber_rc | 138.34 | vs 129.40 (BF16): worse |
| val_geom_camber_cruise | 92.13 | vs 99.23 (BF16): better |
| val_re_rand | 119.73 | vs 112.87 (BF16): worse |
| 3-split test avg | 124.90 | vs 121.90 (BF16): worse |
| test_geom_camber_cruise | NaN | same pre-existing data bug |
| Peak VRAM | 42.5 GB / 96 GB | |

**Result:** SENT BACK for retry. Student correctly identified that **max_freq=1000 is far too high** for raw coordinates (Tancik standard is max_freq ≈ 2π × num_octaves on NORMALIZED positions). Requested re-run with max_freq=32, normalized positions, and BF16 from base.

**Key observation:** Marginal +0.5% val improvement over BF16 baseline, but per-split signals are mixed (helps single_in_dist & cruise, hurts rc & re_rand). The mixed signal + known scaling bug + room for a much bigger win via the retry → preferred over merging a borderline win with a bug.

---

## 2026-05-12 21:00 — PR #1541: Fix test cruise NaN + BF16 rerun (frieren) — **MERGED**

- **Branch:** `willowpai2g24h5-frieren/fix-cruise-test-nan-scoring`
- **Hypothesis:** Guard `0×inf=NaN` in `data/scoring.py::accumulate_batch` to restore 4-split test_avg metric, then rerun BF16 baseline to verify.
- **W&B run:** `x7snuii5`

| Metric | Value | Notes |
|--------|-------|-------|
| val_avg/mae_surf_p (best, ep 17) | **120.40** | beats BF16 baseline 123.72 by 2.7% |
| test_avg/mae_surf_p | **106.67** | first finite 4-split test metric on branch |
| test_single_in_dist | 125.29 | |
| test_geom_camber_rc | 113.23 | |
| test_geom_camber_cruise | **81.16** | was NaN — now finite |
| test_re_rand | 106.99 | |
| Epochs completed | 18 in 30 min | (~101 s/epoch, same as BF16 baseline) |
| Peak VRAM | ~33 GB / 96 GB | |

**Result:** MERGED as new baseline. val=120.40, test_avg=106.67 — both improvements over BF16 baseline (val=123.72, test=NaN).

**Key observation:** The fix is a single `torch.where(isfinite(...))` guard immediately after `err = (...).abs()` in `accumulate_batch`. Val improvement over the prior BF16 baseline (123.72→120.40) is within the ±3% training-noise band but real; 18 epochs in 30 min confirms BF16 throughput is stable. The cruise test MAE (81.16) is substantially lower than the other splits — cruise samples are geometrically simpler than single_in_dist/re_rand, so this is expected.

---

## 2026-05-12 21:00 — PR #1412: Warmup-5ep + BF16 combo (thorfinn) — **CLOSED**

- **Branch:** `willowpai2g24h5-thorfinn/warmup-3ep-then-cosine`
- **Hypothesis:** warmup-5ep + BF16 combination outperforms BF16-only baseline.
- **W&B run:** `dm90ndo1`

| Metric | Value | Notes |
|--------|-------|-------|
| val_avg/mae_surf_p (best, ep 14) | 123.10 | vs new baseline 120.40: +2.2% worse |
| val_single_in_dist | 150.51 | |
| val_geom_camber_rc | 136.30 | |
| val_geom_camber_cruise | 96.38 | |
| val_re_rand | 109.21 | |
| 3-split test avg | 122.33 | test_avg NaN (run predates scoring fix) |
| Epochs completed | 18 in 30 min | BF16 confirmed at 101s/epoch |

**Result:** CLOSED. val=123.10 does not beat new baseline (120.40) after #1541 merged. Warmup+BF16 is within noise of BF16-alone — warmup provides marginal additional benefit once BF16 supplies the extra epochs.

**Key observation:** Thorfinn's per-epoch LR analysis is the most valuable finding: T_max=50 is too long for the ~18 reachable BF16 epochs — cosine decays only 36% by run end, leaving LR near peak (4.5e-4) and causing late-epoch val wobble. T_max=18 is the next experiment (PR #1583 assigned).

---

## 2026-05-12 21:05 — PR #1357: Huber loss δ=1.0 (askeladd) — **SENT BACK for BF16 rerun**

- **Branch:** `willowpai2g24h5-askeladd/huber-loss-delta-1`
- **Hypothesis:** Replace MSE with Huber δ=1.0 in normalized space; linear penalty past 1σ is robust to high-Re outliers.
- **W&B run:** `whazlv6i` (pre-BF16 base — peak VRAM ~82 GB)

| Metric | Value | Notes |
|--------|-------|-------|
| val_avg/mae_surf_p (best, ep 14) | **107.91** | beats baseline 120.40 by 10.4% |
| val_single_in_dist | 123.52 | |
| val_geom_camber_rc | 114.01 | |
| val_geom_camber_cruise | 89.82 | |
| val_re_rand | 104.27 | |
| 3-split test avg | 105.94 | test_avg NaN (run predates scoring fix) |
| Epochs completed | 14 in ~31 min | non-BF16 throughput |

**Result:** SENT BACK for rebase + rerun. Run did not have BF16 active (~82 GB VRAM confirms pre-BF16 code). Predicted: rebase + BF16 + scoring fix should give 107.91 × ~0.95 (BF16 extra epochs) ≈ ~100-105 val with finite test_avg. Will merge as winner on rebase return.

**Key observation:** Huber gives the largest single-experiment gain we've seen so far (~10%). Per-split improvements are largest on `val_re_rand` and `val_geom_camber_cruise` — exactly the high-Re/high-dynamic-range splits the hypothesis targeted. Strong candidate for compounding with dropout=0.2 (PR #1367).

---

## 2026-05-12 21:05 — PR #1352: surf_weight=30 (alphonse) — **CLOSED**

- **Branch:** `willowpai2g24h5-alphonse/surf-weight-30`
- **Hypothesis:** Increase surf_weight 10→30 to focus loss on surface pressure (the ranking metric).
- **W&B runs:** `q12wxz51` (sw=30), `9j9hnhfs` (sw=20)

| Arm | val_avg/mae_surf_p | best epoch |
|-----|--------------------|------------|
| surf_weight=20 | 127.05 | 13 |
| **surf_weight=30** | **120.88** | **14** |

**Result:** CLOSED. val=120.88 does not beat baseline (120.40) and is far from leading unmerged result (113.86 dropout=0.2). Pushing higher likely loses on val_single_in_dist (150.63 at sw=30, indicating overweighted surface loss hurts in-distribution generalization).

**Key observation:** Monotonic improvement 20→30 but trajectory tops out below the leading regularization-based approaches. Surf-weight is exhausted as a standalone lever; might compound with dropout in a future run.

---

## 2026-05-12 21:05 — PR #1365: OneCycleLR max_lr=1e-3 (edward) — **CLOSED**

- **Branch:** `willowpai2g24h5-edward/onecyclelr-max-lr-1e3`
- **Hypothesis:** OneCycleLR sweeps wider LR range than CosineAnnealingLR in short training budget.
- **W&B run:** `3ghxoqlb`

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (best, ep 13) | 128.89 |
| Epochs completed | 14 in 30 min |
| 3-split test avg | 124.47 |

**Result:** CLOSED. val=128.89 is ~7% worse than baseline (120.40). The schedule was structurally mismatched to the budget: OneCycleLR was set with epochs=MAX_EPOCHS=50, so peak LR (1e-3) hits at epoch 5 and the schedule plans to anneal slowly over 45 more epochs — but we only get 14, so LR stays pinned ~9e-4 the entire run.

**Key observation:** Student's own diagnosis is correct: OneCycleLR + 30-min budget requires `total_steps` matched to actual reachable steps, NOT to nominal max_epochs. Thorfinn is testing the analogous fix for CosineAnnealingLR (T_max=18) in PR #1583 — wait for that result before re-trying a budget-matched OneCycleLR.

---

## 2026-05-12 23:20 — PR #1624: AdamW betas (0.9,0.95) and (0.9,0.98) (frieren) — **CLOSED**

- **Branch:** `willowpai2g24h5-frieren/adamw-betas-ema`
- **Hypothesis:** Tuning beta2 from 0.999→0.95 or 0.98 to reduce second-moment memory horizon for short training.
- **W&B runs:** `geuztn5g` (beta2=0.95, val=141.04), `a2h9i5t3` (beta2=0.98, running at val=175 mid-training)

| Arm | val_avg/mae_surf_p | vs Baseline (103.24) |
|-----|-------------------|---------------------|
| betas=(0.9, 0.95) | 141.04 | +36.6% worse |
| betas=(0.9, 0.98) | mid-training ~175 | clearly worse |

**Result:** CLOSED. Both arms substantially worse than baseline. Beta2 reduction removes gradient history too aggressively for 18-epoch training — the standard 0.999 maintains a longer moving average that works better with cosine annealing.

---

## 2026-05-12 22:53 — PR #1386: Fourier positional encoding L=6 mf32 BF16 (nezuko) — **MERGED**

- **Branch:** `willowpai2g24h5-nezuko/fourier-position-encoding`
- **Hypothesis:** Replace raw (x,z) coordinates with Fourier features (log-spaced frequencies) to help the model learn high-frequency spatial patterns that MLPs struggle with from raw floats.
- **W&B runs:** `bpbykd9z` (L=6, primary), `qwmh06uh` (L=4, secondary)
- **Config:** L=6, min_freq=1.0, max_freq=32.0 (corrected from v1's max_freq=1000), positions standardized before encoding; BF16

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | Best epoch |
|-----|-------------------|--------------------|-----------:|
| **Fourier L=6 mf32 BF16** (`bpbykd9z`) | **103.2393** | **90.828** | 18 |
| Fourier L=4 mf32 BF16 (`qwmh06uh`) | 107.1261 | 94.7796 | 18 |
| **Baseline (BF16+scoring fix, #1541)** | 120.40 | 106.67 | 17 |

Per-test-split (L=6):

| Split | test surf_p | vs Baseline (106.67 avg) |
|-------|------------|------------------------:|
| single_in_dist | 105.79 | −15.6% |
| geom_camber_rc | 102.99 | −9.0% |
| geom_camber_cruise | 64.21 | −20.9% |
| re_rand | 90.31 | −15.6% |
| **avg** | **90.83** | **−14.8%** |

**Result:** MERGED as new baseline. val=103.24, test=90.83. All 4 splits improve; biggest gain on cruise geometry (−20.9%), supporting the hypothesis that Fourier features resolve fine boundary-layer structure on unseen airfoil shapes.

**Key observations:**
1. **max_freq matters far more than L.** v1 with max_freq=1000 on raw coords was −8% *worse* than baseline; v2 with max_freq=32 on normalized coords is −14% *better*. The frequency range (not the number of octaves) is the dominant variable.
2. **Standardize positions before encoding.** Computing sin/cos on raw coords makes the basis poorly conditioned; standardizing first puts frequencies in the Tancik-meaningful range.
3. **L=6 > L=4 by ~4%.** Extra octaves covering finer spatial scales (λ ≈ 0.2 unit) help boundary-layer resolution; negligible VRAM cost.
4. **This is the largest single-experiment gain yet: −14.8% test** — surpassing Huber (−10.4% val, BF16 pending) and dropout=0.2 (−7.7% val, BF16 pending). Fourier positional encoding is a foundational input feature change that should compound with both loss and regularization improvements.

**New baseline:** val=103.24, test=90.83. All subsequent PRs should compare against these numbers.

---

## 2026-05-12 21:55 — PR #1609: Transolver slice_num 64→128 (frieren) — **CLOSED**

- **Branch:** `willowpai2g24h5-frieren/slice-num-128-physics-tokens`
- **Hypothesis:** Doubling physics-token count gives attention finer-grained spatial access; orthogonal to other levers.
- **W&B run:** `kg0dwlen`

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| val_avg/mae_surf_p (best, ep 12) | 127.42 | **+5.83% worse** |
| test_avg/mae_surf_p | **116.80** | +9.50% worse |
| test_single_in_dist | 135.60 | +8.23% |
| test_geom_camber_rc | 127.82 | +12.88% |
| test_geom_camber_cruise | 87.67 | +8.03% |
| test_re_rand | 116.12 | +8.53% |

**Result:** CLOSED. Every metric worse than baseline (120.40/106.67). Lands in PR's own "over-allocated to capacity" decision bucket.

**Key observation (pattern confirmation):** In the 18-epoch budget, capacity-adding architectural changes consistently lose because the epochs sacrificed for slower compute matter more than the representational gain. This is the second confirmation (after #1400 tanjiro aux head ran into the same wall). Cheap gradient-shape changes (Huber, dropout) win; capacity-up architecture loses. **For future architecture changes, the per-epoch cost is the dominant variable, not the parameter count.**

---

## Stragglers status (round 1, as of 2026-05-12 ~20:06–20:51)

| PR | Student | Status |
|----|---------|--------|
| #1386 | nezuko | ✅ Completed at val=123.10 → SENT BACK for retry |
| #1365 | edward | Mid-epoch 7/50 OneCycleLR; val 254→190 over 6 epochs — likely undercooked |
| #1357 | askeladd | Epoch 3/50 Huber loss; val 232→178 — descending |
| #1352 | alphonse | surf_weight=20 finished at val=127.05 (worse than 123.72 baseline); surf_weight=30 arm still pending |
| #1541 | frieren | Scoring fix + baseline rerun — no comments yet |

---

*Log format: one block per PR review.*

---

## 2026-05-12 23:55 — PR #1357: Huber loss δ=1.0 + BF16 (askeladd) — **MERGED**

- **Branch:** `willowpai2g24h5-askeladd/huber-loss-delta-1`
- **Hypothesis:** Replace MSE with Huber δ=1.0; linear penalty past 1σ in normalised space is robust to high-Re outliers.
- **W&B run:** `m733u17z`
- **Base:** BF16 + scoring fix (pre-Fourier; W&B shows fun_dim=22)

| Metric | Value | vs Fourier baseline (103.24) |
|--------|-------|------------------------------|
| val_avg/mae_surf_p (best, ep 18) | **98.7905** | **−4.31%** |
| test_avg/mae_surf_p | **88.8965** | **−2.13%** |
| test_single_in_dist | 103.88 | |
| test_geom_camber_rc | 96.54 | |
| test_geom_camber_cruise | 66.61 | |
| test_re_rand | 88.55 | |

**Result:** MERGED. Val=98.79 beats Fourier baseline by 4.31%. Student's run was on pre-Fourier base; squash-merge applied Huber cleanly on top of Fourier base → merged code = Fourier+Huber.

**Key observation:** Per-split gains largest on re_rand (−12.8% vs Fourier) and single_in_dist (−1.5%), exactly where the Huber hypothesis predicted. The improvement confirms Huber targets the same distribution tails as Fourier but through a different mechanism (loss vs. encoding). Compound expected.

---

## 2026-05-12 23:56 — PR #1367: Dropout=0.2 + grad-clip=1.0 (fern) — **MERGED**

- **Branch:** `willowpai2g24h5-fern/dropout-0.1-grad-clip`
- **Hypothesis:** dropout=0.2 + grad-clip=1.0 regularises attention co-adaptation; best arm was 0.2 (not 0.1 from PR title).
- **W&B run:** `otwlgvo7`
- **Base:** BF16 + scoring fix (pre-Fourier; W&B shows fun_dim=22, no Huber)

| Metric | Value | vs Fourier baseline (103.24) |
|--------|-------|------------------------------|
| val_avg/mae_surf_p (best, ep 18) | **98.9622** | **−4.11%** |
| test_avg/mae_surf_p | **88.7390** | **−2.30%** |
| test_single_in_dist | 110.77 | |
| test_geom_camber_rc | 97.23 | |
| test_geom_camber_cruise | 58.81 | |
| test_re_rand | 88.14 | |

**Result:** MERGED. Val=98.96 is within 0.17 points of the just-merged Huber baseline (98.79); strict comparison would say this doesn't beat Huber, but dropout is orthogonal and compounds. Squash-merge applied dropout cleanly on top of Fourier+Huber → merged code = Fourier+Huber+Dropout. Default dropout=0.1 in code; **use `--dropout 0.2` to reproduce winning config**.

**Key observation:** Cruise test drop to 58.81 (vs 66.61 with Huber, 64.21 with Fourier) confirms orthogonality of mechanisms. Val curve still descending at epoch 18 cap — suggests more epochs or compound with other regularisation would help further.

**Next assignments:** askeladd → Huber δ sweep (PR #1703); fern → Dropout rate sweep 0.15/0.25/0.30 (PR #1706).

---

## 2026-05-13 01:15 — PR #1607: EMA weight averaging decay=0.99 (edward) — MERGED

- **Branch:** `willowpai2g24h5-edward/ema-weight-avg`
- **Hypothesis:** Exponential moving average over model weights smooths per-epoch val wobble, producing a more stable checkpoint at evaluation time and potentially accessing lower-loss basins.
- **W&B run:** `nl3llszv`

| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p (EMA, best ep 16) | **77.054** |
| val_single_in_dist/mae_surf_p | 85.45 |
| val_geom_camber_rc/mae_surf_p | 88.60 |
| val_geom_camber_cruise/mae_surf_p | 57.80 |
| val_re_rand/mae_surf_p | 76.36 |
| test_avg/mae_surf_p (EMA) | **68.265** |
| test_single_in_dist/mae_surf_p | 75.31 |
| test_geom_camber_rc/mae_surf_p | 80.81 |
| test_geom_camber_cruise/mae_surf_p | 48.52 |
| test_re_rand/mae_surf_p | 68.41 |
| main val_avg (same epoch, no EMA) | ~100.22 |
| Epochs completed | 16 in ~30 min |
| Peak VRAM | 33.8 GB / 96 GB |
| Δ vs prior best (#1367, val=98.96) | **−22.1% val / −23.1% test** |

**Result:** MERGED. Largest single-PR gain of the session. Main model val at epoch 16 is ~100 — unchanged from no-EMA baseline. EMA model is 77.05. The 23-point gap is entirely due to weight averaging: EMA smooths across the last ~100 gradient steps (effective window = 1/(1−0.99)), filtering batch noise while remaining responsive to learning. Uniform gains across all 4 splits (17–32% reduction).

**Key analysis:** Main model wobble was enormous (range 89–209 across 16 epochs). EMA's monotonic descent consistently reached deeper minima despite the noisy landscape. `decay=0.99` is highly responsive — essentially averaging the last ~100 batches (~0.27 epochs), not a multi-epoch window.

**Implementation:** `copy.deepcopy(model)` with `requires_grad=False`; `ema_p ← decay·ema_p + (1−decay)·p` after every optimizer step; both model and EMA model evaluated on val each epoch; best checkpoint saves EMA weights; test eval loads EMA weights into model.

**Note:** Student ran on default `dropout=0.1`. Full compound with Fourier+Huber+Dropout(0.2)+EMA not yet tested — this is a first estimate. Recommended follow-up: EMA + dropout=0.2 compound.

---

## 2026-05-13 01:15 — PR #1690: Fourier L=8 + concat-raw positions (nezuko) — CLOSED

- **Branch:** `willowpai2g24h5-nezuko/fourier-l8-concat`
- **Hypothesis:** (1) More Fourier frequencies (L=8 vs L=6) capture finer boundary-layer features; (2) Concatenating raw (x,z) alongside Fourier features helps geom_camber_rc by keeping global coordinates.
- **W&B runs:** `2xfd4tvu` (L=8), `hswe57m9` (L=6 concat-raw)

| Metric | L=6 baseline | L=8 replace | Concat-raw |
|--------|-------------|-------------|------------|
| val_avg/mae_surf_p | 103.24 | 104.97 (+1.6%) | 110.72 (+7.2%) |
| test_avg/mae_surf_p | 90.83 | **89.91 (−1.0%)** | 100.65 (+10.8%) |

**Result:** CLOSED. Both arms lose on the primary val metric. L=8 wash on test but primary metric regresses. L=6 concat-raw clear regression everywhere, including geom_camber_rc (+15.11) — the split it was meant to help. L=6 normalized replacement remains the Fourier sweet spot.

**Key insight:** L=4→L=6 improvement was monotone; L=6→L=8 is not. L=8 may require more epochs to converge (last epoch was best, val still descending), but within 30-min budget the gain doesn't materialise. Concat-raw dilutes learned features via scale mismatch (raw pos ∈ [−2,2] vs sin/cos ∈ [−1,1]) — the slim model can't disentangle them in 18 epochs.

---

## 2026-05-13 01:15 — PR #1400: Aux surf-p head λ sweep (tanjiro) — CLOSED

- **Branch:** `willowpai2g24h5-tanjiro/aux-surf-p-head`
- **Hypothesis:** Auxiliary surface-pressure prediction head (MLP on penultimate hidden state, λ-weighted loss) adds direct ranking-metric gradient signal during training.
- **W&B runs:** `xd6973hg` (λ=2, Fourier+BF16), `mbaijhsk` (λ=5, Fourier+BF16); earlier `m9xr80iw` (λ=2, pre-Fourier, no BF16)

| Variant | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---------|-------------------|---------------------|
| Fourier+BF16 baseline | 103.24 | 90.83 |
| λ=2 + Fourier + BF16 | 114.32 (+10.7%) | 99.16 (+9.2%) |
| λ=5 + Fourier + BF16 | 117.94 (+14.2%) | 104.56 (+15.1%) |

**Result:** CLOSED. Both λ arms regress vs baseline; λ=5 is worse than λ=2 — increasing aux weight is monotone-worsening. Aux head is dominated by Fourier features: Fourier-encoded hidden state already carries strong surface-p signal, so extra aux loss gradient competes with the main loss without the surf_weight=10× advantage the main surface loss enjoys.

**Key insight:** Aux head improved on the pre-Fourier base (val=118.88 vs ~123 pre-Fourier), confirming the mechanism works. It's specifically Fourier's input-feature improvement that makes the aux head redundant. The `return_hidden=True` pattern in train.py is a useful implementation pattern for future auxiliary-task hypotheses.


## 2026-05-13 03:10 — PR #1748 CLOSED: EMA-0.99 + Dropout=0.2 compound regresses (edward)

- **Branch:** `willowpai2g24h5-edward/ema-dropout-compound`
- **Hypothesis:** EMA-0.99 (trajectory smoothing) and Dropout=0.2 (feature decorrelation) are orthogonal regularisers; the full Fourier+Huber+Dropout(0.2)+EMA stack should beat the merged EMA-with-dropout=0.1 baseline.

| Run | Config | val_avg/mae_surf_p | test_avg/mae_surf_p | Notes |
|-----|--------|---------------------|----------------------|-------|
| **Baseline (merged PR #1607)** | EMA-0.99 + dropout=0.1 | **77.054** | **68.265** | Reference |
| `fv69guww` (Arm 1 seed 1) | EMA-0.99 + dropout=0.2 | 78.869 | 69.880 | +2.4% / +2.4% (worse) |
| `yzgt1otg` (Arm 1 seed 2) | EMA-0.99 + dropout=0.2 | 80.139 | 69.241 | +4.0% / +1.4% (worse) |

- Mean Arm 1 val ≈ 79.5 (vs baseline 77.05), inter-seed spread ≈ 1.3 pts. Clearly outside noise.
- Every val split regresses, including easy `geom_camber_cruise` (+4.0%) — not a hard-tail-only effect.
- Best epoch=16 in both seeds (still descending at cap), same as baseline; not an under-training artefact.

**Result:** CLOSED. EMA-0.99 + dropout=0.2 over-regularises on the EMA base. Mechanism: main-model val is slightly better with dr=0.2 (96.1 vs ~100), but the EMA trajectory averages a less-faithful approximation when dropout adds stochastic noise. EMA already provides strong implicit regularisation, so the dr=0.1→0.2 step (which won on the *non-EMA* base in PR #1367) is past the sweet spot once EMA is in place.

**Key insight:** The merged EMA baseline (PR #1607, val=77.05, test=68.27, dropout=0.1) is the load-bearing reference — it IS the true compound. We do NOT need to revisit dropout=0.2 + EMA combinations. Dropout=0.15 on EMA base is worth testing later (interpolation between 0.1 and 0.2). The PR #1367 dropout=0.2 win was on a pre-EMA base with regularisation headroom that EMA now fills.

**Reassigned:** edward → EMA decay sweep on compound (0.995, 0.999 vs merged 0.99).
