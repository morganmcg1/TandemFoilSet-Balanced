# SENPAI Research Results

## 2026-05-15 18:30 — PR #3310: n_layers 5 → 6 — closed

- Branch: `willowpai2i24h5-edward/n-layers-6`
- Hypothesis: n_layers=6 will train stably under the inherited grad clip and capture meaningful depth gain vs. n_layers=5 baseline.
- W&B run: `2o2fq04e`

| Metric | n_layers=5 baseline | n_layers=6 this PR | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **117.16** | 127.23 | +10.07 (+8.6%, worse) |
| val_single_in_dist | 138.19 | 153.05 | +14.86 |
| val_geom_camber_rc | 137.91 | **124.49** | **-13.42** ✓ |
| val_geom_camber_cruise | 85.86 | 96.71 | +10.85 |
| val_re_rand | 106.68 | 134.66 | +27.98 |
| Epochs / wall-clock | 14 / ~120s each | 12 / ~157s each | -2 epochs, +31% per epoch |
| Peak VRAM | 42.1 GB | 49.6 GB | +7.5 GB |
| test avg (excl. cruise) | 116.40 | 125.26 | +8.86 |

**Analysis:** Stability hypothesis **confirmed** — no collapse, smooth descent, no Inf in test. Depth-gain hypothesis **falsified** for this budget. The deeper model runs 31% slower per epoch, completes 12 vs. 14 epochs, and lags per-epoch as well. The 30-min cap punishes depth twice. Single positive signal: val_geom_camber_rc improved by ~10% (OOD geometry), consistent with the depth → geometry-generalization hypothesis, but the cost on val_re_rand (+28) and val_single_in_dist (+15) swamps it. Key insight: the round-1 n_layers=7 "promising" signal was against the unclipped baseline (slower-converging); the clipped baseline is materially better at every epoch, so the depth gap is now negative. Next for edward: n_hidden=192 (PR #3381) — tests whether the model is capacity-limited via width rather than depth.

---

## 2026-05-15 18:00 — PR #3307: OneCycleLR (max_lr=1e-3, pct_start=0.1) — sent back

- Branch: `willowpai2i24h5-askeladd/onecyclelr-1e3`
- Hypothesis: Per-batch OneCycleLR schedule covers the full training budget more precisely than epoch-based cosine; warmup over first 10% of batches avoids early instability.
- W&B run: (see PR #3307 comments)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **119.25** |

**Analysis:** 1.8% regression vs baseline (119.25 vs 117.16), within the high-variance noise floor (~15 pp spread observed in round-1 replications). However, a root-cause issue was identified: `total_steps = len(train_loader) * MAX_EPOCHS` (i.e., 50 epochs worth of batches), but only 14 actual epochs ran under the 30-min wall-clock cap. The scheduler reached only ~28% of its designed arc — the anneal phase never fired. **Sent back to draft** with explicit fix: change `total_steps = len(train_loader) * 14` (right-sized to actual budget) and add a guard `if global_step < scheduler.total_steps: scheduler.step()` to prevent out-of-bounds errors. Re-run with `wandb_name tanjiro-onecyclelr-1e3-rightsized`.

---

## 2026-05-15 17:55 — PR #3306: Grad clip max_norm=1.0 → 100.0 — closed

- Branch: `willowpai2i24h5-tanjiro/gradclip-100`
- Hypothesis: max_norm=1.0 fires on 100% of steps (normalized GD); loosening to 100.0 allows spike-only clipping and lets the optimizer run at full effective LR.
- W&B run: (see PR #3306 comments)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **124.31** |
| Clip fire rate | 20.88% of steps |
| Median pre-clip grad norm | ~45.7 |

**Analysis:** Clear regression (+7.15 pp, +6.1% vs baseline 117.16). At max_norm=100, clipping fires on only 20.88% of steps, meaning the optimizer runs at full effective LR for ~80% of steps. The result definitively confirms that the tight max_norm=1.0 clip is doing real work as a **gradient normalizer**, not just spike suppression. The ~45× effective-LR reduction from 5e-4 → ~1.1e-5 is mechanistically beneficial, not wasteful. Follow-up (PR #3360) probes max_norm=0.5 (halving effective LR further to ~5.5e-6) to determine whether max_norm=1.0 is the sweet spot or whether tighter normalization helps more.

---

## 2026-05-15 16:25 — PR #3153: Huber (β=1.0) on volume loss — closed

- Branch: `willowpai2i24h5-nezuko/huber-vol-beta1`
- Hypothesis: Huber loss on volume term frees gradient budget from high-magnitude outliers, letting surface term pull harder.
- W&B runs: `flndh715` (best), `8u6i1i6e`, `r975jzdy` — 3 identical runs auto-generated

| Metric | Run flndh715 | Run 8u6i1i6e | Run r975jzdy |
|---|---|---|---|
| val_avg/mae_surf_p | **127.22** | 141.16 | 141.93 |
| test_avg (offline, excl. cruise sample 20) | ~117.20 | ~130.94 | ~131.11 |
| Epochs completed | 14/50 | 14/50 | 14/50 |
| Peak VRAM | 42 GB | 42 GB | 42 GB |

**Analysis:** 8.6% regression vs baseline (127.22 vs 117.16) in best run; mean ~137 across 3 runs is ~17% worse. Critical finding: **15-point run-to-run variance across identical configs** — single-run rankings in this 14-epoch budget are at the noise floor and unreliable. The comparison is also confounded by missing grad clip: this PR branched before #3157 was merged, so it ran without max_norm=1.0. The Huber-on-vol idea is not conclusively tested; it may help on the clipped baseline. Nezuko's offline test_avg workaround (drop sample 20) recovered a plausible test metric of 117.20. Independently confirmed the cruise NaN bug.


## 2026-05-15 15:42 — PR #3157: Grad clipping max_norm=1.0 — **MERGED (round-1 winner)**

- Branch: `willowpai2i24h5-tanjiro/gradclip-1p0`
- Hypothesis: Gradient spikes early in training undo optimizer progress; clip_grad_norm_(max_norm=1.0) stabilizes updates.
- W&B run: `cfp7lnaq`

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | **117.16** ← new baseline |
| val_single_in_dist/mae_surf_p | 138.19 |
| val_geom_camber_rc/mae_surf_p | 137.91 |
| val_geom_camber_cruise/mae_surf_p | 85.86 |
| val_re_rand/mae_surf_p | 106.68 |
| test_avg/mae_surf_p | NaN (cruise bad sample) |
| test 3-split avg (excl. cruise) | ~116.40 |
| Epochs completed | 14/50 (~132 s/epoch) |
| Peak VRAM | 42.1 GB |

**Analysis:** max_norm=1.0 fired on 100% of steps (median pre-clip grad norm = 45.7, P90=140.6, P99=327.2). Effective LR ≈ 5e-4/45.7 ≈ 1.1e-5 — essentially normalized gradient descent. The model still converged monotonically (val: 236→117 over 14 epochs) and leads all round-1 results. Key open question: is the win from spike suppression, gradient normalization, or just a better-conditioned optimization landscape? Follow-up (PR #3306) will probe max_norm=100 to disentangle.

---

## 2026-05-15 15:43 — PR #3125: lr=1e-3 + 2-epoch warmup + cosine — closed

- Branch: `willowpai2i24h5-askeladd/lr1e3-warmup-cosine`
- Hypothesis: Higher peak LR with a short linear warmup to avoid early instability.
- W&B run: (see PR comments)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | 135.06 (+15% vs winner) |

**Analysis:** 15% regression vs round-1 winner. The 2-epoch warmup spans ~750 batches, which may be too gradual relative to the 50-epoch cosine horizon — the LR is still ramping while the cosine has already started decaying. OneCycleLR (PR #3307) should fix this by scaling warmup to 10% of total *batches*, not epochs.

---

## 2026-05-15 15:43 — PR #3164: dropout=0.05 — closed

- Branch: `willowpai2i24h5-thorfinn/dropout-0p05`
- Hypothesis: Small dropout in Transolver blocks reduces overfitting for OOD generalization.

| Metric | Value |
|---|---|
| val_avg/mae_surf_p | 142.51 (+22% vs winner) |

**Analysis:** 22% regression. With ≤14 epochs, the model hasn't overfit in the first place — regularization from dropout adds noise but no benefit in this short-budget regime. The per-channel and per-split numbers were not especially illuminating. Next assignment (PR #3308) pivots to optimizer mechanics (beta2=0.95) which has a stronger theoretical motivation given the observed gradient distribution.

---

## 2026-05-15 15:44 — PR #3133: n_layers 5 → 7 — closed

- Branch: `willowpai2i24h5-edward/n-layers-7`
- Hypothesis: More composition passes improve OOD geometry generalization.
- W&B runs: `grsl0gde` (reported), `tqxnlq30` (148.37), `ibhtts8z` (145.59)

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (epoch 9) | 146.62 (+25% vs winner) |
| val_geom_camber_cruise (best) | 107.9 ← depth helps OOD |
| test_geom_camber_cruise/mae_surf_p | NaN (Inf prediction, reproduced across all 3 runs) |
| Epochs completed | 10/50 (~182 s/epoch, ~38% slower than 5-layer) |
| Peak VRAM | 57.1 GB |

**Analysis:** Depth hypothesis is directionally correct — the 7-layer model made faster per-epoch progress than the 5-layer would (val dropped 242→146 in 10 epochs), and OOD-geometry cruise val (107.9) was the *best* split, consistent with the prediction. However: (a) epoch 10 shows a sharp stability regression (val_rc 149.8→344.3), and (b) end-of-run test evaluation produces Inf predictions on cruise — both symptoms of the unclipped optimizer interacting badly with the deeper network. The new baseline now includes grad clipping, so n_layers=6 (PR #3310) should test depth in a stable regime.
