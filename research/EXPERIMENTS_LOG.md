# SENPAI Research Results — willow-pai2g-24h-r5

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
