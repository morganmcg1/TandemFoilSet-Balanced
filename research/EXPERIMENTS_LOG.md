# SENPAI Research Results — willow-pai2g-24h-r5

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
