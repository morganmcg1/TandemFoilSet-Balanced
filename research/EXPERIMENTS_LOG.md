# SENPAI Research Results — Charlie pai2g 24h r3

Advisor branch: `icml-appendix-charlie-pai2g-24h-r3`.
Records of reviewed and merged experiment PRs. Add a new section under the
appropriate heading whenever an experiment terminal-completes.

---

## Round 2 — build on merged stack (ongoing)

All experiments in this round must rebase on `icml-appendix-charlie-pai2g-24h-r3` (includes grad_clip=1.0, wd=1e-3, OneCycleLR, EMA=0.999) before running. Current baseline: **val_avg/mae_surf_p = 112.546** (PR #1520).

---

### 2026-05-12 19:53 — PR #1520: OneCycleLR + EMA weights (fern)
**Branch:** `charliepai2g24h3-fern/onecycle-lr-ema` | **Status: MERGED** ⭐

- **Hypothesis:** Replace CosineAnnealingLR (T_max=50, only ~5% annealed in 14 epochs) with OneCycleLR (auto-adapts to actual steps). Add EMA(0.999) weights for evaluation to eliminate checkpoint-selection jitter.
- **val_avg/mae_surf_p: 112.546** (epoch 14/14) — **−2.5% vs baseline 115.403**.
- **Per-split:** single=125.10, rc=136.04, **cruise=86.31** (best camber), re_rand=102.73.
- **Test (3-split proxy):** single=113.89, rc=118.86, re_rand=99.84 → **110.862** (−3.7% vs baseline 115.13).
- **Analysis:** Val curve was strictly monotone decreasing (epoch 14 still the best at cap). EMA eliminated the regression fern's own baseline saw at epochs 13–14 (115.40 → 119.37 → 126.42). OneCycleLR peak LR = 5e-3 reached at epoch 3; lr=4.3e-3 at cap (barely in cosine anneal phase). val_re_rand improved 6.4% (109.76 → 102.73) — EMA smoothing particularly helps the Re-generalization split. Both effects (OneCycleLR + EMA) are compounding.
- **Artifacts:** `models/model-charliepai2g24h3-fern-onecycle-ema-decay999-20260512-191518/{metrics.jsonl,metrics.yaml}`

---

### 2026-05-12 19:54 — PR #1484: Huber loss delta=0.5 and delta=1.0 (alphonse)
**Branch:** `charliepai2g24h3-alphonse/huber-pressure-loss` | **Status: SENT BACK**

- **Hypothesis:** Huber loss (delta=1.0 in normalized space) clips high-Re gradient extremes and improves `val_avg/mae_surf_p` by 3–8%.
- **val_avg/mae_surf_p:** delta=0.5: **108.097**, delta=1.0: **108.104** (both epoch 14; tie within noise).
- **Per-split (delta=0.5 best):** single=146.2, rc=113.3, cruise=79.0, re_rand=93.8.
- **Test (3-split proxy):** delta=0.5: single=124.8, rc=98.4, re_rand=90.4 → ~104.55. delta=1.0: single=111.3, rc=105.3, re_rand=102.3 → ~106.27.
- **Analysis:** Best raw number of all round-1 runs (108.10 vs 115.40 baseline), but ran on pre-merge base (no grad_clip + wd, no OneCycleLR + EMA). Merge conflict blocked direct merge. Key concern: delta=0.5 helps cruise/re_rand but hurts single_in_dist (146.2 vs 122.6 at d=1.0) — aggressive clipping removes signal needed for high-Re single-foil. Bug analysis of cruise NaN matches tanjiro's independent trace.
- **Why sent back:** Merge conflicts (DIRTY state). Pre-merge base comparison unfair to new baseline 112.546. Instructed: rebase, run both arms (d=0.5, d=1.0) with full merged stack (OneCycleLR + EMA + clip + wd), pass criterion val_avg < 112.546.
- **Artifacts:** `models/model-huber-delta-{1p0,0p5}-20260512-*/metrics.yaml`

---

### 2026-05-12 20:XX — PR #1543: Log-cosh loss on merged stack (fern)
**Branch:** `charliepai2g24h3-fern/logcosh-loss` | **Status: WIP (assigned)**

- **Hypothesis:** Log-cosh loss [smooth, threshold-free heavy-tail robustification] is a cleaner alternative to Huber (no delta tuning, no gradient discontinuity). `L(r) = log(cosh(r))` → gradient = tanh(r) → saturates to ±1 for large residuals automatically.
- **Expected delta:** −3% to −8% on val_avg/mae_surf_p vs baseline 112.546.
- **Artifacts:** TBD

---

## Round 1 — broad coverage (assigned 2026-05-12)

Hypotheses sourced from `/research/RESEARCH_IDEAS_2026-05-12_18:00.md`.

**Cross-round findings (apply to all round 1 results):**
- All 5 reviewed runs hit the 30-min timeout. With `--epochs 50` and CosineAnnealingLR `T_max=50`, only 7-14 epochs ran → LR barely annealed (~93-95% of peak).
- `test_geom_camber_cruise/mae_surf_p` is NaN for all runs. Root cause traced by tanjiro (PR #1494): `splits_v2/.test_geom_camber_cruise_gt/000020.pt` contains 761 `+Inf` values in `y[:, 2]`. In `data/scoring.py`, the subtraction `pred - y` happens before the sample-skip mask is applied, so `Inf * 0 = NaN` poisons the accumulator. File is read-only; use a safe re-eval side script (zero-fill non-finite `y` before subtraction) or the 3-split proxy.
- grad_clip=1.0 fires on 100% of training batches (real norms 41-115). This is unit-norm SGD + AdamW adaptive scaling, not "spike clipping" — but it works.

---

### 2026-05-12 18:56 — PR #1491: Gradient clipping + weight_decay tuned (fern)
**Branch:** `charliepai2g24h3-fern/grad-clip-adamw-tuned` | **Status: MERGED** ⭐

- **Hypothesis:** grad_clip=1.0 + weight_decay 1e-4→1e-3 would stabilize training on high-Re outliers.
- **val_avg/mae_surf_p: 115.403** (epoch 12/14)
- **Per-split:** single=133.09, rc=129.76, **cruise=88.99** (best), re_rand=109.76
- **Test (3-split proxy):** single=116.98, rc=119.26, re_rand=109.15 → proxy avg ~115.1
- **Analysis:** Clipping fired on 100% of batches (norms 41-115). Produces the smoothest val trajectory of round 1 (249 → 115 over 12 epochs, nearly monotone). The wd=1e-3 + clip combination outperforms all other round-1 variants. This is the new baseline.
- **Artifacts:** `models/model-grad-clip-wd1e-3-20260512-181000/metrics.jsonl`

---

### 2026-05-12 18:56 — PR #1495: AoA + NACA camber jitter augmentation (thorfinn)
**Branch:** `charliepai2g24h3-thorfinn/geometry-aoa-augmentation` | **Status: SENT BACK**

- **Hypothesis:** Online jitter of AoA (±0.5°) and NACA camber (±0.002) improves OOD camber splits.
- **val_avg/mae_surf_p: 129.694** (epoch 12/14)
- **Per-split:** single=155.30, rc=141.33, **cruise=102.93** (OOD best), re_rand=119.22
- **Test (3-split proxy):** single=139.75, rc=129.04, re_rand=122.90 → ~130.56
- **Analysis:** 12% worse than fern's run, but same epoch budget. Camber OOD splits are NOT the worst — single-foil in-dist (155.3) is worst, possibly because extreme high-Re raceCar pressures dominate. Cannot isolate augmentation effect without equal-budget no-aug control. Cosine T_max mismatch (same as all round 1). Sent back: rebase on #1491 baseline + fix T_max.
- **Artifacts:** `models/model-geom-aoa-augment-20260512-181104/metrics.jsonl`

---

### 2026-05-12 18:53 — PR #1492: mlp_ratio 2→4 wider FFN (frieren)
**Branch:** `charliepai2g24h3-frieren/mlp-ratio-4-wider-ffn` | **Status: SENT BACK**

- **Hypothesis:** Restoring mlp_ratio to the paper's default (4) improves FFN capacity.
- **val_avg/mae_surf_p: 144.334** (epoch 11/13)
- **Per-split:** single=183.46, rc=153.62, cruise=105.23, re_rand=135.03
- **Test (3-split proxy):** single=155.33, rc=139.70, re_rand=125.49 → ~140.18
- **Analysis:** 25% worse than fern's run. Same 30-min timeout issue. mlp_ratio=4 is ~21% slower per epoch, so fewer epochs completed. Without proper cosine annealing the comparison is unfair. Sent back: rebase on #1491 + set --epochs 12 to match actual budget.
- **Artifacts:** `models/model-mlp-ratio-4-20260512-180817/metrics.jsonl`

---

### 2026-05-12 19:09 — PR #1494: FiLM conditioning on log(Re) (tanjiro)
**Branch:** `charliepai2g24h3-tanjiro/re-film-conditioning` | **Status: SENT BACK**

- **Hypothesis:** Inject FiLM (γ·h + β) per TransolverBlock conditioned on log(Re); should help cross-Re generalization (val_re_rand).
- **val_avg/mae_surf_p: 129.94** (epoch 12/14) — 12.6% worse than #1491 baseline.
- **Per-split:** single=156.91, rc=140.57, cruise=106.23, **re_rand=116.04 (best)**.
- **Test (safe re-eval):** single=138.96, rc=123.33, cruise=90.24 (199/200 samples), re_rand=120.40 → **test_avg=118.23**.
- **FiLM diagnostics:** γ/β weight norms grow monotonically from zero (block0 0→5.97, block4 0→3.01 over 12 epochs). Conditioning IS being learned. val_re_rand becomes the best-of-4 split — consistent with the FiLM hypothesis.
- **Why sent back, not closed:** Ran on pre-merge base (no grad_clip + wd=1e-3); not a fair comparison to merged baseline. Same cosine T_max=50 mismatch. Need rebase + --epochs 14 re-run.
- **Bonus:** Tanjiro's bug analysis on the cruise NaN is the source of the safe re-eval pattern now in BASELINE.md.
- **Artifacts:** `models/model-re-film-conditioning-20260512-182128/{metrics.jsonl,metrics.yaml,test_safe_eval.log}`

---

### 2026-05-12 19:22 — PR #1493: PhysicsAttention slice_num 64→128 (nezuko)
**Branch:** `charliepai2g24h3-nezuko/more-slices-128` | **Status: SENT BACK**

- **Hypothesis:** Doubling slice_num gives PhysicsAttention more token capacity to represent surface vs. volume regions in 74-242K-node meshes.
- **val_avg/mae_surf_p: 138.317** (epoch 10/11) — 19.9% worse than #1491 baseline.
- **Per-split:** single=175.88, rc=147.04, cruise=108.51, re_rand=121.83.
- **Test (3-split proxy):** single=146.80, rc=135.49, re_rand=123.74 → ~135.01.
- **Memory:** Peak 54.5 / 96 GB — slice_num=128 is cheap. Room for slice_num=192 or 256 in a later round.
- **Cruise NaN trace:** Independently identified the same 761-Inf bug as tanjiro (PR #1494). Clearest write-up of the boolean→float cast mechanics. Credited alongside tanjiro.
- **Why sent back, not closed:** Ran on pre-merge base (no grad_clip + wd=1e-3); not a fair comparison to merged baseline. Same cosine T_max=50 mismatch (11 epochs only). Need rebase on #1491 + --epochs 11 re-run.
- **Artifacts:** `models/model-more-slices-128-20260512-180855/{metrics.jsonl,metrics.yaml}`

---

### 2026-05-12 18:53 — PR #1490: Scale model n_hidden=256, n_head=8 (edward)
**Branch:** `charliepai2g24h3-edward/scale-model-256` | **Status: SENT BACK**

- **Hypothesis:** n_hidden 128→256, n_head 4→8 (~2.54M params) improves capacity.
- **val_avg/mae_surf_p: 172.262** (epoch 6/7; 30-min cap after 7 epochs)
- **Per-split:** single=199.15, rc=194.97, cruise=131.46, re_rand=163.47
- **Test:** NaN overall; 3-split proxy: single=191.41, rc=186.78, re_rand=159.10 → ~179.09
- **Analysis:** Severely under-budgeted — ~260 s/epoch means only 7 epochs in 30 min. Model trending down (172 → 176 at epoch 7) but far from converged. Also: model is 2.54M not the predicted ~6M (mlp_ratio=2 was not changed). OOM risk (83.9GB peak). Sent back: scale down to n_hidden=192 + set --epochs 10 + rebase on #1491.
- **Artifacts:** `models/model-scale-model-256-20260512-180850/metrics.jsonl`
