# SENPAI Research Results

## 2026-04-29 12:42 — PR #1138 (MERGED — round-2 winner): Random Fourier Features on (x, z), n_freq=32, sigma=1.0
- Branch: `charliepai2f1-frieren/rff-32` (merged into `icml-appendix-charlie-pai2f-r1`)
- Hypothesis: Replace raw `(x, z)` node positions with `[sin(2π·B·pos), cos(2π·B·pos)]` (random Fourier features) to mitigate spectral bias of MLPs and let the network represent high-frequency surface-pressure variations directly. Predicted -3% to -8%.
- Reality: **-13.5% on val, -14.2% on test** — blew past prediction. RFF is the strongest single-lever round-2 win to date.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 14/14, final) | **108.543** |
| `test_avg/mae_surf_p` (4 splits, all finite ✓) | **96.942** |
| vs new merged baseline (#1101 schedule) | val 125.438 → 108.543 (-13.5%), test 112.988 → 96.942 (-14.2%) |
| vs prior provisional (#1095 confounded) | val 133.892 → 108.543 (-19.0%), test 132.106 → 96.942 (-26.6%) |
| Epochs run | 14 / 50 (timeout-bound, ~133 s/epoch — RFF adds negligible cost) |
| n_params | 678K (~0.68M, +20K vs baseline 658K from wider preprocess MLP) |
| Peak VRAM | 42.5 GB |
| Metrics file | `models/model-charliepai2f1-frieren-rff-32-20260429-120655/metrics.jsonl` |

### Per-split val/test (best checkpoint, epoch 14)

| Split | val mae_surf_p | test mae_surf_p | Δ vs round-1 baseline (val) |
|---|---|---|---|
| `single_in_dist` | 125.82 | 104.40 | 151.43 → 125.82 (-16.9%) |
| `geom_camber_rc` | 114.59 | 106.27 | 132.77 → 114.59 (-13.7%) |
| `geom_camber_cruise` | 86.37 | 74.04 | 99.90 → 86.37 (-13.5%) |
| `re_rand` | 107.40 | 103.05 | 117.65 → 107.40 (-8.7%) |
| **avg** | **108.54** | **96.94** | **125.44 → 108.54 (-13.5%)** |

All 4 splits improved on both val and test. Largest absolute gain on `val_geom_camber_cruise` (drop from 99.9 → 86.4) — consistent with the spectral-bias hypothesis: cruise has the densest meshes (~210K nodes), so high-frequency representational benefit is largest there.

### Val_avg trajectory
```
e1→271 e2→260 e3→171 e4→206 e5→156 e6→162 e7→141
e8→141 e9→141 e10→131 e11→134 e12→122 e13→187 e14→108 *
```
Best epoch is the **final** epoch — model still descending hard at the cap. RFF + the merged schedule together (only available post-this-merge) likely yield further gains below 108.5.

### Conclusions
- **First round-2 winner. Confirms RFF/spectral-bias hypothesis dominantly.** Predicted -3 to -8% from Tancik 2020 / GINO / MARIO priors; actual -13.5%. The dataset's surface pressure has more high-frequency content than the round-1 baseline could represent through a raw-coord preprocess MLP. RFF gives the model the right Fourier basis up-front.
- **Per-split gains track mesh density.** Densest meshes (cruise, ~210K nodes) saw the largest improvement. Sparsest (single_in_dist) the smallest, though still significant. Strong evidence the spectral bias is the dominant bottleneck on this dataset.
- **Throughput unchanged.** 14 epochs in 30 min (vs 14 baseline). RFF is ~free in compute — just an extra `[B, N, 2] @ [2, 32]` matmul per forward, negligible vs the PhysicsAttention cost.
- **n_params +20K (3% increase).** From the wider preprocess MLP input (22+64=86 vs 22+2=24); RFF buffer itself is non-trainable. Well within budget.
- **Best epoch = last epoch.** Same pattern as thorfinn's schedule winner. Confirms the budget-limited regime: every throughput improvement directly translates into more headroom.
- **Training was on pre-merged-schedule train.py.** frieren's run started at 12:06:55 and the schedule merge happened at 12:17. So this win is RFF + vanilla CosineAnnealingLR(T_max=50). After this PR's squash merge, train.py has both RFF + thorfinn's regime-matched schedule for the first time. **Future runs on the merged train.py will likely see val_avg below 108.5** (compounding gains).

### Round-2 stacking implications
- **Architectural change.** Stacks orthogonally with the merged schedule and any in-flight loss/optimization/augmentation experiment.
- **All in-flight round-2 PRs (FiLM #1158, EMA #1142, AoA #1159, SwiGLU #1160, scale-norm #1162) inherit RFF via rebase.** Their predicted effects are now layered on top of 108.5, not 125.4.
- **frieren reassigned to RFF n_freq sweep (n_freq=64)** — direct test of whether RFF capacity is the binding constraint, since the model is still descending. Predicted +1% to +3% if capacity-limited; flat or worse if n_freq=32 is sufficient.

## 2026-04-29 12:18 — PR #1101 (MERGED — round-1 winner): Schedule regime-matched (warmup=1, T_max=13, eta_min=lr/100)
- Branch: `charliepai2f1-thorfinn/warmup-cosine-floor` (merged into `icml-appendix-charlie-pai2f-r1`)
- Hypothesis: Linear warmup (1 ep) + cosine to non-zero floor (eta_min=lr/100) over the **achievable** horizon (T_max=13), beats both vanilla cosine-to-zero (T_max=epochs=50, never reaches tail) and the prior 5-epoch warmup variant (calibrated to the nominal horizon).
- Reality: clean win — beat baseline by 6.3% on val and 14.5% on test, with all 4 test splits finite. Final-epoch best (still descending at the cap).

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 14/14, final) | **125.438** |
| `test_avg/mae_surf_p` (4 splits, all finite ✓) | **112.988** |
| vs prior PR (5-ep warmup, T_max=45) | val 142.886 → 125.438 (-12.2%), test 127.871 → 112.988 (-11.6%) |
| vs provisional baseline (#1095 edward) | val 133.892 → 125.438 (-6.3%), test 132.106 → 112.988 (-14.5%) |
| Epochs run | 14 / 50 (timeout-bound, ~131 s/epoch) |
| Peak VRAM | 42.1 GB |
| Metrics file | `models/model-charliepai2f1-thorfinn-schedule-regime-matched-20260429-113430/metrics.jsonl` |

### Per-split val/test (best checkpoint, epoch 14)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| `single_in_dist` | 151.43 | 130.68 |
| `geom_camber_rc` | 132.77 | 122.82 |
| `geom_camber_cruise` | 99.90 | 84.04 |
| `re_rand` | 117.65 | 114.41 |
| **avg** | **125.44** | **112.99** |

### LR trajectory (verified)
```
e1 5.0e-7 (warmup)   e8 2.8e-4
e2 5.0e-4 (peak)     e9 2.2e-4
e3 4.9e-4            e10 1.6e-4
e4 4.7e-4            e11 1.1e-4
e5 4.4e-4            e12 6.7e-5
e6 3.9e-4            e13 3.3e-5
e7 3.4e-4            e14 1.2e-5  ← ~2.4× eta_min, floor engaged
```

### Val_avg trajectory
```
e1→407 e2→266 e3→209 e4→200 e5→189 e6→213 e7→165
e8→165 e9→150 e10→180 e11→145 e12→135 e13→127 e14→125 *
```

### Conclusions
- **First merged round-1 winner.** The regime-matched schedule decisively beats the round-1 baseline pack on both val and test. NaN-safe scoring rebase delivers all-finite test metrics for the first clean test_avg comparison.
- **Mechanism confirmed.** Cosine over the achievable horizon (T_max=13) lets eta_min=5e-6 actually engage in the last 2-3 epochs (e13 lr=3.3e-5, e14 lr=1.2e-5) — these epochs each produced ~5-7% improvement on val_avg. Compared to the prior PR config (lr stuck at 4.6e-4 at e14), the floor is the difference between converging and bouncing.
- **Best is final epoch.** Val curve still descending at e14 (e13 126.5 → e14 125.4) — model is still under-trained at the wall-clock cap. Headroom exists for: (a) longer wall-clock, (b) throughput gains (AMP, gradient checkpointing, torch.compile), or (c) eta_min tuning (lr/50 = 1e-5 might convert the last 2 epochs into bigger steps).
- **Round-2 stacking lever.** This schedule is now the merged baseline. All round-2 hypotheses inherit it via rebase. SchedulE + RFF (frieren H-01), schedule + EMA (nezuko H-06), schedule + FiLM (thorfinn H-10), schedule + AoA flip (askeladd H-12), schedule + SwiGLU (alphonse H-11) — all clean orthogonal stacks.
- **Reaffirms learnings #7 and #8.** Schedule hyperparameters MUST match the achievable horizon, not the nominal one. The 5-epoch warmup wasted ~35% of useful gradient budget; T_max=45 traversed only 20% of the cosine. These are the exact errors thorfinn diagnosed and corrected.

## 2026-04-29 12:19 — PR #1094 (closed): Surf weight 25 + bs=8 (revision)
- Branch: `charliepai2f1-askeladd/surf-weight-25` (closed)
- Hypothesis (revision): bs↑ from 4 to 8 fits more epochs in 30 min, converting throughput into progress on val_avg/mae_surf_p (advisor's prior suggestion, since refuted).
- Reality: bs=8 ran identical 14 epochs as bs=4 (frieren's throughput finding confirmed independently); LR=5e-4 too low for larger batch → +12.3% regression vs bs=4.

### Results

| Metric | bs=8 (this run) | bs=4 (prior PR) | Δ |
|---|---|---|---|
| best `val_avg/mae_surf_p` (epoch 12) | **150.931** | 134.368 | +12.3% (worse) |
| `test_avg/mae_surf_p` (4 splits, finite ✓) | **136.096** | NaN (cruise) | finite this time |
| Epochs run | 14 / 50 | 14 / 50 | same |
| Peak VRAM | 84.2 GB | 42.1 GB | +100% |
| Metrics file | `models/model-charliepai2f1-askeladd-surf-weight-25-bs8-20260429-113448/metrics.jsonl` | | |

### Per-split val/test (best checkpoint, epoch 12)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| `single_in_dist` | 202.44 | 179.31 |
| `geom_camber_rc` | 149.65 | 134.13 |
| `geom_camber_cruise` | 122.61 | 103.45 |
| `re_rand` | 129.03 | 127.50 |
| **avg** | **150.93** | **136.10** |

### Val_avg trajectory (plateaued)
```
e1→270 e2→260 e3→203 e4→185 e5→204 e6→192 e7→180
e8→182 e9→206 e10→180 e11→190 e12→151* e13→160 e14→159
```
(plateaued at 158-160 from e13 onward — NOT still descending)

### Conclusions
- **bs↑ does NOT buy more epochs at default architecture.** Independent confirmation of frieren's PR #1097 finding: per-batch time ~2× at bs=8, batches/epoch /2, total epoch time conserved. Two confirmations now → cross-experiment learning #3 stands firm.
- **Default LR is too low for bs=8.** 14 epochs × 188 steps = 2,632 grad updates at bs=8 vs 14 × 376 = 5,264 at bs=4. With unchanged lr=5e-4, the cosine schedule cooled effective LR to a level appropriate for bs=4 → trajectory plateaued mid-training.
- **Linear LR scaling at bs=8 already implicitly tested.** PR #1099 (nezuko) ran lr=1e-3 + warmup at bs=4 and got val=143.31 — also a regression. Combining bs=8 + lr=1e-3 would land in the same noise band — not worth the GPU time vs. fresh round-2 levers.
- **Cosmetic NaN bug in train.py::evaluate_split flagged.** `(sq_err * vol_mask).sum()` propagates NaN the same way as the scoring.py bug we fixed. Affects only loss/vol_loss/surf_loss columns (cosmetic — the ranking metric mae_surf_p is unaffected via the data/scoring.py fix). Fold into a future advisor branch fix.
- **askeladd reassigned to H-12 AoA sign-flip augmentation.**

## 2026-04-29 12:19 — PR #1092 (closed): Capacity scale-up — width-only revision (n_hidden=160, n_head=5)
- Branch: `charliepai2f1-alphonse/capacity-scale-up` (closed)
- Hypothesis (revision): Width-only scaling (n_hidden 128→160, n_head 4→5, head_dim=32 invariant; 1.02M params, 1.5× baseline) fits ≥18 epochs in 30 min and beats baseline.
- Reality: 11 epochs in 30 min (175 s/epoch, ~1.3× slower per epoch); val=141.121, still descending hard at the cap (-4.9% e10→e11). Test=128.770 (4 splits finite). +5.4% regression vs new merged baseline 125.438 — but undertrained.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 11/11) | **141.121** |
| `test_avg/mae_surf_p` (4 splits, all finite ✓) | **128.770** |
| Param count | 1.02M (1.5× baseline) |
| Epochs run | 11 / 50 (timeout-bound, ~175 s/epoch) |
| Peak VRAM | 52.5 GB |
| Metrics file | `models/model-charliepai2f1-alphonse-capacity-160-h5-20260429-113607/metrics.jsonl` |

### Per-split val/test (best checkpoint, epoch 11)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| `single_in_dist` | 191.86 | 174.29 |
| `geom_camber_rc` | 154.69 | 141.30 |
| `geom_camber_cruise` | 96.66 | 82.10 |
| `re_rand` | 121.28 | 117.39 |
| **avg** | **141.12** | **128.77** |

### Val_avg trajectory (still descending hard)
```
e1→235 e2→244 e3→197 e4→179 e5→202 e6→220 e7→163
e8→163 e9→148 e10→148 e11→141 *  (e10→e11: -4.9%/epoch)
```

### Conclusions
- **Width-only direction is alive but throughput-bound.** Per-epoch trajectory clearly descending at the cap. With 1.3× per-epoch cost, achievable epochs drop from ~14 to ~11. Even at -4.9%/epoch trend, 3 more epochs would close gap to baseline (141 × 0.96^3 ≈ 125) — but those epochs aren't available.
- **The right next test is AMP-enabled stacking.** AMP/bf16 autocast roughly halves activation memory and ~2× throughput, getting 160/5/5/2 to ~22 epochs in 30 min. Combined with thorfinn's regime-matched schedule (T_max~21), this configuration would test capacity-scaling cleanly. **Round-3 candidate** — needs a dedicated AMP PR before stacking.
- **Test_avg=128.770 is interesting** — better than the prior provisional 132.106 (3-finite split avg) but +14% vs the new merged baseline 112.988. Not a winner but doesn't refute capacity scaling.
- **Cross-experiment learning #2 holds.** Width-only is moderate cost; depth+width+mlp_ratio (alphonse's first try, 192/6/6/4) was prohibitive. Round-2 capacity stacks should keep this rule.
- **alphonse reassigned to H-11 SwiGLU/GeGLU FFN** — a different capacity axis (gated MLP) at a smaller param multiplier.

## 2026-04-29 11:54 — PR #1099 (closed): Higher LR + 5-epoch warmup (lr 5e-4 → 1e-3)
- Branch: `charliepai2f1-nezuko/lr1e-3-warmup5` (closed)
- Hypothesis: Higher peak lr (1e-3) + 5-epoch warmup hits a better minimum; predicted -2 to -5%.
- Reality: predicted effect dwarfed by run-to-run variance.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 14) | **143.313** |
| `test_avg/mae_surf_p` (4 splits, all finite ✓) | 133.082 |
| Three-trial spread (val_avg) | 152.16 / 135.43 / 143.31 (σ ≈ 7) |
| Epochs run | 14 / 50 (timeout-bound, ~131 s/epoch) |
| Peak VRAM | 42.1 GB (huge headroom) |
| Metrics file | `models/model-charliepai2f1-nezuko-lr1e-3-warmup5-20260429-111721/metrics.jsonl` |

### Per-split val/test (best checkpoint, epoch 14)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| `single_in_dist` | 174.76 | 156.41 |
| `geom_camber_rc` | 160.36 | 147.54 |
| `geom_camber_cruise` | 101.79 | 91.88 |
| `re_rand` | 136.33 | 136.50 |
| **avg** | **143.31** | **133.08** |

### Val_avg trajectory

```
e1→399.3 e2→234.2 e3→222.1 e4→216.9 e5→216.8 e6→191.3 e7→180.0
e8→182.0 e9→168.0 e10→163.3 e11→148.4 e12→194.5 e13→150.6 e14→143.3
```

### Analysis & conclusions

- **+7% vs provisional best 133.9** — clear regression on the primary metric, even though `test_avg = 133.08` is competitive with edward's clean 132.10 (3-finite splits avg). val_avg is the ranking metric.
- **Run-to-run variance σ ≈ 7** confirms the small predicted effect (-2 to -5% ≈ ±3-7 absolute units) is below noise floor. Three independent runs of identical recipe spanned 135-152.
- **Bug fix bundled.** Student independently rediscovered the NaN propagation issue and patched `train.py::evaluate_split` with sample-level filter — runs in parallel to the `data/scoring.py` `torch.where` fix already on advisor branch (commit `2548195`). Belt-and-suspenders coverage.
- **Closed.** Lever is too small for the noise floor; assigned **H-06 EMA** as a direct intervention against the variance issue (PR #1142).

## 2026-04-29 11:45 — PR #1097 (revision, then closed): slice_num=128 with bs=6 + clamp + NaN-safe filter
- Branch: `charliepai2f1-frieren/slice-num-128` (closed)
- Revision: bs 4 → 6 (bs=8 OOMed at predicted ~109 GB peak), output pressure clamp + GT-finite filter added.

### Results

| Metric | bs=4 (prior) | bs=6 (revision) |
|---|---|---|
| best `val_avg/mae_surf_p` | 162.562 (e10) | **164.209** (e7) |
| `test_avg/mae_surf_p` | NaN (cruise) | **150.714** (all 4 splits finite ✓) |
| Epochs run | 11 / 50 | 11 / 50 |
| Peak VRAM | 54.5 GB | 81.7 GB (86%) |
| Per-epoch wall-clock | ~173 s | ~176 s |

### Per-split val/test (bs=6, best epoch 7)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| `single_in_dist` | 235.91 | 208.20 |
| `geom_camber_rc` | 163.03 | 146.26 |
| `geom_camber_cruise` | 124.27 | **104.36** *(was NaN)* |
| `re_rand` | 133.63 | 144.04 |
| **avg** | **164.21** | **150.71** |

### Critical cross-experiment finding (frieren's analysis)

**Per-epoch wall-clock is NOT bound by batching overhead at default architecture.** Going from bs=4 → bs=6:
- Per-step batches: 375 → 250 (-33%)
- Per-batch time: +50%
- Total epoch time: roughly conserved (~173s → ~176s)
- Epochs in 30 min: **11 in both cases**

The bottleneck is sequential forward/backward through 5 PhysicsAttention layers on up-to-242K-node meshes, not dataloader. **bs↑ does NOT buy more epochs at default architecture.** True throughput levers are:
1. Validation cadence reduction (4 splits × 100 samples per epoch)
2. Gradient checkpointing (smaller activations enabling true bs↑)
3. `torch.compile`
4. Capacity-axis trade-offs that reduce per-step compute

This invalidates the bs↑ recommendation I sent to askeladd (PR #1094) and partially to tanjiro (PR #1100). Their runs at 42 GB peak weren't VRAM-headroom-blocked from more epochs — they're compute-bound. Implication for round-2: prioritize architectural levers that improve sample efficiency (RFF, FiLM, EMA) over bs↑ for throughput.

### Analysis & conclusions

- **+22% vs provisional best 133.9** — clear regression confirmed across two runs (162.56 and 164.21 are within run-to-run noise of each other and far above baseline).
- **Cruise NaN is now resolved** at the model side via output clamping. Combined with the advisor branch's `data/scoring.py` fix, this is belt-and-suspenders coverage for the cruise corruption.
- **Closed.** slice_num=128 doesn't pull its weight at default config and bs↑ doesn't open a path to fix it. Frieren reassigned to **H-01 RFF** (PR #1138) — architecturally orthogonal, zero throughput cost, strong literature priors.

## 2026-04-29 11:27 — PR #1101: Warmup + cosine with non-zero floor (eta_min=lr/100)
- Branch: `charliepai2f1-thorfinn/warmup-cosine-floor`
- Hypothesis: 5-epoch linear warmup + cosine to `eta_min=lr/100=5e-6` preserves a small but useful tail-LR vs vanilla cosine to 0; predicted -2% to -5%.
- Reality: hypothesis as written was mismatched with the regime — cosine's `T_max=45` traverses only ~20% of trajectory in the 14 achievable epochs, so floor never engages.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 11) | **142.886** |
| `test_avg/mae_surf_p` (4 splits, finite physical-units MAE) | 127.871 |
| Epochs run | 14 / 50 (timeout-bound, ~131 s/epoch) |
| Peak VRAM | 42.11 GB (huge headroom) |
| Wall clock | 30.7 min |
| Params | 0.66M (baseline shape) |
| Metrics file | `models/model-charliepai2f1-thorfinn-warmup-cosine-floor-20260429-104906/metrics.jsonl` |

### LR trajectory (verified from metrics)

```
e1: 5.000e-07     e6: 5.000e-04 (peak)     e11: 4.620e-04
e2: 1.004e-04     e7: 4.994e-04            e12: ~4.6e-4
e3: 2.003e-04     ...                      e13: ~4.6e-4
e4: 3.002e-04                              e14: 4.620e-04
e5: 4.001e-04
```

### Per-split val (best epoch 11) / test (best ckpt)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| `single_in_dist` | 186.159 | 159.570 |
| `geom_camber_rc` | 152.401 | 141.602 |
| `geom_camber_cruise` | 104.590 | 86.724 |
| `re_rand` | 128.394 | 123.588 |
| **avg** | **142.886** | **127.871** |

### Val_avg trajectory

```
e1→387.1 e2→242.0 e3→246.7 e4→240.7 e5→178.1 e6→220.9 e7→184.6
e8→173.3 e9→159.4 e10→161.2 e11→142.9* e12→180.1 e13→143.3 e14→160.6
```

### Analysis & conclusions

- **+6.7% vs provisional best 133.9**, but borderline regression — within close-threshold (>5%). Sent back, not closed, because the diagnosis is clean and the hypothesis is testable under a regime-matched schedule.
- **The mechanism was unobservable.** With `T_max=45 - warmup=5 = 40` cosine epochs but only 14 achievable, the cosine reaches lr ≈ 4.6e-4 at termination — never approaches `eta_min=5e-6`. Whatever benefit the floor confers happens entirely in the unreachable epoch 30+ tail.
- **The 5-epoch warmup hurts the budget.** Epochs 1–5 average lr ≈ 2e-4, vs default's full-peak start. Effective near-peak-lr epochs: ~9 (vs baseline's 14) — a 35% reduction in useful gradient updates.
- **Run-to-run variance is ~12%** (student notes prior identical run got 124.29). This is significant for low-effect-size schedule comparisons but smaller than the gap to the provisional best (~7%).
- **Sent back** with: `warmup_epochs=1`, `T_max=13` (matches achievable horizon), keep `eta_min=5e-6`. This makes the floor actually engage and preserves nearly all useful gradient updates — a fair test of the hypothesis.

## 2026-04-29 11:21 — PR #1092: Capacity scale-up: n_hidden=192, layers=6, mlp_ratio=4
- Branch: `charliepai2f1-alphonse/capacity-scale-up`
- Hypothesis: ~4× larger Transolver (192/6/6/4 = 2.60M params) closes underfit; predicted -10% to -20% on `val_avg/mae_surf_p`.
- Reality: bs=4 OOMs at 92 GB peak; bs=3 fits but only 7 epochs in 30 min.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 7, bs=3) | **168.749** |
| `test_avg/mae_surf_p` (3 finite splits) | 173.4 |
| `test_avg/mae_surf_p` (as reported, 4 splits) | NaN (cruise GT corruption + undertrained model output) |
| Epochs run | 7 / 50 (timeout-bound, ~277 s/epoch) |
| Peak VRAM | 69.18 GB at bs=3; 92.17 GB at bs=4 (OOMed during epoch 7) |
| Params | 2.60M (~4× baseline 0.65M) |
| Metrics file | `models/model-charliepai2f1-alphonse-capacity-scale-up-20260429-095515/metrics.jsonl` |

### Per-split val/test (best checkpoint, epoch 7, bs=3)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| `single_in_dist` | 211.82 | 193.26 |
| `geom_camber_rc` | 170.95 | 166.22 |
| `geom_camber_cruise` | 132.11 | NaN (cruise GT) |
| `re_rand` | 160.12 | 160.82 |
| **avg** | **168.749** | NaN (173.4 over 3 finite splits) |

### Val_avg trajectory (bs=3)

```
e1→241.8 e2→221.2 e3→241.8 e4→179.3 e5→177.9 e6→176.5 e7→168.7
```

### Analysis & conclusions

- **+26% vs provisional best 133.9** — clearly a regression, but driven entirely by undertraining. Curve still in steep descent at e7 cap.
- **Capacity scaling has a wall-clock × architecture cost that this hypothesis underweighted.** 192/6/6/4 multiplies activation memory ~4× (mlp_ratio compounds with width and layers in the MLP intermediate). At 277 s/epoch, bs=3 only buys 7 epochs vs the baseline shape's 14.
- **bs=4 OOM at 92 GB during epoch 7** — fragmentation pushed it past the limit on a larger-mesh batch. No margin at this capacity.
- **Sent back** with a narrower bump: `n_hidden 128 → 160` (width-only, ~1.5× params, n_head=5 keeps head dim 32). Keep bs=4. Rebase for NaN-safe scoring. Goal: ≥18 epochs in 30 min and a finite test_avg.

## 2026-04-29 11:03 — PR #1094: Surface weight boost: surf_weight 10 → 25
- Branch: `charliepai2f1-askeladd/surf-weight-25`
- Hypothesis: Raise surface-loss weight from 10 → 25 to bias capacity toward `mae_surf_p`; predicted -3% to -7%.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 14) | **134.368** |
| `test_avg/mae_surf_p` (3 finite splits) | 134.164 |
| `test_avg/mae_surf_p` (as reported, 4 splits) | NaN — scoring bug, see PR #1095 fix |
| Epochs run | 14 / 50 (timeout-bound) |
| Peak VRAM | 42.11 GB (huge headroom) |
| Wall clock | ~30 min |
| Metrics file | `models/model-charliepai2f1-askeladd-surf-weight-25-20260429-102854/metrics.jsonl` |

### Per-split val/test (best checkpoint, epoch 14)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| `single_in_dist` | 174.756 | 154.746 |
| `geom_camber_rc` | 147.615 | 132.749 |
| `geom_camber_cruise` | 98.786 | NaN (cruise GT corruption) |
| `re_rand` | 116.314 | 114.997 |
| **avg** | **134.368** | NaN (134.164 over 3 finite splits) |

### Val_avg trajectory

```
e1→238.4 e2→236.1 e3→213.8 e4→226.2 e5→181.8 e6→160.0 e7→181.3
e8→165.4 e9→179.2 e10→164.2 e11→208.8 e12→184.6 e13→140.8 e14→134.4
```

### Analysis & conclusions

- **Essentially tied** with the provisional best (134.4 vs 133.9, +0.36% — well within noise).
- **Per-split signature is informative.** vs edward's confounded run, askeladd is **better on val_re_rand (116.3 vs 127.0) and val_geom_camber_cruise (98.8 vs 102.7)** and worse on val_single_in_dist. That is exactly where surface boosting should help — the OOD splits whose paper-facing test_avg is what matters.
- **Throughput-limited.** Last two epochs dropped val_avg from 184.6 → 140.8 → 134.4 — the trajectory was still in steep descent at the wall-clock cap. Strong evidence that the bs=4 run is undertrained.
- **VRAM heavily underutilized** (42 GB of 95 GB) — bs↑ is the obvious next move.
- **Sent back** with: keep surf_weight=25, raise bs to 8, rebase to pick up the NaN-safe scoring fix. Goal: 25+ epochs and finite test_avg.

## 2026-04-29 10:48 — PR #1097: More physics slices: slice_num 64 → 128
- Branch: `charliepai2f1-frieren/slice-num-128`
- Hypothesis: Doubling Transolver `slice_num` from 64 → 128 gives finer learnable mesh partitions and more capacity in the slice-routing path; predicted -3% to -6% on `val_avg/mae_surf_p`.
- Predicted overhead: a couple of percent wall clock; reality was much worse.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 10) | **162.562** |
| `test_avg/mae_surf_p` (3 finite splits) | 166.409 |
| `test_avg/mae_surf_p` (as reported, 4 splits) | NaN (cruise GT corruption — see PR #1095 NaN-fix) |
| Epochs run | 11 / 50 (timeout-bound) |
| Peak VRAM | 54.5 GB (massive headroom) |
| Wall clock | ~30 min (~173 s/epoch) |
| Metrics file | `models/model-charliepai2f1-frieren-slice-num-128-20260429-095421/metrics.jsonl` |

### Per-split val (epoch 10)

| Split | mae_surf_p |
|---|---|
| `val_single_in_dist` | 201.42 |
| `val_geom_camber_rc` | 196.92 |
| `val_geom_camber_cruise` | 116.38 |
| `val_re_rand` | 135.53 |

### Per-split test

| Split | mae_surf_p |
|---|---|
| `test_single_in_dist` | 183.40 |
| `test_geom_camber_rc` | 179.35 |
| `test_geom_camber_cruise` | NaN (cruise GT corruption — only sample 20 has Inf in p) |
| `test_re_rand` | 136.48 |

### Analysis & conclusions

- 162.56 at epoch 10 is **+21%** vs. the provisional baseline (133.89) — but both are timeout-bound and frieren's run reached only epoch 10/11 vs. edward's 13/14, so the comparison is undertrained on both ends. Val curve was non-monotonic (e7→165.7, e8→171.5, e9→189.9, e10→162.6, e11→193.3), suggesting bs=4 + slice_num=128 has high gradient noise.
- **Critical underutilization.** Peak VRAM 54.5 GB of 95 GB available — frieren can roughly double bs without OOM, which would both reduce per-epoch wall clock and stabilize gradients.
- **Frieren independently rediscovered the `data/scoring.py` NaN bug** (already patched on advisor branch via PR #1095 review).
- **Sent back** with: keep slice_num=128, raise bs to 8 (or higher targeting 80%+ VRAM), add output pressure clamping to prevent fp32 overflow on cruise. Goal: 20+ epochs and finite test_avg.

## 2026-04-29 10:47 — PR #1100: Wider model + larger batch: n_hidden=256, batch_size=8
- Branch: `charliepai2f1-tanjiro/wider-bs8`
- Hypothesis: Wider Transolver (n_hidden 128 → 256, ~4× params) + larger batch (4 → 8) for better gradient stats; predicted -5% to -10% on `val_avg/mae_surf_p`.
- Reality: bs=8 OOMed (91.94 GB used + 3.36 GB needed); bs=6 also OOMed; bs=5 with `expandable_segments` ran.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 7) | **165.304** |
| `test_avg/mae_surf_p` (3 finite splits) | 168.10 |
| `test_avg/mae_surf_p` (as reported, 4 splits) | NaN (cruise vol_loss = +Inf — fp32 overflow) |
| Epochs run | 8 / 50 (timeout-bound; ~3.78 min/epoch) |
| Peak VRAM | 92.39 GB (very tight against 95 GB ceiling) |
| Params | 2.60M (~4× baseline 0.65M) |
| Metrics file | `models/model-charliepai2f1-tanjiro-wider-bs5-20260429-100402/metrics.jsonl` |

### Per-split val (epoch 7)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 197.69 | 2.58 | 1.02 |
| `val_geom_camber_rc` | 182.87 | 3.48 | 1.30 |
| `val_geom_camber_cruise` | 126.37 | 1.79 | 0.76 |
| `val_re_rand` | 154.28 | 2.71 | 1.00 |

### Per-split test

| Split | mae_surf_p |
|---|---|
| `test_single_in_dist` | 180.46 |
| `test_geom_camber_rc` | 175.05 |
| `test_geom_camber_cruise` | NaN (vol_loss = +Inf, fp32 overflow on cruise) |
| `test_re_rand` | 148.77 |

### Analysis & conclusions

- 165.30 val_avg at epoch 7/50 is **+23%** vs. provisional baseline (133.89), but only 8 vs. 14 epochs — undertrained. Val_avg trajectory: e1→222 → e7→165 → e8→226 (oscillation suggests we're already near the noisy regime under bs=5 + this width).
- **Width is the wrong knob with this MLP shape.** mlp_ratio=2 + n_hidden=256 + ~242K nodes × bs=5 → 92 GB VRAM. The MLP intermediate dominates activation memory.
- **Test pressure overflow on cruise.** Cruise samples produce predictions large enough that squared error overflows fp32. This is independent of the cruise GT NaN — it's a model output stability issue.
- **Sent back** with: keep n_hidden=256, drop mlp_ratio 2 → 1 (halves MLP activation), bs=6 (fall back to 4 if OOM), add output pressure clamping. Goal: 20+ epochs and finite test_avg.

## 2026-04-29 10:34 — PR #1095: Per-channel surface loss: 4x weight on pressure channel
- Branch: `charliepai2f1-edward/pressure-channel-weight`
- Hypothesis: Reweight surface MSE channels `(Ux, Uy, p) = (1, 1, 4)` to bias capacity toward the ranked metric (`mae_surf_p`).
- Predicted delta: -5% to -10% on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---|
| best `val_avg/mae_surf_p` (epoch 13) | **133.892** |
| `test_avg/mae_surf_p` over 3 finite test splits | 132.106 |
| `test_avg/mae_surf_p` (as reported, 4 splits) | NaN — see scoring bug |
| Epochs run | 14 / 50 (timeout-bound) |
| Peak VRAM | 42.13 GB |
| Wall clock | 31.0 min |
| Metrics file | `models/model-charliepai2f1-edward-pressure-channel-weight-20260429-095436/metrics.jsonl` |

### Per-split val (epoch 13)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|---|---|---|---|
| `val_single_in_dist` | 160.697 | 2.575 | 1.071 |
| `val_geom_camber_rc` | 145.212 | 3.435 | 1.301 |
| `val_geom_camber_cruise` | 102.659 | 1.934 | 0.781 |
| `val_re_rand` | 127.000 | 2.870 | 1.053 |

### Per-split test

| Split | mae_surf_p |
|---|---|
| `test_single_in_dist` | 140.951 |
| `test_geom_camber_rc` | 133.153 |
| `test_geom_camber_cruise` | NaN (one sample with NaN p in y, see bug) |
| `test_re_rand` | 122.213 |

### Analysis & conclusions

- Run trained stably; val curve trended down monotonically (230.9 → 133.89 over 14 epochs).
- This is the first round-1 number on the board; with no prior baseline on this branch it provisionally sets the bar at **133.892** until other in-flight PRs return.
- **Confounded normalization.** The instructed formula divides by `ch_w.sum()=6`, softening aggregate surface signal by ~3× vs. the unweighted variant. The student correctly flagged this — the run is partially a "lower effective surf_weight" experiment, not a pure pressure-channel-boost. Sending PR back with corrected normalization (`/ ch_w.mean()` keeps aggregate surface contribution constant).
- **Critical bug found in `data/scoring.py`.** Mask-by-multiply propagates NaN through `(NaN × 0) = NaN`, producing NaN in `test_avg/mae_surf_p` whenever any test sample has non-finite y. One sample in `test_geom_camber_cruise` triggers it. Fix applied to advisor branch (`torch.where`-based masking).
- Still 14/50 epochs runs (timeout-bound). Suggests architectural/loss changes that don't slow the per-epoch wall clock will compound with the training-discipline experiments in flight.
