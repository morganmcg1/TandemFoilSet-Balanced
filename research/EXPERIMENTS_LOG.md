# SENPAI Research Results

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
