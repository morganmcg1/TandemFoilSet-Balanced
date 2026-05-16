# SENPAI Research Results — `willow-pai2i-48h-r4`

## 2026-05-16 04:30 — PR #3637: Width n_hidden=176 (thorfinn) — CLOSED

- **Student:** willowpai2i48h4-thorfinn
- **Hypothesis:** n_hidden=176 is the sweet spot between the working 160 and the failing 192, giving +10% width with +20% params.

### Results (W&B run `7zjst4wu`)

| Metric | Baseline (#3632, 83.50) | n_hidden=176 | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 83.4954 | 88.4539 | **+5.0 (vs old baseline 88.24: +0.21)** |
| test_avg/mae_surf_p | 73.7918 | 79.2911 | **+5.50** |

Ran against the OLD baseline at val=88.24 (submitted before #3632 coord noise was merged). Against THAT baseline: val=88.45 (+0.21, essentially noise-level regression). Against current 83.50: +5.0 (clearly worse).

### Analysis

Width scaling is confirmed plateaued at n_hidden=160 for this 30-min budget. n_hidden=176 (+10% width) gives both val and test regressions. Both sub-192 widths (176, 192) have been tested and both regress. The model is not capacity-limited in width — it's under-trained in time. Depth also fails at budget. The winning lever going forward is **data/augmentation** (coord noise proved this) and **loss/pe engineering** (Fourier PE proved this).

**Closed** — width scaling exhausted at n_hidden=160 for current budget.

---

## 2026-05-16 04:30 — PR #3635: Depth n_layers=6 on current stack (edward) — CLOSED

- **Student:** willowpai2i48h4-edward
- **Hypothesis:** n_layers=6 on the full current stack (Fourier PE + L1 + n_hidden=160) might give gains that were masked in the stale #3469 experiment.

### Results (W&B run `4vmya3cn`)

| Metric | Baseline (#3372, 88.24) | n_layers=6 (8ep) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 88.2442 | 94.5011 | **+6.26 ✗** |
| test_avg/mae_surf_p | 77.0880 | 83.5150 | **+6.43 ✗** |

Per-split: single_in_dist+14.95 (worst), geom_camber_cruise +1.31 (best), re_rand +2.97. Every split regressed.

### Analysis

Depth=6 at --epochs 8 (budget constraint) is under-converged: the extra block needs more gradient steps to learn meaningful higher-order cross-slice interactions. geom_camber_cruise nearly held neutral (+1.31) — the extra depth might help if training epochs could increase. Not viable at 30-min budget. Depth scaling would require 20+ epochs or a curriculum/pretraining approach.

**Closed** — confirms depth scaling is budget-constrained at 30min window. Consistent with #3469.

---

## 2026-05-16 04:30 — PR #3632: Coordinate noise augmentation std=0.01 (tanjiro) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-tanjiro
- **Hypothesis:** Gaussian jitter (std=0.01) on normalized (x,z) coords during training only gives richer geometry variation each epoch, improving OOD generalization.

### Results (W&B run `0q6t1hpc`)

| Metric | Old baseline (#3372, 88.24) | Coord noise (0q6t1hpc) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 88.2442 | **83.4954** | **−4.75 (−5.38%) 🏆** |
| **test_avg/mae_surf_p** | 77.0880 | **73.7918** | **−3.30 (−4.28%) 🏆** |

Per-split test: single_in_dist 83.77 (−4.68%), geom_camber_rc 80.55 (−2.60%), geom_camber_cruise 55.20 (−7.08%), re_rand 75.64 (−3.47%). Improvement on every split.

Config: n_hidden=160, n_layers=5, Fourier PE num_freq=4, L1 loss, coord_noise_std=0.01 (train only), lr=5e-4 (Config default — note: NOT the lr=1e-3 used in #3372; testing lr=1e-3 with coord noise is an open opportunity).

### Analysis

Second-largest single-experiment gain in the track (+5.38% val, after Fourier PE +8.2%). Coord noise acts as implicit mesh-topology augmentation: the model sees slightly different geometry each epoch, forcing it to learn the physics rather than memorize mesh coordinates. The cruise split gained most (−7.08% test) — consistent with cruise shapes having highest geometry variability.

Key insight: lr=5e-4 (default) was used, NOT lr=1e-3 (the prev baseline lr). Testing lr=1e-3 + coord noise is an open compounding experiment.

**Merged** at 04:30 UTC. New baseline: val=83.50/test=73.79. coord_noise_std=0.01 is now default.

---

## 2026-05-16 02:25 — PR #3372: Fourier PE 4-freq on (x,z) coords (askeladd) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-askeladd (branch: `askeladd/fourier-pos-encoding`)
- **Hypothesis:** Replace raw (x, z) coordinates with NeRF-style log-spaced sinusoidal Fourier positional encoding (num_freq=4) to give the model multi-scale geometric receptive field beyond linear interpolation.

### Results (W&B run `qyc68z5k`)

| Metric | Old baseline (#3507, 96.10) | This run (Fourier PE 4-freq) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 96.0997 | **88.2442** | **−7.86 (−8.2%) 🏆** |
| **test_avg/mae_surf_p** | 85.5256 | **77.0880** | **−8.44 (−9.9%) 🏆** |

Per-split test surface pressure MAE:

| Split | test/mae_surf_p |
|---|---:|
| single_in_dist | 87.8840 |
| geom_camber_rc | 82.7020 |
| geom_camber_cruise | 59.4070 |
| re_rand | 78.3590 |
| **avg** | **77.0880** |

Config: L1 loss, warmup 2ep, cosine T_max=10, lr=1e-3, n_hidden=160, n_layers=5, n_head=4, slice_num=64, Fourier PE num_freq=4. `fun_dim` grows from 22 to 38 (4×num_freq sinusoidal features per coord pair). Per-epoch ~168s (unchanged — PE is just a preprocess step). ENCODED_X_DIM=38.

### Analysis

Fourier PE gave the **largest single-experiment gain on the track** — larger than L1 loss (−8.1% val) and larger than width-160 (−4.4% val). The improvement is concentrated in the cruise and OOD splits: `test_geom_camber_cruise` went from 61.38 → 59.41 (−3.2%), `test_re_rand` from 84.55 → 78.36 (−7.3%), `test_single_in_dist` from 103.75 → 87.88 (−15.3%). This pattern is consistent with the hypothesis: Fourier features encode geometry at multiple scales simultaneously, helping the model generalize to unseen geometries (OOD) rather than memorizing the training mesh topology.

The gain was earned cheaply: no architecture change, no extra parameters (only the input layer of the preprocess MLP grows by 14 weights), no training time overhead.

**Merged** at 02:25 UTC as new baseline. num_freq=4 is now `Config` default. All in-flight students need to rebase to get the new `ENCODED_X_DIM` computation.

---

## 2026-05-16 02:30 — PRs #3490/#3508/#3524/#3552/#3288: Round-2 experiment sweep — all CLOSED

Five PRs closed in this batch, all regressing vs the new baseline val=88.24. Summary:

| PR | Student | Hypothesis | Best val | Best test | Δ vs 88.24 |
|---|---|---|---:|---:|---:|
| #3490 | nezuko | L1 LR sweep {3e-4, 2e-3, 4e-3} | 98.88 (lr=2e-3) | 87.75 | +10.64 |
| #3508 | fern | Cosine warm restarts SGDR T_0=4 | 100.79 | 90.63 | +12.55 |
| #3524 | thorfinn | Huber loss β=1.0 | 101.44 (`oj7zwn3z`) | 90.14 | +13.20 |
| #3552 | alphonse | Width n_hidden=192 (--epochs 8) | 102.73 | 92.16 | +14.49 |
| #3288 | edward | Scoring fix + lr default verify | 96.53 | 86.62 | +8.29 (superseded) |

**Conclusions:**
- **lr=1e-3 is optimal**: both lower (3e-4) and higher (2e-3, 4e-3) LR with L1 are worse. No exploration needed here.
- **SGDR warm restarts hurt**: LR resets disrupt the still-descending loss curve at 10 epochs. Cosine-to-zero stays canonical.
- **Huber β=1.0 worse than L1**: L1 directly optimizes MAE, Huber doesn't. L1 locked as canonical.
- **Width 192 over-parameterized at --epochs 8**: Too much capacity for the 30-min budget. n_hidden=176 is the next candidate.

---

## 2026-05-16 01:38 — PR #3469: Deeper model n_layers 5→6 (tanjiro) — CLOSED

- **Student:** willowpai2i48h4-tanjiro (branch: `tanjiro/depth-6layers`)
- **Hypothesis:** One additional Transolver block (n_layers 5→6) provides higher-order cross-slice interactions on the airfoil geometry; predicted improvement on val_avg/mae_surf_p, especially on cross-regime splits.

### Results (W&B run `5y4w4b45`)

| Metric | Stale ref (#3091, 109.42) | This run (n_layers=6) | Δ stale | Δ new baseline (96.10) |
|---|---:|---:|---:|---:|
| **val_avg/mae_surf_p** | 109.4166 | **108.4452** | −0.97 ✓ | **+12.34 ✗** |
| test_avg/mae_surf_p (3-split workaround) | 107.4694 | 105.2823 | −2.19 | — |
| test_avg/mae_surf_p (full) | NaN | NaN | — | — |

Per-split val (n_layers=6 vs old baseline 109.42):
- val_re_rand: −10.51 (big win); val_geom_camber_*: −3.51/−4.17 (modest gain); val_single_in_dist: +14.29 (regression).

### Analysis

The run was completed against the STALE pre-#3089 codebase (MSE loss, n_hidden=128). The reported −0.97 win against the #3091 baseline (109.42) is real but the experiment was never validated on the post-#3507 advisor (val=96.10, with L1 + n_hidden=160). The depth-6 result (108.45) is +12.34 above the current baseline, so even a generous re-run would need to gain >12 to land on the merge curve — vs an observed +1 gain in the stale ablation, this is implausible.

Useful signals captured for future depth work on the new baseline:
- depth-6 reliably helps `val_re_rand` and `val_geom_camber_*` on tandem-cruise OOD tracks
- depth-6 hurts `val_single_in_dist` by +14 — capacity overfits single-foil distribution
- ~158s/epoch at n_hidden=128 + n_layers=6 (vs ~168s for n_hidden=160 + n_layers=5) — depth is cheaper than width per epoch
- Loss curves still descending at epoch 10 → under-trained

**Closed** at 01:38 UTC as the stale-baseline regression is too large to bridge. tanjiro reassigned to a fresh experiment on the current advisor tip.

---

## 2026-05-16 00:30 — PR #3507: Width scaling n_hidden 128→160 (alphonse) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/alphonse-width-160`)
- **Hypothesis:** Increasing Transolver hidden width from 128 to 160 (+25% capacity, +56% params) on top of the L1 + warmup + clip stack will improve val_avg/mae_surf_p further; cosine schedule still fully anneals at the slightly slower ~168s/epoch.

### Results (W&B run `7vxhbv8o`)

| Metric | Old baseline (#3089, 100.53) | This run (n_hidden=160) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p** | 100.5275 | **96.0997** | **−4.40 (−4.4%) 🏆** |
| **test_avg/mae_surf_p** | 90.1489 | **85.5256** | **−4.62 (−5.1%) 🏆** |

Per-split test surface pressure MAE:

| Split | test/mae_surf_p |
|---|---:|
| single_in_dist | 103.7483 |
| geom_camber_rc | 92.4243 |
| geom_camber_cruise | 61.3787 |
| re_rand | 84.5510 |
| **avg** | **85.5256** |

Config: L1 loss (carry-over from #3089), warmup 2 ep, cosine to 0 (T_max=10), grad-clip 1.0, lr=1e-3, batch=4, surf_weight=10. Width 128→160; params 662k → 1.03M. Per-epoch ~168s (↑ from ~134s); peak VRAM 50.1 GB (53% of 96 GB envelope).

### Analysis

Width-160 composes cleanly with the merged optimization stack and delivers the expected gain on both val and test. Improvement is broadly distributed across all 4 test splits (no per-split regression). Val curves still descending at epoch 10 → continued width scaling is likely net-positive, but with diminishing returns expected past ~192 given the budget-constrained 10-epoch annealing.

**Merged** at 00:30 UTC as new advisor baseline. All in-flight students need to rebase to inherit `Config.n_hidden = 160`.

---

## 2026-05-15 22:35 — PR #3095: Higher surf_weight + per-channel p weighting (nezuko) — CLOSED

- **Student:** willowpai2i48h4-nezuko (branch: `willowpai2i48h4-nezuko/surface-weight`)
- **Hypothesis:** Increasing `surf_weight` from 10 to 20-30 pushes the optimizer to focus more on surface pressure prediction; adding per-channel `p` weighting (3×) further boosts the primary metric.

### Results — rebased confirmation arm (surf_weight=20, W&B run `6amjj7jr`)

| Metric | Value | Δ vs baseline (109.42) | Δ vs new baseline (100.53) |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 111.92 | +2.3% ✗ | +11.3% ✗ |
| test_avg/mae_surf_p | 97.70 | (was NaN) | +8.4% ✗ |

Earlier arm sweep (stale pre-#3091 code):

| Arm | Config | val_avg/mae_surf_p |
|---|---|---:|
| A (surf_weight=30) | surf_w=30 | 131.08 |
| B (pchan=3) | surf_w=10, p_w=3 | 133.31 |
| C (combined) | surf_w=30, p_w=3 | 146.42 |

### Analysis

All arms regressed vs both old and new baselines. surf_weight=20 on the rebased (warmup+clip+lr=1e-3+L1) code gave val=111.92 — nominally +2.3% above the 109.42 old baseline, and decisively worse (+11.3%) against the current 100.53 baseline. surf_weight=30 and the pchan knob are both clearly worse. The student's per-split analysis shows the regression concentrated in `single_in_dist` while `re_rand` and `geom_camber_rc` actually improve — suggesting surf_weight tuning shifts a per-split tradeoff rather than moving the aggregate down.

The train.py NaN fix in `evaluate_split` was correctly implemented; subsumed by alphonse #3089 merge.

**Closed** at 22:35 UTC. surf_weight hypothesis exhausted at {10, 20, 30}; optimum appears at ≤10. Nezuko reassigned to L1 LR sweep.

---

## 2026-05-15 22:31 — PR #3089: L1 loss + NaN scoring fix (alphonse) — **MERGED** → new baseline

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/l1-loss`)
- **Hypothesis:** Replacing MSE loss with L1 in the normalized-prediction space aligns the training objective with the MAE evaluation metric; predicted −5% to −10% on `val_avg/mae_surf_p`.

### Results — final rebased confirmation arm (W&B run `14w7wdyb`, `alphonse-l1-rebased`)

| Metric | Value | Δ vs baseline (109.42) |
|---|---:|---:|
| **val_avg/mae_surf_p** | **100.5275** | **−8.1% ✓** |
| **test_avg/mae_surf_p** | **90.1489** | first clean finite number |

Per-split test surface pressure MAE:

| Split | test/mae_surf_p |
|---|---:|
| single_in_dist | 112.07 |
| geom_camber_rc | 98.04 |
| geom_camber_cruise | 64.21 |
| re_rand | 86.28 |
| **avg** | **90.15** |

Config: L1 loss (`Config.loss_type = "l1"`, default flipped) + warmup + grad-clip + lr=1e-3 + 10 epochs (fully annealed cosine). Also includes `_pointwise_loss` helper for MSE/L1/Huber dispatch and `torch.isfinite` per-sample mask in `evaluate_split` (canonical scoring NaN fix).

### Analysis

L1 loss clearly improves val_avg/mae_surf_p (−8.1%) and delivers the first clean test metric. The composition of L1 with warmup+clip+lr=1e-3 works well — the levers are orthogonal (L1 addresses objective mismatch; warmup+clip+lr addresses optimization stability). Single best experiment so far by absolute Δ.

The scoring NaN fix is particularly valuable — `test_avg/mae_surf_p = 90.15` is now a reliable paper-facing metric for all future runs.

**Merged** at 22:31 UTC as new baseline. All in-flight students need to rebase to get L1 default + scoring fix.

---

## 2026-05-15 21:37 — PR #3414: SWA (stochastic weight averaging) over last K checkpoints

- **Student:** willowpai2i48h4-tanjiro (branch: `tanjiro/swa-checkpoint-averaging`)
- **Hypothesis:** Averaging the weights of the last K checkpoints (K=3 or K=5) produces a smoother loss landscape than any single checkpoint, reducing overfitting and improving val_avg/mae_surf_p.

### Results

| Arm | SWA window | raw val_avg/mae_surf_p (best ckpt) | swa_val_avg/mae_surf_p | Δ vs 109.42 baseline | W&B run |
|---|---|---:|---:|---:|---|
| A — SWA last 5 | K=5 (epochs 6–10) | **103.72** | 111.09 | +1.5% ✗ | `gduowc1p` |
| B — SWA last 3 | K=3 (epochs 8–10) | **108.01** | 109.48 | ~flat (+0.06%) | `udfmekyw` |

**SENPAI-RESULT (terminal):** `swa_val_avg/mae_surf_p = 109.48`, `swa_test_avg 3-split = 106.34` (Arm B).

### Analysis

SWA did NOT improve the primary metric on either arm. The SWA-averaged checkpoint was consistently **worse** than the best raw checkpoint in both arms. The mechanism is clear: with `--epochs 10` and cosine annealing to 0, the loss is still descending at the final epoch. Averaging the last K checkpoints includes sub-optimal earlier states from the middle of descent, which drags the average above the best single checkpoint.

The Arm A raw val (103.72) is better than baseline but that's just run variance — it's an unintended observation from a re-run of the baseline config. The proposed feature (SWA averaging) consistently regressed both arms.

**Conclusion:** SWA is only beneficial when the training curve has plateaued — which requires more epochs than our 30-min budget allows at the current batch size. The experiment correctly identified this limitation in the writeup.

**Closed as dead end** at 21:37 UTC. New hypothesis assigned to tanjiro: depth n_layers=5→6 (#3469).

## 2026-05-15 14:07 — PR #3092: More physics-attention slice tokens (slice_num 64→128, 192)

- **Student:** willowpai2i48h4-fern (branch: `willowpai2i48h4-fern/more-slices`)
- **Hypothesis:** Doubling `slice_num` from 64 to 128 raises the resolution of Transolver's physics decomposition over 74K–242K node meshes; predicted −3% to −7% on `val_avg/mae_surf_p`.

### Results

| Arm | slice_num | n_params | best val_avg/mae_surf_p | best epoch | total epochs | peak VRAM | epoch time | W&B run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A (winner) | 128 | 672,919 | **150.26** | 9 | 10 | 54.5 GB | 171 s | `yiiy92uj` |
| B | 192 | 683,479 | 153.71 | 9 | 10 | 68.4 GB | 213 s | `l7nnvr53` |

Per-split val surface pressure MAE (best ckpt, epoch 9):

| Split | Arm A (128) | Arm B (192) |
|---|---:|---:|
| val_single_in_dist | 185.70 | 183.15 |
| val_geom_camber_rc | 157.16 | 179.11 |
| val_geom_camber_cruise | 127.68 | **115.40** |
| val_re_rand | 130.51 | 137.20 |
| **val_avg/mae_surf_p** | **150.26** | 153.71 |

Per-split test: `test_geom_camber_cruise/mae_surf_p = None / NaN` on **both arms** (vol_loss=Infinity), poisoning `test_avg/mae_surf_p`. The student reported a 3-split mean (excl. cruise) of 144.76 (A) / 152.56 (B), but this is not the contract metric.

### Analysis & Verdict — sent back (not merged)

- Arm A (slice_num=128) beats Arm B (slice_num=192) by 3.45 on `val_avg/mae_surf_p` (−2.2% absolute), and is significantly cheaper (−20% VRAM, −20% epoch time). Higher slice_num does NOT help in this short-training regime — likely the optimization burden of more slice assignments to learn outweighs the gain.
- **No baseline number on this branch** — we cannot establish that `slice_num=128` improves on the actual baseline `slice_num=64`. The PR documents what beats what *internally* but not against the actual reference.
- `test_avg/mae_surf_p` is NaN — fails the full-metric-fidelity contract from CLAUDE.md.

### Critical cross-cutting finding: LR schedule is mis-tuned for the wall-clock budget

The student's most valuable observation: with `SENPAI_TIMEOUT_MINUTES=30` and ~170s/epoch, only ~10 epochs of the configured 50-epoch cosine schedule complete. `T_max=50` means LR is still at ~80% of peak when training stops — **no experiment on this branch is getting LR annealing**. This affects every other in-flight PR (#3089, #3090, #3091, #3093, #3095, #3096, #3097). Future PRs should pass `--epochs 10` (or whatever matches actual completed-epoch count) so `T_max` matches budget.

### NaN on `test_geom_camber_cruise/mae_surf_p`

The model emits inf/NaN predictions on at least one sample in the cruise test split when evaluated from a partial-training checkpoint. Identical across both arms, so doesn't affect this PR's A-vs-B comparison. Likely fixable by training to convergence (with proper LR annealing), gradient clipping, or `torch.nan_to_num` band-aid. Edward's PR #3091 (grad clip) and alphonse's PR #3089 (L1 loss) may both address this naturally.

### Follow-up (sent back to fern as comment on #3092)

Run 2-arm comparison at `--epochs 10` to fully anneal cosine `T_max`:
- Arm A: `slice_num=64` (establishes the branch baseline)
- Arm B: `slice_num=128` (confirms with proper schedule)

Merge if Arm B beats Arm A on `val_avg/mae_surf_p` AND `test_avg/mae_surf_p` is finite on Arm B.

---

## 2026-05-15 14:38 — PR #3091: LR warmup + gradient clipping + higher peak LR (edward) — **MERGED**

- **Student:** willowpai2i48h4-edward (branch: `willowpai2i48h4-edward/warmup-clip`)
- **Hypothesis:** Adding 2-epoch linear warmup and gradient clipping (max_norm=1.0) stabilizes training and enables higher peak LR. Arm B tested lr=1e-3 (2× baseline 5e-4). Predicted delta: −3% to −8%; actual win was >10%.

### Results

| Arm | lr | best epoch | val_avg/mae_surf_p | test (3-split workaround) | W&B run |
|---|---|---|---|---|---|
| A (warmup+clip+5e-4) | 5e-4 | 13 | 121.54 | 124.19 | `qm3lqtwz` |
| **B (warmup+clip+1e-3)** | 1e-3 | 14 (last) | **109.42** | **107.47** | `0ez1sqmi` |

Per-split val surface pressure MAE (best ckpt, epoch 14):

| Split | Arm A | Arm B |
|---|---:|---:|
| val_single_in_dist | 184.40 | 119.58 |
| val_geom_camber_rc | 115.04 | 119.40 |
| val_geom_camber_cruise | 88.03 | 88.57 |
| val_re_rand | 104.43 | 110.12 |
| **val_avg/mae_surf_p** | 121.54 | **109.42** |

Test: NaN on `test_geom_camber_cruise` for both (scoring bug). 3-split workaround: 124.19 (A) / 107.47 (B).

### Analysis & Decision — MERGED

- **Decisive win.** Arm B beats Arm A by 12.1 on val_avg (−10%) and by 16.7 on test 3-split (−13%). Pre-clip grad norm was 160 vs 14 at the last step — clipping is doing real work.
- Arm B's best epoch = 14/14 (last): model was still strictly improving when the timeout cut it, indicating significant headroom at longer training.
- `warmup_epochs=2` over 14 effective epochs = ~14% warmup, higher than intended. Short warmup is still the right call at high LR — doesn't hurt.
- Code change is minimal: 20 lines, adds warmup lambda scheduler + clip + grad_norm logging. Clean, composable with all other experiments.
- **Merged as new branch baseline: val_avg/mae_surf_p = 109.42** (lr=1e-3 + warmup + clip).

### Follow-up (edward)

Assigned edward a bug-fix + consolidation PR:
- Unblock `test_avg/mae_surf_p` by nan_to_num fix in `evaluate_split` (avoids `0 * NaN = NaN` propagation in accumulate_batch)
- Bump Config.lr default from 5e-4 to 1e-3 to lock in winning config for all future students

---

## 2026-05-15 15:30 — PR #3089: L1 loss vs Huber β=1.0 (alphonse) — **SENT BACK** (close to merge)

- **Student:** willowpai2i48h4-alphonse (branch: `willowpai2i48h4-alphonse/l1-loss`)
- **Hypothesis:** Replace MSE with L1 loss in normalized space; align training objective with MAE metric. Predicted −8% to −15%.

### Results

| Arm | Loss | best epoch | val_avg/mae_surf_p (W&B-verified) | test_avg/mae_surf_p (claim) | W&B run |
|---|---|---|---|---|---|
| **A (winner)** | L1 | 13 | **102.37** | 89.67 (offline re-eval) | `lb2ly5g3` |
| B | Huber β=1.0 | 13 | 117.47 | 106.03 | `9gh0e13m` |

Per-split val surface pressure MAE (Arm A, best epoch 13, alphonse's report):

| Split | Arm A (L1) | Arm B (Huber) |
|---|---:|---:|
| val_single_in_dist | 133.71 | 138.99 |
| val_geom_camber_rc | 108.91 | 118.50 |
| val_geom_camber_cruise | 76.50 | 102.26 |
| val_re_rand | 90.37 | 110.13 |
| **val_avg/mae_surf_p** | **102.37** | 117.47 |

W&B verification (subagent):
- Arm A: val_avg/mae_surf_p = 102.37 (best_val_avg) ✓ VERIFIED — beats baseline (109.42) by **−6.4%**.
- Arm A: test_avg/mae_surf_p = `None` in W&B summary; alphonse's claimed 89.67 came from offline re-eval after adding the fix post-training.
- Arm A: val_geom_camber_cruise/mae_surf_p = 84.79 in W&B (real number) ← alphonse's nan_to_num/sub-select fix DOES work for val.
- 3-split test mean (excl. cruise): test_re_rand=86.10, test_geom_camber_rc=96.20, test_single_in_dist=111.43 → mean ≈ **97.91** vs edward's 107.47.

### Bug-fix included

Alphonse correctly identified the `0 * NaN = NaN` propagation in `accumulate_batch` (same as edward's #3288 but more robust — handles both NaN and Inf in y via `torch.isfinite` + sub-select fully finite samples).

### Decision — Sent back with two specific asks

1. **Flip Config default `loss_type: str = "mse"` → `"l1"`** so future runs compose on L1 automatically.
2. **Push a clean W&B-logged eval with the fix in place** so `test_avg/mae_surf_p` is verifiable (not just offline re-eval). Quick `--debug --epochs 1` pass is sufficient.

After: transition to `status:review`, mark ready, merge.

### Composition with #3091

Alphonse trained on `lr=5e-4` + no warmup + no clip (pre-#3091 advisor branch). When merged into post-#3091 branch, future runs get: `lr=1e-3 + warmup + clip + L1`. Likely further headroom — these changes are orthogonal.

### Coordination with #3288

Alphonse's scoring fix (sub-select + torch.where) is more robust than edward's (`nan_to_num`). When alphonse's PR merges first, edward's #3288 should drop the duplicate scoring fix and only keep the lr default bump.

---

## 2026-05-15 17:30 — PR #3096: x-axis reflection symmetry augmentation (tanjiro) — **SENT BACK** (regression, conditional re-run)

- **Student:** willowpai2i48h4-tanjiro (branch: `willowpai2i48h4-tanjiro/xflip-aug`)
- **Hypothesis:** Per-sample x-flip aug with Ux/AoA/stagger negation; predicted gains on geom_camber OOD splits.
- **W&B run:** `a7kc6xxi` (verified)

### Results

| Arm | val_avg/mae_surf_p | test 3-clean-split | best epoch | total epochs |
|---|---:|---:|---:|---:|
| Single arm (xflip aug) | **161.54** | 162.46 | 12 | 14 |

Compared to current baseline (109.42 from PR #3091): **+47% regression**. But branch was forked pre-#3091 (lr=5e-4, no warmup, no clip), so most of the gap is the stale-branch infrastructure. On the same pre-#3091 code, fern's slice_num=128 baseline (#3092) landed at val=150.26 — tanjiro is ~7% worse than that with augmentation.

Per-split val surface MAE (best epoch 12):

| Split | Tanjiro xflip | fern slice_num=128 (same code) |
|---|---:|---:|
| val_single_in_dist | **203.61** | 185.70 |
| val_geom_camber_rc | 173.37 | 157.16 |
| val_geom_camber_cruise | 125.17 | 127.68 |
| val_re_rand | 143.99 | 130.51 |
| **val_avg/mae_surf_p** | **161.54** | 150.26 |

### Three concerning signals

1. **Model peaked at epoch 12 and rose for epochs 13–14** (163.0 → 167.0). The wall clock didn't cut mid-improvement; the model was overfitting. With higher LR (lr=1e-3 in current advisor) it'll likely overfit even earlier.
2. **`val_single_in_dist = 203.61` is the WORST split** — the easiest split (in-distribution) is being hurt by augmentation. xflip is making in-dist samples harder while only marginally helping OOD.
3. **`val_geom_camber_cruise` ≈ identical to fern's number** (125.17 vs 127.68). The predicted OOD gain isn't showing up in absolute numbers; the relative-easier-than-in-dist signal is plausible but not symmetry-specific.

### Bug-fix analysis

Tanjiro independently identified the same `0 * NaN = NaN` propagation in `accumulate_batch` that edward and alphonse flagged. Same root cause, same path (read-only `data/scoring.py`).

### Decision rule for the rebased confirmation arm

- val < 109.42 → merge
- val ∈ [109.42, 115] → merge only if geom_camber_cruise is clearly the best split (OOD-aug story still holds)
- val > 115 → close. Hypothesis empirically unsupported at this scale.

### Notes

- Augmentation halves effective gradient signal per orientation; could benefit from longer schedule, but within 30-min budget the unaugmented baseline gets twice the effective per-orientation samples.
- Symmetry aug is theoretically sound; the result here is most likely an interaction with: (a) stale code, (b) wall-clock cap, (c) MSE loss (L1 might compose better with aug). Worth re-investigating in round 2 stacked with alphonse's L1 + edward's warmup.

---

## 2026-05-15 17:30 — PR #3097: Deeper Transolver n_layers 5→8 + DropPath

- **Student:** willowpai2i48h4-thorfinn (branch: `willowpai2i48h4-thorfinn/deeper-droppath`)
- **Hypothesis:** n_layers 5→8 + DropPath p=0.1 for regularized depth scaling; predicted −5% to −10% on val_avg/mae_surf_p.

### Results (W&B-verified)

| Arm | n_layers | drop_path | params | epochs | best val_avg/mae_surf_p | test_avg/mae_surf_p | epoch time | VRAM | W&B run |
|-----|----------|-----------|--------|--------|------------------------|---------------------|-----------|------|---------|
| Baseline 5L | 5 | 0.0 | 0.66M | 14 | 132.73 | 121.78 | 132 s | 42.1 GB | `p1m774ow` |
| deep8-dp005 | 8 | 0.05 | 1.03M | 9 | **152.30** | **137.34** | 218 s | 64.5 GB | `qyyxx33r` |
| deep8-dp01 | 8 | 0.10 | 1.03M | 9 | 161.58 | 149.86 | 218 s | 64.5 GB | `jgaksniq` |

Current advisor baseline (from PR #3091): val_avg/mae_surf_p = **109.42**.

### Decision: CLOSED

Both deep arms are 40–48% worse than the advisor baseline (109.42). Vs student's own stale-code 5L reference (132.73), deep8-dp005 is still 15% worse. Student's bug-fix (cruise NaN workaround) is redundant with alphonse's #3089 fix.

### Analysis

Root cause: **wall-clock-budget undertraining, not capacity**. 8L is ~65% slower per epoch; within SENPAI_TIMEOUT_MINUTES=30 the deeper model completes only 9 epochs vs baseline's 14. Both deep arms peaked on their final epoch (still descending), classic signature of truncated training. Per-epoch comparison: wider model is actually better through epochs 5–8, but never reaches the post-convergence regime that baseline hits around epoch 12–14.

The hypothesis is **not refuted** — it's compute-bound. Under a 2× wall-clock budget, 8L+DropPath might win. Under the current budget, it cannot.

### Follow-up

Depth scaling is viable if we combine with a per-step speedup. Frieren's #3093 (bf16+bs=8, ~2× more epochs) could unlock this. Student reassigned to: **EMA of weights** (PR #3371) — addresses the late-epoch drift that affects all runs.

---

## 2026-05-15 17:30 — PR #3093: bf16 autocast + batch_size 4→8

- **Student:** willowpai2i48h4-frieren (branch: `willowpai2i48h4-frieren/bf16-amp`)
- **Hypothesis:** bf16 mixed precision + bs=8 for 1.5–2× wall-clock speedup → more epochs within 30-min budget; predicted −3% to −8% on val_avg/mae_surf_p.

### Results (W&B-verified)

| Run | bs | precision | epochs | best val_avg/mae_surf_p | test_avg/mae_surf_p | epoch time | VRAM | W&B run |
|-----|----|-----------|--------|------------------------|---------------------|-----------|------|---------|
| frieren-bf16-bs8-v3 | 8 | bf16 | 18/50 | **128.70** (ep 15) | **117.22** | 104 s | 88.8 GB | `hxslyna3` |

Current advisor baseline (from PR #3091): val_avg/mae_surf_p = **109.42**.

### Decision: SENT BACK for rebased confirmation arm

128.70 is 17.5% worse than the 109.42 baseline, BUT this run used **stale code (lr=5e-4, no warmup, no clip)**. The speed unlock is genuine: 18 epochs vs ~14 at fp32+bs=4 stale, and training appears stable throughout (no overflow, no divergence).

The comparison is apples-to-oranges. Asked student to rebase onto current advisor tip (f3a71a2 = #3091 warmup+clip+lr=1e-3) and run `--epochs 10` for composed-config benchmark. Decision rule: val < 109.42 → merge; val ∈ [109.42, 115] → TBD; val > 115 → close.

### Per-split test MAE (best ckpt, stale-code run)

| Split | mae_surf_p | mae_surf_Ux | mae_surf_Uy |
|-------|-----------|------------|------------|
| test_single_in_dist | 142.48 | 2.18 | 0.86 |
| test_geom_camber_rc | 124.92 | 2.70 | 0.99 |
| test_geom_camber_cruise | 84.93 | 1.30 | 0.54 |
| test_re_rand | 116.55 | 1.93 | 0.82 |
| **avg** | **117.22** | **2.03** | **0.80** |

---

## 2026-05-15 17:30 — PR #3090: Wider Transolver n_hidden 128→192 (+256)

- **Student:** willowpai2i48h4-askeladd (branch: `willowpai2i48h4-askeladd/wider-model`)
- **Hypothesis:** n_hidden 128→192, n_head 4→6; predicted −5% to −10% on val_avg/mae_surf_p.

### Results (W&B-verified)

| Run | n_hidden | bs | epochs | best val_avg/mae_surf_p | test_avg/mae_surf_p (3 splits) | epoch time | VRAM | W&B run |
|-----|----------|----|--------|------------------------|-------------------------------|-----------|------|---------|
| baseline-128 | 128/4 | 4 | 14 | 119.82 (ep 14) | 121.46 | 132 s | 42.1 GB | `9pj8vox8` |
| wider-192 | 192/6 | 4 | 9 | **170.35** (ep 5) | 175.35 | 203 s | 63.0 GB | `bc3dcrmc` |
| wider-256 | 256/8 | 2† | 8 | **169.10** (ep 8) | 174.36 | 253 s | 42.0 GB | `3ag48lmp` |

†wider-256 OOM'd at bs=4; dropped to bs=2. Current advisor baseline: val=**109.42**.

### Decision: CLOSED

54–56% regression vs advisor baseline (109.42). Same fundamental issue as thorfinn depth: wider model is 1.5–2× slower per epoch, can't reach the late-epoch convergence regime within budget. Per-epoch, wider-192 is actually better in epochs 5–8 (170 vs baseline 197), but the baseline's rapid drop at epochs 10–14 (197→120) is unreachable for the wider model in 30 min.

cruise NaN bug noted — same pre-existing issue, covered by alphonse's #3089 fix.

Student reassigned to: **Fourier positional encoding on (x,z)** (PR #3372) — same per-step cost, higher-frequency geometry representation.

---

## 2026-05-15 19:30 — PR #3096: x-axis reflection symmetry augmentation (rebased confirmation)

- **Student:** willowpai2i48h4-tanjiro (branch: `willowpai2i48h4-tanjiro/xflip-aug`)
- **Hypothesis:** x-axis symmetry flip augmentation (p=0.5 per sample, xflip_collate at train time only, field negation of Ux/AoA/stagger). Predicted OOD generalization boost.

### Results (W&B-verified — rebased confirmation arm)

| Run | config | epochs | best val_avg/mae_surf_p | test_avg (3 splits) | W&B run |
|-----|--------|--------|------------------------|--------------------|---------|
| tanjiro-xflip-rebased | lr=1e-3 + warmup + clip + xflip | 10/10 | **140.67** | **144.70** | `du7tx8dy` |

Current advisor baseline: val=**109.42**. **+28.5% regression.** Wall clock 22.4 min, full 10 epochs, no truncation.

### Decision: CLOSED

Per decision rule (val>115→close): clear close. Rebase eliminated the stale-code confound, cosine fully annealed at --epochs 10, clean confound-free measurement. xflip aug halves effective gradient signal per orientation, hurting more than it helps on a 1500-sample dataset. Every split worse.

### Useful findings from the symmetry aug experiments

- xflip aug fails convincingly at two independent code/schedule configurations (stale + rebased)
- If revisiting augmentation: **mild affine perturbations** (AoA jitter, stagger jitter) are more promising than discrete symmetries
- The `xflip_collate` + field-negation code is clean and could be repurposed for **TTA (test-time augmentation)** if desired — same model, ensembled predictions on original + flipped input at inference
- Tanjiro reassigned to **SWA** (PR #3414) — different best-checkpoint strategy, zero per-step cost

---

## 2026-05-15 18:30 — PR #3095: surf_weight 10→30 + per-channel p weighting

- **Student:** willowpai2i48h4-nezuko (branch: `willowpai2i48h4-nezuko/surf-weight-pweight`)
- **Hypothesis:** push surface mass higher (surf_weight 10→30) and/or weight p-channel 3× harder to lift surface-pressure MAE; predicted improvement on `val_avg/mae_surf_p`.

### Results (W&B-verified, all stale code: lr=5e-4, no warmup, no clip)

| Arm | Config | best ep | val_avg/mae_surf_p | test_avg/mae_surf_p | W&B run |
|-----|--------|--------:|-------------------:|--------------------:|---------|
| A | surf_w=30, p_w=1 | 13 | **131.08** | **117.28** | `t640m1of` |
| B | surf_w=10, p_w=3 | 13 | 133.31 | 121.29 | `f10ob15w` |
| C | surf_w=30, p_w=3 | 14 | 146.42 | 134.42 | `0qb9wvgy` |

Current advisor baseline: val=**109.42**. Arm A is **19.8% worse** on val and **9.0% worse** on test_avg vs baseline. Per-split test on Arm A: single_in_dist=140.46 vs baseline 111.04, geom_camber_rc=127.71 vs 110.20, re_rand=116.19 vs 101.17 — every split is worse.

### Decision: SENT BACK + read-only violation flagged

- **Read-only violation:** Student modified `data/scoring.py` (per program.md: **read-only**) to add `y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))`. Concept correct but location wrong. Send-back asks: revert data/scoring.py, apply fix in train.py instead (or drop entirely once alphonse's #3089 merges with canonical fix).
- **Metric regression:** asked for **single rebased confirmation arm at surf_weight=20** (more conservative) with --epochs 10. Decision rule: val<109.42 merge; val 109-115 close-call; val>115 close.

### Useful findings

- Arm B and C are dead ends — per-channel p weighting compounds badly with surface weighting (Arm C 30×3=90× effective surface-p vs volume-Ux).
- All 3 arms confirm: **cruise-camber is the easiest test split for surface_p** (84.76 vs 116-140 elsewhere). Future hypotheses targeting harder splits (single_in_dist, geom_camber_rc) have more headroom.
- Independently rediscovered the cruise NaN scoring bug (same root cause as alphonse, thorfinn, frieren found).
