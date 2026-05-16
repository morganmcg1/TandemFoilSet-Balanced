# SENPAI Research Results — Round 5 (charlie-pai2i-24h-r5)

Advisor branch: `icml-appendix-charlie-pai2i-24h-r5`. Primary ranking metric: `val_avg/mae_surf_p` (lower better). Test-time decision metric: `test_avg/mae_surf_p`.

## 2026-05-15 14:25 — PR #3266: Per-sample scale-invariant loss to equalize Re-regime gradients [MERGED — round-5 baseline anchor]

- Branch: `charliepai2i24h5-frieren/per-sample-instance-norm-targets`
- Student: charliepai2i24h5-frieren
- Hypothesis: divide each sample's `vol_loss_per_sample + surf_weight * surf_loss_per_sample` by that sample's field std (computed in normalized-y space) before averaging across the batch, so the gradient contribution is equalized across the ~40× per-sample y_std variation between low-Re and high-Re samples.
- Status: MERGED as the round-5 anchor baseline. First terminal result on this branch.

### Results

| Split | val_mae_surf_p | test_mae_surf_p |
|---|---:|---:|
| single_in_dist | 142.1946 | 125.8483 |
| geom_camber_rc | 136.1165 | 130.8474 |
| geom_camber_cruise | 95.7637 | 85.2854 |
| re_rand | 121.4363 | 115.4971 |
| **avg** | **123.8778** | **114.3695** |

Metric artifacts:
- `models/model-charliepai2i24h5-frieren-per_sample_instance_norm_targets-20260515-132755/metrics.jsonl`
- `models/model-charliepai2i24h5-frieren-per_sample_instance_norm_targets-20260515-132755/metrics.yaml`

### Analysis

The per-split pattern matches the predicted mechanism. `geom_camber_cruise` has the lowest field std in the corpus (program.md notes cruise per-sample y std mean 164, max 506 vs raceCar single mean 458 max 2077) and was therefore most under-weighted by the original global-MSE loss. Under the scale-invariant loss, cruise becomes the easiest split (lowest val surf_p, 95.76). This is exactly the directional claim of the hypothesis.

Run hit the 30-min wall clock at epoch 14/50; val_avg/mae_surf_p was still falling at ~-2.6 per epoch with no plateau. The cosine schedule was set to T_max=50 so LR did not fully decay (was at ~89% of peak when training stopped). The student's own follow-up #2 (`--epochs 14` with matched cosine T_max) is a likely compounding improvement we can run later.

### Critical operational bug found

`data/scoring.py::accumulate_batch` has a NaN-propagation bug: one sample (`test_geom_camber_cruise/000020.pt`) has 761 NaN in the p channel of y. The intended sample-level skip via `y_finite = isfinite(y).all(dim=-1)` is defeated downstream by `err = |pred - NaN| = NaN`, then `NaN * 0 = NaN` slips past the masked sum, producing NaN `test_avg/mae_surf_p`. Because `data/` is read-only, the fix lives in `train.py::evaluate_split`: pre-skip samples with non-finite y from the mask and `torch.nan_to_num` the y before passing to `accumulate_batch`. This workaround is included in the merged baseline and propagated to the seven other in-flight round-5 PRs via PR comments.

## 2026-05-15 15:35 — PR #3265: FiLM Re/AoA/NACA conditioning every Transolver block [SENT BACK — beat baseline but had merge conflicts]

- Branch: `charliepai2i24h5-fern/film-flow-condition-every-block`
- Student: charliepai2i24h5-fern
- Hypothesis: per-block FiLM scale+shift modulation on global flow vector (log Re, AoAs, NACAs, gap, stagger) injected after each Transolver block's residual, giving every layer direct access to the global flow conditioning rather than only the input MLP.
- Status: WINNER on absolute numbers (val 122.27 / test 112.17 vs baseline 123.88 / 114.37) but the branch was cut before PR #3266 landed, so squash-merge had conflicts in `train.py`. Sent back for rebase + re-run; FiLM should compound cleanly with the scale-invariant loss in the merged baseline.

### Results

| Metric | Baseline (PR #3266) | FiLM (#3265) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 123.8778 | **122.2673** | -1.30% |
| test_avg/mae_surf_p | 114.3695 | **112.1684** | -1.92% |

Per-split val surface pressure MAE:

| Split | Baseline | FiLM | Δ |
|---|---:|---:|---:|
| single_in_dist | 142.19 | 141.32 | -0.6% |
| geom_camber_rc | 136.12 | 138.27 | +1.6% |
| geom_camber_cruise | 95.76 | 98.62 | +3.0% |
| **re_rand** | **121.44** | **110.87** | **-8.7%** ← big OOD win |

Per-split test surface pressure MAE:

| Split | Baseline | FiLM | Δ |
|---|---:|---:|---:|
| single_in_dist | 125.85 | 120.73 | -4.1% |
| geom_camber_rc | 130.85 | 128.81 | -1.6% |
| geom_camber_cruise | 85.29 | 87.81 | +3.0% |
| re_rand | 115.50 | 111.32 | -3.6% |

Best epoch 13/50, run cut at epoch 14 by 30-min wall clock. Model is 829K params (FiLM adds 166K on top of baseline 663K), no OOM, no instability.

### Analysis

The FiLM mechanism delivers exactly the predicted Re-rand OOD improvement (-8.7% on val_re_rand, the single largest per-split win), confirming the hypothesis: explicit per-layer modulation by log Re + AoA + geometry is meaningfully different from baking those features into the input MLP. Three of four test splits improve. Cruise val regresses by 3% — small enough that compounding with the scale-invariant baseline (which had cruise as its strongest split) is likely to net out positive.

Predicted -8% to -15%; actual -1.3% to -1.9% averaged. Less impressive than predicted on val_avg, but the test_avg win is meaningful and the re_rand win is exactly the hypothesis prediction. Compounding with scale-invariant loss is the natural next step.

Sent back for rebase onto current advisor branch + re-run with FiLM stacked on top of scale-invariant loss + NaN fix.

## 2026-05-15 15:35 — PR #3272: Surface arc-length Fourier PE for surface pressure accuracy [CLOSED — regression]

- Branch: `charliepai2i24h5-askeladd/surface-arclen-positional-encoding`
- Student: charliepai2i24h5-askeladd
- Hypothesis: inject sin/cos Fourier features of the signed-arc-length input dims into the per-node embedding (only on surface nodes), giving the slice-attention an explicit periodic boundary-following positional structure beyond the raw MLP-projected features.
- Status: CLOSED. Both arms (n_freqs=8 and n_freqs=16) regressed >10% from baseline. Surface PE direction not productive for this benchmark.

### Results

| Arm | params | val_avg/mae_surf_p | test_avg/mae_surf_p | vs baseline (val) |
|---|---:|---:|---:|---:|
| baseline (PR #3266) | 662,359 | 123.8778 | 114.3695 | — |
| Arm A (n_freqs=16) | 670,951 | 151.123 | 138.528 | +21.9% worse |
| Arm B (n_freqs=8) | 666,847 | **139.838** | **125.336** | **+12.9% worse** |

Arm B beats Arm A on every split, indicating the larger PE bandwidth (n_freqs=16) is over-parameterised for the small arc-length signal each foil carries. Even Arm B (the winner of the ablation) is >12% worse than baseline on val_avg.

### Analysis

The arc-length signed-distance features are already in dims 2–3 of the raw input x and pass through the preprocessor MLP. Adding a sin/cos Fourier expansion of the same dims as a separate additive embedding did not help the model — the preprocessor is apparently already extracting the relevant frequencies. The hypothesis assumed under-utilisation of arc-length, but the data shows the opposite. Surface arc-length Fourier PE is not a productive direction for this benchmark.

Important caveat: this PR ran on vanilla MSE (the branch was cut before PR #3266 landed). Even applying the scale-invariant loss on rebase is very unlikely to close a >12% gap; the scale-invariant loss alone moved baseline numbers by only ~1.3% on val_avg in the merged result. Closed as clear regression.

## 2026-05-15 16:23 — PR #3281: EMA model weights for checkpoint + test eval [MERGED — new round-5 baseline]

- Branch: `charliepai2i24h5-frieren/ema-weights-checkpoint`
- Student: charliepai2i24h5-frieren
- Hypothesis: maintain a shadow EMA copy of the model (decay=0.999, ~1000-step averaging window ≈ 2.7 epochs) updated after every `optimizer.step()`; use the EMA weights for validation, checkpoint selection, and final test eval. Stacks on top of the merged scale-invariant loss baseline (#3266).
- Status: MERGED as the new round-5 baseline. Strongest result so far, with the gain concentrated on the OOD splits exactly as the flat-minimum hypothesis predicts.

### Results

| Metric | Baseline (PR #3266) | EMA (#3281) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 123.8778 | **114.1704** | **-7.84%** |
| test_avg/mae_surf_p | 114.3695 | **102.0813** | **-10.74%** |

Per-split val surface pressure MAE:

| Split | Baseline | EMA | Δ |
|---|---:|---:|---:|
| single_in_dist | 142.19 | 138.48 | -2.6% |
| geom_camber_rc | 136.12 | 130.84 | -3.9% |
| geom_camber_cruise | 95.76 | 84.60 | -11.7% |
| re_rand | 121.44 | 102.77 | **-15.4%** ← big OOD win |

Per-split test surface pressure MAE:

| Split | Baseline | EMA | Δ |
|---|---:|---:|---:|
| single_in_dist | 125.85 | 121.81 | -3.2% |
| geom_camber_rc | 130.85 | 115.66 | **-11.6%** |
| geom_camber_cruise | 85.29 | 71.60 | **-16.1%** |
| re_rand | 115.50 | 99.26 | **-14.0%** |

Best epoch 14/50, run cut at epoch 14 by 30-min wall clock. ~0.4 GB extra VRAM for shadow weights, no other overhead.

Metric artifacts:
- `models/model-charliepai2i24h5-frieren-ema_weights_checkpoint-20260515-153007/metrics.jsonl`
- `models/model-charliepai2i24h5-frieren-ema_weights_checkpoint-20260515-153007/metrics.yaml`

### Analysis

EMA does exactly what Morningstar et al. (TMLR 2024) describes: averaging the noisy SGD/AdamW iterate gives a flatter-minimum weight that generalizes better to OOD samples. The gradient of improvement matches the predicted shape — in-distribution (-2.6% / -3.2%) is the smallest gain, OOD splits are 4-7× larger (re_rand val -15.4%, cruise test -16.1%). The 4-track score is now substantially more balanced (val splits range 84.6–138.5 vs baseline 95.8–142.2).

The val curve was still monotonically dropping at ~-4/epoch when the wall clock cut training (epoch 14, val=114). This confirms the "EMA helps most when training is undercooked" intuition: the EMA is already living near the flat minimum that the raw weights have not yet found. Compounding with the cosine-T_max fix (next round) is the obvious follow-up.

Implementation is exactly the spec: deepcopy at init, requires_grad_(False), update_ema after every optimizer.step() with decay=0.999, EMA used for val + checkpoint + test eval. No other hyperparameters touched; per-sample scale-invariant loss and `evaluate_split` NaN sanitization preserved.

## 2026-05-15 16:32 — PR #3271: Signed-log pressure target transform [CLOSED — regression]

- Branch: `charliepai2i24h5-thorfinn/log-transform-pressure-target`
- Student: charliepai2i24h5-thorfinn
- Hypothesis: apply `signed_log(p) = sign(p) * log(1 + |p|/eps)` to the pressure target before normalization, then invert at eval. Idea: compress the ~2500× Re-driven dynamic range into a more uniform scale so the loss signal isn't dominated by high-Re extremes.
- Status: CLOSED. Both arms (eps=1.0 and eps=0.1) regressed >15% from baseline. Direction not productive for this benchmark.

### Results

| Metric | Baseline (PR #3266) | eps=1.0 | eps=0.1 (best arm) |
|---|---:|---:|---:|
| val_avg/mae_surf_p | 123.88 | 151.72 (+22.5%) | 144.12 (+16.3%) |
| test_avg/mae_surf_p | 114.37 | 136.54 (+19.4%) | 132.17 (+15.6%) |

### Analysis

The signed-log transform did the opposite of what was hoped: it broke the model's ability to track absolute pressure scale. The single_in_dist split (largest Re range) was supposed to benefit most from log compression — but it regressed worst (val 142.2 → 213.3, +50%). The transform is decoupling the pressure prediction from its physical magnitude in a way that the post-hoc invert can't recover.

The student's two-arm sweep was well-designed (eps=1.0 vs eps=0.1 explored the compression-strength axis cleanly). The NaN workaround was correctly applied. Just the hypothesis was wrong. Closed.

## 2026-05-15 15:52 — PR #3267: Separate surface decoder head [SENT BACK — beat student's own baseline but not the merged baseline]

- Branch: `charliepai2i24h5-tanjiro/separate-surface-decoder-head`
- Student: charliepai2i24h5-tanjiro
- Hypothesis: add a dedicated MLP head for surface-node predictions, separate from the volume head. Surface and volume signals differ (surface has sharp gradients, volume is smoother), so separating heads should let each task learn its own optimal mapping without interference.
- Status: SENT BACK. The hypothesis works in the student's own ablation (-5.3% val_avg vs their no-scale-inv-loss baseline) but the absolute val_avg = 128.70 is +12.7% worse than the merged baseline (114.17) because the student's branch was cut before PR #3281 landed.

### Results

| Metric | Student's own baseline | Surface head (this PR) | Δ (within PR) | Merged baseline #3281 |
|---|---:|---:|---:|---:|
| val_avg/mae_surf_p | 135.84 | 128.70 | -5.3% | **114.17** |
| test_avg/mae_surf_p | 122.49 | 119.01 | -2.8% | **102.08** |

Within-PR per-split val (surface head vs student's single-head baseline):
- val_single_in_dist: 164.16 → 148.88 (-9.3%) ← largest within-PR gain, matches the "sharper local gradients" mechanism
- val_geom_camber_rc: 142.77 → 140.23 (-1.8%)
- val_geom_camber_cruise: 111.85 → 105.88 (-5.3%)
- val_re_rand: 124.60 → 119.81 (-3.8%)

### Analysis

The mechanism is real — surface and volume have meaningfully different signal characteristics, and a dedicated head improves both within-PR val (-5.3%) and test (-2.8%). But the student's PR runs neither the per-sample scale-invariant loss (#3266) nor the EMA weights (#3281), both of which are now part of the merged baseline. Their no-scale-inv single-head baseline (135.84) is +9.6% worse than #3266 baseline (123.88), and +18.9% worse than the EMA baseline (114.17). The surface-head improvement (-5.3% within-PR) is real but smaller than the cumulative improvement from those two merged changes (-7.84%).

The natural next step is a rebase + re-run on the current advisor branch, which inherits scale-invariant loss + EMA + NaN fix. The surface head's win on val_single_in_dist (-9.3%) is exactly the split where EMA has the smallest gain (-2.6%), suggesting the two mechanisms might compound positively. Sent back with detailed rebase instructions.

## 2026-05-15 16:48 — PR #3268: NACA camber mixup augmentation [CLOSED — net aggregate regression with fundamental mesh-mismatch issue]

- Branch: `charliepai2i24h5-alphonse/naca-camber-mixup`
- Student: charliepai2i24h5-alphonse
- Hypothesis: mix two samples' global conditions (dims 13–23 of x, including camber M) and targets with a Beta(α, α) lambda, p=0.5 of the time. Idea: generate intermediate-camber virtual samples to help the OOD camber holdouts (rc, cruise).
- Status: CLOSED. Both arms (α=0.2 and α=0.4) achieve real OOD gains on the targeted splits but with too-large in-dist regression to net positive on aggregate. Student correctly identified the underlying issue (mesh stays anchored to one sample while conditions interpolate).

### Results

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | vs student's no-mixup baseline | vs merged baseline #3281 |
|---|---:|---:|---:|---:|
| baseline (no-mixup, this PR) | 136.50 | 123.96 | — | +19.5% val, +21.4% test |
| α=0.2, p=0.5 | 139.07 | 126.52 | +1.9% val | +21.7% val, +23.9% test |
| α=0.4, p=0.5 | 136.84 | 126.33 | +0.3% val | +19.7% val, +23.6% test |

Per-split test mae_surf_p — both arms split improvements asymmetrically:
- α=0.2 wins on camber_rc (-9.2%) but loses elsewhere
- α=0.4 wins on camber_cruise (-10.5%) but loses elsewhere
- Both regress on in_dist (+9.9% / +10.3%) and re_rand (+6.1% / +6.5%)

### Analysis

The student's mesh-mismatch diagnosis is sharp: input-space mixup interpolates global conditions + target fields but **keeps the original mesh**, so every mixed batch carries a consistent geometry/condition mismatch signal that hurts in-dist samples. The asymmetry between α=0.2 (favours wider camber gap, rc) and α=0.4 (favours tighter camber gap, cruise) is real and useful intuition — no single fixed α handles both gaps.

The hypothesis is partially supported on the targeted axis but the technique is structurally limited for this dataset. The fundamental fix is to mix in latent space (manifold mixup) after the model has encoded the geometry — assigned as alphonse's next experiment (PR #3347).

The student also discovered and patched the NaN-propagation bug independently, and committed a `test_corrected` re-evaluation pattern for the α=0.4 arm (which ran before the patch landed). Both consistent with PR #3266's workaround.

Closed; alphonse reassigned to manifold mixup (PR #3347).

## 2026-05-15 17:26 — PR #3337: Surface-pressure L1 auxiliary loss [ASSIGNED — frieren]

- Branch: `charliepai2i24h5-frieren/surface-pressure-l1-aux-loss`
- Student: charliepai2i24h5-frieren
- Hypothesis: add `aux_weight * mean(|pred_surf_p - y_surf_p|)` (L1 on surface-pressure channel only, in normalized-y space) to the existing per-sample scale-invariant MSE. Directly aligns training objective with the MAE eval metric.
- Two arms: aux_weight ∈ {1.0, 3.0}. Expected -3% to -7% on val_avg.

## 2026-05-15 17:25 — PR #3346: Cosine T_max=15 + 1-epoch warmup + LR=7e-4 [ASSIGNED — thorfinn]

- Branch: `charliepai2i24h5-thorfinn/cosine-tmax-fix-warmup-lr7e-4`
- Student: charliepai2i24h5-thorfinn
- Hypothesis: aligned cosine schedule (T_max=15 matching the ~14-epoch wall-clock budget) + 1-epoch linear warmup + raised peak LR=7e-4. Canonical schedule fix expected to give a clean -2% to -5% recovery after the signed-log dead-end.

## 2026-05-15 17:27 — PR #3347: Manifold mixup at random Transolver block [ASSIGNED — alphonse]

- Branch: `charliepai2i24h5-alphonse/manifold-mixup`
- Student: charliepai2i24h5-alphonse
- Hypothesis: instead of input-space mixup (which couples the original mesh to interpolated conditions and hurts in-dist), mix latent representations at a uniformly-random block index in [0, 3) using Beta(α, α) lambda. The mixed latent lives on the learned manifold and contains no geometry/condition mismatch signal.
- Two arms: α ∈ {0.4, 0.2}, both with p=0.5 and max_block=3. Direct follow-up to the closed PR #3268.

## 2026-05-15 17:45 — PR #3269: Multi-scale slice attention (per-layer slice_num) [CLOSED — within seed noise]

- Branch: `charliepai2i24h5-nezuko/multi-scale-slice-attention`
- Student: charliepai2i24h5-nezuko
- Hypothesis: vary `slice_num` per layer — hourglass (`[32, 64, 128, 64, 32]`) or ascending (`[32, 48, 64, 96, 128]`) — to give different layers different physics-token capacity. Hourglass should help camber-OOD splits (more low-level slices to resolve sharp surface features).
- Status: CLOSED. Best arm (hourglass run 1) lands at `val_avg/mae_surf_p = 120.49` vs the merged EMA baseline `114.17` (+5.5% worse). A second hourglass run with the same config lands at `127.94`. Seed-to-seed Δ ≈ 7.5 is more than 2× larger than the apparent hourglass-vs-uniform effect ≈ 3.4 — within noise. Student's own analysis explicitly acknowledges the variance band overwhelms the effect.

### Results

| Arm | slice_num | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---|---:|---:|
| Baseline (uniform) | [64,64,64,64,64] | 123.92 | 111.80 |
| Hourglass run 1 | [32,64,128,64,32] | **120.49** | **109.17** |
| Hourglass run 2 (same config) | [32,64,128,64,32] | 127.94 | 117.65 |
| Ascending | [32,48,64,96,128] | 125.13 | 115.92 |

Metric artifacts (all committed under `models/`):
- `models/model-charliepai2i24h5-nezuko-multiscale-slice-hourglass-20260515-132756/metrics.jsonl`
- `models/model-charliepai2i24h5-nezuko-multiscale-slice-hourglass-20260515-152957/metrics.jsonl`
- `models/model-charliepai2i24h5-nezuko-multiscale-slice-ascending-20260515-141315/metrics.jsonl`
- `models/model-charliepai2i24h5-nezuko-multiscale-slice-baseline-uniform-20260515-163129/metrics.jsonl`

### Analysis

The mechanistic prediction (hourglass should help camber-OOD splits with finer surface features) failed in the right place to falsify the hypothesis: cruise-camber under hourglass run 1 *regressed* (90.99 vs 81.78 baseline), the opposite of the predicted direction. The split that improved most under hourglass run 1 (`single_in_dist`: 121.84 vs 138.77 baseline) is also the easiest split and where the multi-scale story shouldn't dominate. Combined with the seed-pair variance, no defensible win.

Useful pattern: the student's `retest_checkpoint.py` (re-evaluate saved checkpoints with the NaN fix and append `event:"test_corrected"` to `metrics.jsonl`) is a reusable artifact for future PRs that ran before a fix landed.

Closed; nezuko reassigned.

## 2026-05-15 17:46 — PR #3270: Transolver capacity scale-up (256h / 8l / 8h, bs=2, lr=3e-4) [CLOSED — undertrained at 30-min wall clock]

- Branch: `charliepai2i24h5-edward/transolver-capacity-scale-up`
- Student: charliepai2i24h5-edward
- Hypothesis: increase Transolver capacity by 6× (n_hidden=128→256, n_layers=5→8, n_head=4→8), with bs=2 to fit VRAM and lr=3e-4 to manage gradient noise.
- Status: CLOSED. Large model `val_avg/mae_surf_p = 165.12` vs merged EMA baseline `114.17` is a +44.6% regression. Large model completes only 5 epochs in 30 min wall clock (vs 14 for the baseline), and the val curve is still descending steeply at the timeout — undertrained, not broken. At fixed wall-clock budget, capacity scale-up loses.

### Results

| Arm | params | sec/epoch | epochs in 30 min | best_val_avg | best_epoch |
|---|---:|---:|---:|---:|---:|
| Baseline (128h, 5l, 4h, bs=4) | 0.66M | 132s | 14 | 134.48 | 13 |
| Large (256h, 8l, 8h, bs=2) | 3.94M | 388s | 5 | 165.12 | 5 |

Large is worse on every single val and test split — no silver lining. Peak VRAM 64.4 GB (32 GB headroom) — bottleneck is iteration cost, not memory.

Metric artifacts:
- `models/model-charliepai2i24h5-edward-transolver-large-256h-8l-fixed-20260515-152720/metrics.jsonl`
- `models/model-charliepai2i24h5-edward-transolver-baseline-128h-5l-20260515-162442/metrics.jsonl`

### Analysis

The hypothesis isn't necessarily wrong (val curve at epoch 5 was still falling −24/epoch ≈ −12%/epoch), but it's untestable under the current budget. The student's follow-up suggestions — bf16 mixed precision (to roughly halve per-epoch time), intermediate sizes (192h/6l/6h), and constant-depth variants (256h/5l/8h) — are reasonable, but bf16 is the highest-leverage next move because it unlocks capacity-revisit attempts as a whole. Reassigning edward to bf16 mixed precision as a fresh hypothesis.

Closed.

## 2026-05-15 17:47 — PR #3315: Cautious AdamW optimizer [SENT BACK — beat OLD baseline but lost to NEW EMA baseline; rebase to compound]

- Branch: `charliepai2i24h5-askeladd/cautious-adamw-optimizer`
- Student: charliepai2i24h5-askeladd
- Hypothesis: replace AdamW with Cautious AdamW (Liang et al. 2024): gate each update component by the sign-agreement mask `(m * g > 0)` where `m` is the EMA momentum and `g` is the original gradient, mean-rescaled with `clamp(min=1e-3)`. Should help noisy gradient landscapes by skipping disagreed-on update directions.
- Status: SENT BACK. Result is a real win on the OLD baseline (−4.72% val / −7.46% test vs #3266 123.88 / 114.37) but lost ground vs the NEW EMA baseline (#3281, 114.17 / 102.08). The mechanism (update-direction gating) is orthogonal to EMA (weight averaging), so a rebase + re-run should compound — predicted −2% to −5% on top of the EMA baseline.

### Results

| Comparison | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---:|---:|
| Old baseline (PR #3266) | 123.88 | 114.37 |
| Cautious AdamW (this PR) | **118.03** | **105.83** |
| New merged baseline (PR #3281, EMA + scale-inv) | **114.17** | **102.08** |

Per-split (val/test): big OOD wins (geom_camber_cruise −12.15% val, re_rand −15.24% val), in-dist regressions (+1.98%, +2.90% on single_in_dist and geom_camber_rc). The localization matches the Cautious-mask mechanism: it gates noisy update directions, which dominate the OOD-flavored splits.

Mask agreement: ~0.62 across all 14 epochs, stable. This is in the predicted range (0.5–0.85) and confirms the optimizer is doing real work (~38% of update components are gated to zero each step). The flat trajectory (no rise to 0.7–0.85 as in LLM pre-training) is consistent with the multi-split foil-regression loss surface staying noisy throughout training rather than smoothing.

Metric artifacts:
- `models/model-charliepai2i24h5-askeladd-cautious_adamw-20260515-164009/metrics.jsonl`
- `models/model-charliepai2i24h5-askeladd-cautious_adamw-20260515-164009/metrics.yaml`

### Analysis

The OOD wins (cruise −12.15%, re_rand −15.24% on val) come exactly where EMA's wins are smallest, suggesting the two mechanisms are complementary on the same axes. The in-dist regressions (single_in_dist +1.98%, geom_camber_rc +2.90%) point at this being a regularization-flavored effect — gated updates miss some easy improvements on the smoother in-dist distribution but generalize better to held-out test. EMA on top should soften the in-dist regression while preserving the OOD wins.

Sent back with explicit rebase instructions and predicted target val_avg ≈ 108–112.


## 2026-05-15 17:55 — PR #3373: bf16 mixed precision (AMP) [ASSIGNED — edward]

- Branch: `charliepai2i24h5-edward/bf16-mixed-precision`
- Student: charliepai2i24h5-edward
- Hypothesis: wrap the forward pass in `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`; backward + optimizer.step stay in fp32. Expected ~30-50% per-epoch wall-clock reduction → more effective epochs within the 30-min cap → measurable val_avg improvement (predicted -2% to -5%). Compute unlock makes future capacity-revisit experiments (the 256h/8l/8h sweep that #3270 couldn't fairly test) tractable.
- Two arms: bf16 + baseline-config (primary), bf16 + batch_size=8 (ablation, uses speed gain to double batch).

## 2026-05-15 17:56 — PR #3374: Stochastic depth (DropPath) regularization [ASSIGNED — nezuko]

- Branch: `charliepai2i24h5-nezuko/stochastic-depth`
- Student: charliepai2i24h5-nezuko
- Hypothesis: per-block DropPath during training (drop residual contribution with probability `p_drop`), all blocks active at eval. Each forward pass samples a sub-network → implicit ensemble + per-block robustness. Compounds with EMA (averaged across sub-network gradients → flat region in weight space). Predicted -1.5% to -4% val_avg, largest gains on `single_in_dist` (currently worst split at 138.48).
- Two arms: uniform p_drop=0.1 across blocks (primary), linearly-increasing 0→0.1 by depth (DeiT-style "drop path"; secondary).

## 2026-05-15 19:31 — PR #3337: Surface-pressure L1 auxiliary loss [MERGED — new round-5 baseline]

- Branch: `charliepai2i24h5-frieren/surface-pressure-l1-aux-loss`
- Student: charliepai2i24h5-frieren
- Hypothesis: add a pooled L1 term on the pressure channel of surface nodes (in normalized-y space) to the per-sample scale-invariant MSE objective. L1 aggregation `Σ|err|/n_surf` is shape-identical to the eval-time `mae_surf_p` metric, so gradient direction is sign-of-error which directly minimizes MAE. Compounds with EMA (loss-side vs parameter-trajectory mechanism).
- Status: MERGED. Best round-5 single-PR improvement.

### Results

| Metric | Baseline (#3281) | Arm A (w=1.0) | Δ% | Arm B (w=3.0) | Δ% |
|---|---:|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | 114.1704 | **106.8550** | **−6.41%** | 109.9254 | −3.72% |
| `test_avg/mae_surf_p` | 102.0813 | **96.8671** | **−5.11%** | 98.5929 | −3.42% |

Per-split val (Arm A, winner): single_in_dist −7.68%, geom_camber_rc −7.43%, geom_camber_cruise −3.79%, re_rand −5.55%.

Metric artifacts:
- `models/model-charliepai2i24h5-frieren-surf_p_l1_aux_w1.0-20260515-172842/metrics.jsonl`
- `models/model-charliepai2i24h5-frieren-surf_p_l1_aux_w3.0-20260515-182332/metrics.jsonl`

### Analysis

Per-split ordering matched the heavy-tail prediction: largest gains on the splits with the largest |p| dynamic range (single_in_dist and geom_camber_rc), smallest on cruise (smallest |p| range). Arm B's collapse on single_in_dist (−2.74% vs Arm A's −7.68%) suggests w=3.0 over-rides the per-sample scale-invariance specifically on surface pressure, undoing the regime balancing that PR #3266 provides. The two ideas compose at moderate weighting (w=1.0) but not at high weighting.

Cumulative round-5: −13.74% val_avg (123.88 → 106.86) and −15.30% test_avg (114.37 → 96.87) over the pre-round-5 baseline.

## 2026-05-15 19:35 — PR #3267: Separate surface decoder head (rebased) [CLOSED — clear regression on merged baseline]

- Branch: `charliepai2i24h5-tanjiro/separate-surface-decoder-head`
- Student: charliepai2i24h5-tanjiro
- Hypothesis (rebased re-run, after pre-EMA send-back): add a dedicated 2-layer surface decoder head alongside the volume head; the surface MLP's extra capacity gives sharper pressure prediction. Pre-rebase ablation showed −5.3% val vs single-head sibling baseline. Predicted compound effect vs #3281 EMA baseline: clear `single_in_dist` win, possibly clearing val_avg=110.
- Status: CLOSED. Rebased run was worse than the merged baseline on every split (+5.83% val_avg, +7.41% test_avg).

### Results

| Metric | Merged baseline (#3281) | Rebased surface head | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 114.1704 | 120.8263 | +5.83% worse |
| `test_avg/mae_surf_p` | 102.0813 | 109.6483 | +7.41% worse |

Per-split val: single_in_dist +7.6%, geom_camber_rc +7.7%, cruise +4.0%, re_rand +2.5%. Worst regressions exactly where the pre-rebase ablation showed the largest wins.

Metric artifact: `models/model-charliepai2i24h5-tanjiro-separate_surface_decoder_head_v2-20260515-172556/metrics.jsonl`.

### Analysis

The pre-rebase win came primarily from the surface head reducing cross-task interference in the single-shared-output regime. EMA + scale-invariant loss already provides much of that decoupling implicitly: EMA's ~2.7-epoch averaging smooths surface predictions on a similar time-scale, and the per-sample scale-invariant loss equalizes gradient contributions across the velocity/pressure magnitude gap. With those two mechanisms in the baseline, the surface head's specialization becomes redundant — and the extra ~50K head parameters (+7.5%) compete for the same 14-epoch undertrained budget. Closed as a standalone idea; the surface-pressure-only aux-loss alternative (frieren's PR #3337) captures the same intent at zero parameter cost.

## 2026-05-15 19:50 — PR #3422: Huber loss replacement for surface-pressure aux term [ASSIGNED — frieren]

- Branch: `charliepai2i24h5-frieren/surf-pressure-huber-aux`
- Student: charliepai2i24h5-frieren
- Hypothesis: replace the merged L1 surface-pressure aux with Huber to smooth out the discontinuous gradient at zero (frieren's PR #3337 noted oscillation in train_surf_p_l1 at convergence). Keeps the L1 tail behavior for the heavy-pressure splits while giving stable small-residual gradients. Predicted −1% to −3% on top of the new merged baseline.
- Two arms: Huber δ=1.0 (transitions to linear at 1 normalized std), Huber δ=0.5 (more L1-like). Weight held at 1.0 for clean comparison to merged result.

## 2026-05-15 19:55 — PR #3425: Schedule-Free AdamW [ASSIGNED — tanjiro]

- Branch: `charliepai2i24h5-tanjiro/schedule-free-adamw`
- Student: charliepai2i24h5-tanjiro
- Hypothesis: replace AdamW + CosineAnnealingLR(T_max=50) with `schedulefree.AdamWScheduleFree` to fix the schedule-budget mismatch (current cosine decays only 14% by epoch 14 wall-clock cap). SF-AdamW maintains an implicit polynomial-average iterate that adapts to any stopping time. Pairs with EMA (different time-scale averaging filters). RANK #1 in the latest research agenda.
- Two arms: SF-AdamW lr=5e-4 (drop-in at merged-baseline LR), SF-AdamW lr=7e-4 (test the higher-LR cliff that schedule-free typically tolerates).

## 2026-05-15 20:26 — PR #3265: FiLM per-block global-condition modulation [MERGED — new round-5 baseline]

- Branch: `charliepai2i24h5-fern/film-flow-condition-every-block`
- Student: charliepai2i24h5-fern
- Hypothesis (rebased): per-block FiLM scale+shift conditioning on the global flow vector (log Re, AoAs, NACAs, gap, stagger) injected after each of the 5 Transolver blocks' residuals. Gives every layer direct access to the global regime state rather than relying on the input MLP alone. Compounds with scale-invariant loss (orthogonal: loss-side vs architecture-side) and EMA (orthogonal: parameter-averaging vs conditioning).
- Status: MERGED. Code includes EMA + scale-inv + FiLM (pre-#3337 run, but mergeable CLEAN with #3337 code). Note: validated metrics are for FiLM-on-#3281 baseline; merged code also includes #3337 surf-L1.

### Results (rebased run on #3281 baseline, tip 1211ee5)

| Metric | #3281 baseline | Rebased FiLM | Δ% |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 114.1704 | **103.0171** | **−9.77%** |
| `test_avg/mae_surf_p` | 102.0813 | **92.1617** | **−9.74%** |

Per-split val: single_in_dist −13.5% (122.19), geom_camber_rc −7.7% (120.72), cruise −22.0% (76.89), re_rand −16.8% (92.26). Largest gains on cruise and re_rand (unseen camber/Re regime) where explicit condition injection matters most.

Metric artifacts:
- `models/model-charliepai2i24h5-fern-film_flow_condition_every_block_rebased-20260515-184444/metrics.jsonl`

### Analysis

FiLM addresses a different failure mode than the prior wins: the model's spatial pathway (Transolver sliced attention) must simultaneously encode physical geometry and regime adaption. By providing an explicit per-layer condition pathway, FiLM lets the spatial path focus on geometry while regime-specific scaling handles the aerodynamic regime. The 24% re_rand win is the clearest signal: re_rand varies the Reynolds number, which is the most "global" condition (pure Re change, same geometry), and FiLM provides exactly that explicit Re-dependent scaling.

Note: merged code includes #3337 surf-L1. Next validation run (askeladd #3315 rebase) will establish the compound FiLM + surf-L1 + Cautious-AdamW + EMA metric.

## 2026-05-15 20:20 — PR #3347: Manifold mixup (follow-up to #3268) [CLOSED — regression, mesh-correspondence problem]

- Branch: `charliepai2i24h5-alphonse/manifold-mixup`
- Student: charliepai2i24h5-alphonse
- Hypothesis: mix latent representations at a random Transolver block (not input space), solving the mesh-mismatch problem of input-space mixup (PR #3268). Predicted that latent mixing would preserve OOD camber wins from #3268 without the in-dist regression.
- Status: CLOSED. Both arms regressed +7.4% / +8.1% val_avg vs #3281 baseline.

### Results

| Arm | val_avg | Δ% | test_avg | Δ% |
|---|---:|---:|---:|---:|
| Arm A (α=0.4, p=0.5) | 122.57 | **+7.36%** | 109.63 | **+7.40%** |
| Arm B (α=0.2, p=0.5) | 123.44 | **+8.11%** | 110.73 | **+8.47%** |

### Analysis

Student correctly diagnosed root cause: manifold mixup assumes canonical position correspondence across the batch (position i in sample A corresponds to position i in sample B). For variable-mesh point clouds with padding-up-to-max collation, this assumption is false — position i in sample A is a completely different physical node than position i in sample B. Mixing creates non-physical gradient targets. The ~30% slower convergence (val_avg 122 vs baseline 114 at epoch 14) is the fingerprint of half-wasted gradient signal per mixup-active batch. The slice-token mixup follow-up (mixing in the learned global-feature space where tokens ARE permutation-invariant) is the right next step for this family.

## 2026-05-15 20:28 — PR #3315: Cautious AdamW (second send-back — rebase onto #3337) [SENT BACK — needs rebase again]

- Updated: askeladd's rebased Cautious AdamW run gave val_avg=103.02 on #3281 baseline (−9.77%). Very strong result. However, #3337 (surf-L1) merged after askeladd's run, and the branch now has merge conflicts (DIRTY). Sending back for third-and-final rebase to include surf-L1 + FiLM.
- Predicted outcome: val_avg ~96–100 (−3% to −6% on current 103.02 baseline).

## 2026-05-15 20:28 — PR #3432: SEMA / Switch EMA [ASSIGNED — fern]

- Branch: `charliepai2i24h5-fern/sema-switch-ema`
- Student: charliepai2i24h5-fern
- Hypothesis: after each epoch's EMA accumulation, copy EMA weights back into the live model (`model.load_state_dict(ema_model.state_dict())`). This ensures gradient steps always start from the flat-minimum region found by EMA rather than the raw noisy iterate. RANK #2 in current research agenda. Zero extra compute.
- Two arms: SEMA every epoch (freq=1, warmup=5 epochs), SEMA every 2 epochs (freq=2, warmup=5).

## 2026-05-15 20:30 — PR #3433: Per-domain target normalization [ASSIGNED — alphonse]

- Branch: `charliepai2i24h5-alphonse/per-domain-target-norm`
- Student: charliepai2i24h5-alphonse
- Hypothesis: replace global y_mean/y_std normalization with per-domain statistics (raceCar single / raceCar tandem / cruise computed from training data at startup). Equalizes baseline gradient magnitudes across domains, specifically targeting the single_in_dist anomaly (worst split at val=122.19 despite being in-distribution). RANK #3 in current research agenda.
- Two arms: per-domain hard labels (gap==0 → single), per-domain + per-channel (Ux/Uy/p separate).

## 2026-05-15 21:26 — PR #3373: bf16 mixed-precision AMP [MERGED — new round-5 baseline]

- Branch: `charliepai2i24h5-edward/bf16-mixed-precision`
- Student: charliepai2i24h5-edward
- Hypothesis: wrap forward pass in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`; cast pred and sq_err back to fp32 before all reductions. Expected ~20-30% per-epoch speedup → more effective epochs in 30-min cap.
- Status: MERGED. Arm A (batch_size=4, bf16) wins. Arm B (batch_size=8) regressed.

### Results

| Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | epochs | sec/epoch | peak VRAM |
|---|---:|---:|---:|---:|---:|
| Baseline (#3281, fp32) | 114.1704 | 102.0813 | 14 | ~125s | 42 GB |
| Arm A (bf16, bs=4) | **99.1251** | **89.1198** | **19** | **~98s** | **33 GB** |
| Arm B (bf16, bs=8) | 122.4724 | 110.8305 | 18 | ~105s | 66 GB |

Per-split val (Arm A vs #3281): single_in_dist 119.86 (−13.5%), geom_camber_rc 112.45 (−14.1%), geom_camber_cruise 74.04 (−12.5%), re_rand 90.15 (−12.3%). Uniform improvement across all splits.

Note: run was on tip `1211ee5` (pre-surf-L1, pre-FiLM). Standalone bf16 result (99.13) already beats current best FiLM baseline (103.02). Full compound expected in high-80s/low-90s val_avg.

Metric artifacts:
- `models/model-charliepai2i24h5-edward-bf16-baseline-config-20260515-182527/metrics.jsonl`
- `models/model-charliepai2i24h5-edward-bf16-batch8-20260515-192457/metrics.jsonl`

### Analysis

bf16 delivers exactly the predicted ~21% per-epoch speedup, 5 additional effective training epochs in the wall-clock budget (14 → 19), and 9 GB VRAM reduction (42 → 33 GB). The mechanism is purely a compute unlock: more passes over the data with the same training recipe. The improvement is broad-based (10-14% per split) consistent with "same training, more data passes" rather than any per-split optimization. Arm B regression is well-explained: batch_size=8 at bf16 has fewer optimizer steps per epoch, but doesn't save wall-clock (because compute per step is larger), so total gradient steps drops by ~50%. Per-sample scale-invariant loss already equalizes residual gradient noise; removing additional noise at batch=8 isn't helpful. VRAM headroom (33 GB) enables capacity-revisit experiments (n_hidden=192/256).

## 2026-05-15 21:26 — PR #3346: Cosine T_max=15 + 1-epoch warmup + LR=7e-4 [CLOSED — regression]

- Branch: `charliepai2i24h5-thorfinn/cosine-tmax-fix-warmup-lr7e-4`
- Student: charliepai2i24h5-thorfinn
- Hypothesis: align cosine schedule to wall-clock budget (T_max=15) + 1-epoch warmup + raised peak LR=7e-4. Predicted −2% to −5% on val_avg.
- Status: CLOSED. All 3 seeds regress +6.45% to +9.30% vs merged baseline (#3281).

### Results

| Run | val_avg/mae_surf_p | Δ vs #3281 baseline |
|---|---:|---:|
| Baseline (#3281) | 114.1704 | — |
| Best seed (20260515-192741) | 120.6238 | +6.45% worse |
| Seed 2 (20260515-182633) | 123.4285 | +9.26% worse |
| Seed 3 (20260515-202725) | 123.4680 | +9.30% worse |

Cross-seed std ≈ 1.6; the underperformance well outside noise. Schedule executed correctly (verified via per-epoch LR logging).

### Analysis

Student's post-mortem is comprehensive and correct:
1. **Warmup epoch wastes budget.** With only ~14 epochs of wall-clock, epoch 1 training at LR=7e-7 is a ~7% throughput cost with zero return.
2. **The "flat" baseline cosine is a feature, not a bug.** `CosineAnnealingLR(T_max=50)` truncated at epoch 14 gives LR≈4.7e-4 throughout — essentially constant. This maximizes early-training gradient signal in the undercooked regime.
3. **eta_min=7e-6 is too aggressive.** Reaches LR=4.13e-5 at epoch 14 while baseline is still at 4.45e-4.

The "drop the warmup, T_max=19 (match bf16 horizon)" variant is the natural follow-up, assigned to thorfinn as PR #3465.

## 2026-05-15 21:29 — PR #3315: Cautious AdamW on full merged stack [MERGED — new best, current baseline]

- Branch: `charliepai2i24h5-askeladd/cautious-adamw-optimizer`
- Student: charliepai2i24h5-askeladd
- Hypothesis (third rebase — onto b5760af with FiLM + surf-L1 + EMA + scale-inv): Cautious AdamW gates ~38% of updates each step via `(m * g > 0)` agreement mask; mechanisms are orthogonal to all four merged techniques.
- Status: MERGED. Decisive win, new round-5 best.

### Results

| Comparison | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---:|---:|
| #3337 surf-L1 baseline | 106.8550 | 96.8671 |
| #3265 FiLM baseline | 103.0171 | 92.1617 |
| **Cautious AdamW + full stack (this run)** | **90.3428** | **80.1674** |
| Δ vs #3265 (pre-merge best) | **−12.31%** | **−13.01%** |

Per-split val vs #3265: single_in_dist −10.05% (109.91), geom_camber_rc −14.37% (103.37), geom_camber_cruise −14.57% (65.69), re_rand −10.69% (82.40). All 8 val/test cells improve by 10–15%.

Mask agreement: mean ≈ 0.620, flat across all 13 training epochs and across all three run variants (standalone, +EMA, +EMA+surf-L1+FiLM). This is direct evidence the cautious mechanism operates on disjoint state from all other merged techniques.

Metric artifacts:
- `models/model-charliepai2i24h5-askeladd-cautious_adamw_on_ema_v2-20260515-203741/metrics.jsonl`
- `models/model-charliepai2i24h5-askeladd-cautious_adamw_on_ema_v2-20260515-203741/metrics.yaml`

### Analysis

Five compounding wins in round 5: scale-inv → EMA → surf-L1 → FiLM → Cautious AdamW. Cumulative: −27.06% val_avg (123.88 → 90.34), −29.91% test_avg (114.37 → 80.17). The 12.31% jump from FiLM→Cautious AdamW was larger than predicted (96–100 expected; 90.34 delivered), driven by FiLM landing alongside surf-L1 between the second and third rebase — the compound FiLM+surf-L1+cautious stack is multiplicatively better than any two-mechanism compound tested so far.

Key finding: EMA + FiLM together stabilize the iterate trajectory enough that cautious masking becomes a net positive on every split — in the standalone run, single_in_dist regressed (+2%) while OOD splits won big (−15%); with the full stack, every split wins (10–15% uniformly). The interaction is mechanistically clear: FiLM reduces per-step regime-conditioning variance (less gradient noise from condition-switch), EMA smooths the iterate to avoid sharp minima, and cautious masking then gates the remaining ~38% of disagreed-on directions.

Steep epoch-13 descent (94.7 → 90.3 in final epoch) plus flat mask agreement curve signals training is still in cold-start. bf16 (#3373) will compound here, giving ~6 more effective epochs.

## 2026-05-15 21:30 — New assignments (3 idle students after merges and close)

- **PR #3463 (edward): Capacity revisit with bf16** — sweep n_hidden=192 (Arm A) and n_hidden=256 (Arm B) at batch_size=4 with bf16 and the full merged stack. bf16 brought peak VRAM to 33 GB, making these tests fair (previous #3270 capacity run completed only 5/50 epochs in the same budget). Expected −2% to −8% on best arm.
- **PR #3465 (thorfinn): Schedule T_max alignment** — T_max=19 (match bf16 wall-clock epoch count), no warmup, eta_min=lr*0.05 (Arm A); T_max=25, eta_min=lr*0.1 (Arm B). Direct follow-up to the #3346 negative result: the "drop warmup, match T_max to budget" variant identified in thorfinn's own post-mortem.
- **PR #3466 (askeladd): Bernoulli pressure residual** — predict `p − p_Bernoulli(Re, AoA)` instead of raw p. Removes the analytic dynamic-range component; model specializes on the viscous residual. Arm A: free-stream Bernoulli only (per-sample scalar subtraction). Arm B: free-stream + chord-position correction. Highest-novelty unexplored direction; targets the single_in_dist gap (val=109.91 after Cautious AdamW win).

## 2026-05-15 22:33 — PR #3425: Schedule-Free AdamW [SENT BACK — strong standalone, conflicts with Cautious AdamW; needs head-to-head rebase]

- Branch: `charliepai2i24h5-tanjiro/schedule-free-adamw`
- Student: charliepai2i24h5-tanjiro
- Hypothesis: replace AdamW + CosineAnnealingLR with `schedulefree.AdamWScheduleFree` — implicit polynomial-average iterate, no schedule-budget mismatch. RANK #1 in latest research agenda.
- Status: SENT BACK. Standalone result is the strongest single-arm round-5 gain yet (val_avg=87.24, −18.39% on #3337 baseline). But measured on a baseline without FiLM and pre-Cautious AdamW. Sent back for direct head-to-head replacement of Cautious AdamW on the current merged stack.

### Standalone results (Arm A, lr=5e-4, on tip `ac5df20` — pre-FiLM, pre-Cautious-AdamW, pre-bf16)

| Comparison | val_avg/mae_surf_p | test_avg/mae_surf_p |
|---|---:|---:|
| #3337 baseline (tested-against) | 106.8550 | 96.8671 |
| **Arm A (SF-AdamW lr=5e-4)** | **87.2407** | **78.4670** |
| Arm B (SF-AdamW lr=7e-4) | 89.5096 | 81.2350 |
| Δ Arm A vs #3337 | **−18.39%** | **−19.00%** |

Per-split val (Arm A vs #3337): single_in_dist −21.51% (largest gain — directly attacked the worst split), geom_camber_rc −20.52%, geom_camber_cruise −15.85%, re_rand −13.60%. All splits improve, in-dist splits gain most.

LR ablation: lr=5e-4 beats lr=7e-4 by 2.5% on val_avg. The higher-LR arm converges faster for the first 2 epochs only, then arm A overtakes. The "drop the cosine schedule unlocks higher LR" prediction was wrong for this problem.

Metric artifacts:
- `models/model-charliepai2i24h5-tanjiro-sf_adamw_lr5e-4-20260515-203510/metrics.jsonl`
- `models/model-charliepai2i24h5-tanjiro-sf_adamw_lr7e-4-20260515-212710/metrics.jsonl`

### Decision rationale

SF-AdamW and Cautious AdamW (current merged #3315) both replace the optimizer — they cannot stack. Code-level: SF-AdamW's `AdamWScheduleFree` instantiation conflicts with #3315's `CautiousAdamW` class + instantiation. Mechanistically: SF-AdamW averages iterates via polynomial mean (smooth in time); Cautious AdamW gates per-step update components (per-step quality filter). These are competing approaches to the same problem (noisy iterate quality).

The decision: do a head-to-head on identical full-merged-stack baseline. Send back for rebase that REPLACES Cautious AdamW with SF-AdamW (delete the CautiousAdamW class, remove the cosine scheduler, keep bf16 + FiLM + surf-L1 + EMA + scale-inv unchanged). If the rebased run beats current best (90.34 val), revert #3315 and merge SF-AdamW. If not, close — Cautious AdamW's per-step gating interacts with FiLM better than SF-AdamW's polynomial averaging.

Predicted rebased outcome on full merged stack: val_avg **82–86**, test **74–78**. Two arms requested: lr=5e-4 (direct) and lr=5e-4 + warmup_steps=500 (student's suggestion #3, matching the 5-10% of total steps recommendation).

## 2026-05-15 23:20 — PR #3374: Stochastic depth regularization [CLOSED — robust 3-seed negative result]

- Branch: `charliepai2i24h5-nezuko/stochastic-depth`
- Student: charliepai2i24h5-nezuko
- Hypothesis: per-block DropPath regularization (`p_drop=0.1`) to address overfitting on single_in_dist; arm A uniform, arm B linear-by-depth (DeiT schedule).
- Status: CLOSED. Three seeds at identical config all regressed; both arms regressed across every val and test split; central single_in_dist-improves-most prediction was contradicted.

### Results vs the student's tested baseline (PR #3281: EMA, val_avg=114.17)

| Run | val_avg/mae_surf_p | Δ | test_avg/mae_surf_p | Δ |
|---|---:|---:|---:|---:|
| Arm A: uniform p=0.1 | 130.33 | **+14.16%** | 117.88 | +15.47% |
| Arm B: linear 0→0.1 | 122.68 | **+7.45%** | 111.04 | +8.78% |
| (PR-predicted range) | 110–113 | −1.5 to −4% | — | — |

Per-split: ALL splits regressed in both arms. `geom_camber_rc` (not single_in_dist) regressed the most in absolute val terms for arm A (+27.92), indicating the baseline's worst-split MAE is not primarily driven by feature-memorization that drop-path could disrupt. Seed variance ±2 on val_avg confirmed the negative result is robust (3 seeds at arm A: 126.53 / 129.01 / 130.33; all >baseline).

Per-epoch wall-clock: arms were ~2-3% *slower* than baseline, not faster as predicted. `timm.layers.DropPath` scales the residual by `1/keep_prob` rather than skipping the residual subgraph entirely, so random-tensor generation overhead dominates the marginal forward-pass cost saved.

Metric artifacts:
- `models/model-charliepai2i24h5-nezuko-stochastic-depth-uniform-20260515-213318/metrics.jsonl`
- `models/model-charliepai2i24h5-nezuko-stochastic-depth-linear-20260515-203439/metrics.jsonl`

### Analysis

The hypothesis is fundamentally fine theory in a longer-training regime, but at p_drop_max=0.1 and ~14 epochs wall-clock cap, the regularization is too strong relative to training time. Best epoch was 14/50 in every run, identical to baseline — the network never gets to recover capacity from the regularization. This matches risk #1 in the hypothesis ("p_drop too high → underfitting in 14-epoch regime"), but the data falsifies the prediction that linear-by-depth would land near baseline.

The current merged baseline has moved much further out of reach (90.34 vs 114.17 tested-against), so a rebase or p=0.05 follow-up would still leave the result ~3-5% below baseline — not worth the GPU. Closed; nezuko reassigned to Fourier-embedded FiLM (#3519).

## 2026-05-15 23:25 — New assignment for newly-idle nezuko

- **PR #3519 (nezuko): Fourier-embedded FiLM conditioning** — replace the raw 11-dim flow-condition vector with a multi-frequency sin/cos Fourier embedding (Tancik et al. 2020 style) before feeding into the FiLM MLP. Concat raw+Fourier for residual safety. Arm A: 4 frequencies, sigma=1 (Tancik default). Arm B: 8 frequencies, sigma=2 (rich basis). Targets the single_in_dist split by giving FiLM richer angular resolution on AoA and Re×geometry interactions. Orthogonal to all 7 in-flight experiments.

## 2026-05-16 00:00 — PR #3466: Bernoulli pressure residual [MERGED — new best val_avg=86.09]

- Branch: `charliepai2i24h5-askeladd/bernoulli-pressure-residual`
- Student: charliepai2i24h5-askeladd
- Hypothesis: predict the viscous residual `p − p_B` where `p_B = 0.5·V_∞²` is the freestream Bernoulli reference (per-sample scalar), instead of raw `p`. Remove the analytic dynamic-range component so the network specializes on viscous spatial structure (BL, separation, vortex shedding).
- Status: MERGED. Seventh compounding win in round 5. New cumulative −30.49% val_avg / −32.22% test_avg.

### Results vs current merged baseline (PR #3315 + bf16 + FiLM + surf-L1 + EMA + scale-inv, val_avg=90.34)

| Arm | val_avg/mae_surf_p | Δ val | test_avg/mae_surf_p | Δ test |
|---|---:|---:|---:|---:|
| Baseline merged | 90.3428 | — | 80.1674 | — |
| **A: Free-stream Bernoulli** | **86.0948** | **−4.70%** | **77.5066** | **−3.32%** |
| B: + chord correction (raw x) | 122.1765 | +35.24% | 121.3164 | +51.33% |

### Per-split (Arm A)

| Split | val_baseline | val_armA | val Δ | test_baseline | test_armA | test Δ |
|---|---:|---:|---:|---:|---:|---:|
| single_in_dist | 109.9053 | **102.0390** | **−7.16%** | 96.3817 | 93.0730 | −3.43% |
| geom_camber_rc | 103.3709 | 100.1206 | −3.14% | 91.8343 | 90.1840 | −1.80% |
| geom_camber_cruise | 65.6940 | 62.9941 | −4.11% | 54.3760 | 52.3967 | −3.64% |
| re_rand | 82.4009 | 79.2254 | −3.85% | 78.0776 | 74.3726 | −4.74% |
| **avg** | **90.3428** | **86.0948** | **−4.70%** | **80.1674** | **77.5066** | **−3.32%** |

All 8 val and test cells improved. Largest gain on `single_in_dist` (the previous worst split). Gains scale with V_∞-range mixedness — largest on regime-mixed splits (single_in_dist, re_rand), smallest on narrow-V_∞ splits (geom_camber_rc which is only raceCar tandem).

Metric artifacts:
- `models/model-charliepai2i24h5-askeladd-bernoulli_freestream-20260515-215606/metrics.jsonl`
- `models/model-charliepai2i24h5-askeladd-bernoulli_chord-20260515-223334/metrics.jsonl`

### Analysis

The student's pre-pass identified an interesting paradox: subtracting `p_B` *increases* the total p-channel std by 1.71× (raw std=679 → residual std=1160), because per-sample means are anticorrelated with p_B (raceCar inverted-foil samples have suction-dominant means when V_∞ is largest, `Corr(per_sample_p_mean, p_B) = −0.583`). Their post-result analysis correctly reframed why Arm A wins despite this: **the model isn't optimizing per-pixel std — it's optimizing a normalized-space MSE/L1 with a finite-capacity Transolver. Subtracting p_B doesn't reduce variance; it shifts variance into a more learnable functional form.** The network no longer has to internally compute `V_∞²/2` from the conditioning vector and apply it everywhere; that piece is now handled outside the network, freeing capacity for the viscous residual.

Arm B failed because the chord-position formula was applied to global mesh x-coordinate (range ~[-9.55, 11.34], spanning ~20 chord lengths), producing high-frequency oscillations. The hypothesis spec explicitly anticipated this fragility ("if this gets complex, just use the simple formula" → it didn't work). Proper chord-position-aware Cp would require per-foil chord-boundary detection; deferred as follow-up.

Best epoch 17/50 — training still in cold-start at the wall-clock cap.

## 2026-05-16 00:05 — PR #3433: Per-domain target normalization [CLOSED — both arms regressed]

- Branch: `charliepai2i24h5-alphonse/per-domain-target-norm`
- Student: charliepai2i24h5-alphonse
- Hypothesis: normalize y-targets with per-domain (racecar_single / racecar_tandem / cruise) means and stds instead of global stats; addresses the in-distribution split's gap.
- Status: CLOSED. Both arms regressed on the tested baseline (FiLM #3265 val=103.02), with `val_single_in_dist` regressing most under arm B (+8.0%) — opposite of the prediction.

### Results vs tested baseline (PR #3265 FiLM, val_avg=103.02)

| Arm | val_avg | Δ | test_avg | Δ |
|---|---:|---:|---:|---:|
| A: scalar std per domain | 114.82 | **+11.5%** | 103.01 | +11.8% |
| B: per-channel std per domain | 106.37 | **+3.2%** | 95.83 | +4.0% |

Per-split Arm B: `val_single_in_dist` +8.0% (target split worsened), `val_geom_camber_rc` +4.7%, `val_geom_camber_cruise` −5.0% (only split that helped), `val_re_rand` +2.1%.

### Analysis

Per-domain *empirical* output normalization is the wrong direction. The regime mismatch the hypothesis aimed to address is the V_∞²-driven per-sample std variance, which is more cleanly addressed by the analytic Bernoulli prior (#3466 merged the same day, attacking the same problem from a physics-informed angle and winning). Arm B's per-channel directionality was better than scalar arm A (matching DomainNorm literature), but neither inverted sign at 14 epochs in the cold-start regime.

The student's domain-distribution analysis (racecar_single=599 train, racecar_tandem=457, cruise=443 train) and per-domain p-channel std numbers (904.80 / 782.40 / 383.65) are useful empirical baselines we should retain. Sharp post-mortem.

## 2026-05-16 00:05 — PR #3422: Huber loss for surf-pressure aux [CLOSED — both arms regressed]

- Branch: `charliepai2i24h5-frieren/surf-pressure-huber-aux`
- Student: charliepai2i24h5-frieren
- Hypothesis: replace L1 surf-pressure aux loss with Huber (smooth at small residuals) to reduce optimization oscillation.
- Status: CLOSED. Both arms regressed (+5-8%) on the tested baseline (#3337 L1, val=106.86). Hypothesis falsified: L1's *constant gradient magnitude* at small residuals is the active mechanism, not its noise tolerance.

### Results vs tested baseline (PR #3337 L1, val_avg=106.86)

| Run | val_avg | Δ | test_avg | Δ | Best epoch |
|---|---:|---:|---:|---:|---:|
| Arm A — Huber δ=1.0, w=1.0 | 115.04 | **+7.66%** | 103.34 | +6.68% | 14 |
| Arm B — Huber δ=0.5, w=1.0 | 112.62 | **+5.39%** | 101.86 | +5.16% | 14 |

### Analysis

The student's train-loss-trajectory data is the gem of this experiment: L1 trajectory std=0.0144, Huber δ=1.0 std=0.0050, Huber δ=0.5 std=0.0056. **Huber smooths the training trajectory 2-3× tighter — but the smoother training loss does NOT predict better validation.** This is a useful counter-example to "smooth loss landscapes generalize better." The L1 aux's value is its constant gradient at small residuals (no "give up" behavior near zero), pushing the network to refine even well-fit pressure predictions. Huber's quadratic regime explicitly damps this signal.

Lower δ doesn't help: as δ → 0, Huber → L1, so we just recover the merged baseline. Scheduled δ (Huber early, L1 late) is a future experiment, but out of scope for this rebase.

## 2026-05-16 00:30 — New assignments for 3 idle students after Bernoulli merge

- **PR #3545 (alphonse): EMA decay annealing** — start with decay=0.99 (~100-step window), ramp to 0.999 (~1000-step window). Arm A: linear ramp first 5 epochs. Arm B: cosine ramp over full 19 epochs. Targets the cold-start regime where the merged 0.999 EMA window is too long for early-training fast iterate movement.
- **PR #3547 (askeladd): Cp normalization on top of Bernoulli residual** — extend her own merged win by additionally dividing by `p_B` (Arm A: full Cp = (p − p_B) / p_B; Arm B: half-Cp dividing by V_∞ instead of V_∞²). Her own pre-pass identified this as the natural physics-informed follow-up to address the V_∞²-correlated per-sample std variance.
- **PR #3548 (frieren): AoA-jitter test-time augmentation** — first eval-only experiment in round 5. K-ensemble of inference with small AoA perturbations. Arm A: K=4, σ=0.1°. Arm B: K=8, σ=0.05° + jitter on other continuous cond dims. Targets OOD splits without any training cost. Compounds with every merged mechanism.


## 2026-05-16 02:30 — PR #3465: CosineAnnealingLR T_max alignment [MERGED — new best]

- Branch: `charliepai2i24h5-thorfinn/schedule-tmax-alignment`
- Student: charliepai2i24h5-thorfinn
- Hypothesis: align `CosineAnnealingLR(T_max=25, eta_min_factor=0.10)` to the actual bf16 training budget (~17–19 epochs), fixing the pre-bf16 default `T_max=50` that caused LR to hover near peak throughout the wall-clock window with near-zero floor.
- Status: MERGED as new best. val_avg=75.40, test_avg=65.86. **−12.42% vs PR #3466 Bernoulli residual baseline (86.09)**.

### Results (Arm B — T_max=25, eta_min_factor=0.10 — winner)

| Split | val_mae_surf_p | test_mae_surf_p | Δ val vs #3466 |
|---|---:|---:|---:|
| single_in_dist | 84.8767 | 73.8438 | −16.8% |
| geom_camber_rc | 85.9009 | 77.2500 | −14.2% |
| geom_camber_cruise | 57.6499 | 46.6446 | −8.5% |
| re_rand | 73.1886 | 65.6982 | −7.6% |
| **avg** | **75.4040** | **65.8592** | **−12.42%** |

Arm A (T_max=19, eta_min_factor=0.05): val_avg=80.12 — also beats prior baseline but loses to Arm B.

Metric artifacts:
- `models/model-charliepai2i24h5-thorfinn-tmax25_em10-20260515-232319/metrics.jsonl`
- `models/model-charliepai2i24h5-thorfinn-tmax19_em05-20260515-222432/metrics.jsonl`

### Analysis

Massive, clean win on a pure schedule fix. The cosine curve with T_max=50 was spending most of its decay budget *after* the wall-clock cutoff — so the optimizer ran at near-peak LR for all 17 training epochs. T_max=25 pushes the cosine descent into the actual training window; LR at epoch 17 was 1.79e-4 (well above the floor) and still productive. `eta_min_factor=0.10` ensured the floor (5e-5) is non-zero, preserving late-epoch gradient signal. Arm B's longer T_max=25 > T_max=19 won because T_max=19 reached eta_min exactly at the wall-clock cutoff, leaving the last few epochs at near-zero LR.

Cautious mask mean flat at 0.617 across all epochs — schedule realignment did not perturb the cautious gating equilibrium. All 8 cells (val+test across 4 splits) improved, biggest gain on single_in_dist (−16.8% val), geom_camber_rc (−14.2% val). Best epoch still 17/50 — training still undercooked (descent active at −2.74/epoch at cutoff). This motivates the two new assignments: higher peak LR (#3581) and torch.compile() (#3582).

Cumulative round-5 improvement now **−39.15% val_avg** (123.88 → 75.40) and **−42.45% test_avg** (114.37 → 65.86) over the pre-round-5 floor. **Eight compounding wins**.

## 2026-05-16 02:30 — PR #3432: SEMA — copy EMA weights back each epoch [CLOSED — both arms regressed]

- Branch: `charliepai2i24h5-fern/sema-ema-copy-back`
- Student: charliepai2i24h5-fern
- Hypothesis: SEMA (Stochastic EMA) from Kaddour et al. — periodically copy EMA weights back into the live model at frequency=1 or frequency=2 epochs, after a warmup period. Expected to keep the live model near "flat minimum" regions discovered by EMA averaging.
- Status: CLOSED. Both arms regressed +22-33% vs FiLM baseline (#3265, val=103.02). Arm A +33.4%, Arm B +22.8%.

### Results vs tested baseline (PR #3265 FiLM, val_avg=103.02)

| Arm | val_avg | Δ | test_avg | Best epoch |
|---|---:|---:|---:|---:|
| Arm A: sema_freq=1, warmup=5 | 137.38 | **+33.4%** | 124.17 | 14 |
| Arm B: sema_freq=2, warmup=5 | 126.54 | **+22.8%** | 115.07 | 13 |

### Analysis

Clear negative, well-analyzed by student. The mechanism inversion: EMA decay=0.999 at batch_size=4 gives a ~2.7-epoch effective averaging window. At epoch 6 (first SEMA copy), EMA reflects the model state from epochs 3–5 — so the copy is a 2–3 epoch **reset**, not a flat-region refinement. Each SEMA copy bleeds away recent gradient progress. Student ran 3 independent arm A replications (132.08, 137.38, 141.25 — mean 136.9); all far above baseline. Robust negative result.

Arm B (freq=2) being less bad than Arm A (freq=1) confirms the mechanistic explanation: less frequent resets waste less gradient progress. Neither is remotely competitive.

Implication: SEMA may become viable if (a) EMA decay is lowered to ≈0.99 so EMA is closer to the live model at copy time, or (b) training is much longer. PR #3545 (EMA decay annealing) addresses the same cold-start from a different angle and is still WIP.

## 2026-05-16 02:30 — New assignments after T_max win and SEMA close

- **PR #3581 (thorfinn): Peak LR sweep on T_max=25 aligned schedule** — test lr=7e-4 (Arm A) and lr=1e-3 (Arm B) with the fixed T_max=25/eta_min_factor=0.10 schedule. Descent still active at −2.74/epoch at epoch 17; higher LR may unlock faster per-epoch improvement. Cautious AdamW masking (0.62) stabilizes high-LR runs by zeroing ~38% of update components.
- **PR #3582 (fern): torch.compile() for more effective epochs** — JIT-compile the model graph (Arm A: default mode; Arm B: reduce-overhead) to reduce per-epoch wall-clock time by 10–25%. Expected to unlock 2–4 additional effective epochs within the 30-min cap. At −2.74/epoch descent rate, this translates to 5.5–11 point improvement. First pure speed experiment in round 5.

## 2026-05-16 02:50 — PR #3463: Capacity revisit n_hidden=192/256 [SENT BACK — strong win on old baseline, needs rebase + re-run]

- Branch: `charliepai2i24h5-edward/capacity-revisit-with-bf16`
- Student: charliepai2i24h5-edward
- Hypothesis: With bf16's VRAM savings (#3373), the tractable capacity budget shifts. Test n_hidden=192 and n_hidden=256 (vs baseline 128) on the full merged stack to see if the model is undersized.
- Status: SENT BACK. Arm A (n_hidden=192) is a strong −11.16% win against the pre-Bernoulli/pre-T_max baseline (#3315 at 90.34). Tested at val_avg=80.26 / test_avg=71.27 with uniform 8.5–13.5% improvement across all 8 val+test cells. Mergeable=CONFLICTING vs current 75.40 baseline; needs rebase + re-run with T_max=25 + Bernoulli + full stack.

### Results vs tested baseline (PR #3315 Cautious AdamW, val_avg=90.34)

| Arm | n_hidden | n_params | Epochs | val_avg | Δ | test_avg | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A** | **192** | **1.84M** | **14** | **80.26** | **−11.16%** | **71.27** | **−11.10%** |
| B | 256 | 3.26M | 12 | 91.27 | +1.03% | 81.89 | +2.15% |

Per-split Arm A (vs #3315): single_in_dist −13.43%, geom_camber_rc −8.52%, geom_camber_cruise −11.68%, re_rand −11.01%. **Uniform broad-based win**.

### Analysis

Arm A is a clean, broad-based capacity win. Three load-bearing signals:
1. **Uniform gain across all 8 cells** (8.5–13.5%) rules out a "wider helps in-distribution at OOD's cost" interpretation. Added width helps every regime.
2. **Cautious mask invariant** at 0.614 (vs 0.620 at n=128) — optimizer dynamics are width-agnostic. Argues for orthogonal compounding with all other merged mechanisms.
3. **Arm A descent at cutoff: −4.31/epoch** (Arm B: −6.78/epoch). Both arms still descending sharply — wall-clock-bound. Arm B at 161s/epoch fits only 12 epochs (vs 14 for Arm A); the n=256 loss is undertraining, not capacity hurting.

VRAM: Arm A 46.78 GB / Arm B 58.13 GB — both well within 96 GB budget. Capacity is not VRAM-bound, only wall-clock-bound.

### Why send back instead of merge?

Tested baseline (90.34) is pre-Bernoulli (which gave −4.70%) and pre-T_max (which gave −12.42%). Current merged baseline is 75.40. Even with the strong −11.16% Arm A win, 80.26 > 75.40 in absolute terms. The capacity win must compound with the new schedule alignment + Bernoulli residual to be merge-eligible.

Predicted full-stack outcome (sub-multiplicative compounding): n_hidden=192 + T_max=25 + Bernoulli ≈ 75.40 × (1 − 0.11) ≈ **67** val_avg. Even with more conservative compounding, low-70s is realistic.

Arm B (n_hidden=256) is dropped from the re-run — wall-clock-undertrained regardless of stack changes; spending GPU on a single solid Arm A confirmation is higher value.
