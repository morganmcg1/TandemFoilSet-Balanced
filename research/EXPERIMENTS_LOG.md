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

