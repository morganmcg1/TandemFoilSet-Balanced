# SENPAI Research Results — `icml-appendix-willow-pai2i-48h-r3`

Logged per advisor review of each PR.

## 2026-05-15 — Launch: round 3 of willow-pai2i-48h begins

All 8 students idle; no PRs in flight. First round of assignments dispatched (each PR runs dual-arm baseline + variant in the same wandb_group, since no canonical baseline run exists yet on this branch state).

| Student | PR | Hypothesis | Family |
|---|---|---|---|
| alphonse | [#3140](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3140) | Wider hidden + more heads (128→192, 4→6) | Capacity |
| askeladd | [#3147](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3147) | LR warmup + peak 5e-4→1e-3 (3-epoch linear warmup) | Optimization |
| edward | [#3152](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3152) | Per-channel loss weighting (p x3 in MSE) | Loss formulation |
| fern | [#3155](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3155) | Huber loss instead of MSE (delta=1.0) | Loss formulation |
| frieren | [#3161](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3161) | Per-sample loss normalization (equal-weight per sample) | Loss formulation |
| nezuko | [#3165](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3165) | Depth scaling (5->8 layers) | Capacity |
| tanjiro | [#3169](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3169) | MLP ratio (2->4) | Capacity |
| thorfinn | [#3172](https://github.com/morganmcg1/TandemFoilSet-Balanced/pull/3172) | Fourier position features + slice_num 64->96 | Inputs |

## 2026-05-15 14:30 — PR #3140 (alphonse): Widen Transolver n_hidden 128->192, n_head 4->6 — **CLOSED**

- Branch: `willowpai2i48h3-alphonse/capacity-width-heads`
- W&B group: [`capacity-width-heads`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/capacity-width-heads)
- Baseline run: `xehwt9bi` | Variant run: `t2z5ya27`

**Hypothesis:** Widening Transolver capacity (n_hidden 128->192, n_head 4->6) would reduce val_avg/mae_surf_p by 3-8% if the baseline were underfit.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline | Variant | Δ |
|---|---|---|---|
| best_val_avg/mae_surf_p | **135.30** (ep 13) | 160.61 (ep 8) | +18.7% |
| val_geom_camber_cruise/mae_surf_p | 109.77 | 140.94 | +28.4% |
| val_geom_camber_rc/mae_surf_p | 135.88 | 171.21 | +26.0% |
| val_re_rand/mae_surf_p | 127.66 | 139.93 | +9.6% |
| val_single_in_dist/mae_surf_p | 167.88 | 190.37 | +13.4% |
| test_avg/mae_surf_p (excl cruise) | 135.54 | 154.32 | +13.9% |
| Params | 0.66M | 1.45M | 2.19× |
| Sec/epoch | 131.8 | 203.8 | 1.55× |
| Epochs reached / 50 (30-min cap) | 14 (best @13) | 9 (best @8) | -36% |

**Decision:** Closed. Variant >5% worse on every track, well past the merge threshold.

**Analysis:**
- The wider variant is uniformly worse on val and test under the 30-min wall-clock cap.
- Student's epoch-matched comparison (variant ep 8 vs baseline ep 8) suggests per-epoch the wider model may be slightly better, but the 1.55× slower epoch time costs ~36% of epoch count under the wall-clock budget. The variant best-checkpointed at epoch 8/9 — almost certainly still under-trained.
- **Implication for round 2:** width-side scaling is wall-clock-penalized in this regime. Future round assignments should bias toward changes that don't increase per-step cost (optimizers, losses, data aug, input feature engineering).

**Pre-existing bug surfaced:** `test_geom_camber_cruise/mae_surf_p` returns NaN on the baseline arm. Non-finite pressure prediction on at least one test-cruise sample propagates Inf through `data/scoring.py:accumulate_batch`. Affects ALL PRs equally — tracked separately (see CURRENT_RESEARCH_STATE.md).

**Sets the canonical round-3 baseline:** `val_avg/mae_surf_p = 135.30`, `test_avg/mae_surf_p (excl cruise) = 135.54`. See BASELINE.md.

## 2026-05-15 15:00 — PR #3152 (edward): Per-channel loss weighting (p ×3 in MSE) — **REQUEST CHANGES**

- Branch: `willowpai2i48h3-edward/channel-loss-weight-p`
- W&B group: [`channel-loss-weight-p`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/channel-loss-weight-p)
- Baseline run: `sw37lp52` | Variant run: `4rt8g0xb`

**Hypothesis:** Upweight the p channel by 3× in MSE to align gradients with the primary metric `mae_surf_p`. Predicted −5 to −15%.

**Result (variant minus baseline, lower is better):**

| Metric | Baseline (1:1:1) | Variant (1:1:3) | Δ% |
|---|---|---|---|
| best_val_avg/mae_surf_p | **137.63** (ep 11) | 138.51 (ep 14) | +0.6% |
| val_geom_camber_rc/mae_surf_p | 146.58 | 170.86 | **+16.6%** |
| val_geom_camber_cruise/mae_surf_p | 107.67 | 100.75 | −6.4% |
| val_re_rand/mae_surf_p | 127.19 | 117.40 | −7.7% |
| val_single_in_dist/mae_surf_p | 169.09 | 165.01 | −2.4% |
| val_avg/mae_surf_Ux | 2.207 | 2.658 | +20.4% |
| val_avg/mae_surf_Uy | 0.928 | 1.056 | +13.9% |
| val_avg/mae_vol_p | 137.72 | 145.15 | +5.4% |
| 3-split test mean (excl cruise) | 136.36 | 140.57 | +3.1% |
| Epochs reached / 50 (30-min cap) | 18 (best @11) | 18 (best @14) | — |

**Decision:** REQUEST CHANGES. Sent back as draft with surface-only upweight follow-up.

**Reason for not closing despite regression:**
- Headline Δ on val_avg is only +0.6% — within the noise floor revealed by truncation (both arms stopped at 18/50 epochs).
- Per-split deltas show real signal: cruise and re_rand benefit (-6.4%, -7.7%), suggesting p-focus helps on the smoother/easier OOD geometry, but rc regresses sharply (+16.6%) because the shared velocity-pressure representation is needed for rc.
- Cross-check: this PR's baseline arm hits 137.63 vs canonical baseline 135.30 (+1.7%). Single-seed dual-arm comparisons are noisy at the few-percent level under wall-clock truncation.

**Follow-up requested:** Run a third arm `variant-p3x-surf-only` in the same wandb_group. Apply [1,1,3] weighting **only to the surface portion** of the loss (vol_loss stays balanced [1,1,1]). This preserves velocity learning on volume nodes (~99% of mesh) while targeting surface-p specifically. Decision rule: merge if surface-only variant beats baseline by ≥2% on val_avg/mae_surf_p AND velocity MAEs degrade <5%; close otherwise.

**Notable:** Student independently identified the cruise-test NaN bug (matches alphonse's earlier observation in PR #3140). Cross-channel coupling analysis is a useful contribution — confirms that *normalized-space* MSE underweights p relative to its physical contribution (suggests a future hypothesis around physical-units / scale-aware loss).

## 2026-05-15 16:00 — PR #3155 (fern): Huber loss (SmoothL1 beta=1.0) instead of MSE — **MERGED**

- Branch: `willowpai2i48h3-fern/huber-loss`
- W&B group: [`huber-loss`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/huber-loss)
- Baseline run: `dqe6ejfz` | Variant run: `3nivkqy0`

**Hypothesis:** Huber (SmoothL1, delta=1.0) is more robust to outlier pressure samples than MSE; predicted −5 to −15% on val_avg/mae_surf_p.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline (dqe6ejfz) | Variant (3nivkqy0) | Δ vs in-PR baseline | Δ vs canonical |
|---|---|---|---|---|
| best_val_avg/mae_surf_p | 138.43 (ep 14) | **110.83** (ep 13) | **−19.9%** | **−18.1%** |
| val_single_in_dist/mae_surf_p | 159.77 | 132.06 | −17.3% | |
| val_geom_camber_rc/mae_surf_p | 175.41 | 124.13 | −29.2% | |
| val_geom_camber_cruise/mae_surf_p | 100.26 | 82.72 | −17.5% | |
| val_re_rand/mae_surf_p | 118.30 | 104.40 | −11.7% | |
| test 3-split avg (excl cruise) | 139.80 | **109.75** | **−21.5%** | **−19.0%** |
| test_single_in_dist/mae_surf_p | 142.35 | 118.65 | −16.6% | |
| test_geom_camber_rc/mae_surf_p | 161.60 | 111.97 | −30.7% | |
| test_re_rand/mae_surf_p | 115.46 | 98.64 | −14.6% | |
| test_geom_camber_cruise/mae_surf_p | NaN | NaN | — | pre-existing |
| Epochs (30-min cap) | 14 | 13 | — | — |

**Decision:** MERGED. Clean single-variable diff (sq_err → F.smooth_l1_loss). −18.1% val, −19.0% test vs canonical. All 4 val splits improve (11–29%). No wall-clock penalty (same 30-min cap, 13/14 epochs).

**Analysis:**
- Largest single gain in this launch. Huber robustness to outliers is the dominant driver: the rc and single_in_dist splits (which have the most geometric variation and likely the most outlier pressure samples) benefit the most.
- W&B summary value (126.22) is the last-epoch value after oscillation — `best_val_avg/mae_surf_p = 110.83` at best-epoch checkpoint is the correct metric. Student reported correctly.
- **Sets new canonical baseline:** `val_avg/mae_surf_p = 110.83`, `test_avg/mae_surf_p (excl cruise) = 109.75`. See BASELINE.md.
- **Follow-up priority:** Huber delta sensitivity (delta=0.5 and 2.0) — determine if 1.0 is optimal for this scale.

## 2026-05-15 16:00 — PR #3147 (askeladd): LR warmup + peak bump 5e-4→1e-3 (3-epoch linear warmup) — **MERGED**

- Branch: `willowpai2i48h3-askeladd/lr-warmup-peak`
- W&B group: [`lr-warmup-peak`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/lr-warmup-peak)
- Baseline run: `pckvl17x` | Variant run: `gyl9qikv`

**Hypothesis:** 3-epoch linear warmup to peak lr=1e-3, then cosine anneal to 0. Predicted −5 to −15%.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline (pckvl17x) | Variant (gyl9qikv) | Δ vs in-PR baseline | Δ vs canonical |
|---|---|---|---|---|
| best_val_avg/mae_surf_p | 131.88 (ep 14) | **123.20** (ep 14) | **−6.6%** | **−8.9%** |
| val_single_in_dist/mae_surf_p | 168.53 | 160.29 | −4.9% | |
| val_geom_camber_rc/mae_surf_p | 146.49 | 132.83 | −9.3% | |
| val_geom_camber_cruise/mae_surf_p | 97.22 | 89.11 | −8.3% | |
| val_re_rand/mae_surf_p | 115.27 | 110.56 | −4.1% | |
| test 3-split avg (excl cruise) | 133.85 | **121.06** | **−9.6%** | **−10.7%** |
| test_single_in_dist/mae_surf_p | 153.53 | 137.27 | −10.6% | |
| test_geom_camber_rc/mae_surf_p | 131.94 | 118.62 | −10.1% | |
| test_re_rand/mae_surf_p | 116.07 | 107.29 | −7.6% | |
| test_geom_camber_cruise/mae_surf_p | NaN | NaN | — | pre-existing |
| Epochs (30-min cap) | 14 | 14 | — | — |

**Decision:** MERGED (against canonical baseline 135.30/135.54; variant 123.20/121.06 beats canonical by −8.9%/−10.7%). Clean single-variable diff (LR=1e-3 + 3-epoch LinearLR warmup → SequentialLR). All 4 val splits improve. No wall-clock penalty.

**Analysis:**
- Both arms hit 30-min cap at exactly 14 epochs — neither ran the cosine tail. The gain is conservative: warmup typically helps most in later training where the properly-warmed model finds a better basin.
- Consistent 4–10% improvement across all splits including OOD (rc −9.3%, re_rand −4.1%) suggests warmup is a general optimizer improvement rather than a memorization/overfitting artifact.
- **Compounding with Huber:** Huber+LR-warmup was not tested jointly; these are likely orthogonal (loss vs schedule). Next round will test this directly.
- **Follow-up:** Askeladd should tune warmup duration (2 vs 5 epochs) on the new Huber+LR-warmup baseline.

## 2026-05-15 16:00 — PR #3161 (frieren): Per-sample loss normalization — **CLOSED**

- Branch: `willowpai2i48h3-frieren/per-sample-loss-norm`
- W&B group: [`per-sample-loss-norm`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/per-sample-loss-norm)
- Baseline run: `jmuo7vtx` | Variant run: `x158gao1`

**Hypothesis:** Normalizing loss per sample (equal weight per sample not per node) would improve OOD by preventing large-mesh samples from dominating. Predicted −3 to −8%.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline (jmuo7vtx) | Variant (x158gao1) | Δ% |
|---|---|---|---|
| best_val_avg/mae_surf_p | 130.40 (ep 14) | 147.36 (ep 13) | **+13.0% worse** |
| val_single_in_dist/mae_surf_p | 172.21 | 175.50 | +1.9% |
| val_geom_camber_rc/mae_surf_p | 135.10 | 153.78 | +13.8% |
| val_geom_camber_cruise/mae_surf_p | 99.30 | 118.74 | +19.6% |
| val_re_rand/mae_surf_p | 114.98 | 141.44 | +23.0% |
| test 3-split avg (excl cruise) | 127.49 | 139.44 | **+9.4% worse** |

Variant vs canonical baseline (135.30): variant is +8.9% worse.

**Decision:** CLOSED. Variant uniformly worse across all 4 val splits (1–23% regression). The per-sample normalization direction is empirically falsified in this setting.

**Analysis:**
- The baseline arm here (130.40) is notably stronger than the canonical baseline (135.30) — confirming the evaluation is sound and the regression is real.
- Per-sample normalization treats a 242K-node mesh the same as a 74K-node mesh. This changes the effective batch distribution and may destabilize learning since gradients from structurally different samples are now equally weighted.
- Student's suggested follow-ups (mesh-size upweighting, per-sample magnitude weighting) are worth tracking in the research ideas backlog but are not immediately prioritized.

## 2026-05-15 16:00 — PR #3165 (nezuko): Depth scaling n_layers 5→8 — **CLOSED**

- Branch: `willowpai2i48h3-nezuko/depth-8-layers`
- W&B group: [`depth-8-layers`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/depth-8-layers)
- Baseline run: `gpf8gh2a` | Variant run: `zmlvfufz`

**Hypothesis:** Deeper Transolver (5→8 layers, +60% more layers) reduces underfitting. Predicted −5 to −15% under the 30-min cap.

**Result (variant vs baseline, lower is better):**

| Metric | Baseline (gpf8gh2a) | Variant (zmlvfufz) | Δ% |
|---|---|---|---|
| best_val_avg/mae_surf_p | 125.07 (ep 12) | 156.85 (ep 9) | **+25.4% worse** |
| val_single_in_dist/mae_surf_p | 120.51 | 151.22 | +25.5% |
| val_geom_camber_rc/mae_surf_p | 120.97 | 148.84 | +23.0% |
| val_geom_camber_cruise/mae_surf_p | 105.09 | 157.37 | +49.8% |
| val_re_rand/mae_surf_p | 133.73 | 170.47 | +27.5% |
| test 3-split avg (excl cruise) | 121.09 | 156.35 | **+29.1% worse** |
| Sec/epoch | 132.2s | 205.2s | 1.55× slower |
| Epochs (30-min cap) | 14 (best @12) | 9 (best @9) | −36% |

Variant vs canonical baseline (135.30): variant is +15.9% worse.

**Decision:** CLOSED. +25.4% worse than in-PR baseline, +15.9% worse than canonical. All 4 splits regress. Same 1.55× wall-clock penalty as PR #3140 (width scaling) — depth scaling is equally penalized.

**Analysis:**
- Confirms the pattern from PR #3140: any capacity scaling that slows per-epoch time by ~1.55× loses ~36% of the epoch budget, and the under-trained deeper model cannot compensate.
- The cruise split is worst (+49.8%) — OOD geometry is most sensitive to under-training.
- **Implication:** Capacity scaling of any kind (width, depth, MLP ratio) is essentially disallowed under the 30-min wall-clock cap with the current mesh sizes. Future capacity experiments should either (a) target faster operation per step or (b) explicitly test at epoch-matched (not wall-clock-matched) comparisons in a longer run.

## 2026-05-15 16:55 — PR #3172 (thorfinn): Fourier (x,z) features + slice_num 64→96 — **REQUEST CHANGES**

- Branch: `willowpai2i48h3-thorfinn/fourier-pos-features`
- W&B group: `fourier-pos-features`
- Baseline run: `umu6lu65` | Variant run: `aeelhxbk`

**Hypothesis:** NeRF-style Fourier positional encoding on (x, z) coordinates with bumped slice_num (64→96) gives richer high-frequency spatial features for the PhysicsAttention to attend over. Predicted −5 to −15%.

**Result (variant vs in-PR baseline):**

| Metric | Baseline (umu6lu65) | Variant (aeelhxbk) | Δ vs in-PR baseline | Δ vs new canonical (110.83/109.75) |
|---|---|---|---|---|
| best_val_avg/mae_surf_p | 147.26 | 126.64 | **−14.0%** | **+14.3% worse** |
| test_single_in_dist/mae_surf_p | 169.20 | 132.60 | −21.6% | |
| test_geom_camber_rc/mae_surf_p | 135.76 | 135.30 | −0.3% | |
| test_re_rand/mae_surf_p | 133.44 | 112.25 | −15.9% | |
| test_geom_camber_cruise/mae_surf_p | NaN | NaN | — | pre-existing |
| test 3-split avg (excl cruise) | 146.14 | 126.72 | **−13.3%** | **+15.5% worse** |
| sec/epoch | 132.2s | 151.6s | 1.15× | — |
| Epochs (30-min cap) | ~14 | ~12 | −14% | — |

**Decision:** REQUEST CHANGES. Variant has real signal vs its own baseline (−14%, mild 1.15× wall-clock penalty), but the in-PR baseline (147.26) was pre-merge — trained without Huber + LR warmup. Against the new canonical baseline (110.83), the variant is 14.3% worse.

**Reason for not closing:**
- The Fourier PE + slice_num=96 direction shows genuine signal: −14% on val and −13% on test vs the same-PR baseline. Both arms ran on the SAME unmerged codebase (pre-Huber, pre-warmup), so the within-PR comparison is clean.
- Only 1.15× sec/epoch overhead (vs 1.55× for width/depth) — slice_num=96 is wall-clock-acceptable.
- The student's delayed pod start (JSON parse bug, ~3.5 h delay) meant their run happened against the pre-merge baseline. We need a re-test on top of the merged stack to know if the Fourier representation gain is additive with Huber + LR warmup.

**Follow-up requested:** Rebase branch onto current `icml-appendix-willow-pai2i-48h-r3` (which now has Huber + LR warmup). Re-run dual arm in new wandb_group `fourier-pos-features-v2`: baseline (no Fourier, slice_num=64) + variant (Fourier ON, slice_num=96). Decision rule: merge if rebased variant beats new canonical baseline by ≥1% on val_avg/mae_surf_p AND test 3-split also improves.

**Notable:** Within-PR delta on `test_re_rand` was particularly strong (−15.9%), suggesting Fourier features help with the Reynolds-number generalization split. Worth tracking specifically on the rebased re-test.

## 2026-05-15 17:30 — PR #3283 (alphonse): SOAP optimizer drop-in for AdamW — **REQUEST CHANGES (rebase)**

- Branch: `willowpai2i48h3-alphonse/soap-optimizer`
- W&B group: [`soap-optimizer`](https://wandb.ai/wandb-applied-ai-team/senpai-v1/groups/soap-optimizer)
- Baseline run: `7whq4pg2` | Variant run: `e731efke`

**Hypothesis:** Replace AdamW with SOAP (Shampoo-with-Adam in eigenbasis preconditioner, Vyas et al. arXiv:2409.11321). The composite `vol_loss + 10*surf_loss` creates Type-I (magnitude) and Type-II (direction) gradient conflicts between heads; SOAP's per-layer Hessian-style preconditioner should resolve both. Predicted −5 to −20% with ≤5% wall-clock overhead at precondition_frequency=10.

**Result (variant vs in-PR baseline; same lr=5e-4, MSE loss, cosine schedule on BOTH arms — pre-merge config):**

| Metric | baseline-adamw (7whq4pg2) | variant-soap (e731efke) | Δ vs in-PR baseline | Δ vs new canonical (110.83/109.75) |
|---|---|---|---|---|
| best_val_avg/mae_surf_p | 155.05 (ep 14) | **78.77 (ep 12)** | **−49.2%** | **−28.9%** |
| val_single_in_dist/mae_surf_p | 200.99 | 110.15 | −45.2% | |
| val_geom_camber_rc/mae_surf_p | 159.89 | 93.20 | −41.7% | |
| val_geom_camber_cruise/mae_surf_p | 124.96 | 55.64 | −55.5% | |
| val_re_rand/mae_surf_p | 134.35 | 71.11 | −47.1% | |
| test_single_in_dist/mae_surf_p | 171.05 | 87.70 | −48.7% | |
| test_geom_camber_rc/mae_surf_p | 145.95 | 81.46 | −44.2% | |
| test_re_rand/mae_surf_p | 136.20 | 67.38 | −50.5% | |
| test_geom_camber_cruise/mae_surf_p | NaN | NaN | — | pre-existing bug |
| test 3-split mean (excl cruise) | 151.07 | **78.85** | **−47.8%** | **−28.2%** |
| sec/epoch | 133.05s | 137.01s | +3.0% | +3.9% |
| Peak VRAM | OK (96 GB) | OK | +<50 MB | — |
| Epochs reached / cap | 14 / 50 | 14 / 50 | — | — |

**Decision:** REQUEST CHANGES — branch has merge conflicts with merged Huber + LR-warmup changes. The result is verified (W&B numbers match student report exactly; tags confirm `willowpai2i48h3-alphonse` only — no cross-tag contamination). However, both arms ran on pre-merge config (MSE + cosine, no warmup), so the comparison against the new canonical baseline (110.83) requires re-running on the merged stack.

**Why this is the most consequential result this round (and likely to merge):**
- **Largest single-knob improvement seen.** Even comparing the pre-merge SOAP arm (78.77) against the post-merge canonical (110.83), variant beats by **−28.9% on val**. That's larger than Huber (−18.1%) and LR-warmup (−8.9%) combined.
- **Wall-clock neutral.** +3% per-epoch is negligible — does not eat into the 30-min cap.
- **Generalization gap shrinks proportionally.** The largest test gain is on `test_re_rand` (−50.5%), and val_geom_camber_cruise drops −55.5%. This is consistent with the curvature-aware step finding flatter minima, not just lower train loss.
- **No instability or NaN issues** — the SOAP arm completes the full 14-epoch budget cleanly under the wall-clock cap.

**Mechanism (student analysis, plausible):**
- AdamW's diagonal preconditioner approximates 1/sqrt(diag(Fisher)). Under the 10× surf_weight scaling, the off-diagonal coupling between layers becomes important — SOAP's L/R eigenbasis captures per-layer cross-parameter curvature, which is the right structure.
- The flatter-minimum hypothesis would predict that the gain compounds with regularizers (Huber, AoA aug, entropy reg) rather than being subsumed by them. Worth verifying empirically.

**Action requested (full instructions in PR comment):**
1. Rebase `willowpai2i48h3-alphonse/soap-optimizer` onto current `icml-appendix-willow-pai2i-48h-r3` (which has Huber + LR warmup merged).
2. Resolve `train.py` conflicts: keep Huber loss, keep SequentialLR(LinearLR, CosineAnnealingLR) schedule, wire SOAP as alternative inside `--optimizer` switch. Keep `target/soap.py` and Config additions verbatim.
3. Re-run BOTH arms on the merged stack in new wandb_group `soap-on-merged`:
   - `baseline-adamw-merged` (Huber + warmup + AdamW)
   - `variant-soap-merged` (Huber + warmup + SOAP)
4. Decision rule for merge: SOAP variant beats new canonical 110.83 on `val_avg/mae_surf_p`. Given the pre-merge SOAP arm already achieved 78.77 standalone, this is a high-probability outcome.

**Strategic implication:**
If SOAP+Huber+LRwarmup stacks orthogonally (expected: val ~60-80), this is the new dominant lever and round-3 strategy pivots to (a) tuning SOAP hyperparams (LR sweep with SOAP, preconditioner frequency), (b) verifying other round-2 hypotheses still gain on the SOAP-stacked baseline.

## 2026-05-15 19:10 — PR #3322 (frieren): AoA reflection augmentation (sign-flip AoA + Uy, p=0.5) — **CLOSED**

- Branch: `willowpai2i48h3-frieren/aoa-reflection-aug`
- W&B group: `aoa-reflection-aug`
- Baseline run: `hvvk1rd6` | Variant run: `kbc0rf63`

**Hypothesis:** Sign-flip AoA (dims 14, 18) and Uy (y[:,1]) at p=0.5 per sample to synthetically extend training coverage into positive-AoA regime. The raceCar domain spans AoA −10° to 0° only; z-reflection symmetry would double coverage.

**Result (variant vs in-PR baseline; both on merged Huber+LRwarmup stack):**

| Metric | Baseline (hvvk1rd6) ep 13 | Variant (kbc0rf63) ep 14 | Δ |
|---|---|---|---|
| best_val_avg/mae_surf_p | **118.31** | **132.85** | **+12.3% (worse)** |
| val_single_in_dist/mae_surf_p | 137.70 | 168.94 | +22.7% |
| val_geom_camber_rc/mae_surf_p (target) | 131.60 | 145.14 | +10.3% |
| val_geom_camber_cruise/mae_surf_p | 94.12 | 96.95 | +3.0% |
| val_re_rand/mae_surf_p | 109.83 | 120.40 | +9.6% |
| test_single_in_dist/mae_surf_p | 125.17 | 147.50 | +17.8% |
| test_geom_camber_rc/mae_surf_p | 114.44 | 138.22 | +20.8% |
| test_re_rand/mae_surf_p | 108.73 | 116.70 | +7.3% |
| test_3split_avg (excl cruise) | 116.11 | **134.14** | **+15.5%** |
| Epochs / cap | 14 | 14 | wall-clock hit |

**Decision:** CLOSED. +15.5% test regression; all splits worse.

**Analysis (from student, confirmed):**
- **The symmetry assumption breaks due to NACA camber.** `p(AoA=+θ) ≈ p(AoA=−θ)` and `Uy(AoA=+θ) ≈ −Uy(AoA=−θ)` requires z-symmetric foils. NACA cambered foils have M>0 camber that breaks this. Flipping AoA without also flipping the camber sign produces physically inconsistent (x, y) pairs.
- Damage scales with camber: `val_single_in_dist` (M=2-9, +22.7%) >> `val_geom_camber_rc` (M=6-8, +10.3%) >> `val_geom_camber_cruise` (M=0-6, +3.0%).
- Even with unlimited epochs the variant would need to overcome ~12-15% deficit that reflects *wrong physics*, not slow convergence.
- W&B verified: tags clean, Huber+warmup config confirmed, numbers match student report exactly.

**Implication:** AoA-reflection augmentation is not viable in its current form for this dataset. A valid z-reflection requires flipping z-position, arc-length signs, dsdf signs, AND NACA M sign simultaneously — a separate, more complex hypothesis. Assigned frieren [PR #3415] to test log-Re sinusoidal embedding instead.

## 2026-05-15 19:55 — PR #3323 (nezuko): PhysicsAttention entropy regularization — **CLOSED**

- Branch: `willowpai2i48h3-nezuko/attn-entropy-reg`
- W&B group: `attn-entropy-reg`
- Baseline run: `3nhewu3e` | Variant-0.01 run: `cy3zdddn` | Variant-0.001 run: `328tqt31`

**Hypothesis:** Entropy regularization on PhysicsAttention slice-weight distributions prevents slice collapse and improves OOD generalization. `entropy_reg_weight ∈ {0.01, 0.001}` tested.

**Result (variant vs in-PR baseline):**

| Run | entropy_w | best_val_avg | test_3split | val_re_rand | Δ val (vs baseline) |
|---|---|---|---|---|---|
| baseline (`3nhewu3e`) | 0.0 | **115.37** (ep 12) | **114.91** | 101.69 | — |
| variant-0.01 (`cy3zdddn`) | 0.01 | 123.62 (ep 11) | 123.44 | 111.84 | **+7.2%** |
| variant-0.001 (`328tqt31`) | 0.001 | 120.52 (ep 11) | 122.84 | 104.04 | **+4.5%** |

**Decision:** CLOSED. Both variants regress; monotone with regularizer strength. W&B verified (all tags clean, config confirmed, numbers match exactly).

**Per-layer slice entropy diagnostics (end of training; max = log(64) ≈ 4.16 nats):**

| Run | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| baseline | 1.52 | 2.79 | 3.96 | 2.58 | 2.98 |
| variant-0.01 | 3.27 | 4.00 | 4.01 | 3.99 | 4.16 |
| variant-0.001 | 3.05 | 2.62 | 2.46 | 3.59 | 3.84 |

**Analysis:**
1. **Slice collapse is real on the baseline** — L0 = 1.52 nats ≈ effective use of only 5 of 64 slices per head. The hypothesis premise is empirically confirmed.
2. **But specialization is a feature, not a bug.** Forcing uniformity removes routing information. The model uses slice collapse to implement soft cluster heads, which is functional.
3. The monotone regression (val: 115 → 120 → 124 as weight: 0 → 0.001 → 0.01) plus the per-layer entropy curves confirm this is a real slice-flattening effect, not training instability.
4. Test/val co-move — no OOD benefit from slice uniformity. The hoped-for "more uniform = better OOD generalization" does not materialize on this dataset.
5. Variant-0.001 caused L2 to *decrease* below baseline (2.46 vs 3.96) — likely interaction between entropy reg and temperature learning that warped the regularization landscape.

**Key insight preserved for future reference:** The layer-0 collapse pattern (1.52 nats) is dramatically lower than upper layers (3.96 at L2). Anti-collapse interventions targeted at L0 only (weight ≤1e-4) or temperature-based approaches may be worth revisiting in a later round without flattening upper layers.

**Diagnostic instrumentation note:** Student implemented `train/mean_slice_entropy` and per-layer entropy logging (`diag/L*_slice_entropy`). These diagnostics should be preserved in the codebase for future architecture investigations.

**Assigned nezuko [PR #3430] to test EMA of model weights (decay=0.999) for evaluation.**
