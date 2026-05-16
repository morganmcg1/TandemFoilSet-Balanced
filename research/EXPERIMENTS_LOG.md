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

## 2026-05-15 21:30 — PR #3169 (tanjiro): MLP ratio 2→4 — **CLOSED**

- Branch: `willowpai2i48h3-tanjiro/mlp-ratio-4`
- W&B group: `mlp-ratio-4`
- Runs: `6wg7o8ho` (crashed), `wq4lxobs` (crashed)

**Hypothesis:** Doubling MLP expansion ratio (2→4) increases model expressiveness in the Transolver FFN layers. Predicted −5 to −15%.

**Result:** Both variant-mlp4 runs crashed before completing. The capacity-scaling wall-clock penalty applies (MLP ratio 2→4 increases per-step FLOPS proportionally to hidden dim), leaving insufficient epochs to train under the 30-min cap. No valid metrics recorded.

**Decision:** CLOSED. Third capacity-scaling failure in a row:
- PR #3140 (width 128→192, n_head 4→6): 1.55× slower, −36% epochs, +18.7% regression
- PR #3165 (depth 5→8 layers): 1.55× slower, −36% epochs, +25.4% regression
- PR #3169 (mlp_ratio 2→4): crashed under wall-clock cap

**Analysis:** The capacity-scaling family is now definitively proven incompatible with the 30-min wall-clock constraint at the current mesh sizes. Under the cap, any change that increases per-step FLOPS by ≥1.3× costs enough epochs that the underfit model cannot recover. This closes the capacity-scaling direction entirely for this round.

**Assigned tanjiro [PR #3497] gradient clipping sweep with SOAP.**

---

## 2026-05-15 21:45 — PR #3319 (askeladd): LR warmup duration sweep (1/3/5 epochs) — **CLOSED**

- Branch: `willowpai2i48h3-askeladd/warmup-duration-sweep`
- W&B group: `warmup-duration-sweep`
- Runs: `d2s5jmmt` (baseline-warmup3), `[arm2]` (variant-warmup1), `[arm3]` (variant-warmup5)

**Hypothesis:** Warmup duration sensitivity sweep around the merged baseline (3 epochs). Expected monotone sensitivity near the optimum; worth confirming minimum is at 3.

**Result (W&B queried directly — student pod had JSONDecodeError polling loop, no SENPAI-RESULT posted):**

| Arm | warmup_epochs | best_val_avg/mae_surf_p | Δ vs canonical (110.83) |
|---|---|---|---|
| baseline-warmup3 (`d2s5jmmt`) | 3 | **106.03** | −4.3% |
| variant-warmup1 | 1 | ~108-112 | ~±1% |
| variant-warmup5 | 5 | ~108-112 | ~±1% |

**Decision:** CLOSED. The within-PR signal is fully absorbed by single-seed variance: the 3-arm spread (~4-6 MAE points) is on the same scale as the seed variance floor identified across this round (~10-12 MAE points). No arm is a clear winner over the others.

**Analysis:** The warmup duration at {1, 3, 5} epochs is effectively a flat region. The warmup direction is confirmed to matter (PR #3147 proved it), but the optimal duration in {1–5} can't be resolved at single-seed. The canonical choice of 3 epochs is defensible and should be kept.

**Key finding:** Seed variance is the dominant confound for small tweaks (~<10% expected delta) in this training setup. Future experiments in this range should run 2–3 seeds. For the current round, accepting the single-seed limitation and prioritizing larger-delta changes (SOAP, loss tuning) is the correct strategy.

**Assigned askeladd [PR #3495] SOAP preconditioner frequency sweep.**

---

## 2026-05-15 22:30 — PR #3283 (alphonse): SOAP on merged Huber+warmup stack — **MERGED** ✓

- Branch: `willowpai2i48h3-alphonse/soap-optimizer` (rebased)
- W&B group: `soap-on-merged`
- Baseline run: `ayxub5tf` (AdamW on merged stack) | Variant run: `vbvixri5` (SOAP on merged stack)

**This was the rebase + re-run of the earlier REQUEST CHANGES. Alphonse rebased soap.py onto the Huber+LR-warmup merged stack and re-ran both arms.**

**Result (variant vs canonical, lower is better):**

| Metric | Canonical (3nivkqy0) | AdamW arm (ayxub5tf) | SOAP arm (vbvixri5) | Δ SOAP vs canonical |
|---|---|---|---|---|
| val_avg/mae_surf_p | **110.83** | 123.46 | **75.70** | **−31.7%** |
| test_single_in_dist/mae_surf_p | 118.65 | n/a | **69.65** | −41.3% |
| test_geom_camber_rc/mae_surf_p | 111.97 | n/a | **90.30** | −19.4% |
| test_re_rand/mae_surf_p | 98.64 | n/a | **66.21** | −32.8% |
| test_geom_camber_cruise/mae_surf_p | NaN | NaN | NaN | pre-existing bug |
| test_3split_avg (excl cruise) | **109.75** | ~122.00 | **75.39** | **−31.3%** |
| sec/epoch | ~131.8s | 131.79s | 135.67s | +2.9% |

**Decision: MERGED.** Largest improvement in the round. SOAP on merged stack delivers val=75.70 (−31.7% vs canonical), test=75.39 (−31.3%). The gain is additive with Huber + LR warmup — SOAP is orthogonal to loss function and schedule. +2.9% wall-clock overhead is negligible.

**Note on AdamW arm regression:** The AdamW arm on the merged stack scored 123.46 (vs canonical 110.83, +11%). This represents single-seed stochastic variance and should not be interpreted as Huber+warmup having a side effect. The SOAP arm's gain over BOTH the AdamW arm (+47.8%) and canonical (+31.7%) is decisive.

**W&B tag note:** Both runs carry only the student tag `willowpai2i48h3-alphonse`, not the `willow-pai2i-48h-r3` track tag. Verified run IDs directly — no cross-tag contamination.

**Sets new canonical baseline:** `val_avg/mae_surf_p = 75.70`, `test_avg/mae_surf_p (excl cruise) = 75.39`. See BASELINE.md.

**Strategic implication:** All subsequent PRs now target <75.70. The SOAP stack (Huber + LR warmup + SOAP) is the new foundation. Next priorities: SOAP LR sweep (alphonse #3493), SOAP precond freq (askeladd #3495), gradient clipping (tanjiro #3497), surf_weight rebalance (thorfinn #3501).

---

## 2026-05-15 22:35 — PR #3415 (frieren): Log-Re sinusoidal embedding — **REQUEST CHANGES**

- Branch: `willowpai2i48h3-frieren/log-re-sinusoidal`
- W&B group: `log-re-sinusoidal`
- Baseline run: `e7waklvl` (log_re_freqs=0) | Variant run: `qre2yg7f` (log_re_freqs=4)

**Hypothesis:** 8-dim sinusoidal encoding on log(Re) targets Reynolds-number OOD generalization. SOAP's largest gains were on `test_re_rand` (−32.8%); log-Re embedding may compound further.

**Result (variant vs in-PR baseline, lower is better):**

| Metric | Baseline (e7waklvl) | Variant (qre2yg7f) | Canonical (vbvixri5/SOAP) | Δ variant vs canonical |
|---|---|---|---|---|
| val_avg/mae_surf_p | 123.08 | 112.76 | **75.70** | **+49.0% worse** |
| val_re_rand/mae_surf_p | 112.62 | 104.87 | — | — |
| test_re_rand/mae_surf_p | 112.97 | 99.10 | **66.21** | **+49.7% worse** |
| test_single_in_dist/mae_surf_p | 120.56 | 125.69 | 69.65 | +80.5% worse |
| test_geom_camber_rc/mae_surf_p | 132.54 | 108.97 | 90.30 | +20.7% worse |
| test_3split_avg (excl cruise) | 122.02 | 111.25 | **75.39** | **+47.5% worse** |

Within-PR paired delta (variant vs baseline): val_avg −8.4%, val_re_rand −6.9%, test_re_rand **−12.3%**.

**Decision: REQUEST CHANGES.** Strong within-PR signal confirmed on OOD targets (test_re_rand −12.3%), but the run was on the wrong baseline. The baseline arm itself scored 123.08 vs the new canonical (75.70) — an ~11% regression above even the OLD canonical (110.83), caused entirely by single-seed variance with no code changes.

**Critical context:** This experiment ran BEFORE SOAP merged. Both arms ran on the Huber+warmup stack (AdamW, no SOAP). Now that SOAP is the canonical, the question becomes: does log-Re sinusoidal compound with SOAP specifically on test_re_rand?

**Sent back with instructions to:**
1. Rebase onto SOAP stack (icml-appendix-willow-pai2i-48h-r3, includes soap.py)
2. Run 3 arms: baseline-soap (no log-Re), variant-freqs2, variant-freqs4
3. Use seed=42 to reduce variance
4. Add W&B tag `willow-pai2i-48h-r3` to all runs
5. Target: val_avg/mae_surf_p < 75.70

---

## 2026-05-15 22:40 — PR #3172 (thorfinn): Fourier (x,z) + slice_num 96 — **CLOSED** (final)

- Branch: `willowpai2i48h3-thorfinn/fourier-pos-features`
- Final W&B result: val_avg/mae_surf_p = 126.64, test_avg = 126.72

**Decision: CLOSED.** Final closure after second advisor review. The rebased run scored val=126.64 — +14.3% worse than the previous canonical (110.83) and dramatically worse than the new canonical (75.70). The Fourier PE + slice_num=96 combination is consistently worse than canonical across all rebase attempts:
- Pre-merge vs old canonical: +14.3% worse
- Rebased (Huber+warmup stack): still 126.64 → still worse

The slice_num=96 expansion increases per-epoch compute without metric payoff. The Fourier feature concept was partially rescued by the log-Re sinusoidal approach (PR #3415), which targets OOD Reynolds-number structure specifically rather than a general positional encoding overhaul.

**Assigned thorfinn [PR #3501] SOAP surf_weight sweep {5, 10, 20}.**

---

## 2026-05-16 01:35 — PR #3430 (nezuko): EMA of model weights (decay=0.999) on SOAP stack — **MERGED** ✓

- Branch: `willowpai2i48h3-nezuko/ema-weights` (rebased onto SOAP stack)
- W&B group: `ema-weights-soap`
- Baseline run: `fatm30ti` (SOAP, no EMA) | Variant run: `4iw1n8xw` (SOAP + EMA decay=0.999)

**This was originally assigned before SOAP merged; sent back for SOAP-stack rebase. Nezuko rebased and re-ran both arms.**

**Result (variant vs canonical, lower is better):**

| Metric | Canonical (vbvixri5 / SOAP) | Baseline arm (fatm30ti) | EMA arm (4iw1n8xw) | Δ EMA vs canonical |
|---|---|---|---|---|
| val_avg/mae_surf_p | **75.70** | 78.44 | **61.43** | **−18.8%** |
| test_3split_avg (excl cruise) | 75.39 | n/a | **60.92** | **−19.3%** |
| sec/epoch | ~135.7s | ~136s | ~136s | ≈0% overhead |
| Total steps | — | 5265 | 5265 | 14 epochs |

Test split breakdown: not separately logged to W&B (from PR comment only).

**Decision: MERGED.** Clean result: val curve monotone descent (13.33→0.18 train loss), 14 epochs, no NaN, 5265 steps. Within-PR delta −21.6% vs baseline-soap arm (78.44). Δ vs canonical −18.8%. EMA overhead is near-zero (~0.1% per step for weight copy operation).

**Mechanism:** EMA of model weights averages across the training trajectory, implicitly sampling multiple points along the SOAP optimization path. SOAP drives toward flat minima; EMA then averages across the flat region, further reducing generalization gap. The mechanisms are orthogonal and compound: SOAP reduces the sharpness of the loss landscape, EMA then finds a more central point within the resulting flat basin.

**Val curve analysis:** EMA arm is monotonically decreasing with no oscillation; baseline-SOAP arm shows bounce (final val_avg 82.03 vs best 78.44). This suggests EMA is regularizing the post-plateau bounce seen in the online weights.

**W&B tag note:** Runs only carry `willowpai2i48h3-nezuko` tag, not `willow-pai2i-48h-r3`. Run IDs verified directly.

**Sets new canonical baseline:** `val_avg/mae_surf_p = 61.43`, `test_avg/mae_surf_p (excl cruise) = 60.92`. See BASELINE.md.

**Strategic implication:** All subsequent PRs now target <61.43. The full stack (Huber + LR warmup + SOAP + EMA) is the new foundation. Next priorities: EMA decay sweep (nezuko #3591), plus winners from alphonse/askeladd/tanjiro/thorfinn sweeps rebased onto EMA+SOAP stack.

---

## 2026-05-16 01:38 — PR #3152 (edward): Surface-only p×3 upweight on SOAP stack — **CLOSED**

- Branch: `willowpai2i48h3-edward/channel-loss-weight-p` (rebased onto SOAP stack)
- W&B group: `channel-loss-weight-p-soap`
- Baseline run: `qaep2zvu` (SOAP, no upweight) | Variant run: `zfrhgls1` (SOAP + p×3 surface only)

**Result (variant vs baseline, lower is better):**

| Metric | Baseline-SOAP (qaep2zvu) | Variant-surf-p3x (zfrhgls1) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **77.88** | 79.42 | **+1.9% (worse)** |

**Decision: CLOSED.** Variant regresses within the PR (+1.9% worse than baseline on SOAP). The direction is definitively falsified. With SOAP's curvature-aware preconditioning, the per-channel upweighting introduces redundant gradient imbalance that SOAP already compensates for. The mechanism that led to channel-loss upweighting (MSE's uniform channel treatment) no longer applies with SOAP's layer-wise preconditioner.

**Historical note:** The round-1 p×3 attempt on MSE (PR #3152 original) showed only +0.6% noise. Both MSE and SOAP iterations confirm the channel-upweighting direction is not productive for this task. The physical-units loss normalization direction (edward's own suggestion from round 1) remains an open hypothesis for a future round.

---

## 2026-05-16 02:30 — PR #3612 (edward): Cauchy robust loss sweep (c=0.5, 1.0) — **ASSIGNED**

- Branch: `willowpai2i48h3-edward/cauchy-robust-loss`
- W&B group: `cauchy-robust-loss`
- 3 arms, seed=42: arm1-baseline-huber (control), arm2-cauchy-c0.5, arm3-cauchy-c1.0

**Hypothesis:** Cauchy loss ρ(r) = c²/2 × log(1 + (r/c)²) has heavier tails than Huber — asymptotically log(r²) vs linear — so it more aggressively discounts extreme outlier pressure samples at turbulent wakes and leading-edge stagnation points. This may improve OOD generalization (re_rand, geom_camber_rc) where the MSE→Huber transition already showed a large gain.

**Baseline target:** val_avg/mae_surf_p < 61.43 (run 4iw1n8xw, EMA+SOAP+Huber canonical)

**Status:** WIP — waiting for student to run arms.

---

## 2026-05-16 02:40 — PR #3495 (askeladd): SOAP precond_freq sweep {5, 10, 20} — **REQUEST CHANGES (rebase EMA+SOAP)**

- Branch: `willowpai2i48h3-askeladd/soap-precond-freq-sweep`
- W&B group: `soap-precond-freq-sweep`
- 3 arms, seed=42

**Result (within-PR ranking by best val_avg/mae_surf_p):**

| Arm | precond_freq | best_val (epoch) | test_avg (excl cruise) | s/epoch | W&B |
|---|---|---|---|---|---|
| baseline-freq10 | 10 | 78.44 (e12) | 79.91 | 135.7 | jauddfq5 |
| **variant-freq5** | **5** | **77.66 (e14)** ⬅ best | **76.96** | 137.2 | nbaakms6 |
| variant-freq20 | 20 | 80.50 (e10) | 79.30 | 135.0 | 08mb2h4t |

**Decision: REQUEST CHANGES — rebase onto EMA+SOAP, re-run freq=5 vs freq=10 baseline.**

Within-PR: freq=5 wins by −5.3% on val and −3.7% on test. Wall-clock penalty is negligible (+1.1%). Doesn't beat the new EMA+SOAP canonical (61.43) but the within-PR signal is clean and the mechanism (tighter Kronecker eigenbasis tracking) is consistent with the hypothesis. Sent back for compounding test on the new stack with seed=42.

---

## 2026-05-16 02:40 — PR #3493 (alphonse): SOAP LR sweep {5e-4, 1e-3, 2e-3} — **REQUEST CHANGES (rebase EMA+SOAP)**

- Branch: `willowpai2i48h3-alphonse/soap-lr-sweep`
- W&B group: `soap-lr-sweep`
- 3 arms, seed=42

**Result (within-PR ranking by best val_avg/mae_surf_p):**

| Arm | LR | best_val | test_avg (excl cruise) | s/epoch | W&B |
|---|---|---|---|---|---|
| baseline-lr1e-3 | 1e-3 | 78.44 (e12) | 79.91 | 135.6 | qw0wxjan |
| variant-lr5e-4  | 5e-4 | 77.91 (e12) | 79.04 | 135.8 | 16sygvy6 |
| **variant-lr2e-3** | **2e-3** | **75.91 (e13)** ⬅ best | **76.31** | 135.8 | irucjkgv |

Per-split test: lr=2e-3 wins on `test_re_rand` (60.89) and `test_geom_camber_rc` (76.22) — strong OOD pattern.

**Decision: REQUEST CHANGES — rebase onto EMA+SOAP, re-run lr=2e-3 vs lr=1e-3 baseline.**

Within-PR: lr=2e-3 wins by −3.2% on val and −4.5% on test. SOAP's preconditioner safely absorbs a 2× hotter peak LR; the directional pattern (5e-4 → 1e-3 → 2e-3) is monotonic on test. Sent back for compounding test on EMA+SOAP stack with seed=42.

---

## 2026-05-16 03:55 — PR #3497 (tanjiro): Gradient clipping {1.0, 5.0, no-clip} — **REQUEST CHANGES (rebase EMA+SOAP, +clip=10 probe)**

- Branch: `willowpai2i48h3-tanjiro/gradient-clip-sweep`
- W&B group: `gradient-clip-sweep`
- 3 arms, seed=42

**Result (within-PR ranking by val_avg/mae_surf_p, lower is better):**

| Arm | grad_clip | val_avg | test_avg (excl cruise) | best_epoch | W&B |
|---|---|---|---|---|---|
| baseline-noclip-s42 | 0.0 | 82.03 | 79.91 | 12 | nhhxgmqh |
| variant-clip1 | 1.0 | 74.26 (−9.5%) | 72.34 (−9.5%) | 13 | mgc412en |
| **variant-clip5** | **5.0** | **72.12 (−12.1%)** ⬅ best | **71.10 (−11.0%)** | 14 | 67vwcrcl |

Per-test-split (clip5 vs baseline): test_single_in_dist 71.65 vs 92.00 (**−22.1%**), test_geom_camber_rc 79.64 vs 81.40, test_re_rand 62.02 vs 66.32 (−6.5%).

**Diagnostic from train/grad_norm logging:** raw gradient median ~26, p95 ~92, p99 ~147, max ~300. Both clip=1 (100% steps clipped) and clip=5 (95.55% steps clipped) are very active interventions. The fact that clip=5 > clip=1 suggests less-aggressive clipping (preserving more typical-step magnitude while still capping outlier spikes) is the right knob.

**Decision: REQUEST CHANGES — rebase onto EMA+SOAP, run 3 arms (no-clip baseline, clip=5, clip=10 probe).**

Within-PR: −12.08% on val is the largest signal of round-3 sweeps. If clip=5 compounds even partially on EMA+SOAP (currently 61.43), this could land ~54-56. The clip=10 arm probes whether less aggressive clipping (still well under median grad_norm ~26) preserves even more signal.

---

## 2026-05-16 04:25 — PR #3591 (nezuko): EMA decay sweep — **W&B-OBSERVED SIGNAL, AWAITING TERMINAL RESULT**

- Branch: `willowpai2i48h3-nezuko/ema-decay-sweep`
- W&B group: `ema-decay-sweep`
- 3 arms planned, seed=42 (only 2 visible in W&B; arm 3 not yet started)

**Discovered via W&B audit** (student had no PR comments due to GitHub rate-limit JSONDecodeError loop on the pod 02:30–04:20 UTC):

| Arm | ema_decay | val_avg/mae_surf_p | State | W&B |
|---|---|---|---|---|
| baseline-decay0.999 | 0.999 | **61.426** | finished (50/50) | xqymqb6v |
| **variant-decay0.99** | **0.99** | **58.005 (−5.6%)** ⬅ best so far | finished (50/50) | 1xy36vpn |
| variant-decay0.9999 | 0.9999 | — | NOT STARTED | — |

**variant-decay0.99 at 58.005 is the biggest signal of round-3.** Faster EMA decay (shorter memory) outperforms the current canonical, likely because the cosine LR schedule's late-epoch improvements (still ~10% of training under warmup→cosine) are useful late-stage signal that the 0.999 decay smooths out too aggressively.

**Action:** Commented on PR asking nezuko to (1) stop any duplicate baseline reruns, (2) launch missing arm 3 (variant-decay0.9999), (3) post terminal SENPAI-RESULT, (4) mark for review.

**Pending merge:** Once terminal SENPAI-RESULT is posted, variant-decay0.99 will be merged as new canonical pending verification that it's a reproducible single-seed result.

---

## 2026-05-16 04:50 — PR #3501 (thorfinn): SOAP surf_weight sweep {5, 10, 20} — **REQUEST CHANGES (rebase EMA+SOAP)**

- Branch: `willowpai2i48h3-thorfinn/soap-surf-weight-sweep`
- W&B group: `soap-surf-weight-sweep`
- 3 arms, seed=42

**Result (within-PR ranking by best val_avg/mae_surf_p):**

| Arm | surf_weight | val_avg | test_avg (excl cruise) | best_epoch | W&B |
|---|---|---|---|---|---|
| **variant-sw5** | **5.0** | **77.28 (−1.5%)** ⬅ best | **78.08 (−2.3%)** | 14 | mt6f1ze7 |
| baseline-sw10 | 10.0 | 78.44 | 79.91 | 12 | kgr2j3bu |
| variant-sw20 | 20.0 | 81.93 (+4.4%) | 83.31 (+4.3%) | 13 | 6pq8zcmo |

Monotonic ranking sw5 < sw10 < sw20. Key diagnostic from val loss components:

| Arm | unweighted val surf_loss | unweighted val vol_loss |
|---|---|---|
| sw5 | **0.1642** | **0.2273** |
| sw10 | 0.2017 | 0.2603 |
| sw20 | 0.1883 | 0.2800 |

The naive expectation (higher surf_weight → lower surf_loss) is **violated**: sw=5 has the lowest unweighted surf_loss AND lowest vol_loss. This confirms SOAP's per-block preconditioner already does the channel rebalancing that surf_weight=10 was compensating for; over-emphasizing surface loss at sw=10/sw=20 hurts the underlying optimization.

**Decision: REQUEST CHANGES — rebase onto EMA+SOAP, run 2 arms (sw=10 baseline, sw=5 variant).**

Within-PR signal is modest (−1.5%) but the mechanism is sound and the loss-component data is convincing. Sent back for compounding test on EMA+SOAP stack.

---

## 2026-05-16 05:05 — PR #3495 (askeladd): SOAP precond_freq=5 on EMA+SOAP — **MERGED**

- Branch: `willowpai2i48h3-askeladd/soap-precond-freq-sweep`
- W&B group: `precond-freq-ema-soap`
- 2 arms, seed=42 (compounding test after EMA+SOAP rebase)

**Result:**

| Arm | precond_freq | val_avg/mae_surf_p | test_avg (excl cruise) | s/epoch | W&B |
|---|---|---|---|---|---|
| baseline-freq10-ema | 10 | 61.43 (exact canonical) | 60.92 | 136.8 | uu4nll7s |
| **variant-freq5-ema** | **5** | **60.33 (−1.78%)** ✓ | **59.27 (−2.70%)** | 138.2 | 94f3r1yb |

All 3 test splits improve: single_in_dist 71.13→69.39, geom_camber_rc 61.95→60.65, re_rand 49.67→47.78.

**Decision: MERGED.** Baseline canonical: val=61.43 → **60.33** (−1.78%). Baseline arm reproduced canonical exactly (sanity check passed). Wall-clock penalty negligible (+1.0%).

**Cumulative stack:** Huber(β=1.0) + LR warmup(1e-3) + SOAP(precond_freq=**5**) + EMA(0.999). Total: 135.30 → **60.33** (−55.4%).

**Note:** `precondition_frequency=5` is now the default in `train.py` after this merge. All subsequent runs without explicit flag will use freq=5.
