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
