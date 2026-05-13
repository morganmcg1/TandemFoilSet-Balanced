# SENPAI Research State

- **As of:** 2026-05-13 (updated cycle 19)
- **Round:** willow-pai2g-48h-r4 (advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none — controlled 24/48 h Charlie-vs-Willow logging ablation, hard cap `SENPAI_TIMEOUT_MINUTES=30`)

## Current baseline

**`val_avg/mae_surf_p = 97.9914`** — PR #1795 (decoupled LR surf_head, surf_head_lr=5e-3), merged 2026-05-13 05:30.
**Test 3-split mean: 99.5856** (slight regression on 3-split vs prior; 4-split test_avg=88.53 includes now-finite cruise).

## Improvement trajectory

| Cycle | PR | Change | val_avg/mae_surf_p | Δ |
|-------|-----|--------|-------------------|---|
| 2 | #1502 | BIVW (per-sample IVW) | 126.0751 | baseline |
| 3 | #1528 | BIVW + zero-init surf-head | 119.2987 | −5.37% |
| 5 | #1527 | Test NaN guard (eval only) | (val unchanged) | infra |
| 10 | #1558 | Huber surface loss δ=0.5 | 98.1642 | −17.72% |
| 18 | **#1795** | **Decoupled LR surf_head (5e-3)** | **97.9914** | **−0.18%** |

## Current research focus

**Cycle 18.** Four orthogonal mechanisms now merged (BIVW + surf-head + Huber + decoupled surf_head LR) giving 26.4% total improvement from original baseline. Three wall-clock-bound capacity/optimization failures now characterized:

- **Warmup + cosine LR** (PR #1497): +17.98%. 5 warmup epochs consume the most productive early steps; T_max=50 never decays in 14 epochs.
- **Wider MLP (ratio=4)** (PR #1498): +24.97%. ~19% slower per-epoch (152s vs 128s) → 12 epochs vs 14 → underfit.
- **Grad-clip + higher LR** (PR #1499): +1.45%. Huber already compresses grad norms 5×; clip is redundant.

Combined principle: **under the 30-min cap, baseline epoch 14 is still improving. Any change that costs ≥10% per-epoch loses ≥1 epoch and regresses unless it accelerates convergence proportionally.**

Three additional stack-on-Huber attempts also closed with consistent negatives:

- **Smaller Huber delta** (PR #1627): δ=0.5 is the local optimum — δ=0.2/0.3 both regress >15%, δ=1.0 also regressed.
- **Huber on volume** (PR #1650): all arms regressed +8.7% to +19.9%. Surface Huber + volume MSE is the correct recipe.
- **Per-channel BIVW** (PR #1580): +29.6% regression. Scalar BIVW's p-dominated coupling is load-bearing.
- **Frozen 1/var(p) sampler** (PR #1746): +272% regression. var(p) spans 8 OOM; unbounded inverse weights collapse effective dataset.

Plus additional closed experiments: surf_weight sweep (sw=10 already optimal), EMA weights (budget too short, 14-ep descent phase not plateau).

These collectively define a clear principle: **the Huber+BIVW+surf-head+decoupled-LR baseline is a tight local optimum**. New gains will require either (a) compositions that hit a different mechanism, (b) capacity unlocks via efficiency rather than parameter count, (c) data-side levers with bounded weight functions, or (d) convergence acceleration through per-group optimization.

**Active directions:**

1. **Decoupled LR extension** — merged at 5e-3 (−0.18%); trend not exhausted. Push to 7e-3/1e-2 with 2-epoch head warmup to tame late oscillation. #1949 thorfinn (WIP).
2. **Adaptive Huber δ** — self-tuning EMA of p75 per-batch residuals. Fixed δ=0.5 may be sub-optimal early (large residuals → too few in L1) and late (small residuals → too many in MSE). #1950 fern (WIP).
3. **SWA late-epoch averaging** — average last K checkpoints post-training, bypassing EMA's early-training contamination issue. #1951 askeladd (WIP).
4. **BF16 capacity unlock** — #1572 frieren (WIP, stale).
5. **Pressure-channel emphasis** — #1496 alphonse (WIP, stale).
6. **Per-channel Huber delta** — δ_p vs δ_ux/uy. #1922 nezuko (WIP).
7. **Encoder LR re-tune** — encoder LR was calibrated pre-Huber/pre-decoupled-head; re-sweep {3e-4, 7e-4} stacked on surf_head_lr=5e-3. #1974 edward (NEW).
8. **Re-curriculum via loss multiplier** — per-sample `w = 1 + α × |normalized(log_Re)|` applied as loss multiplier; avoids sampler no-op structural failure from #1868. #1978 tanjiro (NEW).

## Key insights

**Cycle 11 (PR #1580):** Per-channel BIVW failed (+29.6%). Scalar BIVW is *implicitly* a p-variance-driven Re-curriculum (p dominates pooled variance). Decoupling removes the beneficial coupling. **Lesson:** scalar BIVW coupling across channels is load-bearing for `mae_surf_p` — preserve in any reweighting work.

**Cycle 13 (PR #1650):** Huber on volume regressed all arms. Surface and volume play different roles: surface is the evaluated readout; volume is the encoder's *supervisory signal*. Scale information in volume MSE is more valuable than outlier robustness. **Lesson:** surface Huber + volume MSE is the correct recipe.

**Cycle 14 (PR #1627):** Huber delta sweep (δ=0.2, 0.3) both regress. At δ=0.5 only ~2% of residuals in the L1 regime; smaller deltas over-flatten mid-magnitude gradients that drive MAE. **Lesson:** δ=0.5 is a narrow sweet spot — do not re-sweep without changing other levers.

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 1496 | alphonse | pressure-channel-prioritized-loss | WIP | Huber default correction sent; use --huber_delta 0.5 |
| 1572 | frieren | bf16-mixed-precision | WIP | Huber default correction sent; add --huber_delta 0.5 |
| 1922 | nezuko | per-channel-huber-delta | WIP | δ_p=0.5, δ_ux/uy ∈ {1.0, 2.0}; tests if global δ over-flattens Ux/Uy distributions |
| 1949 | thorfinn | surf-head-lr-warmup | WIP | surf_head_lr ∈ {7e-3, 1e-2} + 2-ep head warmup; extends PR #1795 trend |
| 1950 | fern | adaptive-huber-delta | WIP | EMA of p75 per-batch residuals; self-tuning δ. Arms: p75 and p90 |
| 1951 | askeladd | swa-late-epoch | WIP | Avg last K checkpoints post-training; avoids EMA's early-training drag |
| 1974 | edward | encoder-lr-retune | WIP (NEW) | Re-tune encoder LR {3e-4, 7e-4} stacked on surf_head_lr=5e-3; encoder LR stale since pre-Huber |
| 1978 | tanjiro | re-loss-weight | WIP (NEW) | Per-sample loss multiplier 1+α×|norm(log(Re))|; loss-side Re-curriculum avoiding #1868 sampler no-op |

## Working hypotheses

1. **BIVW** — confirmed (PR #1502, −5.4%).
2. **BIVW + surf-head** — confirmed (PR #1528, −5.4% additional).
3. **Huber surface loss delta=0.5** — confirmed (PR #1558, **−17.7%**). The largest gain yet.
4. **Per-channel BIVW** — **rejected** (PR #1580, +29.6% regression). Scalar BIVW's p-dominated coupling is beneficial — do not break it.
5. **Grad-clip + higher LR on Huber base** — **rejected** (PR #1499, +1.45% regression). Huber already compresses grad norms 5×; clip is redundant.
6. **Huber on volume loss** — **rejected** (PR #1650, best +8.7% regression). Volume loss is the encoder's supervisory signal; Huber removes scale information the encoder needs. Surface Huber + volume MSE remains the correct recipe.
7. **Smaller Huber delta** — **rejected** (PR #1627). δ=0.3 (+15.6%) and δ=0.2 (+17.2%) both regress. δ=0.5 is a narrow local optimum.
8. **surf_weight tuning on Huber baseline** — **rejected** (PR #1720, all arms +7-21% regression). Optimum is at sw=10; hypothesis was wrong about Huber requiring higher surf_weight. Volume MSE starvation mechanism identified.
9. **Frozen p-variance stratified sampling** — **rejected** (PR #1746, +272% regression). Variance dynamic range is 8 OOM; 1/var(p) sampler collapses effective training set to a handful of low-Re samples. Conceptually sound but wrong functional form.
9a. **log(Re) quantile bucketing** — **rejected** (PR #1868, +8.4% regression). Quantile boundaries produce equal-count buckets; 1/count weights then ≈ uniform — structural no-op. Only adds ±2% perturbation to existing domain weights. Correct mechanism is loss-side multiplier: PR #1978.
9b. **Re-curriculum via per-sample loss multiplier** — testing (PR #1978 tanjiro). `w = 1 + α × |normalized(log_Re)|` applied as per-sample loss multiplier. Avoids sampler structural cancellation. Tests if Re-tail focus improves re_rand + geom_camber_rc.
10. **BF16/AMP** — testing (#1572); primarily for capacity headroom.
11. **Wider MLP (ratio=4)** — **rejected** (PR #1498, +24.97% regression). 19% slower per-epoch → 12 vs 14 epochs → underfit. Confirms wall-clock-bound principle.
11a. **Slice_num=128** — **rejected** (PR #1501, +19.30% regression). +37% per-epoch cost → 10 vs 14 epochs. Fourth wall-clock-bound capacity failure. Pareto frontier confirmed: depth=5/14ep is optimal; all capacity expansions on depth+slice axes lose.
11b. **Shallower depth (n_layers=4)** — **rejected** (PR #1881, +8.39% regression). −14% per-epoch cost gained 2 extra epochs (16 vs 14) but capacity loss from 1 fewer TransolverBlock dominated. Regression uniform across all 4 splits → pure underfitting. Depth=5/14ep is Pareto frontier — both perturbations on the depth axis confirm this.
11c. **n_head 4→8** — **rejected** (PR #1924, +18.84% regression). Wall-clock overhead +31% (175 s vs 133 s per epoch) → lost 3 epochs (11 vs 14). Per-epoch quality was better at epoch 11 (−9.3%) but budget loss dominated. Fifth wall-clock-bound failure. Pareto frontier confirmed for all capacity-axis perturbations.
11d. **Encoder LR re-tune** — testing (#1974 edward, NEW). Encoder LR stale at 5e-4 since pre-Huber era; sweep {3e-4, 7e-4} stacked on surf_head_lr=5e-3.
16. **Per-channel Huber delta** — testing (#1922 nezuko, NEW). δ_p=0.5, δ_ux/uy ∈ {1.0, 2.0}. Tests if global δ=0.5 is over-flattening Ux/Uy mid-magnitude gradients that drive velocity MAE.
12. **Warmup schedule** — **rejected** (PR #1497, +17.98% regression). Wall-clock-bound training (~14 epochs) makes warmup a liability — 5 warmup epochs consume the most productive early steps. The CosineAnnealingLR(T_max=50) baseline is effectively flat at lr=5e-4 for 14 epochs and wins. No instability observed in baseline; the hypothesis was wrong.
13. **Pressure-channel emphasis** — WIP (#1496); on Huber base.
14. **EMA model weights** — **rejected** (PR #1808, +7.8-16.2% regression). 14-epoch budget too short; model in descent phase, not noisy-plateau. EMA window contaminates evaluation with early-training weights. Budget mismatch, not hypothesis failure.
15. **Decoupled LR for surf_head vs encoder** — **confirmed** (PR #1795, −0.18%). surf_head_lr=5e-3 (10×encoder) is the winning arm; monotonic improvement trend across 1e-3→3e-3→5e-3 not yet exhausted. Extending: PR #1949 thorfinn tests 7e-3/1e-2 with warmup.
16. **Adaptive Huber δ** — testing (#1950 fern, NEW). EMA of p75 per-batch residuals; self-tuning warm start at δ=0.5.
17. **SWA late-epoch averaging** — testing (#1951 askeladd, NEW). Average last K checkpoints to avoid EMA's early-training drag.

## Closed / rejected hypotheses

- **PR #1503** (standalone surf-head, no BIVW) — 6.2% worse.
- **PR #1500** (n_hidden=256 at FP32) — budget failure; BF16 unlock in progress (#1572).
- **PR #1580** (per-channel BIVW) — 29.6% regression. Scalar BIVW implicitly p-dominated; decoupling removed the Re-curriculum.
- **PR #1499** (grad-clip + higher LR on Huber base) — 1.45% val regression. Huber already compresses grad norms; clip is redundant. Test marginal (3-split: −1.1%). Closed.
- **PR #1650** (Huber on volume loss) — all arms regressed (+8.7% to +19.9%). Volume MSE is the encoder's supervisory signal; Huber removes needed scale information. Principle: surface Huber + volume MSE is the correct recipe. Closed.
- **PR #1627** (Huber delta sweep δ=0.2, 0.3) — both arms regressed (+15.6%, +17.2%). With δ=1.0 also worse (+1.3%), δ=0.5 is a narrow local optimum. Principle: do not re-sweep delta without first changing other levers.
- **PR #1497** (5-epoch linear warmup + CosineAnnealingLR) — +17.98% regression. With T_max=50 but only ~14 epochs run, warmup costs productive steps without delivering cosine tail benefit. Principle: under 30-min cap, T_max must ≤ epochs_actually_run for scheduling to matter.
- **PR #1746** (Frozen p-variance stratified sampler) — +272% regression. var(p) dynamic range is 8 OOM; 1/var(p) sampler collapses effective training set to a few low-Re samples. Principle: inverse-variance sampling weights on this corpus must be tempered, log-compressed, or quantile-bucketed.
- **PR #1498** (Wider MLP, mlp_ratio=2→4) — +24.97% val regression. Third wall-clock-bound failure. Closed.
- **PR #1501** (slice_num=128) — +19.30% val regression. Fourth wall-clock-bound failure. Pareto frontier characterized: depth=5/width=128/slice=64/~14ep. Closed.
- **PR #1881** (n_layers=4) — +8.39% val regression. Both depth perturbations confirm depth=5/14ep is Pareto. Closed.
- **PR #1720** (surf_weight {5, 15, 30}) — all arms +7-21% regression. sw=10 already optimal; higher sw starves volume MSE. Closed.
- **PR #1808** (EMA weights) — +7.8-16.2% regression. Budget too short for EMA; model in descent phase. Closed. Follow-up: SWA late-epoch (#1951).
- **PR #1868** (log(Re) quantile bucketing) — +8.4% regression. Structural no-op: quantile bounds → equal counts → 1/count weights ≈ uniform. Max/min ratio 1.02×; only ±2% perturbation of existing domain weights. Mechanism itself is broken. Follow-up: loss-side multiplier (#1978).
- **PR #1924** (n_head=8) — +18.84% regression. +31% per-epoch wall-clock → 11 vs 14 epochs. Fifth wall-clock-bound failure. Pareto frontier fully characterized.

## Potential next directions

- **Capacity scaling (n_hidden=256) + BF16 + Huber** — full composition when #1572 merges
- **Per-channel Huber delta** (different delta per channel for p vs Ux/Uy) — channels have different residual distributions; one global δ may be suboptimal
- **Adaptive Huber** — set delta from a moving p95 quantile of residuals rather than fixed
- **EMA per-channel variance** (tanjiro suggestion) — reduce within-batch noise for any per-channel weighting future attempt
- **Per-node loss weighting by node curvature/geometry** — surface nodes near sharp leading/trailing edges may dominate MAE; weight by local curvature or distance-to-feature
- **Test-time geometry augmentation** — multiple rotations/reflections averaged
- **Heavier surface decoder** — currently single MLP; deeper surface-specific decoder atop encoder
- **Re-examine encoder bottleneck** — if capacity scaling stalls, encoder representation may be the ceiling; consider attention head/dim sweep
- **Cross-channel attention in surf_head** — current surf_head is channel-independent MLP; add cross-channel mixing

## Known issues

- ~~**test_avg/mae_surf_p = NaN**~~ — Fixed via PR #1527 (merged). 3-split test mean now reportable.
- **Rate limit impact**: GitHub GraphQL rate limits caused multiple student pods to see "No assigned PRs" and idle for 30–60 min between training arms. Pods recover within ~5-15 min.
- **Slice-attention VRAM**: n_hidden=256 needs BF16 (#1572) to be fairly evaluated.
- **Multiple WIPs need rebase**: #1496–#1499, #1501 all need rebase onto advisor branch post-Huber-merge (baseline notes sent to each).
- **Huber default trap**: PR #1558 left dataclass `huber_delta: float = 1.0` but the winning 98.16 baseline used `--huber_delta 0.5` explicitly. The δ=1.0 arm was only −1.3% vs prior, not −17.7%. ALL rebased PRs must use `--huber_delta 0.5` explicit, NOT rely on defaults. Correction sent to alphonse/askeladd/edward/frieren/nezuko on 2026-05-13 ~00:14 UTC.
