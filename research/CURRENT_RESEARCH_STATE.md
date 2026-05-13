# SENPAI Research State

- **As of:** 2026-05-13 (updated cycle 14)
- **Round:** willow-pai2g-48h-r4 (advisor branch `icml-appendix-willow-pai2g-48h-r4`)
- **Most recent human-team direction:** (none — controlled 24/48 h Charlie-vs-Willow logging ablation, hard cap `SENPAI_TIMEOUT_MINUTES=30`)

## Current baseline

**`val_avg/mae_surf_p = 98.1642`** — PR #1558 (Huber surface loss delta=0.5), merged 2026-05-12 22:00.
**Test 3-split mean: 98.7537** (cruise still NaN, but all other splits now reportable via #1527 guard).

## Improvement trajectory

| Cycle | PR | Change | val_avg/mae_surf_p | Δ |
|-------|-----|--------|-------------------|---|
| 2 | #1502 | BIVW (per-sample IVW) | 126.0751 | baseline |
| 3 | #1528 | BIVW + zero-init surf-head | 119.2987 | −5.37% |
| 5 | #1527 | Test NaN guard (eval only) | (val unchanged) | infra |
| 10 | **#1558** | **Huber surface loss δ=0.5** | **98.1642** | **−17.72%** |

## Current research focus

**Cycle 14.** Three orthogonal mechanisms are merged (BIVW + surf-head + Huber) giving 26% total improvement. Four "stack-on-Huber" attempts have now closed cleanly with consistent negative results:

- **Smaller Huber delta** (PR #1627): δ=0.5 is the local optimum — δ=0.2/0.3 both regress >15%, δ=1.0 also regressed.
- **Huber on volume** (PR #1650): all arms regressed +8.7% to +19.9%. Surface Huber + volume MSE is the correct recipe.
- **Per-channel BIVW** (PR #1580): +29.6% regression. Scalar BIVW's p-dominated coupling is load-bearing.
- **Grad-clip + higher LR** (PR #1499): +1.45% regression. Huber already compresses grad norms 5×; redundant.

These four closures collectively define a clear principle: **the Huber+BIVW+surf-head baseline is a tight local optimum**. New gains will require either (a) compositions that hit a different mechanism, (b) capacity unlocks, or (c) data-side levers.

**Active directions:**

1. **Capacity expansions rebased onto Huber** — slice_num=128 (#1501 nezuko), wider MLP (#1498 edward), BF16 (#1572 frieren).
2. **LR schedule on Huber base** — warmup + cosine on new optimum (#1497 askeladd, rebasing).
3. **Pressure-channel emphasis** — still recovering from rate-limit stall (#1496 alphonse).
4. **surf_weight retuning** — Huber lowered surface loss magnitude; the surf:vol ratio may have shifted. #1720 fern testing {5, 15, 30}.
5. **Frozen p-variance stratified sampling** — makes BIVW's implicit Re-curriculum explicit at data-loader level. #1746 tanjiro testing.
6. **Decoupled LR for surf_head vs encoder** — surf_head is 0.026M params with zero-init last layer; may benefit from higher LR than the encoder. #1765 thorfinn (NEW).

## Key insights

**Cycle 11 (PR #1580):** Per-channel BIVW failed (+29.6%). Scalar BIVW is *implicitly* a p-variance-driven Re-curriculum (p dominates pooled variance). Decoupling removes the beneficial coupling. **Lesson:** scalar BIVW coupling across channels is load-bearing for `mae_surf_p` — preserve in any reweighting work.

**Cycle 13 (PR #1650):** Huber on volume regressed all arms. Surface and volume play different roles: surface is the evaluated readout; volume is the encoder's *supervisory signal*. Scale information in volume MSE is more valuable than outlier robustness. **Lesson:** surface Huber + volume MSE is the correct recipe.

**Cycle 14 (PR #1627):** Huber delta sweep (δ=0.2, 0.3) both regress. At δ=0.5 only ~2% of residuals in the L1 regime; smaller deltas over-flatten mid-magnitude gradients that drive MAE. **Lesson:** δ=0.5 is a narrow sweet spot — do not re-sweep without changing other levers.

## Live PRs

| # | Student | Slug | Status | Notes |
|---|---------|------|--------|-------|
| 1496 | alphonse | pressure-channel-prioritized-loss | WIP | Huber default correction sent; use --huber_delta 0.5 |
| 1497 | askeladd | warmup-cosine-lr | WIP | Huber default correction sent |
| 1498 | edward | wider-mlp-ratio (2 to 4) | WIP | Huber default correction sent |
| 1501 | nezuko | more-slices (64 to 128) | WIP | Confirmed huber_delta=0.5 explicit; running |
| 1572 | frieren | bf16-mixed-precision | WIP | Huber default correction sent; add --huber_delta 0.5 |
| 1720 | fern | surf-weight-tuning-on-huber | WIP | surf_weight ∈ {5, 15, 30} on Huber baseline |
| 1746 | tanjiro | frozen-p-variance-stratified-sampling | WIP | Pre-compute p-var weights over corpus; WeightedRandomSampler |
| 1795 | thorfinn | decoupled-lr-surf-head | WIP | surf_head_lr ∈ {1e-3, 3e-3, 5e-3}; encoder lr fixed at 5e-4 |
| 1808 | askeladd | ema-model-weights | WIP (NEW) | EMA shadow weights for eval; decay ∈ {0.999, 0.995} |

## Working hypotheses

1. **BIVW** — confirmed (PR #1502, −5.4%).
2. **BIVW + surf-head** — confirmed (PR #1528, −5.4% additional).
3. **Huber surface loss delta=0.5** — confirmed (PR #1558, **−17.7%**). The largest gain yet.
4. **Per-channel BIVW** — **rejected** (PR #1580, +29.6% regression). Scalar BIVW's p-dominated coupling is beneficial — do not break it.
5. **Grad-clip + higher LR on Huber base** — **rejected** (PR #1499, +1.45% regression). Huber already compresses grad norms 5×; clip is redundant.
6. **Huber on volume loss** — **rejected** (PR #1650, best +8.7% regression). Volume loss is the encoder's supervisory signal; Huber removes scale information the encoder needs. Surface Huber + volume MSE remains the correct recipe.
7. **Smaller Huber delta** — **rejected** (PR #1627). δ=0.3 (+15.6%) and δ=0.2 (+17.2%) both regress. δ=0.5 is a narrow local optimum.
8. **surf_weight tuning on Huber baseline** — testing (PR #1720 fern). surf_weight ∈ {5, 15, 30}.
9. **Frozen p-variance stratified sampling** — testing (PR #1746 tanjiro). Makes BIVW's implicit Re-curriculum explicit at data-loader level; removes within-batch estimation noise from batch-of-4 estimator.
10. **BF16/AMP** — testing (#1572); primarily for capacity headroom.
11. **Capacity (MLP width, slices)** — WIP (#1498, #1501); on Huber base with explicit huber_delta=0.5.
12. **Warmup schedule** — **rejected** (PR #1497, +17.98% regression). Wall-clock-bound training (~14 epochs) makes warmup a liability — 5 warmup epochs consume the most productive early steps. The CosineAnnealingLR(T_max=50) baseline is effectively flat at lr=5e-4 for 14 epochs and wins. No instability observed in baseline; the hypothesis was wrong.
13. **Pressure-channel emphasis** — WIP (#1496); on Huber base.
14. **EMA model weights** — testing (#1808 askeladd). EMA shadow copies evaluated at inference; decay ∈ {0.999, 0.995}. Orthogonal to all training changes — only affects which checkpoint is evaluated.
14. **Decoupled LR for surf_head vs encoder** — testing (PR #1765 thorfinn). surf_head_lr ∈ {2e-4, 1e-3, 3e-3}; encoder lr fixed at 5e-4. Hypothesis: zero-init surf_head head may benefit from higher LR than the encoder.

## Closed / rejected hypotheses

- **PR #1503** (standalone surf-head, no BIVW) — 6.2% worse.
- **PR #1500** (n_hidden=256 at FP32) — budget failure; BF16 unlock in progress (#1572).
- **PR #1580** (per-channel BIVW) — 29.6% regression. Scalar BIVW implicitly p-dominated; decoupling removed the Re-curriculum.
- **PR #1499** (grad-clip + higher LR on Huber base) — 1.45% val regression. Huber already compresses grad norms; clip is redundant. Test marginal (3-split: −1.1%). Closed.
- **PR #1650** (Huber on volume loss) — all arms regressed (+8.7% to +19.9%). Volume MSE is the encoder's supervisory signal; Huber removes needed scale information. Principle: surface Huber + volume MSE is the correct recipe. Closed.
- **PR #1627** (Huber delta sweep δ=0.2, 0.3) — both arms regressed (+15.6%, +17.2%). With δ=1.0 also worse (+1.3%), δ=0.5 is a narrow local optimum. Principle: do not re-sweep delta without first changing other levers.
- **PR #1497** (5-epoch linear warmup + CosineAnnealingLR) — +17.98% regression. With T_max=50 but only ~14 epochs run, warmup costs productive steps without delivering cosine tail benefit. Principle: under 30-min cap, T_max must ≤ epochs_actually_run for scheduling to matter.

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
