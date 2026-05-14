# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-14 07:50
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** None received — controlled 24/48h Charlie-vs-Willow logging ablation.


## Current baseline (11th shift)

**PR #2801 (Pinball τ=0.55 for pressure channel)** merged 2026-05-14 07:15:
- **`val_avg/mae_surf_p` = 43.0923** (best seed 1 `xkaghm9f`)
- **`test_avg/mae_surf_p` = 37.1943**
- Per-split test: single_in_dist=43.00, geom_camber_rc=49.86, geom_camber_cruise=21.22, re_rand=34.70
- Two-seed mean: val=43.684, test=37.272 (test extremely tight, ±0.08)
- **New merge bar: val < 43.09, test < 37.19, all four test splits finite**

## Baseline progression

| Merge | Time | val | test | Δ vs prior |
|---|---|---:|---:|---:|
| PR #1504 (mask-aware) | 2026-05-12 21:52 | 119.450 | 109.669 | round-1 start |
| PR #1505 (Huber β=0.5 surf) | 2026-05-13 00:00 | 113.794 | 101.782 | −4.7% / −7.2% |
| PR #1715 (bf16 AMP) | 2026-05-13 02:00 | 89.597 | 79.907 | −21.3% / −21.5% |
| PR #1810 (torch.compile dynamic=True) | 2026-05-13 05:15 | 67.831 | 59.784 | −24.3% / −25.2% |
| PR #1910 (vol-Huber β=0.5) | 2026-05-13 07:30 | 65.469 | 57.837 | −3.5% / −3.3% |
| PR #1692 (grad_clip max_norm=1.0) | 2026-05-13 12:00 | 60.093 | 53.370 | −8.2% / −7.7% |
| PR #1589 (AdamW betas 0.9, 0.95) | 2026-05-13 16:03 | 59.970 | 52.363 | −0.2% / −1.9% |
| PR #2017 (weight_decay 1e-4 → 2e-4) | 2026-05-13 16:10 | 58.883 | 51.078 | −1.8% / −2.4% |
| PR #2516 (Lion optimizer) | 2026-05-13 20:05 | 50.193 | 43.501 | −14.8% / −14.8% |
| PR #2562 (Lion lr=7.5e-5) | 2026-05-13 22:30 | 45.433 | 39.509 | −9.5% / −9.2% |
| **PR #2801 (Pinball τ=0.55 pressure)** | **2026-05-14 07:15** | **43.092** | **37.194** | **−5.1% / −5.9%** |

**Cumulative: −63.9% val, −66.1% test from round-1 start.** Still compute-bound (best=last on all 11 merges).

## Current research focus (round 9)

**Building on the pinball τ=0.55 win (PR #2801).** The asymmetric pressure loss win revealed two productive axes: (1) the τ-scale can be pushed higher, and (2) the velocity channels may also benefit from asymmetric loss. Simultaneously, fern's pre-implementation discovery opened a new structural axis (orthogonal init bug fix).

Current focus: **loss geometry exploration** (τ-scan, channel coverage) and **architectural correctness** (latent bug fix).

### Round-9 in-flight (newly assigned)

1. **Pinball τ=0.60 pressure (alphonse #2853)**: scale up asymmetry — test whether stronger directional bias helps further
2. **Orthogonal init restore for in_project_slice (frieren #2854)**: fix the latent bug where trunc_normal_ clobbers the intended orthogonal init — zero compute cost, pure architectural correctness
3. **Pinball τ=0.55 on Ux/Uy velocity channels (tanjiro #2855)**: extend the winning pressure loss to all 3 channels — test velocity field bias hypothesis

### Round-8 WIPs still in flight

4. **Sobolev loss on pressure gradient (thorfinn #2811)**: physics-aware loss term (early val=60 mid-training, trending poor)
5. **LayerScale (nezuko #2812)**: CaiT-style residual gating init=1e-4 (early val=52 mid-training, trending poor)
6. **FiLM-style Re-conditioning (edward #2816)**: per-block Re conditioning (early val=83, very early/inconclusive)
7. **σ-scan for Linear init (fern #2817)**: σ=0.01/0.05 scan after discovering trunc_normal_ σ=0.02 already in baseline

## Round 1 portfolio (current)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #1504–#2017 | various | 8 stacked improvements | **MERGED** (baseline history above) |
| #2516 | edward | Lion optimizer | **MERGED** 2026-05-13 20:05 (val=50.19) |
| #2562 | tanjiro | Lion lr=7.5e-5 | **MERGED** 2026-05-13 22:30 (val=45.43) — 10th shift |
| **#2801** | **askeladd** | **Pinball loss τ=0.55 for pressure** | **MERGED** 2026-05-14 07:15 (val=43.09) — **11th shift** |
| #2561 | edward | Lion beta2=0.95 | **CLOSED** (+14.8%) |
| #2520 | thorfinn | n_head 4→8 | **CLOSED** (+24%) |
| #2504 | frieren | QK-RMSNorm | **CLOSED** (+14%) |
| #2628 | tanjiro | Lion lr=1e-4 | **CLOSED** (+1.9% overshoot) |
| #2501 | askeladd | β_p=0.625 | **CLOSED** (+6.8%) |
| #2565 | fern | max_norm=0.5 | **CLOSED** (stale) |
| #2564 | nezuko | GC (stale) | **CLOSED** (stale) |
| #2505 | alphonse | SiLU activation | **CLOSED** (+18.9%) |
| #2633 | edward | Lion β1=0.95 | **CLOSED** (+4.83pt; var-vs-mean #2) |
| #2631 | thorfinn | Lion warmup 5ep | **CLOSED** (+4.44%; var-vs-mean #3) |
| #2629 | frieren | Lion wd=3e-3 | **CLOSED** (+1.68pt wd axis) |
| #2700 | edward | Lion β1=0.85 | **CLOSED** (+8.3%; β1 fully bracketed) |
| #2693 | tanjiro | CosineAnnealingWarmRestarts | **CLOSED** (+17.7%; schedule retired) |
| #2713 | frieren | Lion β2=0.999 | **CLOSED** (+5.69%; β2 fully bracketed) |
| #2694 | askeladd | Charbonnier loss | **CLOSED** (+1.3% loss-shape saturated) |
| #2726 | alphonse | Lookahead(Lion) | **CLOSED** (+6.77pt; var-vs-mean #6) |
| #2743 | askeladd | p_weight=2.0 | **CLOSED** (+5.22pt; Lion sign discards magnitude) |
| #2751 | frieren | Re-jitter σ=0.05 | **CLOSED** (+12.4pt; conditioning inconsistency) |
| #2752 | tanjiro | Gradient accumulation 2× | **CLOSED** (+8.87pt; var-vs-mean #7) |
| #2712 | thorfinn | SWA avg epochs 26-35 | **CLOSED** (variance characterized; mean misses) |
| #2753 | nezuko | Per-layer LR decay | **CLOSED** (stale; harness issue) |
| #2762 | edward | GC on Lion | **CLOSED** (+6.4%; sign-incompatible) |
| #2763 | fern | max_norm=0.5 v2 | **CLOSED** (stale; harness issue) |
| #2800 | alphonse | RMSNorm | **CLOSED** 2026-05-14 07:40 (+13.6%; mean-centering load-bearing; var-vs-mean #8) |
| #2803 | frieren | Param-group wd | **CLOSED** 2026-05-14 07:40 (+7.7%; Lion wd interaction; axis retired) |
| #2805 | tanjiro | LN γ-init=0.5 | **CLOSED** 2026-05-14 07:40 (+35.4%; var-vs-mean #9 — γ never recovered) |
| **#2811** | **thorfinn** | **Sobolev loss on ∇p** | **WIP 2026-05-14 05:55** |
| **#2812** | **nezuko** | **LayerScale (init=1e-4)** | **WIP 2026-05-14 05:55** |
| **#2816** | **edward** | **FiLM-style Re-conditioning** | **WIP 2026-05-14 06:30** |
| **#2817** | **fern** | **σ-scan (σ=0.01/0.05) for Linear init** | **WIP 2026-05-14 06:30** |
| **#2853** | **alphonse** | **Pinball τ=0.60 pressure** | **WIP NEW 2026-05-14 07:45** |
| **#2854** | **frieren** | **Orthogonal init restore (in_project_slice)** | **WIP NEW 2026-05-14 07:45** |
| **#2855** | **tanjiro** | **Pinball τ=0.55 Ux/Uy velocity channels** | **WIP NEW 2026-05-14 07:45** |

**Merged:** 11 | **Closed:** 54 | **WIP:** 7 | **Idle:** 0

## Key meta-findings from round 1

1. **Compute is permanently binding** — best=last at every merge. The 30-min cap has been the dominant constraint since bf16 (PR #1715).
2. **Variance-vs-mean decoupling confirmed (9 instances)** — β1=0.85/0.95, β2=0.999, warmup, Lookahead, grad-accum, RMSNorm, grad-accum, LN γ-init=0.5 all show variance reduction with mean regression. Pattern: any mechanism reducing optimizer step frequency, representation capacity, or initial activation scale trades mean improvement for variance reduction. At 35-ep compute-bound cap, the mean cost is never recovered.
3. **Lion β1 axis FULLY BRACKETED** — β1=0.85 (+8.3%) and β1=0.95 (+4.83pt) both regress; β1=0.90 confirmed optimal.
4. **Lion β2 axis FULLY BRACKETED** — β2=0.95 (+14.8%) and β2=0.999 (+5.69%) both regress; β2=0.99 confirmed optimal.
5. **Schedule-shape axis FULLY RETIRED** — warmup, warm restarts, all variants lose to cosine T_max=50 with implicit residual.
6. **Per-channel amplitude axis RETIRED under Lion** — p_weight=2.0 failed; per-channel β closed; Lion's sign() discards gradient magnitude, so amplitude-based loss scaling has no effect on capacity allocation.
7. **Conditioning-variable jitter axis RETIRED** — jittering log(Re) creates supervised inconsistency; valid augmentation requires conditional invariance in outputs.
8. **GC + sign() is sign-incompatible** — Gradient Centralization's row-mean subtraction forcibly inverts ~half of coordinate update directions each step under Lion. GC needs magnitude-based optimizers.
9. **Pinball τ=0.55 WIN (PR #2801)** — directional asymmetric loss for under-prediction bias works under Lion. Mechanism: τ=0.55 penalizes residuals with y>pred 10% more; OOD splits (re_rand −8.4%, cruise −11.6%) benefit most. Lion's sign() preserves the directional signal from pinball (unlike amplitude-scaling which gets discarded). This opens the τ-axis and the channel-coverage axis.
10. **trunc_normal_(std=0.02) already in baseline** — fern discovered Transolver._init_weights already applies BERT-style init. Latent bug: in_project_slice orthogonal init clobbered by subsequent apply().

## Currently retired axes

- **Scalar capacity** (n_hidden, n_layers, slice_num, mlp_ratio) — failed at all 3 baselines
- **Schedule shape** — T_max, eta_min, warmup-then-flat, warm restarts — all retired
- **Noise injection** (dropout, DropPath) — regularization stack already saturated
- **Lion betas** — β1 fully bracketed at 0.90; β2 fully bracketed at 0.99
- **Lion LR** — 1e-4 overshoots; 7.5e-5 sweet spot
- **Per-channel Huber β** — both directions failed; global β=0.5 robust
- **n_head=8** — capacity loss + overhead
- **QK-RMSNorm** — Q/K magnitudes carry physics-discriminative signal
- **surf_weight** — fully bracketed (5/10/20); 10 optimal
- **EMA weights** — EMA-lag on cooling cosine cancels smoothing
- **SiLU activation** — Lion sign neutralizes SiLU advantage; GELU selective gating useful in slice-attention
- **Charbonnier loss** — loss-shape axis saturated; Huber β=0.5 robust under Lion+clip
- **CosineAnnealingWarmRestarts** — cycle restart cost irrecoverable at 35-ep cap
- **Lookahead** — momentum state continuity destroyed at each sync-back
- **Gradient accumulation 2×** — step-count halving catastrophic at 35-ep cap
- **Per-channel amplitude weighting** — Lion sign() discards magnitude; amplitude weights have no effect on capacity allocation
- **Conditioning-variable jitter (log(Re))** — creates supervised inconsistency; degraded all splits
- **SWA (last-10 ckpt averaging)** — variance reduction works (4-10× tighter std) but mean still misses bar at current baseline; pairs best with a stronger base run, revisit after future merges
- **Gradient Centralization on Lion** — sign-incompatible: GC's row-mean subtraction forcibly inverts coordinate update directions when combined with sign() optimizer; GC needs Adam-style magnitude updates

## Potential next research directions

### Immediate (round 9 in flight)

1. **Pinball τ=0.60 pressure (alphonse #2853)** — scan τ axis upward; if current τ=0.55 under-corrected the bias
2. **Orthogonal init restore for in_project_slice (frieren #2854)** — zero-cost bug fix; potentially high impact on slice mechanism quality
3. **Pinball τ=0.55 Ux/Uy velocity (tanjiro #2855)** — if velocity fields also have under-prediction bias, extending coverage may win
4. **Sobolev loss on ∇p (thorfinn #2811)** — still running; early val high but may converge
5. **LayerScale (nezuko #2812)** — still running; early val uncertain
6. **FiLM-style Re-conditioning (edward #2816)** — still running
7. **σ-scan for Linear init (fern #2817)** — σ=0.01/0.05

### Medium-term (if round-9 clears)

8. **Pinball τ=0.65 pressure** — if τ=0.60 wins, continue scan
9. **Per-sample Re embedding (MLP-encoded)** — dedicated re_rand OOD lever; learn Re → embedding rather than raw log(Re) concat
10. **Y-flip augmentation** — flow-symmetric BCs admit clean mirror augmentation (conditionally invariant → valid)
11. **τ per-split tuning** — re_rand improved most; try separate τ for different geometry classes
12. **Pinball τ < 0.5 for single_in_dist** — probe whether in-distribution has over-prediction bias (single_in_dist barely improved at τ=0.55)
13. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens
14. **Separate Re embedding network** — richer Re representation for the OOD split
