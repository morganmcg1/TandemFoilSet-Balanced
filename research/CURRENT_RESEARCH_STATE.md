# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-14 13:35
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** None received — controlled 24/48h Charlie-vs-Willow logging ablation.


## Current baseline (13th shift)

**PR #2882 (trunc_normal_ init std=0.07)** merged 2026-05-14 12:15:
- **`val_avg/mae_surf_p`** = 36.5754 (σ=0.07, W&B `gj8qijiv`, single seed)
- **`test_avg/mae_surf_p`** = 30.6438 (σ=0.07, W&B `gj8qijiv`, single seed)
- Per-split test surf_p: single_in_dist=35.87, geom_camber_rc=43.28, geom_camber_cruise=16.30, re_rand=27.12
- Default init_std=0.07 (train.py updated on advisor branch)
- **New merge bar: val < 36.58, test < 30.64, all four test splits finite (single-seed basis; directional)**

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
| PR #2801 (Pinball τ=0.55 pressure) | 2026-05-14 07:15 | 43.092 | 37.194 | −5.1% / −5.9% |
| PR #2817 (σ=0.05 init) | 2026-05-14 09:21 | 40.820 | 35.247 | −5.3% / −5.4% |
| **PR #2882 (σ=0.07 init)** | **2026-05-14 12:15** | **36.575** | **30.644** | **−10.4% / −13.1% ← LARGEST** |

**Cumulative: −69.4% val, −72.1% test from round-1 start.** Still compute-bound (best=last on all 13 merges).

## Current research focus (rounds 10–11)

**Three major wins in rapid succession: σ=0.05, σ=0.07 (init-scale axis)**. σ=0.07 alone gave the largest single-PR improvement in the launch (−13.1% test). This confirms the init-scale axis is strongly load-bearing under compute-bound Lion, with the optimizer behaving differently in small vs large-norm initialization regimes.

**Mechanism insight (13th shift)**: the "start near convergence neighbourhood" mental model was wrong. The correct story: larger σ pushes Lion into a different, higher-L2 basin that generalizes better within 35 epochs. Small σ traps the optimizer in a small-norm regime where it never fully escapes within the compute budget.

**Current focus**: 
1. Pin down the σ-axis peak (tanjiro #2908: σ=0.06/0.09 interior scan)
2. Test whether other axes compound with σ=0.07 (in-flight PRs using σ=0.05 or σ=0.02 need rebase/follow-up)

**Context for in-flight experiments**: PRs #2865, #2866, #2886, #2895, #2897, #2902 were assigned on σ=0.05 or σ=0.02 baseline. Their deltas should be interpreted relative to their original bar. If any beat their original bar, follow-up tests on σ=0.07 baseline are warranted.

### Round-10/11 in-flight (full grid)

Init-scale axis:
1. **σ interior scan: σ=0.06/0.09 (tanjiro #2908)** — NEW, assigned 2026-05-14 12:20; pins down σ-axis peak around σ=0.07 winner.

Physics loss axis:
2. **Pressure-Poisson auxiliary loss (askeladd #2909)** — NEW, assigned 2026-05-14 12:20; ∇²p = f(∇u), λ=0.01; targets geom_camber_rc.
   - #2866 divfree (nezuko): CLOSED 2026-05-14 13:20. λ calibration off 3 orders of magnitude (|∇·u| observed 3-19 vs assumed 1e-3); full λ range tested; mechanism alive (div drops 20%) but doesn't help surf_p. Axis cleanly exhausted.

Conditioning / OOD axis:
4. **γ-only FiLM-AoA (thorfinn #2886)** — WIP, targets camber_rc.
5. **γ-only FiLM-Re (edward #2865)** — rebased (CLEAN); waiting for re-run on σ=0.07 baseline.

Data augmentation axis:
6. **Y-flip augmentation (fern #2895)** — WIP.

Regularization axis:
7. **Weight-decay scan (alphonse #2897)** — SENT BACK 2026-05-14 13:35. σ=0.05 results: wd=1e-3 beats OLD bar (−8.0%/−9.6%, every test split improves, OOD strongest) but misses 13th-shift bar by 2-4%. Re-running wd=1e-3 + wd=7e-4 on σ=0.07 baseline to test compounding.
8. **Stochastic depth DropPath (nezuko #2926)** — NEW, assigned 2026-05-14 13:25; depth-scaled DropPath rates 0.1 (s1) and 0.2 (s2); block-level regularization complementary to init-scale and weight-decay; σ=0.07 init; 5-layer ramp 0.0→drop_path across blocks.

Architectural axis:
9. **SwiGLU FFN (frieren #2902)** — WIP.

Closed (previously active):
- **Pinball τ=0.60 pressure (alphonse #2853)** — CLOSED: τ-axis bracketed
- **Divergence-free auxiliary loss (nezuko #2866)** — CLOSED 2026-05-14 13:20: λ calibration 3 OOM off; all λ range tested; volume ∇·u constraint doesn't help surf_p; axis exhausted.

Architectural / capacity axis:
3. **SwiGLU FFN (frieren #2902)** — NEW, assigned 2026-05-14 11:55; gated FFN replacing GELU+Linear; +25% params; tests whether FFN is saturation point. Uses `--init_std 0.05`.
4. **γ-only FiLM-Re (edward #2865)** — CONFLICTING merge state; sent back for rebase 2026-05-14 10:33. Terminal results posted: mean val=41.13 (MISSES new bar by 0.31) but mean test=34.83 BEATS new bar by 1.19%. Strong re_rand −7.2% and cruise −9.6%; single_in_dist regresses +6.3%. Compound with σ=0.05 is the next test.

Conditioning / OOD axis (NEW):
5. **γ-only FiLM-AoA (thorfinn #2886)** — NEW, assigned 2026-05-14 10:45; mirrors edward's FiLM-Re pattern but conditions on AoA; targets geom_camber_rc (47.72, hardest split). Uses `--init_std 0.05` (matches current baseline).

Init-scale axis:
6. **σ-scan continuation: σ=0.07 and σ=0.10 (tanjiro #2882)** — WIP, runs ON the new σ=0.05 baseline with `--init_std` arg; directly extends winning axis.

Input-encoding axis:
7. **Re-Fourier features at input (askeladd #2863)** — WIP

Data augmentation axis (NEW #2895):
8. **Y-flip augmentation (fern #2895)** — NEW, assigned 2026-05-14 11:05; flip (y_node, AoA, Uy) with p=0.5 per batch; 2× effective data for free. Uses `--init_std 0.05`.

Regularization axis (NEW #2897):
9. **Weight-decay scan on σ=0.05 baseline (alphonse #2897)** — NEW, assigned 2026-05-14 11:05; wd=5e-4 and wd=1e-3 vs current wd=2e-4; σ=0.05 init starts at higher L2, may be under-regularized. Uses `--init_std 0.05`.

Closed this round:
- **Orthogonal init restore (frieren #2854)** — CLOSED 2026-05-14 11:50: +4.5% val on old bar (+10.3% on new). Orthogonal element magnitudes ~9× surrounding trunc_normal_; sharp early slice softmax starves attention block. Best epoch 29-31 confirms slow recovery.
- **AoA-Fourier features at input (thorfinn #2867)** — CLOSED 2026-05-14 10:35: +20.8% val regression. K=8 frequency ladder creates ~44 cycles across ±0.175 rad AoA range — high-frequency aliasing.
- **Pinball τ=0.60 pressure (alphonse #2853)** — CLOSED 2026-05-14 11:00: +11.5% val regression. τ-axis single optimum at 0.55 confirmed; τ=0.60 overshoots to over-prediction bias. τ axis fully bracketed and retired.
- **Pinball Ux/Uy extension (tanjiro #2855)** — CLOSED 09:21: regression (+4.5% val); velocity channels unbiased. Axis retired.

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
| #2811 | thorfinn | Sobolev loss on ∇p | **CLOSED** 2026-05-14 07:50 (val +18%) |
| #2812 | nezuko | LayerScale (init=1e-4) | **CLOSED** 2026-05-14 07:55 (var-vs-mean #10) |
| #2816 | edward | FiLM-style Re-conditioning | **CLOSED** 2026-05-14 08:15 (+50% params unjustified; re_rand OOD mechanism confirmed) |
| **#2817** | **fern** | **σ=0.05 init confirmation** | **MERGED 2026-05-14 09:21 (12th shift: mean val=40.82, test=35.25)** |
| **#2853** | **alphonse** | **Pinball τ=0.60 pressure** | **WIP stale_wip 2026-05-14 07:45; GPU active** |
| **#2854** | **frieren** | **Orthogonal init restore (in_project_slice)** | **WIP stale_wip 2026-05-14 07:45; GPU active** |
| #2855 | tanjiro | Pinball τ=0.55 Ux/Uy velocity channels | **CLOSED** 2026-05-14 10:05 (val +4.5%; velocity unbiased — diagnostic conclusive) |
| **#2863** | **askeladd** | **Re-Fourier features at input (NeRF-style log Re)** | **WIP 2026-05-14 08:30; GPU ~39GB** |
| **#2865** | **edward** | **γ-only FiLM-Re (drop β; param-efficient follow-up)** | **WIP CONFLICTING; sent back for rebase 2026-05-14 10:33** |
| **#2866** | **nezuko** | **Divergence-free auxiliary loss (∇·u=0)** | **WIP** |
| #2867 | thorfinn | AoA-Fourier features at input (targets camber_rc) | **CLOSED** 2026-05-14 10:35 (+20.8% val — frequency aliasing on narrow AoA range) |
| **#2882** | **tanjiro** | **σ-scan continuation: std=0.07 and std=0.10** | **WIP 2026-05-14 10:10** |
| **#2886** | **thorfinn** | **γ-only FiLM-AoA: per-block AoA conditioning (targets camber_rc OOD)** | **WIP 2026-05-14 10:45** |
| #2853 | alphonse | Pinball τ=0.60 pressure | **CLOSED** 2026-05-14 11:00 (+11.5% — τ peak at 0.55, axis retired) |
| **#2895** | **fern** | **Y-flip data augmentation (flow y-equivariance, 2× data free)** | **WIP 2026-05-14 11:05** |
| **#2897** | **alphonse** | **Weight-decay scan on σ=0.05: wd=5e-4 and wd=1e-3** | **WIP 2026-05-14 11:05** |
| #2853 | alphonse | Pinball τ=0.60 pressure | **CLOSED** 2026-05-14 11:00 (+11.5%) |
| #2854 | frieren | Orthogonal init for in_project_slice | **CLOSED** 2026-05-14 11:50 (+4.5%) |
| #2863 | askeladd | Re-Fourier K=8 at input | **CLOSED** 2026-05-14 12:10 (+5.2%; scalar Re aliasing) |
| **#2882** | **tanjiro** | **σ=0.07/0.10 scan** | **MERGED** 2026-05-14 12:15 (13th shift, −10.4% val/−13.1% test) |
| **#2902** | **frieren** | **SwiGLU FFN: replace GELU+Linear with gated FFN** | **WIP 2026-05-14 11:55** |
| **#2908** | **tanjiro** | **σ interior scan: σ=0.06/0.09** | **WIP NEW 2026-05-14 12:20** |
| **#2909** | **askeladd** | **Pressure-Poisson auxiliary loss (λ=0.01)** | **WIP NEW 2026-05-14 12:20** |

**Merged:** 13 | **Closed:** 63 | **WIP:** 8 | **Idle:** 0

## Key meta-findings from round 1

1. **Compute is permanently binding** — best=last at every merge. The 30-min cap has been the dominant constraint since bf16 (PR #1715).
2. **Variance-vs-mean decoupling confirmed (10 instances)** — β1=0.85/0.95, β2=0.999, warmup, Lookahead, grad-accum, RMSNorm, LN γ-init=0.5, LayerScale init=1e-4 all show variance reduction with mean regression. Pattern: any mechanism reducing optimizer step frequency, representation capacity, or initial activation scale trades mean improvement for variance reduction. At 35-ep compute-bound cap, the mean cost is never recovered.
3. **Lion β1 axis FULLY BRACKETED** — β1=0.85 (+8.3%) and β1=0.95 (+4.83pt) both regress; β1=0.90 confirmed optimal.
4. **Lion β2 axis FULLY BRACKETED** — β2=0.95 (+14.8%) and β2=0.999 (+5.69%) both regress; β2=0.99 confirmed optimal.
5. **Schedule-shape axis FULLY RETIRED** — warmup, warm restarts, all variants lose to cosine T_max=50 with implicit residual.
6. **Per-channel amplitude axis RETIRED under Lion** — p_weight=2.0 failed; per-channel β closed; Lion's sign() discards gradient magnitude, so amplitude-based loss scaling has no effect on capacity allocation.
7. **Conditioning-variable jitter axis RETIRED** — jittering log(Re) creates supervised inconsistency; valid augmentation requires conditional invariance in outputs.
8. **GC + sign() is sign-incompatible** — Gradient Centralization's row-mean subtraction forcibly inverts ~half of coordinate update directions each step under Lion. GC needs magnitude-based optimizers.
9. **Pinball τ=0.55 WIN (PR #2801)** — directional asymmetric loss for under-prediction bias works under Lion. Mechanism: τ=0.55 penalizes residuals with y>pred 10% more; OOD splits (re_rand −8.4%, cruise −11.6%) benefit most. Lion's sign() preserves the directional signal from pinball (unlike amplitude-scaling which gets discarded). This opens the τ-axis and the channel-coverage axis.
10. **σ=0.05 init WIN (PR #2817)** — trunc_normal_(std=0.05) beats std=0.02: mean val −6.6%, mean test −5.4%, ALL 4 per-split test splits improve. Mechanism: σ=0.05 starts weights closer to the optimizer's convergence neighbourhood (param L2 ~62 at convergence; σ=0.05 init gives ~25 vs σ=0.02 gives ~10). σ=0.01 fails catastrophically (insufficient time to climb from init L2~5 to convergence within 35-ep budget). σ-axis ongoing (tanjiro #2882 testing σ=0.07/0.10). In_project_slice orthogonal init bug (latent) now being tested by frieren (#2854).
11. **Velocity pinball RETIRED** — tanjiro #2855 confirmed Ux/Uy channels have NO systematic under-prediction bias (signed residuals at final epoch ≈ −0.003, near-zero). τ=0.55 overcorrects unbiased channels → +4.5% val regression. Pinball τ is only effective for channels with directional bias.

## Currently retired axes

- **Scalar capacity** (n_hidden, n_layers, slice_num, mlp_ratio) — failed at all 3 baselines
- **Schedule shape** — T_max, eta_min, warmup-then-flat, warm restarts — all retired
- **Dropout** (per-neuron) — regularization stack already saturated (note: DropPath/stochastic depth has NOT been explicitly tested — #2926 tests this; removed premature DropPath retirement)
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
- **Pinball τ > 0.55 (pressure channel)** — τ-axis has single optimum at τ=0.55. τ=0.60 overshoots: signed residual goes negative (over-prediction), all 4 splits regress +11-17%. Axis fully bracketed (0.50 < **0.55 WIN** > 0.60). No further τ scans warranted.
- **Pinball on velocity channels (Ux/Uy)** — velocity channels unbiased (signed residuals ≈ −0.003); τ≠0.5 overcorrects unbiased channels. Effective only on channels with directional bias.

## Potential next research directions

### Immediate (round 9-10 in flight, 7 PRs)

**Loss geometry axis (2):**
1. Pinball τ=0.60 pressure (alphonse #2853) — push asymmetry harder
2. Divergence-free auxiliary loss (nezuko #2866) — physics-informed ∇·u=0

**Architectural / capacity axis (2):**
3. Orthogonal init restore (frieren #2854) — zero-cost latent bug fix
4. γ-only FiLM-Re (edward #2865) — sent back for rebase; compound with σ=0.05 pending

**Conditioning / OOD axis (NEW #2886):**
5. γ-only FiLM-AoA (thorfinn #2886) — per-block AoA conditioning, identity init; targets camber_rc OOD split. Avoids aliasing failure of #2867 by conditioning via trunk FiLM rather than input-side Fourier features.

**Input-encoding axis:**
6. Re-Fourier features at input (askeladd #2863) — NeRF-style log(Re); ~+2K params

**Init-scale axis:**
7. σ-scan continuation: σ=0.07/0.10 (tanjiro #2882) — directly extends winning σ-axis; runs on σ=0.05 baseline

### Medium-term (next 1-2 rounds)

9. **Pinball τ=0.65 pressure** — if τ=0.60 wins, continue scan
10. **Re-Fourier + γ-FiLM-AoA combined** — if #2863 and #2886 both win, test stacking (orthogonal conditioning variables)
11. **γ-FiLM-Re + γ-FiLM-AoA dual conditioning** — if both edward #2865 rebase and thorfinn #2886 win, combine for joint (Re, AoA) per-block scale
11. **Y-flip augmentation** — flow-symmetric BCs admit clean mirror augmentation
12. **Pressure-Poisson auxiliary loss** — ∇²p coupling to velocity gradients (extends nezuko's divfree mechanism)
13. **τ per-split tuning** — re_rand improved most under τ=0.55; try class-specific τ
14. **Pinball τ < 0.5 for single_in_dist** — probe whether in-distribution has over-prediction bias
15. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens
16. **Fourier-encoded Re feeding into FiLM γ** — richer Re embedding as conditioning input (compound of arms 5 and 6)
17. **GroupedFiLM** — share γ MLPs across 2 consecutive blocks (further param reduction)

### Plateau-protocol escalations (if 5 consecutive misses arrive)

18. **Token-mixing alternative** — replace PhysicsAttention with gated linear attention or MLP-mixer block
19. **Hybrid loss: pinball + Charbonnier (smooth-MAE)** — Charbonnier-then-pinball or composite weighting
20. **Pretrain-then-finetune at higher Re** — explicit OOD-targeted curriculum
