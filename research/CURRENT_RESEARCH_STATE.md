# SENPAI Research State — Willow-pai2g-48h-r3

- **Date:** 2026-05-14 05:45
- **Advisor branch:** `icml-appendix-willow-pai2g-48h-r3`
- **Target task:** TandemFoilSet (CFD surrogate, predict (Ux, Uy, p) on 2D irregular meshes)
- **Primary metric:** `val_avg/mae_surf_p` (selection) and `test_avg/mae_surf_p` (paper-facing)
- **Most recent direction from human team:** None received — controlled 24/48h Charlie-vs-Willow logging ablation.

## Current baseline (10th shift)

**PR #2562 (Lion lr=7.5e-5)** merged 2026-05-13 22:30:
- **`val_avg/mae_surf_p` = 45.433** (seed 2 `srveevtx`)
- **`test_avg/mae_surf_p` = 39.509**
- Per-split test: single_in_dist=42.56, geom_camber_rc=53.48, geom_camber_cruise=24.00, re_rand=37.99
- **New merge bar: val < 45.43, test < 39.51, all four test splits finite**

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
| **PR #2562 (Lion lr=7.5e-5)** | **2026-05-13 22:30** | **45.433** | **39.509** | **−9.5% / −9.2%** |

**Cumulative: −62.0% val, −64.0% test from round-1 start.** Still compute-bound (best=last on both seeds at all 10 merges).

## Current research focus (round 8)

**Moving beyond Lion hyperparameter tuning.** The variance-vs-mean decoupling pattern is now fully confirmed across 6 experiments — any Lion trajectory-stabilization mechanism regresses mean while reducing variance. New round focuses on **structural/geometric axes** orthogonal to training trajectory:

1. **RMSNorm (alphonse #2800)**: replace LayerNorm throughout — normalization architecture change, ~7-10% per-layer compute saving, removes mean-centering step
2. **Pinball loss τ=0.55 (askeladd #2801)**: asymmetric directional loss bias for pressure channel — tests whether pressure is systematically under-predicted; directional not amplitude
3. **Param-group wd (frieren #2803)**: exclude norms/biases from weight decay — standard modern practice, prevents wd from shrinking LN γ→0
4. **LN γ-init=0.5 (tanjiro #2805)**: initialize LayerNorm scale at 0.5 (DeepNorm-style) — changes initial optimization geometry, not training trajectory

Still in flight from prior round:
5. **SWA (thorfinn #2712)**: average last-10-epoch checkpoints — free improvement if late checkpoints are diverse enough
6. **max_norm=0.5 (fern #2763)**: re-test gradient clip on new baseline — had the best OOD single-split signal of the round (−3.8pt camber_rc)
7. **Per-layer LR decay (nezuko #2753)**: α=0.85 geometric decay — deeper transformer blocks get smaller LR
8. **Gradient Centralization (edward #2762)**: zero-mean gradient before momentum update; different from variance-vs-mean axis (GC modifies gradient direction, not trajectory frequency)

## Round 1 portfolio (current)

| PR | Student | Hypothesis | Status |
|---|---|---|---|
| #1504–#2017 | various | 8 stacked improvements | **MERGED** (baseline history above) |
| #2516 | edward | Lion optimizer | **MERGED** 2026-05-13 20:05 (val=50.19) |
| #2562 | tanjiro | Lion lr=7.5e-5 | **MERGED** 2026-05-13 22:30 (val=45.43) — 10th baseline shift |
| #2561 | edward | Lion beta2=0.95 | **CLOSED** (+14.8%, beta2 analogy wrong) |
| #2520 | thorfinn | n_head 4→8 | **CLOSED** (+24%, capacity loss) |
| #2504 | frieren | QK-RMSNorm | **CLOSED** (+14%, Q/K magnitude signal) |
| #2628 | tanjiro | Lion lr=1e-4 | **CLOSED** (+1.9% val, overshoot) |
| #2501 | askeladd | β_p=0.625 | **CLOSED** (+6.8% val — per-channel β fully closed) |
| #2565 | fern | max_norm=0.5 | **CLOSED** 2026-05-14 03:25 (stale; reassigned fresh) |
| #2564 | nezuko | Gradient Centralization | **CLOSED** 2026-05-14 03:00 (stale; reassigned fresh) |
| #2505 | alphonse | SiLU activation | **CLOSED** (+18.9% val; Lion sign neutralizes SiLU advantage) |
| #2633 | edward | Lion beta1=0.95 | **CLOSED** (+4.83pt val; variance −85% but mean shifted) |
| #2631 | thorfinn | Lion warmup 5ep | **CLOSED** (+4.44% val; warmup cost too high at 35-ep cap) |
| #2629 | frieren | Lion wd=3e-3 | **CLOSED** (+1.68pt val; wd axis monotonic-worse) |
| #2700 | edward | Lion beta1=0.85 | **CLOSED** (+8.3% val; β1 axis FULLY BRACKETED, 0.90 optimal) |
| #2693 | tanjiro | CosineAnnealingWarmRestarts T_0=12 | **CLOSED** (+17.7% val; schedule-shape axis retired) |
| #2713 | frieren | Lion β2=0.999 | **CLOSED** (+5.69% val; β2 axis FULLY BRACKETED, 0.99 optimal) |
| #2694 | askeladd | Charbonnier loss ε=0.5 | **CLOSED** (+1.3% miss bar; loss-shape axis saturated) |
| #2726 | alphonse | Lookahead(Lion) k=5 α=0.5 | **CLOSED** 2026-05-14 05:30 (+6.77pt val; variance −93%; 6th variance-vs-mean decoupling) |
| #2743 | askeladd | p_weight=2.0 (volume) | **CLOSED** 2026-05-14 05:30 (+5.22pt val; pressure worst-hit; Lion sign discards magnitude weight) |
| #2751 | frieren | Re-feature jitter σ=0.05 | **CLOSED** 2026-05-14 05:30 (+12.4pt val +27%; conditioning jitter creates inconsistency) |
| #2752 | tanjiro | Gradient accumulation 2× | **CLOSED** 2026-05-14 05:30 (+8.87pt val; variance −95%; step-count cost irrecoverable) |
| **#2712** | **thorfinn** | **SWA (average epochs 26-35)** | **WIP** |
| **#2753** | **nezuko** | **Per-layer LR decay α=0.85** | **WIP** |
| **#2762** | **edward** | **Gradient Centralization on Lion** | **WIP** |
| **#2763** | **fern** | **max_norm=0.5 on new baseline (fresh)** | **WIP** |
| **#2800** | **alphonse** | **RMSNorm (replace LayerNorm)** | **WIP NEW 2026-05-14 05:45** |
| **#2801** | **askeladd** | **Pinball loss τ=0.55 for pressure** | **WIP NEW 2026-05-14 05:45** |
| **#2803** | **frieren** | **Param-group wd (no wd on norms/biases)** | **WIP NEW 2026-05-14 05:45** |
| **#2805** | **tanjiro** | **LayerNorm γ-init=0.5 (DeepNorm-style)** | **WIP NEW 2026-05-14 05:45** |

**Merged:** 10 | **Closed:** 47 | **WIP:** 8 | **Idle:** 0

## Key meta-findings from round 1

1. **Compute is permanently binding** — best=last at every merge. The 30-min cap has been the dominant constraint since bf16 (PR #1715).
2. **Variance-vs-mean decoupling confirmed (6 instances)** — β1=0.85/0.95, β2=0.999, warmup, Lookahead, grad-accum all show variance −68%–95% with mean regression +4.4pt–8.9pt. Pattern: any mechanism reducing optimizer step frequency (or equivalent trajectory diversity) trades mean improvement for variance reduction. At 35-ep compute-bound cap, the mean cost is never recovered.
3. **Lion β1 axis FULLY BRACKETED** — β1=0.85 (+8.3%) and β1=0.95 (+4.83pt) both regress; β1=0.90 confirmed optimal.
4. **Lion β2 axis FULLY BRACKETED** — β2=0.95 (+14.8%) and β2=0.999 (+5.69%) both regress; β2=0.99 confirmed optimal.
5. **Schedule-shape axis FULLY RETIRED** — warmup, warm restarts, all variants lose to cosine T_max=50 with implicit residual.
6. **Per-channel amplitude axis RETIRED under Lion** — p_weight=2.0 failed; per-channel β closed; Lion's sign() discards gradient magnitude, so amplitude-based loss scaling has no effect on capacity allocation.
7. **Conditioning-variable jitter axis RETIRED** — jittering log(Re) creates supervised inconsistency; valid augmentation requires conditional invariance in outputs.

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

## Potential next research directions

### Immediate (round 8 in flight)

1. **RMSNorm** — free-lunch normalization swap; modern default for transformers
2. **Pinball τ=0.55** — directional pressure bias; if residual diagnostic shows consistent direction, retry τ=0.45 if needed
3. **Param-group wd** — standard practice; zero compute cost
4. **LN γ-init=0.5** — init geometry; recoverable since γ is learnable
5. **SWA** — checkpoint averaging; free improvement if checkpoints diverse
6. **max_norm=0.5** — had best OOD single-split signal this round
7. **Per-layer LR decay** — untested structural lever
8. **Gradient Centralization** — geometric constraint on gradient direction (distinct from trajectory frequency)

### Medium-term

9. **Per-sample Re embedding** — dedicated re_rand OOD lever separate from conditioning variable
10. **Surface-anchored cross-attention** — boundary nodes as queries against volume tokens
11. **Y-flip augmentation** — flow-symmetric BCs admit clean mirror augmentation (conditionally invariant → valid)
12. **Label smoothing / target noise** — augment TARGET not INPUT (avoids conditioning inconsistency)
13. **Stochastic depth / DropPath with very low rate** — test if any regularization is still beneficial
14. **Separate embedding network for Re** — rather than concatenating log(Re), learn a dedicated Re embedding; avoids the conditioning-inconsistency problem of jitter
