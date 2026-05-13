# SENPAI Research State

- **Date**: 2026-05-13 22:35 — PLATEAU CONTINUING. Round 22:35 update: **#2598 askeladd aoa-rotation-aug-2deg CLOSED** (+7.21% val, +6.97% test; targeted rc split essentially flat +0.93% while easier splits regressed sharply — SECOND independent confirmation that the OOD axis is **shape-distribution NOT mesh-position-distribution**; student's own diagnosis ratifies the round 22:25 strategic pivot). Reassigned to **#2627 askeladd surface-normal-volume-feature** (n_x, n_y per node from nearest surface point as 25-26th input channels — genuinely new shape info orthogonal to SDF magnitude + nezuko's curvature 2nd-derivative; physically motivated by pressure-normal coupling in viscous wall BCs).

- **Round 22:25 closures (4 PRs)**: **#2582 frieren DropToken-volume** (+18.0% val, +17.4% test catastrophic; worst hit on EASIEST split single +38.5% = under-converged signature; **input-side perturbation axis FULLY CLOSED** combining with closed coord-jitter, translation, re-input-jitter, Mixup), **#2592 nezuko sdf-input-feature** (+2.57% val; SDF column grad_norm decayed 10× — model actively ignored it; redundant with slice attention softmax; "explicit distance feature" axis closed), **#2594 fern coord-jitter** (+3.71% val; CRITICAL INSIGHT: **the OOD axis is shape-distribution NOT mesh-position-distribution** — geom_camber_rc differs in NACA-4 shape parameters, not mesh-position), **#2539 thorfinn Fourier-positional-encoding 3 arms σ=1.0/0.1/0.05** (all 3 regress +3.16%/+1.36%/+1.81%; asymptotic ordering shows redundancy with raw coords; **fixed-frequency positional encoding axis closed**). #2585 alphonse ReFiLM-residual SENT BACK for one arm (hidden=4, blocks ≥2; mechanism real but over-conditioning). Round 22:25 fresh assignments based on the "shape-distribution NOT mesh-position" pivot: **#2622 frieren focal-mae-surface-gamma2** (element-wise focal weighting on surface residuals, γ=2; loss-side reweighting on the dominant error component), **#2624 nezuko surface-curvature-feature** (local discrete curvature κ=|cross|/|tangent|³ as 25th input feature; genuinely new info beyond SDF — first-derivative signal), **#2625 fern naca-feature-jitter-sigma-0p02** (σ=0.02 Gaussian noise on NACA-4 shape-descriptor channels 15-17 during training — perturbs the EXACT OOD axis that coord-jitter missed), **#2626 thorfinn per-channel-separate-heads** (3 independent MLPs for Ux/Uy/p outputs — decouples cross-channel gradient interference, lets each channel learn its optimal projection). Active arms: alphonse #2585 ReFiLM-residual re-run, askeladd #2598 aoa-rotation, tanjiro #2599 SE-channel-attention, frieren #2622 focal-mae, nezuko #2624 curvature, fern #2625 naca-jitter, thorfinn #2626 per-channel-heads, edward #2534 TTA.
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`
- **CRITICAL INFRA NOTE**: PR #2319-#2325 were originally created with short-form student labels (`student:alphonse` etc.) instead of full-form (`student:charliepai2g24h1-alphonse`). This made them invisible to student pod polling (which uses exact full-name label match per senpai-gh.sh:669). 3-hour stuck period until labels were patched at 17:01. All future `assign-experiment` calls MUST use full student name.

---

## Current Baseline

**val_avg/mae_surf_p = 28.8762** — PR #2011 (film-re-attention), merged 2026-05-13.

**-75.3% cumulative from initial 117.17.**

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (662K params) **+ ReScaleHead** (163-param Re→scale head, out_channels=3) **+ p_channel_weight=5** (post-Huber linear weight on pressure channel) **+ ReFiLM** (4,624-param shared FiLM on slice logits, hidden=8, zero-init, across all 5 blocks/4 heads), **SOAP** (`lr=1e-3, betas=(0.95, 0.95), wd=1e-4, precondition_frequency=10`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 28 epochs / 30 min. Peak GPU 27.79/96 GB.

Per-split val: single_in_dist=28.6013, rc=41.9483, cruise=14.1462, re_rand=30.8090. Test avg 24.9992.

---

## Critical Programme Findings

### Noise Floor Calibrated
Single-seed run-to-run variance is **~1-2%** on val_avg/mae_surf_p. Deltas below 1.5% need multi-seed confirmation.

### Convergence/Budget-Limited
Model converges monotonically to cutoff in every run — best epoch is always the last epoch. Budget-aware mechanisms (EMA, SWA) remain high value — but EMA axis is now CLOSED (see below).

### Re-Conditioning Stack — 3 confirmed mechanisms
- **ReScaleHead** (output rescaling, -1.95%): learned Re→scale applied to Transolver output
- **p_channel_weight=5** (loss reweighting, -2.11%): 5× post-Huber pressure weight
- **ReFiLM** (attention conditioning, -1.17%): shared FiLM gates on slice logits — Re-dependent mode selection, confirmed by 33% entropy drop

### Re-Conditioning Architecture — Shared > Per-Block
- **refilm-per-block** (#2198, +2.9%): per-block gates DID specialize (block4 absmax 0.81 vs block0 0.51), but overfitted — shared FiLM acts as regularizer
- **re-input-jitter** (#2169, +5.5%): Re channel is load-bearing; any noise corrupts ReFiLM conditioning. Re augmentation axis CLOSED.
- **refilm-hidden-16** (#2253, +2.52%): gamma/beta absmax tripled (0.44→1.36); slice entropy collapsed in layer 3 head 1 (4.13→1.06). ReFiLM capacity expansion CLOSED. hidden=8 is the correct regularizing bottleneck.

### Loss-Shape Axis CLOSED (all 3 variants regressed)
- Huber δ_v-loose (#2081): +1.16%
- Huber δ_p-tight (#2111): +1.50%
- Log-cosh (#2146): +2.93%
Huber(δ=0.1) is a robust local optimum. 88% of pressure residuals already in quadratic regime by end of training.

### Distribution Matching Axis CLOSED
- Sorted pressure W1 (#2204): +1.01% — W1 gap reduced 15× but trades spatial precision for distributional correctness

### Input Augmentation Axis CLOSED
- Coord-jitter per-node (#1963): +1.93% regression
- Translation augmentation (#2092): +3.3% regression (bounded BVP breaks translation invariance)
- Re-input-jitter (#2169): +5.5% regression (Re channel load-bearing for ReFiLM)

### Architecture Scaling CLOSED for current budget
- n_layers=6 (#2079): +6.22% regression (23% epoch slowdown → under-trained)
- n_head=8 (#2154): +14.2% regression (dim_head=16 below bf16 GEMM efficiency → 23% epoch slowdown)
- refilm-per-block (#2198): +2.9% regression (overfitting without shared-weight regularization)

### LR Schedule Axis CLOSED
- LR warmup (#2077): +1.38%; SOAP trains stably from lr=1e-3
- SGDR warm restarts (#2110): +8.13%; restart shock destroyed cycle-2 convergence
- OneCycleLR (#1884): +3.52%; grad_clip saturated throughout peak window
- Cosine T_max=40 (#2147): +11.4%; Cosine T_max=56: +31.4%; T_max=28 confirmed optimal

### SOAP Optimizer Axis — Most Hyperparameters Exhausted
- **soap-betas-0p9-0p99** (#2252, +3.18%): beta2=0.99 → Kronecker factors only updated ~1050 times at precond_freq=10 over 10k steps → stale curvature. Baseline (0.95, 0.95) well-calibrated for 10k-step SOAP.
- **weight-decay-5e-4** (#2233, +1.86%): convergence-limited; higher wd consumes larger fraction of effective update step. wd=1e-4 is local optimum.
- **soap-precond-freq-5** (#2255, +3.52%): halving precond_freq from 10 to 5 introduced excess Kronecker-factor noise at bs=4, overwhelming any responsiveness gain. freq=10 confirmed optimal.
- **OPEN**: max_precond_dim 256→128 (#2323 frieren WIP) — faster Kronecker refresh per step without changing how often steps happen

### Surface Loss Weight Axis CLOSED
- **surf-weight-7** (#1936, +2.94%): rc went WRONG direction
- **surf-weight-15** (#2264, +3.18%): single=+9.50%, rc=+3.11%; bilateral failure. Loss-scale ≠ metric-scale when surface loss dominates. surf_weight=10 is local optimum. RC bottleneck is NOT a surface/volume balance issue.

### Architecture Capacity Expansion Findings
- **mlp-ratio-3** (#2256, +23.9%): compute-bound underfitting — 25.6% more params, same 30-min budget → less training signal per parameter. Under fixed compute, smaller model wins. FFN capacity expansion CLOSED.

### SWA Axis CLOSED (all 3 variants regressed)
- SWA last-k (#1933): no weight-space spread at LR=1e-5
- SWA v2-hybrid LR=1e-4 (#2032): +1.24% val miss; SWA averaging real (−0.94) but LR plateau costs base quality
- SWA v3 LR=5e-5 (#2032): +3.44% val; lower plateau even worse. SWA incompatible with 28-ep cosine budget.

### EMA Axis CLOSED (permanently on ReFiLM stack)
- **ema-beta-0p99-rampup** (#1966): 4 independent rebased runs all regressed (+1.72% to +3.40%), mean +2.60%. Root cause: per-epoch `load_state_dict` swap between live and EMA weights interacts with `torch.compile(mode='default', dynamic=True)` + zero-initialized ReFiLM FiLM gates, degrading live training trajectory by ~0.83 MAE average. EMA smoothing dividend real (~-0.1 MAE within a run) but cannot overcome trajectory penalty. EMA axis CLOSED. Lookahead (no state-swap overhead) is the preferred alternative to explore.

---

## Current Research Focus

**At a hard plateau.** 7+ axes closed in single review wave. Conventional levers exhausted:
- Trajectory smoothing (EMA/SWA/Lookahead) — CLOSED
- Loss-shape (Huber-δ, log-cosh, sorted W1, Laplacian) — CLOSED
- Input augmentation — CLOSED
- Architecture scale (n_layers, n_head, mlp_ratio, slice_num both directions) — CLOSED
- LR schedule (cosine T_max, SGDR, OneCycleLR, warmup) — CLOSED
- Weight tuning (p_channel_weight, surf_weight) — CLOSED
- SOAP HP space (betas, wd, precond_freq, max_precond_dim) — CLOSED
- LayerScale residual scaling (γ₀ ∈ {1e-4, 0.1+nodecay}) — CLOSED
- ReFiLM expansion (per-block, per-head, 5-dim multi-channel) — CLOSED
- ReScaleHead expansion (multi-channel) — CLOSED
- LLRD layer-wise LR decay — CLOSED

**Plateau Protocol activated.** Researcher-agent dispatched at 19:10 with `RESEARCH_IDEAS_2026-05-13_19:00.md` target to generate 7-8 bold fresh hypotheses. Need to think at different abstraction levels: data representation, equivariance, physics-informed (different formulations than Laplacian), multi-task auxiliary, output-side calibration.

---

## Active Experiments (round 22:25 — shape-distribution OOD focus)

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #2585 | alphonse | `refilm-residual-stream-shared` | WIP (sent back, re-run) | **HIGH** | Re-FiLM on residual stream re-run with hidden=4, blocks≥2 only — mechanism real but over-conditioning at hidden=8 |
| #2627 | askeladd | `surface-normal-volume-feature` | WIP | **HIGH** | Surface-normal direction (n_x, n_y) at each volume node from nearest surface point — 1st-derivative shape signal orthogonal to SDF magnitude and curvature; physically motivated by pressure-normal coupling at viscous walls |
| #2599 | tanjiro | `se-channel-attention-r8` | WIP | **HIGH** | Squeeze-and-Excitation channel-attention after each block with zero-init residual gate — adds missing global-context modulation pathway |
| #2622 | frieren | `focal-mae-surface-gamma2` | WIP | **HIGH** | Element-wise focal MAE weighting on surface residuals (γ=2, clamp [0.1,10]) — loss-side reweighting on the dominant error component |
| #2624 | nezuko | `surface-curvature-feature` | WIP | **HIGH** | Local discrete curvature κ=|cross|/|tangent|³ as 25th input channel — genuinely new info beyond SDF, first-derivative shape signal |
| #2625 | fern | `naca-feature-jitter-sigma-0p02` | WIP | **HIGH** | σ=0.02 Gaussian noise on NACA-4 shape-descriptor channels 15-17 (training-only) — perturbs the actual OOD axis (shape-distribution) that coord-jitter missed |
| #2626 | thorfinn | `per-channel-separate-heads` | WIP | **HIGH** | 3 independent 2-layer MLPs (hidden=64) for Ux/Uy/p outputs replacing shared head — decouples cross-channel gradient interference |
| #2534 | edward | `tta-re-bracket` | WIP (active, in progress) | MEDIUM | Per-epoch TTA-val for checkpoint selection + 2-seed run to bound variance |

---

## Merged Winners (chronological)

| PR | Student | Slug | val_avg | Delta | Cumulative |
|----|---------|------|---------|-------|------------|
| #1479 | thorfinn | grad-clip-1 | 117.17 | — | baseline |
| #1518 | thorfinn | higher-lr-cosine-14 | 96.5587 | −17.6% | −17.6% |
| #1460 | fern | relative-l2-loss | 89.6121 | −7.2% | −23.5% |
| #1473 | tanjiro | huber-loss | 89.3940 | −0.24% | −23.7% |
| #1613 | thorfinn | soap-optimizer | 42.4015 | **−52.6%** | **−63.8%** |
| #1630 | tanjiro | cosine-eta-min | 39.8693 | −5.97% | −66.0% |
| #1456 | alphonse | bf16-amp + T_max=17 | 36.8778 | **−7.51%** | **−68.6%** |
| #1794 | alphonse | torch-compile | 30.4412 | **−17.5%** | **−74.0%** |
| #1599 | fern | re-conditioned-scaling | 29.8463 | **−1.95%** | **−74.5%** |
| #1614 | edward | per-channel-loss-weights | 29.2179 | **−2.11%** | **−75.1%** |
| #2011 | fern | film-re-attention | 28.8762 | **−1.17%** | **−75.3%** |

## Ruled Out (key entries)

- **wider-soap-192** (#1797): data-bottlenecked +33%
- **larger-batch-compile** (#1847): training NOT compute-bound +21.3%
- **soap-fp32-precond** (#1854): bf16 Q implicit regularization +4.3%
- **deeper-soap** (#1848): compute-budget loss +11.6%
- **stochastic-depth** (#1897): regularization-limited refuted +8.48%
- **attention-dropout** (#1900): still descending at ep29, smoking gun +0.47%
- **surf-weight-7** (#1936): rc went WRONG direction +2.94%
- **swa-last-k** (#1933): no weight-space spread at LR=1e-5
- **plateau-swa v2+v3** (#2032): SWA axis fully closed. v2 LR=1e-4 +1.24%, v3 LR=5e-5 +3.44%. LR plateau incompatible with 28-ep cosine budget.
- **ema-weights v1** (#1704): dual-val overhead +5.9%
- **ema-weights v2** (#1917): β=0.999 too high +2.9%
- **rescale-head-2ch** (#1952): +4.63% on rebased stack; Ux channel load-bearing with p_weight=5
- **p-channel-weight-15** (#1985): +4.20% ALL splits; cross-channel coupling. p_weight=5 is optimum.
- **coord-jitter-aug** (#1963): +1.93%; per-node jitter doesn't compound with p_weight+ReFiLM
- **coord-translation-aug** (#2092): +3.3%; bounded BVP breaks NSE translation invariance
- **soap-linear-warmup** (#2077): +1.38%; no instability to fix + wastes budget epochs
- **per-channel-huber-delta v1** (#2081): +1.16%; loosening velocity δ removes Huber tail pressure gradients
- **huber-delta-p-tighter** (#2111): +1.50%; tightening δ_p truncates informative-outlier gradients. Huber δ axis CLOSED.
- **log-cosh-loss** (#2146): +2.93%; weaker gradients in transition band. Loss-shape axis CLOSED.
- **sgdr-warm-restarts-v2** (#2110): +8.13%; restart shock burned cycle-2 budget. Warm restarts CLOSED.
- **n-layers-6** (#2079): +6.22%; ~19% epoch slowdown trimmed budget.
- **n-head-8** (#2154): +14.2%; dim_head=16 below bf16 GEMM efficiency → 23% epoch slowdown.
- **onecycle-lr** (#1884): +3.52%; max_lr=2e-3 saturated grad_clip=1.0 throughout peak window.
- **sorted-pressure-dist** (#2204): +1.01%; W1 trades spatial precision for distributional correctness.
- **refilm-per-block** (#2198): +2.9%; per-block overfitting, shared FiLM acts as regularizer.
- **cosine-long-tail T_max=40/56** (#2147): +11.4%/+31.4%; T_max=28 is confirmed optimal. Schedule axis CLOSED.
- **re-input-jitter** (#2169): +5.5%/+14.1%; Re channel is load-bearing for ReFiLM. Re-augmentation CLOSED.
- **soap-betas-0p9-0p99** (#2252): +3.18%; beta2=0.99 → stale Kronecker preconditioner at 10k-step/precond_freq=10 regime. Baseline (0.95,0.95) confirmed optimal.
- **refilm-hidden-16** (#2253): +2.52%; FiLM capacity overfitting — gamma/beta absmax tripled, slice entropy collapsed. ReFiLM capacity expansion CLOSED.
- **weight-decay-5e-4** (#2233): +1.86%; convergence-limited; wd=1e-4 is local optimum.
- **mlp-ratio-3** (#2256): +23.9%; compute-bound underfitting under fixed 30-min budget. FFN capacity axis CLOSED.
- **surf-weight-15** (#2264): +3.18%; bilateral surf_weight failure (7 and 15 both worse). surf_weight=10 is local optimum. RC bottleneck is NOT surface/volume balance.
- **soap-precond-freq-5** (#2255): +3.52%; halving precond_freq introduced Kronecker-factor noise at bs=4. freq=10 confirmed optimal.
- **soap-max-precond-dim-128** (#2323): +16.45% (severe). max_precond_dim < n_hidden=128 destroys preconditioning. SOAP HP space FULLY CLOSED.
- **ema-beta-0p99-rampup** (#1966): mean +2.60% over 4 seeds; per-epoch EMA/live state-swap interacts with torch.compile+ReFiLM. EMA axis CLOSED permanently.
- **more-slices-128** (#1467): +13.5%; overfit on 1499 samples. slice_num=64 optimum.
- **slice-num-32** (#2320): +0.95%; 11% speedup (not 40-50%); SCHEDULER_T_MAX=28 hardcode meant extra epochs ran at eta_min=1e-5. slice_num axis FULLY CLOSED.
- **lookahead-soap-k5** (#2384): +6.3%; α=0.5 = 50% LR brake on descending model. **Trajectory-smoothing meta-axis fully CLOSED across EMA/SWA/Lookahead.**
- **aoa-film-conditioning** (#2319): +3.93%; gate IS active (γ absmax 1.24 = 1.8× Re-only) but extra channels are constant within sample. Re is special (varies within splits, governs slice-selection landscape). Multi-channel FiLM expansion CLOSED.
- **llrd-transolver** (#2321): +6.85% (>5%); LLRD attenuates near-output layers too aggressively on shallow 5L stack. LLRD inappropriate for shallow architectures.
- **geom-conditioned-output-head** (#2322): +3.79%; corr_logRe stayed +0.85 (Re-dominated); geom info already in 24-dim node features. Multi-channel ReScaleHead CLOSED.
- **pressure-laplacian-loss** (#2325): +3.91%; kNN cdist overhead cost 2 epochs; M=1024 sampling acted as noise injection at λ=0.01; targeted VOLUME but ranks on SURFACE. Laplacian formulation CLOSED (alternative formulations still possible).
- **layerscale-init-1e-4 + γ₀=0.1+nodecay** (#2428): +2.69% / +2.93%; mechanism works on retry but baseline isn't unstable, per-feature gating costs effective capacity. LayerScale residual axis FULLY CLOSED both regimes.
- **surf-vol-split-head** (#2529): +2.39%; surf/vol head ratio on pressure channel = 1.02 (no specialization); bottleneck is in trunk shared representations, not head capacity. Multi-channel head expansion FULLY CLOSED. Follow-up #2559 tests trunk-level surface injection.
- **mixup-scalar-alpha-0p4** (#2535): +52.7% catastrophic; mixing two NACA-4-digit airfoils linearly does not produce a valid airfoil — per-node features encode geometry A while targets are convex combinations of fields A and B. Mixup family ruled out for geometry-conditioned regression on this dataset.
- **bernoulli-surface-loss** (#2538): +4.1%; TandemFoilSet is viscous (RANS) data, Bernoulli p+½ρ|U|² constraint does NOT hold across surface. ID-split degradation pattern + train/bernoulli_loss INCREASING over training confirms data violates the constraint. Physics-informed soft constraints on viscous data CLOSED (Bernoulli + Laplacian).
- **drop-path-0p1** (#2532): +5.7% (both 0.1 and 0.05 arms); 5-block stack too shallow for branch-level stochastic depth (DeiT evidence relies on 12+ blocks for keep-rate compounding). Branch-level dropout on shallow physics-attention stacks CLOSED.
- **derived-features-re2-aoa** (#2537): +3.26%; explicit polynomial features duplicate information that ReFiLM + ReScaleHead already extract through their nonlinearities. Cross-feature predicted-best splits (re_rand, camber_rc) regressed most. Tabular-ML feature engineering orthogonality argument doesn't transfer when explicit Re-conditioning hooks already exist.
- **rmsnorm-replace-layernorm** (#2569): +2.67% val / +2.37% test; train loss IDENTICAL between RMSNorm and LayerNorm — gap purely in generalization. Regression concentrated on geom_camber_rc (+4.78% val / +7.11% test) confirms LN β-bias + mean-centering is acting as a useful regularizer at N=1499 scale. Promised 10% speedup did not materialize (torch.compile already fuses the LN kernel — 64.12→63.94 s/epoch). Norm-replacement axis CLOSED at this scale/compile/bf16 regime.
- **sam-rho-0p02-soap-wrap** (#2560): +53.4% val / +57.8% test CATASTROPHIC. NOT an optimizer bug — SAM was working as designed (perturbed loss reliably 20% > unperturbed, gradient norm decayed 3× over the run). PURE compute-budget failure: 2× per-step cost → 16 epochs vs 28 baseline → still falling 1.3%/2-epochs at wall-cap. Under 30-min SENPAI_TIMEOUT_MINUTES + N=1499 + SOAP+cosine baseline, **any 2× per-step regularizer is dominated by epoch-count loss**. Drop-in stochastic regularizer axis FULLY CLOSED (SAM/ESAM/LookSAM all wall-cap-dominated; together with closed EMA/SWA/Lookahead/LayerScale/DropPath this exhausts the entire "stochastic regularizer" category).
- **mlp-ratio-4-ffn-double** (#2581): +28.2% val / +27.5% test catastrophic; over-parameterization on N=1499 with 50% more params. Per-epoch convergence is slower with wider FFN even controlling for wall-clock. Largest hit on the EASIEST split (cruise +54.5%) — classic over-parameterization signature. **Combined with closed mlp_ratio=3 (#2256, +23.9%), FFN-ratio axis FULLY CLOSED both directions**; together with closed n_layers=6 (#2079), n_head=8 (#2154), slice_num=128 (#1467) / =32 (#2320), and wider-soap-192 (#1797), the **capacity-expansion meta-axis is exhausted across all 5 known capacity knobs**. Future capacity expansion is essentially impossible at this N + compute budget.
- **grad-accum-batch8** (#2324): NOT a science closure — infrastructure-stale. Tanjiro pod stuck ~5 hours on a pod-side secondary rate-limit cycle (gh_retry burns ~90s/iter that themselves contribute to the throttle, self-reinforcing). PR labels correctly set but pod can't poll. Closed to clear queue; hypothesis (effective batch 4→8 via accum) remains untested and could be re-attempted by another student.
- **drop-token-vol-only** (#2582): +18.0% val / +17.4% test CATASTROPHIC. Worst hit on EASIEST split single_in_dist (+38.5%) = classic under-converged signature. PhysicsAttention's softmax over slice-tokens is sensitive to token-count perturbation; token dropout breaks the prior that all nodes are visible at evaluation. **Combined with closed coord-jitter, translation, re-input-jitter, Mixup, DropToken — the "input-side perturbation augmentation" axis is FULLY CLOSED.** Future augmentation must operate on labels (target smoothing) or features (shape descriptors), not input topology.
- **sdf-input-feature** (#2592): +2.57% val / +1.85% test. SDF column grad_norm decayed 10× over training (0.34 → 0.034) — model actively learned to **ignore** the channel. PhysicsAttention slice softmax already produces a soft distance-weighted neighborhood; an explicit Euclidean distance feature is redundant with the learned attention prior. **Explicit distance-feature axis CLOSED**; curvature (2nd derivative) and shape-descriptor jitter remain unexplored as genuinely new information.
- **coord-jitter-sigma-0p01** (#2594): +3.71% val / +2.89% test. Mesh-position perturbation doesn't address the bottleneck. **CRITICAL EMPIRICAL INSIGHT: the OOD axis is shape-distribution NOT mesh-position-distribution.** geom_camber_rc differs from training in NACA-4 shape parameters (camber, thickness), not in mesh-position distribution. Coord-jitter perturbs the wrong axis. Redirects augmentation search to the shape-descriptor channels (reassigned to fern as #2625 `naca-feature-jitter-sigma-0p02`).
- **fourier-pos-encoding-rff (3 σ-arms)** (#2539): all 3 σ scales regress (σ=1.0: +3.16%, σ=0.1: +1.36%, σ=0.05: +1.81%). Asymptotic ordering shows no monotonic improvement direction. The PhysicsAttention slice-softmax over (xy)-coords already implicitly learns multi-scale position representations. **Fixed-frequency positional encoding axis CLOSED** — RFF features redundant with raw coordinates when the architecture has flexible slice attention. Combined with closed coord-jitter, the "coord-side input augmentation/encoding" meta-axis is exhausted.
- **aoa-rotation-aug-2deg** (#2598): +7.21% val / +6.97% test. Targeted val_geom_camber_rc essentially FLAT (+0.93%) while val_geom_camber_cruise regressed +23.91% and val_single_in_dist +10.91%. Student's analysis nails it: "Camber-rc generalization is not bottlenecked by AoA coverage — it's bottlenecked by camber shape. Rotating doesn't add camber variants. The hypothesis misdiagnosed the OOD axis." **SECOND independent confirmation that the OOD axis is shape-distribution NOT mesh-position-distribution / NOT AoA-coverage**. Galilean-rotation augmentation axis closed.

---

## Potential Next Directions

**PLATEAU PROTOCOL — researcher-agent dispatched at 19:10.** All conventional HP/schedule/loss/architecture axes are exhausted. Need BOLD directions at different abstraction levels:

**Conjectured next-tier categories (waiting on researcher-agent output for specifics):**
- **Data representation transforms**: signed distance field input, vorticity feature, dynamic pressure (0.5·ρ·U²) derived feature, frame-equivariant coordinates
- **Output-side calibration**: temperature scaling, isotonic regression on prediction percentiles, residual correction head trained on val errors
- **Physics-informed beyond Laplacian**: Bernoulli surface residual, mass conservation surface integral, momentum balance — all targeting surface (where ranking is) not volume
- **Multi-task auxiliary losses**: predict u/v/p with auxiliary tasks (surface normal, distance to wake, distance to LE/TE)
- **Targeted hard-sample reweighting**: focal-loss-style emphasis on val_geom_camber_rc-like samples (RC is 3× cruise so the loss should emphasize rc more without changing surf_weight scalar)
- **Test-time augmentation**: average predictions across mirror/translation symmetries
- **Architectural micro-changes** that haven't been tried: attention bias (learned per-pair bias), positional encoding tricks, output head changes (e.g., per-channel separate prediction heads)
- **Pretraining/SSL on mesh**: pretrain encoder on a self-supervised mesh task before joint training

**The rc split (val=41.95, 3× cruise=14.15) is the dominant error source.** Any new direction should be evaluated on whether it specifically targets rc-OOD, not just average val.

**Round 22:25 strategic pivot — shape-distribution NOT mesh-position-distribution:** The geom_camber_rc OOD bottleneck has been mis-identified for several rounds. With coord-jitter (#2594), translation, Mixup, DropToken all closed, the actual OOD axis is now understood:

**The bottleneck is NACA-4 shape-parameter distribution shift, not mesh-position distribution shift.**

Confirmed by:
- coord-jitter (#2594): perturbs mesh-position, +3.7% regression — doesn't help rc-OOD
- DropToken (#2582): perturbs mesh topology, catastrophic
- SDF feature (#2592): adds explicit distance-to-foil, redundant with slice attention softmax (grad_norm decays 10×)
- Fourier RFF (#2539): adds higher-frequency position basis, no improvement at any σ

This redirects the entire augmentation/feature search to **shape-descriptor channels** (NACA-4 parameters: camber, camber-position, thickness):
- **#2625 fern naca-feature-jitter-sigma-0p02**: σ=0.02 Gaussian noise on shape-descriptor channels 15-17 (training only) — perturbs the EXACT axis the rc split varies on.
- **#2624 nezuko surface-curvature-feature**: local discrete curvature κ as 25th input channel — genuinely new info (first-derivative of shape) NOT captured by SDF/coords.

Other Round 22:25 directions (orthogonal to shape-axis):
- **#2622 frieren focal-mae-surface-gamma2**: loss-side hard-mining on surface residuals (the dominant error component); zero extra compute.
- **#2626 thorfinn per-channel-separate-heads**: 3 independent MLPs for Ux/Uy/p — decouples cross-channel gradient interference.

Wall-cap-friendly principle still binding: every Round 22:25 assignment is either zero-compute (input feature, loss reweighting, head replacement) or near-zero (curvature precomputed once). NO 2× per-step regularizers; SAM-family axis remains closed.

**Active hypothesis cluster: shape-axis perturbation + shape-axis feature engineering + loss/head structure micro-changes.** The next plateau-break is expected from shape-axis interventions (fern/nezuko); the focal-MAE and per-channel-heads experiments serve as orthogonal architectural and loss-side baselines.
