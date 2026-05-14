# SENPAI Research State

- **Date**: 2026-05-14 01:08 — Round 01:05 update. **3 closures (#2678 alphonse T_max=35 +7.09%, #2671 askeladd surface-only-normal-feature +7.02% / -2.44% cruise, #2662 fern mask-ch15-jitter-ch16+ch17 +7.04%), 3 new assignments (#2699/#2702/#2703)**. Three additional confirmations of the structural finding:
  - **#2678**: T_max=28 is the principled optimum (not budget artefact). Ep28 ticks UP at residual LR vs monotone-descent at eta_min — eta_min FLOOR was doing the work in #2650, not residual LR. T_max>28 axis FULLY CLOSED (28 winner, 35 +7.1%, 40 +11.4%).
  - **#2671**: surface-only mask DOES remove volume-pollution side-channel (val_re_rand recovered from +35% to +5.1%) and DOES reveal a clean cruise gain (-2.44% val, -2.62% test) — but rc regression PERSISTS at +7.0% even without pollution. **The cruise/rc dichotomy at this N=1499/SOAP/30-min budget is partly structural, not purely a pollution artefact**. Collapse from -45%/+27% (#2627) → -2.4%/+7.0% (#2671) confirms most of #2627's drama was pollution amplification.
  - **#2662**: mask-ch15 lost BOTH the cruise win AND the rc regression of #2625. Establishes ch15 was the ENTIRE active axis in symmetric jitter — ch16+ch17 jitter alone is near-noop. The cruise/rc swap mechanism is real and ch15-localized.
  
  Round 01:05 strategic plan: (1) **#2699 alphonse Re-conditional LayerScale** — 5th Re-conditioning hook at per-block residual gating (the only proven winning direction); (2) **#2702 askeladd SWA last 5 epochs** — fundamentally different angle (optimizer-level flat-minima, orthogonal to all 9 falsified directions); (3) **#2703 fern asymmetric one-sided ch15 jitter σ=0.05** — direct attack on rc extrapolation gap (push samples TOWARD M={0.778, 0.889} not inward); (4) wait on remaining 5 WIP (thorfinn #2690 output-bias 4th hook, frieren #2689 oversampling, nezuko #2688 aux-surface-only, tanjiro #2599 SE-attn, edward #2534 TTA-multi-seed).

- **Date**: 2026-05-14 00:46 — Round 00:45 update. **3 closures (#2660 nezuko surface-normal-aux +3.13%, #2659 frieren sample-focal-MAE +15.9%, #2626 thorfinn per-channel-heads-kaiming-h128 +3.95%), 3 new assignments (#2688/#2689/#2690)**. **CRUISE/RC STRUCTURAL DICHOTOMY NOW 6-WAY CONFIRMED** across 6 orthogonal interventions (data-aug, per-ch-heads zero+kaiming, input feature, output aux, focal-MAE). The dichotomy is a **first-class structural finding** about the trunk representation. Any intervention that densifies in-dist neighborhood helps cruise (interpolation) and hurts rc (extrapolation on ch15 M). **The ONLY mechanism that escapes the dichotomy is Re-conditioning at bounded injection points** — PR #2650 ReCondLN improved ALL splits with only mild rc regression (+0.64% val). Round 00:45 strategic plan: (1) test surface-only variants (#2671 + #2688) to isolate volume-pollution vs structural; (2) test data-side oversampling (#2689) qualitatively different from feature/loss/architecture; (3) test 4th Re-conditioning hook (#2690 output bias) extending the only known winning axis; (4) #2678 alphonse T_max=35 budget extension on the winner.

- **Date**: 2026-05-14 00:25 — **NEW BASELINE. PR #2650 alphonse ReConditionalLayerNorm MERGED: val_avg 28.8762 → 28.2414 (-2.20%), test_avg -2.07%.** CIN/adaLN-Zero bounded Re-conditioning of all 3 LN roles — vindicated: val_single_in_dist -4.99% vs residual-stream FiLM which regressed +1.19% on same split. Mechanism is injection-point specific: LN normalisation before γ/β prevents unbounded amplification. Mild rc regression (+0.64% val) is the new bottleneck. Alphonse re-assigned to `recondln-t_max35` — test budget-limit hypothesis: best=last=28 (still falling), a T_max=35 re-run on the new baseline should push further.

- **Date**: 2026-05-14 00:01 — PLATEAU CONTINUING; structural finding now THIRD-CONFIRMED. Round 23:55 update: **#2627 askeladd surface-normal volume feature CLOSED +8.34% val** with the IDENTICAL cruise-WIN (-44.8%) / rc-LOSS (+27.2%) dichotomy. Three orthogonal interventions (input augmentation #2625, output-head architecture #2626, new input feature #2627) all produce essentially identical cruise/rc dichotomy ratios — **the dichotomy is a STRUCTURAL property of the data/model geometry under N=1499 / 30-min budget / Transolver baseline, not specific to any intervention**. Reassigned askeladd to **#2671 surface-only-normal-feature** (zero normals on volume nodes per student's analysis — clean test of whether the dichotomy survives volume-pollution removal, which would reveal whether the issue is feature-pollution or fully structural).

- **🔑 CHANNEL-INDEX CORRECTION (from #2662 fern empirical verification)**: The NACA-4 code is `MPTT`. ch15=M (camber AMPLITUDE), ch16=P (camber POSITION), ch17=T (thickness). Direct check of `splits_v2/`: rc/cruise are held out on **ch15** (amplitude), with rc ch15 values {0.778, 0.889} OOD beyond train cluster. The "rc"/"cruise" split prefix refers to **raceCar/cruise** environment, NOT camber-position channel. The Round 23:41 KEY FINDING is correct in spirit but had the channel index swapped: **rc=EXTRAPOLATION along ch15, NOT ch16**. PR #2662 was sent back with corrected instruction: mask ch15, jitter ch16+ch17.

- **Date**: 2026-05-13 23:45 — PLATEAU CONTINUING with a structural insight unlocking a new sub-axis. Round 23:41 update: **3 closures (#2622, #2624, #2625), 1 send-back (#2626), 3 new assignments**. Closure details: (a) **#2622 frieren element-level focal-MAE (γ=2) CLOSED CATASTROPHIC +18.1% val** — clamp saturated on heavy-tailed pressure residuals; (b) **#2624 nezuko surface-curvature input feature CLOSED +0.91% val** with fat-tailed rc regression (+4.77%); curvature redundant with slice-attention softmax; (c) **#2625 fern NACA-4 jitter σ=0.02 (3-ch) CLOSED +3.45% val** BUT with DRAMATIC dichotomy: **val_geom_camber_cruise -43.71% (massive WIN) / val_geom_camber_rc +24.20% (catastrophic regression)**. Send-back: **#2626 thorfinn per-channel-heads** showed the IDENTICAL parallel dichotomy (cruise -44% / rc +21%) — strong cross-evidence for a structural finding.

- **🔑 KEY PROGRAMME FINDING (Round 23:41)**: **rc camber-position is an EXTRAPOLATION problem; cruise camber is an INTERPOLATION problem.** Two independent experiments produced parallel cruise-WIN / rc-LOSS splits. The rc test split is held out on the camber-position axis (channel 16) — its values lie OUTSIDE the train distribution. Augmenting that axis with σ=0.02 expands training distribution INWARD (densifying in-dist) but does NOT extend to rc test values, so the rc gap relatively widens. cruise is interpolated within train — any generic shape-smoothing helps. **Implication**: input-side shape augmentation MUST be channel-selective — augment channels NOT held out; never augment along the OOD axis itself. This unlocks a new class of OOD-aware input augmentations.

- Round 23:41 new assignments: **#2659 frieren sample-level-focal-mae-gamma1** (γ=1 not 2, per-sample not per-element — avoids the clamp saturation that broke #2622); **#2660 nezuko surface-normal-auxiliary-output-head** (Kendall/Gal/Cipolla multi-task — output-head normal regularization rather than input feature, after curvature input closed); **#2662 fern naca-jitter-ch15-17-only-sigma02** (channel-selective jitter masking ch 16 = camber-position = rc OOD axis; preserves the cruise -43.7% win without rc +24% regression). #2626 thorfinn re-issued as per-channel-heads-kaiming-h128 (kaiming_normal init + head_hidden=128, addresses zero-init under-convergence).

- **Date**: 2026-05-13 23:35 — Round 23:32 update: **#2585 alphonse ReFiLM-residual CLOSED across both arms** (Arm 1 h=8/all-blocks: val_avg +0.06% tied, test_avg -1.70% gain BUT val_rc +2.04%; Arm 2 h=4/blocks-2-4: val_avg +1.31% regressed, val_re_rand +1.85% LOSES previous gain). Student's diagnosis: (1) capacity isn't the lever — |γ|max stayed 0.77-0.85 regardless of hidden size, gradient dominates; (2) depth restriction is a sharp trade — fixes rc but cancels re_rand gain; (3) mechanism remains valid (corr_mod_logre = -0.76 to -0.93) but doesn't translate to val win; (4) val_single_in_dist regresses in BOTH arms → ANY Re-conditioning of the residual stream perturbs in-dist features. **Residual-stream FiLM injection point CLOSED**. Reassigned to **#2650 alphonse re-conditional-layernorm-affine** — same Re-conditioning idea at DIFFERENT injection point (LN's γ/β post-normalization affine, canonical Conditional-InstanceNorm Dumoulin 2017). Bounded modulation by construction (LN normalizes BEFORE γ/β applies) — should avoid the in-dist perturbation issue. 

- **Date**: 2026-05-13 22:35 — Round 22:35 update: **#2598 askeladd aoa-rotation-aug-2deg CLOSED** (+7.21% val, +6.97% test; targeted rc split essentially flat +0.93% while easier splits regressed sharply — SECOND independent confirmation that the OOD axis is **shape-distribution NOT mesh-position-distribution**; student's own diagnosis ratifies the round 22:25 strategic pivot). Reassigned to **#2627 askeladd surface-normal-volume-feature** (n_x, n_y per node from nearest surface point as 25-26th input channels — genuinely new shape info orthogonal to SDF magnitude + nezuko's curvature 2nd-derivative; physically motivated by pressure-normal coupling in viscous wall BCs).

- **Round 22:25 closures (4 PRs)**: **#2582 frieren DropToken-volume** (+18.0% val, +17.4% test catastrophic; worst hit on EASIEST split single +38.5% = under-converged signature; **input-side perturbation axis FULLY CLOSED** combining with closed coord-jitter, translation, re-input-jitter, Mixup), **#2592 nezuko sdf-input-feature** (+2.57% val; SDF column grad_norm decayed 10× — model actively ignored it; redundant with slice attention softmax; "explicit distance feature" axis closed), **#2594 fern coord-jitter** (+3.71% val; CRITICAL INSIGHT: **the OOD axis is shape-distribution NOT mesh-position-distribution** — geom_camber_rc differs in NACA-4 shape parameters, not mesh-position), **#2539 thorfinn Fourier-positional-encoding 3 arms σ=1.0/0.1/0.05** (all 3 regress +3.16%/+1.36%/+1.81%; asymptotic ordering shows redundancy with raw coords; **fixed-frequency positional encoding axis closed**). #2585 alphonse ReFiLM-residual SENT BACK for one arm (hidden=4, blocks ≥2; mechanism real but over-conditioning). Round 22:25 fresh assignments based on the "shape-distribution NOT mesh-position" pivot: **#2622 frieren focal-mae-surface-gamma2** (element-wise focal weighting on surface residuals, γ=2; loss-side reweighting on the dominant error component), **#2624 nezuko surface-curvature-feature** (local discrete curvature κ=|cross|/|tangent|³ as 25th input feature; genuinely new info beyond SDF — first-derivative signal), **#2625 fern naca-feature-jitter-sigma-0p02** (σ=0.02 Gaussian noise on NACA-4 shape-descriptor channels 15-17 during training — perturbs the EXACT OOD axis that coord-jitter missed), **#2626 thorfinn per-channel-separate-heads** (3 independent MLPs for Ux/Uy/p outputs — decouples cross-channel gradient interference, lets each channel learn its optimal projection). Active arms: alphonse #2585 ReFiLM-residual re-run, askeladd #2598 aoa-rotation, tanjiro #2599 SE-channel-attention, frieren #2622 focal-mae, nezuko #2624 curvature, fern #2625 naca-jitter, thorfinn #2626 per-channel-heads, edward #2534 TTA.
- **Most recent research direction from human researcher team**: No directives yet.
- **Advisor branch**: `icml-appendix-charlie-pai2g-24h-r1`
- **CRITICAL INFRA NOTE**: PR #2319-#2325 were originally created with short-form student labels (`student:alphonse` etc.) instead of full-form (`student:charliepai2g24h1-alphonse`). This made them invisible to student pod polling (which uses exact full-name label match per senpai-gh.sh:669). 3-hour stuck period until labels were patched at 17:01. All future `assign-experiment` calls MUST use full student name.

---

## Current Baseline

**val_avg/mae_surf_p = 28.2414** — PR #2650 (re-conditional-layernorm-affine), merged 2026-05-14.

**-75.8% cumulative from initial 117.17.** (-2.20% vs previous 28.8762)

Config: `n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2` (676K params) **+ ReScaleHead** (163-param Re→scale head) **+ p_channel_weight=5** (post-Huber pressure weight) **+ ReFiLM** (4,624-param shared FiLM on slice logits, hidden=8, zero-init) **+ ReConditionalLayerNorm** (13,872 params — shared CIN/adaLN-Zero Re-conditioning of all 3 LN roles: pre-attn, pre-FFN, pre-out; log(Re)→γ+β via Linear(1,8)→GELU→Linear(8,n_hidden), zero-init final layer), **SOAP** (`lr=1e-3, betas=(0.95,0.95), wd=1e-4, precondition_frequency=10`), **`CosineAnnealingLR(T_max=28, eta_min=1e-5)`**, `grad_clip=1.0`, `batch_size=4`, `surf_weight=10.0`, Huber(δ=0.1)+rel-L2 loss, **bf16 AMP**, **torch.compile(mode="default", dynamic=True)**. 28 epochs / 30 min. Peak GPU 27.79/96 GB.

Per-split val: single_in_dist=27.1740, rc=42.2153, cruise=13.6733, re_rand=29.9031. Test avg 24.4827.

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

## Active Experiments (round 01:05 — post-3-closure wave)

Baseline: **PR #2650 ReCondLN val_avg=28.2414**. All experiments below built on the new baseline.

| PR | Student | Slug | Status | Priority | Notes |
|----|---------|------|--------|----------|-------|
| #2699 | alphonse | `re-conditional-layerscale` | WIP (just-assigned) | **HIGHEST** | **5th Re-conditioning hook**: per-block LayerScale α(Re)=1.0+g(logRe) gating residual additions. ~250 params, zero-init residual. Different injection point (per-block depth-modulation) from existing 4 hooks. |
| #2702 | askeladd | `swa-last-5-epochs` | WIP (just-assigned) | **HIGH** | **Optimizer-level flat-minima**: PyTorch built-in SWA on epochs 24-28. Orthogonal to all 9 falsified architecture/feature/loss directions. Reports both SWA and non-SWA metrics. |
| #2703 | fern | `asymmetric-ch15-jitter-sigma05` | WIP (just-assigned) | **HIGHEST** | **Direct attack on rc extrapolation gap**: half-normal `\|N(0,0.05)\|` one-sided positive jitter on ch15 only, clip [0,1]. Pushes samples TOWARD M={0.778, 0.889}. First experiment targeting rc-extrapolation rather than densifying inward. |
| #2688 | nezuko | `surface-only-aux-normal-head` | WIP | **HIGH** | Surface-only AUX OUTPUT normal head — mirror of closed #2671 at the OUTPUT side. After #2671 confirmed pollution-removal works (val_re_rand recovered +35%→+5.1%), this tests whether AUX-output version has same effect. |
| #2689 | frieren | `shape-bin-oversampling-m05` | WIP | **HIGH** | DATA-SIDE intervention: WeightedRandomSampler oversample M≥0.5 by 3×. Densifies train near rc OOD boundary on ch15. Complementary to fern #2703 (data-side weighting vs augmentation). |
| #2690 | thorfinn | `re-conditional-output-bias` | WIP | **HIGH** | 4th Re-conditioning hook — additive per-channel bias after ReScaleHead. Extends winning axis. Same family as #2699 but at output-bias injection point. |
| #2599 | tanjiro | `se-channel-attention-r8` | WIP (pod rate-limit stuck) | MEDIUM | SE channel-attention; pod still on secondary rate-limit cycle. No results yet. |
| #2534 | edward | `tta-re-bracket` (multi-seed) | WIP (active, in progress) | MEDIUM | Per-epoch TTA-val for checkpoint selection + multi-seed run to bound variance. Earlier single-seed arm hit val_avg 29.86 on a +1.0 variance no_tta run; multi-seed will resolve whether TTA wrapping crosses baseline. |

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
| #2650 | alphonse | re-conditional-layernorm-affine | 28.2414 | **−2.20%** | **−75.8%** |

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
- **refilm-residual-stream-shared** (#2585, 2 arms): Arm 1 (h=8, all blocks): val_avg tied +0.06%, test_avg -1.70% but val_rc +2.04%. Arm 2 (h=4, blocks 2-4): val_avg +1.31% regressed, test_avg -0.98%, loses val_re_rand gain. **Residual-stream FiLM injection point CLOSED across 2 arms.** Mechanism remains valid (corr_mod_logre = -0.76 to -0.93, gates open from zero init, slice-logit FiLM unchanged) BUT (a) capacity isn't the lever (|γ|max stayed 0.77-0.85 regardless of hidden 4 vs 8 — gradient dominates not expressivity), (b) depth restriction is a sharp trade (fixing rc cancels re_rand gain — Re-modulation is distributed across blocks not localized), (c) val_single_in_dist regresses in BOTH arms → ANY residual-stream Re-conditioning perturbs in-dist features regardless of placement. The injection point itself is the issue — unbounded amplification of γ⊙h+β before next norm distorts in-dist pipeline.
- **element-level-focal-mae-gamma2** (#2622): +18.1% val / +12.1% test CATASTROPHIC. Clamp saturated immediately on heavy-tailed pressure residuals — |r/δ|^2 hits clip=10 in nearly every batch, and the saturating-clip mass dominates the gradient. Model trains to fit loud outliers at expense of bulk. **Element-level reweighting fundamentally incompatible with the long-tailed surface-p residual distribution.** Reissued at sample granularity + γ=1 as #2659 (sample-level pre-averaging smooths the heavy tail before focal weighting; γ=1 linear avoids quadratic explosion).
- **surface-curvature-input-feature** (#2624): +0.91% val / -0.49% test (small test win, val loss). Critical pattern: **val_geom_camber_rc +4.77% regressed** while val_geom_camber_cruise -2.90% improved — same shape-axis dichotomy as #2625. The slice attention's softmax over element coords is functionally a learned local curvature estimator; explicit κ provides redundant info that confuses the optimizer at OOD shifts. **Explicit curvature input CLOSED**; reassigned to #2660 surface-normal-auxiliary-output-head (output-task regularization, fundamentally different signal pathway than input-feature augmentation).
- **naca-feature-jitter-sigma-0p02 (3-channel)** (#2625): +3.45% val / +? test BUT **DRAMATIC DICHOTOMY**: val_geom_camber_cruise **-43.71% MASSIVE WIN** ✅ / val_geom_camber_rc **+24.20% catastrophic** ❌ / val_single_in_dist +11.59% / val_re_rand -0.96%. **Cross-confirmation from #2626 thorfinn per-channel-heads (cruise -44% / rc +21%) and #2627 askeladd surface-normal (cruise -45% / rc +27%)** — three independent experiments showed parallel cruise-WIN/rc-LOSS splits. **🔑 KEY FINDING: rc camber-AMPLITUDE (ch15, per fern's empirical correction) is an EXTRAPOLATION problem (rc test values 0.778, 0.889 lie outside train cluster); cruise is INTERPOLATION (within train).** Implication: **input-side shape augmentation must be channel-selective** — augment only channels NOT held out in the test splits, never augment along the OOD axis itself. Full-channel jitter CLOSED; channel-selective approach (#2662 fern, mask **ch15** per corrected hypothesis) tests the structural insight.
- **surface-normal-volume-feature** (#2627): +8.34% val / +6.55% test. THIRD independent confirmation of the structural cruise-WIN / rc-LOSS dichotomy: val_geom_camber_cruise -44.8% ✅ / val_geom_camber_rc +27.2% ❌ / val_re_rand +35.0% ❌ (worst). Student's analysis: (a) volume nodes inherit normal from nearest surface point → wake/far-field "ghost orientation" pollution; (b) model learns normal→flow mapping conditional on cruise-regime priors that doesn't generalize; (c) val_re_rand +35% indicates normals used as memorization channel conflated with Re-specific flow features. **Per-node normal feature (with volume-node inheritance) CLOSED.** Reassigned to #2671 surface-only-normal-feature (zero normals on volume nodes — isolates the physical signal at surface elements and removes the pollution side-channel; will reveal whether the cruise/rc dichotomy is fully structural or partly pollution-driven).
- **surface-normal-auxiliary-output-head** (#2660): +0.89% val / +2.62% test vs #2011 (and +3.13% val vs new baseline #2650). 4TH cruise-WIN / rc-LOSS dichotomy: cruise -45.4% / rc +22.7% / re_rand +19%. Aux objective itself works (aux_loss 0.062, pred_mag 0.980). Student's analysis: volume-node target inheritance imposes a 2-dim representation constraint on EVERY volume node, locally inconsistent with the bulk-flow encoding the trunk would otherwise prefer at high-Re/high-magnitude regimes. Reassigned to #2688 surface-only-aux-normal-head (student's follow-up #1).
- **sample-level-focal-mae-gamma1** (#2659): +13.4% val vs #2011 (+15.9% vs new baseline #2650). 5TH dichotomy confirmation: cruise -34% / rc +31% / re_rand +34%. **Focal-MAE FAMILY FULLY CLOSED** at element-level γ=2 with clamp (#2622, +18.1%) AND sample-level γ=1 no clamp (#2659, +15.9%). Student's diagnosis: "'Hard sample within batch' is not 'rc-distribution sample'" — with batch=4, max-MAE sample is gradient noise; 800× sample-weight dynamic range collapsed effective batch to 1-2. Mechanism fundamentally incompatible with N=1499 / batch=4 / heavy-tailed pressure residuals. Reassigned to #2689 shape-bin-oversampling-m05 (data-side: oversample M≥0.5 train samples 3× to densify near rc OOD boundary).
- **per-channel-separate-heads-kaiming-h128 (re-run)** (#2626): +1.63% val vs #2011 (+3.95% vs new baseline #2650). 6TH dichotomy confirmation: cruise -42.7% / rc +21.2%. Kaiming init + h128 fixed under-convergence (~0.9 MAE faster than zero-init/h64 at every epoch) BUT dichotomy preserved — 42.79 vs 42.87 on rc means the per-channel projection rc regression is **structural not convergence-related**. Student's diagnosis ratified: "specialization problem: independent per-channel projections lose the implicit cross-channel coupling (Ux/Uy/p coupled at sharp leading-edge gradients in rc samples)." **Per-channel-heads axis FULLY CLOSED across 2 arms.** Reassigned to #2690 re-conditional-output-bias (4th Re-conditioning hook, extends the only known winning axis).
- **recondln-t-max35** (#2678): +7.09% val vs new baseline #2650. **T_max>28 axis FULLY CLOSED** (28 winner, 35 +7.1%, 40 +11.4% — clean U-shape). Critical diagnostic: ep28 ticked UP (30.24→30.36) at residual LR 1.32e-4 vs #2650's monotone-descent to eta_min=1e-5. Direct evidence the **eta_min FLOOR (not residual LR) was doing the work in #2650's last 3 epochs**. T_max=28 ≈ wall-cap-epochs is the principled setting; extending T_max delays the floor regime past the wall, removing the refinement entirely. val_single_in_dist took biggest hit (+16%) — high-LR tail particularly damages in-dist fitting precision. Reassigned to #2699 re-conditional-layerscale (5th Re-conditioning hook, only proven winning direction).
- **surface-only-normal-feature** (#2671): +7.02% val vs new baseline #2650 (+4.67% vs old #2011). **Pollution side-channel was REAL and big**: val_re_rand recovered from +35% (#2627 all-nodes) → +5.14% here, -8.5pp gap closure. Cruise gain genuine but small (-2.44% val, -2.62% test) — surface normals do carry physical info at viscous walls. **BUT rc regression PERSISTS at +7.0% even without volume pollution** → cruise/rc dichotomy is **partly structural, not purely pollution-driven**. Dichotomy collapse from -45%/+27% (#2627) → -2.4%/+7.0% (#2671): most of #2627's drama was volume-pollution amplifying both effects. **Surface-only INPUT-feature path now CLOSED** — net negative despite mechanism validation. Reassigned to #2702 SWA (optimizer-level flat-minima, fundamentally different angle).
- **naca-jitter-ch16-17-only-sigma02 (mask ch15)** (#2662): +7.04% val vs new baseline #2650. **Establishes ch15 was the ENTIRE active axis in #2625** — symmetric jitter on ch16+ch17 alone is near-noop (val_avg drift +1.6% on cruise, +5.6% on rc, +4.2% on re_rand). Both the -43.7% cruise win AND +24.2% rc regression of #2625 evaporated together when ch15 was masked. **ch15 simultaneously serves as cruise interpolation axis AND rc OOD extrapolation axis** with opposite effects — confirming the structural dichotomy mechanism. Symmetric ch15 jitter densifies inward (helps cruise interpolation, hurts rc extrapolation). Reassigned to #2703 fern asymmetric-ch15-jitter-sigma05 (one-sided positive: PUSH outward toward rc OOD values rather than densify inward).

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

**Round 23:41 structural insight — extrapolation vs interpolation OOD axes:** The dramatic dichotomy in #2625 (NACA jitter, cruise -43.7% / rc +24.2%) and #2626 (per-channel-heads, cruise -44% / rc +21%) reveals that:

1. **cruise camber-shape is INTERPOLATION** — augmenting along the shape axes (channels 15-17) helps because the test distribution lies WITHIN the train distribution. Any generic shape-smoothing works (jitter, capacity expansion at the head level).

2. **rc camber-position is EXTRAPOLATION** — the rc test split is held out specifically on channel 16 (camber-position). Its test values lie OUTSIDE the train distribution on that axis. Augmenting along channel 16 with small σ EXPANDS the train distribution INWARD (densifying in-dist) but does NOT reach rc test values — so the relative gap widens. Capacity expansion at the head level overfits to the in-dist regime, hurting rc.

**Implication for the next wave of experiments**:
- **Channel-selective augmentation** (#2662 fern): mask ch 16, jitter only ch 15+17. Should preserve cruise win without rc regression.
- **Targeted OOD extrapolation strategies for rc** (next-tier ideas):
  - Boundary-aware extrapolation augmentation: scale channel 16 toward extreme values (not random jitter — directional shift)
  - Domain randomization on channel 16 BEYOND train range (controlled extrapolation)
  - Test-time adaptation: gradient-based update on a few rc-like adaptation samples
  - Equivariance/symmetry along channel 16 (if any geometric symmetry maps the OOD to in-dist)
  - Sample-reweighting at training time: upweight train samples nearest to the rc boundary on ch 16

**Next plateau-break candidates (Round 23:41+)**:
- **Channel-selective augmentation** is now the highest-priority axis (proven structural finding).
- **Multi-task auxiliary outputs** (#2660): force trunk to develop normal-aware shape representations — different from input-feature pathway.
- **Sample-level loss reweighting** (#2659): per-sample focal smoother than per-element.
- **LN-affine Re-conditioning** (#2650): bounded modulation regime not yet tested.
- **Surface-normal input feature** (#2627): orthogonal 1st-derivative signal to curvature/SDF.

Wall-cap-friendly principle still binding: every assignment is either zero-compute (input feature, loss reweighting, head replacement) or near-zero. NO 2× per-step regularizers; SAM-family axis remains closed.
