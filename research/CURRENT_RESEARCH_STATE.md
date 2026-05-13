# SENPAI Research State

- 2026-05-13 09:13 — willow-pai2g-48h-r1, round 3 in progress. **CURRENT BEST: test=80.62 (PR #1980 grad-accum=2)**. Cumulative gain from PR #1391: 121.28 → 80.62 = −33.5%. Four CLOSED this cycle: #2050 (EMA decay=0.999, +29.9% mechanism mismatch), #2047 (n_hidden=224, +11.8% width-saturated), #1967 (slice_num=96, +11.3% capacity-up cost dominates), #1969 (decoupled-wd, +1.2% null). Four new round-3 assignments: #2115 (alphonse mesh-node-dropout), #2117 (fern EMA decay=0.95), #2118 (frieren per-axis Fourier Lx=8/Ly=4), #2121 (nezuko slice_num=48).
- No directives from human researcher team yet.

## Current baseline (PR #1980 merged — gradient accumulation accum=2)
**test_avg/mae_surf_p = 80.62** | val = 90.82
Config: bf16 + **batch_size=4** + **accumulation_steps=2** (eff_bs=8) + **Lion lr=1.5e-4** + Fourier L=8 + n_hidden=192, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2. W&B run: 6qxwtm0v.

Per-split: in_dist=82.23, rc=93.60, cruise=61.57, re_rand=85.06.

**Mechanism**: Gradient accumulation improves Lion's sign vote via tighter per-micro-batch padding on variable-length TandemFoilSet meshes. Free: no memory increase, same per-epoch time.

## Previous baselines
- PR #1395 (Lion optimizer): test=83.77 | Lion lr=1.5e-4, no accumulation
- PR #1387 (Fourier+wider): test=93.29 | AdamW lr=7e-4, space_dim=34, n_hidden=192
- PR #1361 (wider-192): test=99.69 (3-seed) | n_hidden=192, AdamW lr=7e-4
- PR #1591 (cosine-aligned): test=111.98 | n_hidden=128
- PR #1391 (bf16+batch-8): test=121.28 | n_hidden=128

## Round-3 status (most recent first)
| Student | PR | Hypothesis | Status | Result |
|---------|-----|-----------|--------|--------|
| alphonse | #2115 | mesh-node-dropout=0.1 | **wip** (new) | Input-side OOD regularizer for mesh-density holdouts. |
| fern | #2117 | ema-decay-095 | **wip** (new) | 14-step half-life EMA, follow-up to #2050. |
| frieren | #2118 | fourier-per-axis-L (Lx=8, Ly=4) | **wip** (new) | Anisotropic mesh Fourier basis, follow-up to #1887. |
| nezuko | #2121 | slice_num-48 | **wip** (new) | Reverse direction of #1967: free per-epoch budget. |
| askeladd | #2088 | lion-lr-2.1e-4-sqrt2 | **wip** | Lion lr 1.5e-4→2.1e-4 sqrt(2) scaling for eff_bs=8. |
| edward | #2010 | swiglu-activation (GELU→SiLU) | **wip** | Training in progress (GPU 87 GB at 100% util, ~iter 51). |
| tanjiro | #2090 | grad-norm-clip-5-on-lion-stack | **wip** | max_norm=5.0 on accumulated grad, rare-event tail stabilizer test. |
| thorfinn | #2030 | drop-path-stochastic-depth | **wip** | DropPath rate=0.1 linear schedule, structural regularizer. |
| alphonse | #2047 | n-hidden-224-rescaled-cosine | **CLOSED** ✗ | +11.8% (budget-truncated; width saturated at 30-min cap). |
| fern | #1969 | decoupled-weight-decay | **CLOSED** ✗ | +1.2% null. Per-split: OOD directionally contradicted. |
| frieren | #2050 | ema-weights-decay-0999 | **CLOSED** ✗ | +29.9%. Mechanism mismatch: rapid-descent regime ≠ stationary trajectory. |
| nezuko | #1967 | slice-num-96 | **CLOSED** ✗ | +11.3% (vs old 83.77 +7.17%). Cost dominates; cosine truncated. |

## Round-2 status (CLOSED — historical)
| Student | PR | Hypothesis | Result |
|---------|-----|-----------|--------|
| alphonse | #1359 | lr-warmup+3e-4 | +5.5%. Warmup redundant for sign-momentum. |
| alphonse | #1945 | n-hidden-256 | +28%. Budget mismatch (12/18 epochs). |
| askeladd | #1771 | schedule-realigned | T_max=14 worse than T_max=18. |
| askeladd | #1877 | lion-bs-8-sqrt2-lr | +6.5%. Step-count starvation. |
| askeladd | #1980 | gradient-accumulation | **MERGED** ✓ test **80.62** new best. |
| askeladd | #2009 | grad-accum-4 | +10.4%. Step starvation at eff_bs=16. |
| edward | #1643 | mlp-ratio-4 | +10%. Per-epoch cost +11% → 13 epochs. |
| edward | #1973 | cosine-eta-min | +2.94%. eta_min=lr/10 overshot refinement. |
| fern | #1796 | weight-decay-1e-3 | +0.8%. wd magnitude lever exhausted. |
| frieren | #1710 | surf-weight-5 | +13%. Dead end. |
| frieren | #1887 | fourier-L-16 | +4.6%. Frequency aliasing. |
| nezuko | #1862 | n-layers-6-fourier-wider | +14.7%. Depth dead. |
| tanjiro | #1798 | grad-norm-clip=1.0 | test=79.91 vs OLD AdamW (wrong baseline). Cannot merge. Mechanism: clip=1.0 fires 100% → AdamW≈Lion via normalization. |
| thorfinn | #1395 | lion-optimizer | **MERGED** ✓ test 83.77. |
| thorfinn | #1876 | n-head-8 | +25.4%. head_dim<32 + per-epoch cost. |
| thorfinn | #1971 | lion-beta2-0999 | +5.99%. Horizon exceeded training budget. |

## Key research findings so far
1. **Throughput matters at 30-min budget**: bf16+batch-8 → 17 epochs → round-1 win.
2. **Schedule alignment**: T_max=actual epochs → −7.67%. T_max=14 over-corrects.
3. **Width × schedule compounds**: n_hidden=192 → −10.97%.
4. **Fourier × width compounds**: NeRF L=8 → −6.42%. High-freq basis helps near-foil.
5. **Lion optimizer biggest lever**: sign-momentum → −15.97% (83.77 from 99.69).
6. **Lion opens memory budget**: AdamW ~94 GB → Lion ~43 GB at n_hidden=192 bs=4.
7. **Depth dead at all tested widths**: horizon-vs-depth tradeoff — per-epoch cost compresses budget. CLOSED permanently.
8. **Surf-weight lever CLOSED**: Default=10 is robust optimum.
9. **LR-warmup CLOSED for Lion**: Sign-momentum inherently stable.
10. **wd magnitude lever CLOSED**: wd=1e-3 under Lion flips per-split signs (OOD exhausted).
11. **n_head=8 CLOSED at n_hidden=192**: head_dim<32 + per-epoch cost. Revisit with n_hidden=256.
12. **mlp_ratio=4 CLOSED**: +11% per-epoch cost → undertraining.
13. **Batch-size lever CLOSED (direct)**: bs=8 = 2.1× fewer steps. Starvation > gradient quality.
14. **Gradient accumulation (accum=2) WINS**: −3.77%, 43 GB unchanged. Mechanism: tighter micro-batch padding reduces sign-vote noise for Lion. **New best: test=80.62**.
15. **LR floor (eta_min=lr/10) CLOSED**: Raises LR at epoch 14 by 75% — overshoots refinement window under truncated cosine.
16. **Lion beta2 horizon lever CLOSED**: beta2=0.999 horizon (~1000 steps) exceeds our 1170-1316 step training budget.
17. **Width scaling has empirical O(n_hidden^2.43) per-epoch cost** (revised from earlier 1.4 projection). n_hidden=256 broke the 30-min budget (153 s/epoch vs 96s at 192); n_hidden=224 forces epochs=12 = 33% budget handicap.
18. **Fourier ceiling lever CLOSED at L=8**: L=16 produces frequency aliasing on irregular CFD mesh (sparse-region Nyquist limit exceeded). Per-axis Fourier L (Lx=8, Ly=4) being tested in #2118.
19. **Gradient accumulation lever CLOSED at accum=2**: accum=4 (eff_bs=16) regresses +10.4% from step starvation. Gradient-quality benefit saturates at eff_bs=8.
20. **Grad-clip=1.0 on AdamW ≈ Lion via different mechanism (analytical finding)**: With raw grad norms 25-550, clip=1.0 firing 100% of batches turns AdamW into sign-of-gradient with per-param scaling. Confirms that **normalization** (not Lion's symbolic search) is what drives the optimizer-class win.
21. **Width lever CLOSED at 30-min budget**: n_hidden=224 #2047 +11.8% (12/12 epochs but val still descending), combined with #1945 256/12-of-18 epochs, conclusively saturates the width lever at this compute envelope. Empirical exponent ≈2.43 makes width>192 infeasible without a budget expansion.
22. **EMA decay=0.999 CLOSED (mechanism mismatch)**: At 18-epoch budget, training is in *rapid descent* throughout — there is no late-cosine stationary trajectory. EMA-0.999 lagged 3-4 epochs behind (diagnostic ratio ended at ~20% vs predicted 1-3%). Direction reversed: short-horizon decay=0.95 reassigned to fern #2117.
23. **Slice-num=96 lever CLOSED**: Capacity-up cost dominates at 30-min cap. Cost grew +16.6% per-epoch (vs predicted <8%), GPU memory doubled, cosine never reached LR floor (13/18 epochs). Direction reversed: slice_num=48 (capacity-down → free per-epoch budget) reassigned to nezuko #2121.
24. **Weight-decay lever CLOSED across both axes**: magnitude (wd=1e-3 #1796 +0.8%) and structure (decoupled-wd #1969 +1.2% null). Lion's per-step wd push is ~1.5e-8 → too small to matter for direct shrinkage. Effects flow through sign-momentum dynamics, which are insensitive to wd magnitude in our regime.

## Active hypotheses in-flight (round 3)
| PR | Student | Hypothesis | Status | Expected gain |
|---|---|---|---|---|
| #2010 | edward | SiLU activation (GELU→SiLU) | Training (87 GB / 100% util) | −0.5% to −2.5% |
| #2030 | thorfinn | DropPath rate=0.1 linear schedule | Starting | −0.5% to −2.5% |
| #2088 | askeladd | Lion lr=2.1e-4 (sqrt(2) scaling for eff_bs=8) | Starting | −0.5% to −2% |
| #2090 | tanjiro | Grad-clip max_norm=5.0 on Lion+grad-accum stack | Starting | −0.5% to −2% |
| #2115 | alphonse | Mesh-node dropout=0.1 (input-mesh OOD regularizer) | Starting | −0.5% to −3% |
| #2117 | fern | EMA decay=0.95 (14-step half-life, tracks descent) | Starting | −0.3% to −1.5% |
| #2118 | frieren | Per-axis Fourier (Lx=8, Ly=4 — anisotropic mesh basis) | Starting | −0.3% to −1.5% |
| #2121 | nezuko | slice_num=48 (capacity-down → free per-epoch budget) | Starting | −0.3% to −1.5% |

## Key open questions
1. **Does mesh-node dropout (#2115) improve geometry OOD splits (rc, cruise)?** Tests whether mesh-density invariance is learnable as a regularizer on irregular meshes.
2. **Does EMA decay=0.95 (#2117) plateau the diagnostic ratio at ~1-3%?** Critical mechanism check. If yes, the 0.95 horizon matches our training. If still drifting, EMA doesn't apply to short-horizon training at all.
3. **Does anisotropic Fourier (Lx=8, Ly=4) (#2118) recover ~half of the L=16 regression?** Tests whether y-axis high-freq channels were the aliasing source.
4. **Does slice_num=48 (#2121) buy 1-2 extra cosine epochs?** Tests budget-for-refinement tradeoff against capacity-down.
5. **Does grad-clip=5.0 on Lion stack compound (#2090)?** Tests rare-event tail clipping (~10-15% fire rate predicted).
6. **Does Lion lr=2.1e-4 (sqrt(2) scaling) beat lr=1.5e-4 (#2088)?** Tests Lion-paper sqrt(B) scaling on eff_bs=8.
7. **Does DropPath (#2030) compose with Lion+grad-accum?** Pure structural regularizer test.
8. **Does SiLU (#2010) beat GELU?** Free activation swap, well-validated in modern stacks.

## Plateau watch
After this round (8 in-flight experiments), we are entering territory where each remaining lever is small (~0.5-2% expected). Specifically the recent closures formed a *block of negative results across orthogonal hypothesis families*: regularization (wd, EMA-long, DropPath pending), capacity (width, slice-num=96), schedule (eta_min, T_max=14). If round 3 produces no winner > 0.5%, **plateau protocol triggers** — escalate strategy tier to:
- **Loss reformulation**: physics-informed loss (PDE residual penalty), focal loss for hard mesh regions, anisotropic surf_weight by split.
- **Architecture rethink**: per-block GLU variants, parallel attention+MLP (PaLM-style), normalization at different positions.
- **Data-side**: training mesh augmentation (rotation, scale, partial occlusion), curriculum on Reynolds number, oversampling hard splits.
- **Optimization-side**: SOAP/Shampoo (preconditioned), Lookahead wrapper around Lion, schedule-free.

## Next milestones (round 3)
- Mesh-node dropout (#2115) — input regularization for geometry OOD
- EMA decay=0.95 (#2117) — short-horizon trajectory smoother
- Per-axis Fourier Lx=8/Ly=4 (#2118) — anisotropic mesh basis
- slice_num=48 (#2121) — capacity-down for cosine refinement
- SiLU activation (#2010, in-flight) — orthogonal free swap
- DropPath rate=0.1 (#2030) — structural regularizer
- Lion lr sqrt(2) scaling (#2088) — most-anticipated optimizer retune
- Grad-clip=5.0 (#2090) — rare-event tail stabilizer
