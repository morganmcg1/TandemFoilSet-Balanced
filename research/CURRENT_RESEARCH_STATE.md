# SENPAI Research State

- **Last updated**: 2026-05-13 21:45 UTC (Wave 17+ Iteration 5: MERGE #2519 tanjiro attn-temp-fixed-sharper √2 — 20th compound win (val −3.68%, test −4.38%, biggest single-experiment win in 18 merges); CLOSE 2 dead-ends (#2510 decoupled-LS-init Outcome C / #2514 reflection-TTA catastrophic +25%); SEND BACK 4 winners for rebase (#2517 Q-bias / #2518 β2=0.99 / #2511 input-gate / #2515 per-block-LS-lr — all above new baseline, need stack-test); ASSIGN 3 new (#2574 alphonse attn-temp-√3 / #2575 askeladd latent-mixup / #2576 tanjiro per-head-τ-multiplier))
- **Track**: `charlie-pai2g-24h-r4` — controlled 24h/48h Charlie-vs-Willow logging ablation. Each individual training run is capped at `SENPAI_TIMEOUT_MINUTES = 30`; host harness controls fleet runtime.
- **Branch**: `icml-appendix-charlie-pai2g-24h-r4`, branched off `icml-appendix-charlie`.
- **Logging**: local JSONL only. **No W&B / wandb experiment logging.**

## Most recent research direction from human researcher team

None received yet on this branch.

## Current best baseline (PR #2519 merged — fixed sharper attention τ=√2 × default, 20th compound win)

- `val_avg/mae_surf_p` = **56.1754** (e12; full stack: ReGLU + inner_dim=288 + learned-freqs no-WD 10× lr + LayerScale γ no-WD 10× lr init=0.1 + LR warmup + surf-ch-weight [0.5,0.5,2.0] + Fourier L=6 learnable + grad-clip-25 + cosine-T_max-14 + L1 + stoch-depth + **fixed attention scale = 1/√(d_head/2) ≈ √2 × default**)
- `test_avg/mae_surf_p` (4-split, NaN-safe) = **48.7149**
- Per-split val: single_in_dist=66.511 / camber_rc=68.819 / camber_cruise=34.782 / re_rand=54.590
- Per-split test: single_in_dist=57.795 / camber_rc=63.594 / camber_cruise=28.422 / re_rand=45.048
- Δ vs prior #2475 baseline (58.3244 / 50.9438): **−3.68% val**, **−4.38% test** (test gain > val gain — strong generalization signal; biggest single-experiment win in 18 merges)
- **Mechanism — fixed scalar captures gain that learnable γ couldn't reach**: #2488 RMSNorm-Q/K-γ activated (γ moved on all 5 blocks) but per-channel std/mean peaked at only 33% (below 50% diversification threshold). The optimizer in 12 epochs cannot diversify a 1280-param per-channel γ enough to manifest the sharpening that a single fixed scalar produces directly. The fixed-scalar formulation works *because* it eliminates degrees of freedom that the optimizer couldn't allocate cleanly.
- **All 8 splits improve uniformly**: not a single-split artifact; sharper attention is the right direction across in-distribution AND OOD.
- **Compound progress**: 100.957 → **56.1754** = **−44.36% over 20 merges**
- **n_params**: **892,637** (unchanged)

## Current research focus

**Wave 17+ Iteration 5 — Sharpening attractor pushed; OOD-side regularization rebooted.**

The #2519 sharper-attention win (val −3.68%, test −4.38%, ALL 8 splits improve) is a turning point. After 18 merges where each win was 0.5-1.5% range, a single-line `scale = 1/√(d_head/2)` change produced the biggest single experiment win since #2370 (learned freqs). This validates two things at once:

1. **Direction**: sharper attention is preferred — confirms #2488 RMSNorm-Q/K-γ's "wants slight sharpening" finding now stripped to fixed cost.
2. **Mechanism**: fixed scalar > learnable γ when the optimizer can't diversify enough params in 12 epochs. Future work should prefer fixed structural changes over learnable per-channel mechanisms when DoF budget is tight.

**Twin frontiers now active**:

**A. Sharpening-attractor exploration (where does it stop?)**:
- `attn-temp-sqrt3` (alphonse #2574 NEW): even sharper τ=√3 × default (1.732×); tests if the attractor is past √2
- `per-head-tau-multiplier` (tanjiro #2576 NEW): 20 params total (4 heads × 5 blocks) on top of √2; tests per-head fine-tuning of sharpening

**B. OOD-side regularization (5 winners in iter-4 all improved single_in_dist; OOD-side plateau)**:
- `latent-mixup` (askeladd #2575 NEW): mixup at slice tokens, α=0.2, prob=0.5 — Manifold Mixup pattern, targeting camber_rc and camber_cruise improvement

**C. Pre-#2519 winners re-testing on new baseline** (4 PRs rebasing — orthogonal mechanism check):
- All four (edward #2517 Q-bias, nezuko #2518 β2=0.99, thorfinn #2511 input-gate, fern #2515 per-block-LS-lr) had val_avg between 56.69 and 58.09 — above the new 56.1754 baseline. Mechanism stack-test required.

**Mechanism findings preserved from earlier iterations**:
- **Anti-correlated init under no-WD 10× lr**: attn γ→sparse attractor (0.04-0.08), mlp γ→dense attractor (0.15-0.19), opposite endpoints despite identical opt treatment. The no-WD 10× lr group lives in a flat loss landscape where init bias compounds rather than relaxes.
- **Trajectory matters more than endpoint** (#2510 closure): starting LayerScale γ AT the attractor produced +5.10% regression. The drift IS the optimization — skipping it loses coupling with other training signals.
- **Y-reflection is NOT a valid prior** (#2514 closure): +25.13% catastrophic regression on TTA. The model has internalized non-symmetric task-relevant structure. Blocks all "y-symmetry as regularizer" hypotheses going forward.

**Wave 17+ Iteration 5 active threads (8/8 students busy):**

| Student | PR | Slug | Hypothesis | Status |
|---------|----|----|---------|--------|
| alphonse | #2574 | attn-temp-sqrt3 | Even-sharper τ=√3 × default (1.732×); continuation of #2519 attractor exploration | NEW |
| askeladd | #2575 | latent-mixup | Latent-space mixup at slice tokens, α=0.2, prob=0.5; OOD-targeted regularization | NEW |
| tanjiro | #2576 | per-head-tau-multiplier | Per-head learnable τ multiplier on top of #2519's √2 baseline (20 params no-WD 10× lr) | NEW |
| edward | #2517 | q-projection-bias | Q-projection learnable bias (640 params no-WD 10× lr); was pre-#2519 win at 56.69, now rebasing | REBASING |
| nezuko | #2518 | adamw-beta2-0.99 | AdamW β2=0.99; was pre-#2519 win at 57.18, now rebasing | REBASING |
| thorfinn | #2511 | input-feature-gate | Per-feature input gate γ_input (44 params no-WD 10× lr); pre-#2519 win at 57.70, now rebasing | REBASING |
| fern | #2515 | layerscale-per-block-lr | Per-block LayerScale lr scaling (5×→15×); pre-#2519 small win at 58.09, now rebasing | REBASING |
| frieren | #2513 | stoch-depth-deep-block-concentrated | Concentrate stoch-depth in deeper blocks `[0,0,0.05,0.10,0.15]` | STALE (no activity since assign — investigate next iter) |

## Key findings from Wave 13/14/15/16/17

**Gate activation axis (FULLY CLOSED at ReLU):**
- SiLU → GELU: **−4.75% val, −2.21% test** (14th compound win)
- GELU → ReLU: **−1.92% val, −4.07% test** (15th compound win)
- ReLU → Squared ReLU: **+5.19%** (CLOSED) — catastrophic
- ReLU → AbsGLU: **+10.35%** (#2385 CLOSED) — bidirectional gate destabilizes; one-sided dead zone is essential
- **Gate axis closed at ReGLU = max(0,x)**; triple-confirmed on negative side (AbsGLU, squared-ReLU both fail)

**Capacity axis (CLOSED):**
- inner_dim=256 (baseline) → 288 (MERGED, 16th win) → 320 (CLOSED #2386: +6.0%, overfit despite full budget)
- **inner_dim axis closed at 288**; bias-variance frontier confirmed between 288 and 320
- n_hidden: closed at 128 (#2371 +19.18%, quadratic scaling under 30-min budget)

**Encoder / Fourier axis:**
- Fixed dyadic freqs: closed at L=6
- Learnable freqs (#2370 MERGED, 17th win): −3.73% — bottom 3 freqs adapted strongly, top 3 gradient-limited
- **Wave 17 follow-ups**: freq init at equilibrium (frieren), 50× lr to unlock top freqs (thorfinn)

**Optimizer-group insight (Wave 17 new axis — refined after #2437):**
- Original framing: "scale/frequency parameters are systematically under-trained at default WD=1e-4 + lr=5e-4"
- **Refined framing (post-#2437)**: the no-WD + high-lr treatment WORKS for **additive scale params** (freqs, LayerScale γ — multiply activations) but FAILS for **softmax-internal scale params** (slice temperature, QK temperature — govern combinatorial routing)
- Freq insight (#2370 MERGED): +17th win (−3.73%)
- Slice-temp at 10× lr (#2437 CLOSED Outcome C): +4.81%; block 0 collapsed uniformly to 0.185, destabilizing token-to-slice routing
- Still testing: LayerScale γ at 10× lr (fern #2436 — additive scale, should win), QK temperature at no-WD + init=0 (tanjiro #2427 — softmax-internal but with proper init), norm/bias at default lr no-WD (nezuko #2465 — broad standard-transformer recipe)

**Sampling axis (CLOSED for camber_rc upsampling):**
- OOD upsampling (#2391): +8.22% — camber_rc bottleneck is geometric extrapolation, NOT data density; upsampling collapses in-dist

## Permanently closed axes (do not re-test)

| Axis | Best setting | Closure reason |
|---|---|---|
| L1 vs MSE | L1 | First merged win |
| Stoch-depth single-knob | [0,0.025,0.05,0.075,0.1] | V-shape; 0.10 optimum |
| Cosine T_max | T_max=14 (per-batch) | #2308: T_max=12 +3.24% |
| LR warmup | epoch-1 linear ramp | Merged |
| Grad-clip | max_norm=25 | {1.0,10,25,50} bracket complete |
| Fourier L (fixed) | L=6 dyadic | L=8 plateau; now learned, see #2370 |
| LayerScale init (symmetric) | γ_l=0.025 | Sweep complete; asymmetric test in #2414 |
| Surf-ch-weight | [0.5,0.5,2.0] | 4× p:v ratio optimum |
| n_head | 4 | n_head=8 +7.81%, n_head=2 +1.24% |
| Normalization | LayerNorm + β | RMSNorm +20.2% catastrophic |
| Depth | n_layers=5 | n_layers=6 +5.43%, compute-bound |
| Slice_num | 64 | slice_num=96 +8.82%, dilution |
| LR axis | lr=5e-4 | 3e-4=+5.95%, 5e-4=optimal, 7e-4=marginal |
| Gate: all except ReGLU | ReGLU (ReLU) | SiLU<GELU<ReLU<AbsGLU<SqReLU all tested; ReLU optimum |
| n_hidden (residual stream) | 128 | #2371: quadratic scaling, compute-bound |
| inner_dim | 288 | #2386: 320 over-fits; 256 under-fits |
| OOD domain upsampling (camber_rc) | equal weights | #2391: extrapolation gap, not density gap |
| EMA weights | off | Fights Fourier high-freq sharpening |
| Per-sample scalar Fourier | concat | #2286 class falsified |
| Hybrid dyadic+RFF σ=1.0 | dyadic L=6 | #2309 redundant low-freq overlap |
| Attn/MLP dropout | off | Stoch-depth redundancy + compute tax |
| Coord jitter | off | std=0.002/0.005 direction-inverted |
| Slice temperature at 10× lr | n/a (don't raise lr) | #2437 CLOSED Outcome C: +4.81%; softmax-internal routing destabilized |
| Learned freqs at 50× lr | n/a (lr ≥ 10× is plenty) | #2435 CLOSED: top freqs are gradient-magnitude-limited, NOT lr-limited |
| Asymmetric LayerScale init (attn>mlp) | symmetric 0.025 | #2414 CLOSED Outcome C: +7.92% vs current; optimizer inverts the asymmetry (final mlp γ > attn γ in all 5 blocks) — the hypothesized direction was wrong |
| F.normalize-based QK-norm | not used | #2377 + #2427 BOTH CLOSED: v1 init confound + v2 magnitude collapse (Q/K unit-sphere destroys per-query magnitude variance needed for attention sharpening); RMSNorm variant tested in #2488 |
| Freqs-xy-separate | unified 6 freqs | #2469 CLOSED Outcome B: directional asymmetry IS real at low freqs (37% asymmetry at freq[1]) but top freqs [8,16,32] stayed pinned in both directions; **top-freq gradient-pin NOT caused by directional cancellation** (mesh-scale aliasing remains the leading hypothesis) |
| Freq-init at equilibrium | dyadic init | #2434 CLOSED Outcome C +4.16%; equilibrium init OVERSHOOTS rather than relaxes — same anti-correlated-init pattern as #2475 LayerScale γ; the no-WD 10× lr group has a flat landscape where init bias compounds |
| FiLM with zero-init γ/β heads | no FiLM | #2453 CLOSED Outcome C +4.45%; FiLM IS active but generalizes in wrong direction — helps single_in_dist (−4.86%), hurts all OOD splits (camber_cruise +16.46%); zero-init heads can't build robust per-cond representations in 12 epochs |
| SWA (last-N epochs averaging) | no SWA | #2476 CLOSED Outcome C +7.79%; cosine T_max=14 + 30-min cap means val_avg drops 9.7% in single epoch (e11→e12); model still mid-descent at training cut-off; no plateau to average over |
| Hybrid fixed RFF + learned freqs | learned freqs only | #2441 CLOSED Outcome D +5.66%; RFF and learned-freqs SHARE low-frequency information rather than orthogonal coverage; camber_cruise regression is smoking gun (both encoders had won there individually); future Fourier work should refine single encoding (per-block/head/channel) not ensemble |
| Bias + norm γ/β + temp blanket no-WD | bias + norm γ/β + temp use default WD | #2465 CLOSED Outcome C +5.16%; bias-WD removal is the regression source (biases drift unbounded without WD anchor); LayerScale γ already provides per-channel offset, so bias-no-WD adds redundant competing path; isolated norm-γ/β-only test not yet done |
| RMSNorm-Q/K with learnable γ | no QK-norm | #2488 CLOSED Outcome B (γ activates but std/mean peaks 33%, below 50% threshold for diversification); competes with LayerScale γ in the residual; Q/K-norm axis fully closed across both F.normalize (#2377/#2427 magnitude collapse) AND RMSNorm (#2488 redundant scaling) |
| LayerScale decoupled init at attractors (attn=0.025, mlp=0.1) | symmetric init=0.1 (#2475) | #2510 CLOSED Outcome C +5.10% val; trajectory matters more than endpoint — the no-WD 10× lr group's flat landscape means the drift IS the optimization, not just a path to a fixed endpoint. Combined with #2414 closure (asymmetric attn>mlp), the LayerScale init axis is fully closed across symmetric AND asymmetric variants. |
| Y-axis reflection (TTA, augmentation, equivariance) | no reflection | #2514 CLOSED catastrophic +25.13% val / +25.37% test; TTA-off shows training was nominal. The tandem-foil dataset is NOT y-symmetric in any task-relevant sense — the model has internalized geometric orientation and flow direction as features. Blocks all "y-symmetry as regularizer" hypotheses (Manifold-Mixup with y-flip, equivariant arch with y-mirror, data aug with y-flip). |

## Prioritized open research themes (Wave 17+)

**Newly active (assigned in this iteration)**:
1. **Decoupled LayerScale init** (alphonse #2510 NEW): `attn=0.025, mlp=0.1` — directly tests #2475's anti-correlated-attractor finding (does starting each path at its natural attractor avoid wrong-init reversal cost?)
2. **Per-block LayerScale lr scaling** (fern #2515 NEW): block 0→5× lr, block 4→15× lr — tests #2475's per-block std/mean heterogeneity (deep blocks have more variable γ)
3. **Per-feature input gate** (thorfinn #2511 NEW): 44 learnable params at input position in no-WD 10× lr group — extends additive-scale theme to a new architecture position
4. **Stoch-depth deep-concentrated** (frieren #2513 NEW): `[0,0,0.05,0.10,0.15]` vs current linear `[0,0.025,0.05,0.075,0.1]` — anti-redundancy theory; deeper blocks should tolerate more drop
5. **Reflection TTA** (askeladd #2514 NEW): y-axis reflection test-time augmentation at val/test eval; risk-free (no retraining, doubles eval cost only)

**Newly active (assigned in this iteration's second batch)**:
6. **Q-projection learnable bias** (edward #2517 NEW): 640 params no-WD 10× lr; additive-scale theme at attention's Q position; never tested
7. **AdamW β2=0.99** (nezuko #2518 NEW): single HP scan from default 0.999; ~10× faster second-moment forget; tests if our 3-group optimizer benefits from faster per-group adaptation
8. **Fixed sharper attention τ=√2** (tanjiro #2519 NEW): no learnable scale; directly tests #2488 RMSNorm-γ "wants mild sharpening" finding without the per-channel-γ overhead

**Future ideas (queued)**:
9. **All-param optimizer sweep**: if LayerScale + norm-bias both win, test joint no-WD config for all 1D params simultaneously
10. **Per-block learned freqs (30 params)**: natural escalation if direction-separation doesn't unlock — block-specific freq pools allow each block its own scale
11. **Slice temp init sweep (not lr sweep)**: if Wave 18 reopens this axis, try init ∈ {0.3, 0.7} at default lr — gradient signal in #2437 suggested sharper is preferred but lr was too aggressive
12. **Different conditioning architecture** (post-#2453): FiLM PER-BLOCK instead of at embedding; or explicit OOD regularization on conditioning head
13. **Auxiliary self-supervised loss**: predict reconstructed coords from internal representations (Kaggle-staple OOD-improver)
14. **Weight standardization on Linear layers**: standardize weight rows before forward pass (used in CIFAR/ImageNet recipes)
15. **Interpolation init sweep for LayerScale**: γ_init=0.05 (halfway between 0.025 and 0.1) — only tested 0.025 and 0.1, the curve shape between them is unknown
