# SENPAI Research State

- **Updated**: 2026-04-28 15:30 UTC
- **Branch**: `icml-appendix-willow-pai2e-r2`
- **Tag**: `willow-pai2e-r2`
- **Most recent human researcher direction**: none; no GitHub Issues open.
- **Lab**: 11 students, 1 GPU each (96 GB), 30 min wall-clock, 50 epochs cap.

## Current baseline

**PR #783 (fern, Huber δ=1.0) — merged 2026-04-28**
- `val_avg/mae_surf_p` = **75.93** at epoch 32/50 (timed out, still improving — significant headroom)
- Per-split val: single=85.84, rc=91.20, cruise=54.68, re_rand=71.99
- Per-split test (3 finite): single=79.35, rc=82.61, re_rand=64.29; cruise=NaN (bug)
- W&B: `2y1lj209`; Config: compound base + `--huber_delta 1.0 --surf_weight 10 --lr 5e-4`

## Current assignments (active WIP PRs)

| Student | PR | Hypothesis | Axis | Status |
|---------|----|-----------|------|--------|
| alphonse | #853 | Huber δ sweep: δ=0.5 and δ=2.0 on compound+Huber base | loss (δ tuning) | WIP (newly assigned) |
| frieren  | #854 | Huber + grad accum (accum_steps=2): double throughput, ~60 epochs in budget | training throughput | WIP (newly assigned) |
| fern     | #855 | Huber + surf_weight sweep: sw=5 and sw=20 vs baseline sw=10 | loss weighting | WIP (newly assigned) |
| askeladd | #821 | tooling: AMP/bf16 + batch_size=16 + NaN-safe eval | infrastructure | WIP |
| nezuko   | #785 | compound + n_hidden=192 (width on compressed base) | architecture (width) | WIP |
| tanjiro  | #786 | compound + RMSNorm (replace LayerNorm) | architecture (normalization) | WIP |
| edward   | #840 | compound + per-sample relative MAE loss | loss (scale heterogeneity) | WIP |
| thorfinn | #841 | compound + slice_num=4 extreme compression (sn4) | architecture (slice floor) | WIP |
| stark    | #842 | compound + SwiGLU param-matched h=168 | architecture (activation) | WIP |
| himmel   | #843 | compound + gradient norm clipping (max_norm sweep 0.5 / 1.0) | optimization (stability) | WIP |
| charlie  | #844 | compound + mlp_ratio=4 (FFN capacity at nh1) | architecture (MLP capacity) | WIP |

**Note on PRs #785, #786, #840–#844**: These were all assigned against the old compound anchor (96.80), not the new Huber baseline (75.93). When they report results, compare against 75.93. If any beat the new baseline they should still be merged.

## Tooling debt

- **Cruise-split NaN bug**: `test/test_geom_camber_cruise/mae_surf_p` returns NaN across all runs. Fix being implemented in PR #821 (askeladd). Until it lands, use `best_val_avg/mae_surf_p` as primary. Students should manually compute clean test_avg over 3 finite splits.
- **Throughput bias**: fp32 + batch=4 → ~56 s/epoch → only 32/50 epochs in 30 min. Huber was still improving at epoch 32. PR #821 (AMP/bf16 + batch=16) and PR #854 (grad accum) both attack this. When either lands, results will be significantly more converged.

## Current research focus

**Primary axis**: Build on the Huber δ=1.0 win. The 21.6% improvement came from a loss reformulation that addresses the core physics challenge (high-Re tail dominance over low-Re bulk). Round 2 explores the loss objective space more deeply:

1. **δ tuning** (#853) — find the optimal linearization threshold for this dataset's residual distribution
2. **Throughput** (#854) — recover the ~20 epochs of training the model is missing due to wall-clock timeout; grad accum doubles epoch capacity with zero VRAM cost
3. **Surface weighting** (#855) — after fixing the gradient distribution problem (Huber), tune the surface vs volume priority to push harder on the primary metric
4. **Legacy in-flight** (#785, #786, #840–#844) — original round-1 sweeps still running on compound base; may still yield interesting signals against 75.93

**Key open question**: How much of the 75.93 val is "undertrained model" vs "fundamental limit of this architecture+loss"? The Huber run was improving monotonically at timeout. PR #854 directly tests this — if grad accum drops val_avg to ~65–70, we know there's headroom and throughput is the bottleneck. If it stays near 75, the architecture is the limit.

## Ruled-out directions (do not repeat)

- Gaussian Fourier PE on (x,z) — closed PR #787, val_avg=100.12, decisive negative
- GeGLU activation in FFN — closed PR #782, val_avg=94.41 (param-matched), decisive negative
- SwiGLU bundled with Fourier PE — confounded; SwiGLU alone in PR #842
- FiLM conditioning — failed in prior round (confounded with Fourier PE)
- OneCycleLR full-schedule — closed PR #784 round 2, val=92.25; gradient-step-limited not LR-schedule-limited
- Fourier-based coordinate encoding in general: all Fourier-related runs regress
- Horizontal flip augmentation — rank 30 in prior senpai round
- Cross-attention decoder — rank 28 in prior senpai round

## Potential next research directions

- **Compound Huber+winners**: once positive results from #853/#855 arrive, layer them on the Huber base.
- **AdamW weight decay tuning**: current `weight_decay=1e-4` may be too strong for the compressed base at low capacity.
- **EMA weights for eval**: exponential moving average of model weights for test eval is cheap and often gives 1–3 pts improvement with no architecture change.
- **Domain-adaptive Huber**: different δ per training domain (cruise vs raceCar single vs raceCar tandem) — optimal threshold likely differs because per-domain residual distributions differ.
- **Per-sample relative-MAE loss** (PR #840 in flight): same motivation as Huber but normalizes per-sample rather than per-residual; may combine additively with Huber.
- **Curriculum learning**: start with single-foil (simplest domain) then introduce tandem, rather than balanced sampling from epoch 1.
- **LR schedule**: cosine warm restart (T_0=8 epochs, T_mult=2) — allows more gradient steps at moderate LR vs hard cosine anneal.
- **If plateau after round 2**: escalate to architecture changes — PerceiverIO-style cross-attention decoder, physics-constrained output layer (divergence-free velocity field), or graph attention network baseline.
