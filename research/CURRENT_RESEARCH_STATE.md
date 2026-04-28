# SENPAI Research State

- **Updated**: 2026-04-28 17:00 UTC
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
| alphonse | #853 | Huber δ sweep: δ=0.5 and δ=2.0 on compound+Huber base | loss (δ tuning) | WIP |
| frieren  | #854 | Huber + grad accum (accum_steps=2): double throughput, ~60 epochs in budget | training throughput | WIP |
| fern     | #855 | Huber + surf_weight sweep: sw=5 and sw=20 vs baseline sw=10 | loss weighting | WIP |
| askeladd | #821 | tooling: AMP/bf16 + batch_size=16 + NaN-safe eval | infrastructure | WIP |
| edward   | #840 | compound + per-sample relative MAE loss | loss (scale heterogeneity) | WIP |
| stark    | #842 | compound + SwiGLU param-matched h=168 | architecture (activation) | WIP |
| himmel   | #843 | compound + gradient norm clipping (max_norm sweep 0.5 / 1.0) | optimization (stability) | WIP |
| charlie  | #844 | compound + mlp_ratio=4 (FFN capacity at nh1) | architecture (MLP capacity) | WIP |
| thorfinn | NEW | — pending assign — | TBD | IDLE |
| tanjiro  | NEW | — pending assign — | TBD | IDLE |
| nezuko   | NEW | — pending assign — | TBD | IDLE |

**Note on PRs #840–#844**: Assigned against the old compound anchor (96.80), not the Huber baseline (75.93). When they report results, compare against 75.93.

## Recently closed (this review pass)

| PR | Student | Result | Verdict |
|----|---------|--------|---------|
| #841 | thorfinn | sn=4, val=98.25 → +29% vs Huber baseline | Closed — slice floor confirmed at sn=16 |
| #786 | tanjiro | RMSNorm, val=109.17 → +43.8% vs Huber baseline | Closed — mean-centering load-bearing in CFD |
| #785 | nezuko | n_hidden=192, val=119.40 @ epoch 18 only | Closed — throughput-bound, wait for AMP/bf16 |

## Tooling debt

- **Cruise-split NaN bug**: `test/test_geom_camber_cruise/mae_surf_p` returns NaN across all runs. Fix implemented by tanjiro in PR #786 (requested as standalone PR). Fix also being implemented in PR #821 (askeladd). Until it lands, use `best_val_avg/mae_surf_p` as primary; students manually compute clean test_avg over 3 finite splits.
- **Throughput bias**: fp32 + batch=4 → ~56 s/epoch → only ~32/50 epochs in 30 min. Huber and sn=4/sn=8 were still improving at timeout. PR #821 (AMP/bf16 + batch=16) and PR #854 (grad accum) both attack this. n_hidden=192 is explicitly blocked by this until PR #821 lands.

## Current research focus

**Primary axis**: Build on the Huber δ=1.0 win. The 21.6% improvement came from a loss reformulation addressing the core physics challenge (high-Re tail dominance over low-Re bulk). Round 2 explores the loss objective space more deeply:

1. **δ tuning** (#853) — find the optimal linearization threshold for this dataset's residual distribution
2. **Throughput** (#854) — recover the ~20 epochs of training the model is missing due to wall-clock timeout
3. **Surface weighting** (#855) — after fixing gradient distribution problem (Huber), tune surface vs volume priority
4. **Legacy in-flight** (#840–#844) — original round-1 sweeps on compound base; compare against 75.93 when they finish

**Key open question**: How much of the 75.93 val is "undertrained model" vs "fundamental limit of this architecture+loss"? PR #854 directly tests this. If grad accum drops val_avg to ~65–70, throughput is the bottleneck. If it stays near 75, the architecture is the limit.

## Settled facts from this round

- **Slice floor at sn=16**: sn=4 (val=98.25) and sn=8 (val=92.5) both regress. Do not go below sn=16.
- **Mean-centering is load-bearing**: RMSNorm (val=109.17) is a decisive negative. CFD's asymmetric pressure distributions require full LN at H=128.
- **Width scaling is throughput-blocked**: n_hidden=192 only reached 18 epochs. Re-visit after AMP/bf16 lands.
- **Huber loss is the key mechanism**: All architecture/normalization tweaks lose to the loss reformulation. Loss-first before architecture.

## Ruled-out directions (do not repeat)

- Gaussian Fourier PE on (x,z) — PR #787, val=100.12, decisive negative
- GeGLU activation in FFN — PR #782, val=94.41 (param-matched), decisive negative
- RMSNorm replacing LayerNorm — PR #786, val=109.17, decisive negative (mean-centering load-bearing)
- SwiGLU bundled with Fourier PE — confounded; SwiGLU alone in PR #842
- FiLM conditioning — failed in prior round (confounded with Fourier PE)
- OneCycleLR full-schedule — PR #784 round 2, val=92.25; gradient-step-limited not LR-schedule-limited
- Fourier-based coordinate encoding in general: all Fourier-related runs regress
- Horizontal flip augmentation — rank 30 in prior senpai round
- Cross-attention decoder — rank 28 in prior senpai round
- slice_num=4 — PR #841, val=98.25, slice floor confirmed at sn=16
- n_hidden=192 — throughput-blocked; re-test after AMP/bf16 (PR #821) lands

## Pending new assignments (thorfinn, tanjiro, nezuko)

Next experiments to assign (from the current hypothesis pool):

1. **AdamW weight decay tuning** — current `weight_decay=1e-4` may be too strong for the compressed base. Sweep wd=1e-5 and wd=0 on the Huber base. Low cost, high expected impact.
2. **EMA weights for eval** — exponential moving average of model weights at test eval. Cheap and often gives 1–3 pts improvement with zero architecture change.
3. **Domain-adaptive Huber δ** — different δ per training domain (cruise vs raceCar single vs raceCar tandem). Optimal threshold likely differs because per-domain residual distributions differ.
4. **Cosine warm restart LR** — CosineAnnealingWarmRestarts(T_0=8, T_mult=2) on Huber base. More gradient steps at moderate LR than hard anneal.
5. **Standalone evaluate_split NaN fix** — tanjiro's bug fix needs a dedicated PR. Tanjiro is best placed to submit it.
6. **Per-sample relative-MAE + Huber combo** — PR #840 (edward) tests relative-MAE alone; after it reports, try combining with Huber.
7. **n_hidden=192 + Huber** — revisit width scaling once AMP/bf16 lands and throughput is better.

## Potential longer-horizon research directions

- **Compound Huber+winners**: once positive results from #853/#855 arrive, layer them on the Huber base.
- **Curriculum learning**: start with single-foil (simplest domain) then introduce tandem, rather than balanced sampling from epoch 1.
- **PerceiverIO-style cross-attention decoder**: if current architecture plateaus at ~70.
- **Physics-constrained output layer**: divergence-free velocity field prior.
- **Graph attention network baseline**: compare against Transolver if Transolver plateaus.
