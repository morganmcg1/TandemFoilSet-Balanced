# SENPAI Research State

- **Date:** 2026-04-28 09:50 UTC
- **Advisor branch:** `icml-appendix-willow-pai2d-r4`
- **Most recent human-team direction:** none received yet on this advisor branch
- **Current best:** PR #576 (nezuko H16 arcsinh × bf16+compile × FiLM × EMA) merged. `val_avg/mae_surf_p=59.31`, `test_avg/mae_surf_p=52.98`. See BASELINE.md for full details and recommended config.

## Active research focus

**Round 0 ongoing — five winners merged.** PR #344 (warmup+cosine+NaN fix), PR #404 (FiLM-on-Re + wd=5e-4), PR #442 (EMA decay=0.99), PR #343 (bf16+torch.compile), and now **PR #576 (arcsinh-on-pressure scale=500)** — a **super-additive compound** with bf16+compile that delivered −26.7% / −27.2% on top of #343. Cumulative round-0 progress vs original baseline: **~52% on val, ~56% on test.**

The arcsinh × bf16+compile super-additivity is the round's headline finding. Mechanism: arcsinh's heavy-tail compression makes the loss landscape easier to navigate, bf16+compile gives more steps to navigate it; the two reinforce. **First demonstration on this branch of super-additive mechanism stacking.**

The four mechanisms now stacked on the merged baseline:
1. **FiLM-on-Re (#404):** cross-Re distribution shift
2. **EMA decay=0.99 (#442):** late-training noise smoothing (small marginal effect at convergence; defensive)
3. **bf16 + torch.compile (#343):** 2.4× throughput (33-37 epochs in 30-min budget)
4. **arcsinh-on-pressure scale=500 (#576):** heavy-tail target compression (super-additive with #343)

## Current themes

1. **Scale-aware losses** — H1 (alphonse, on rebase) attacks heavy-tailed pressure from loss-space; H16 (nezuko, **MERGED**) attacked from target-space angle. H1 may compound with merged arcsinh.
2. **Throughput as a lever** — H6 (askeladd, **MERGED**), now further extended via H22 (thorfinn, reduce-overhead).
3. **Geometry-OOD generalization** — H9 (fern, surface-arc gradient penalty) targets geom-OOD specifically.
4. **Re-conditioning** — H11 merged with 1-D log(Re) FiLM. Richer conditioning (H14 cond5) closed at noise. H20 (frieren, **NEW**, Re-jitter regularization) tests Re-OOD specifically.
5. **Loss alignment with metric** — H3 (tanjiro, Huber on surface) targets MSE-vs-MAE mismatch.
6. **Optimization** — H17 (edward, LLDR) closed at noise. H19 (askeladd, Lion) and H21 (edward, **NEW**, no-decay-1d) test optimization-bucket hypotheses.
7. **Architectural scaling** — H18 (thorfinn, wider) closed at undertraining. H23 (nezuko, **NEW**, RMSNorm replacement) tests an architectural normalization swap.
8. **Robustness tooling** — defensive `nan_to_num`, `--seed` flag, bf16 fp32-fallback, `_raw_module()`, `arcsinh_p_scale` flag.

## Currently in flight

| PR | Student | Hypothesis | Bucket | Predicted Δ | Status |
|----|---------|------------|--------|-------------|--------|
| #342 | alphonse | H1: per-sample y-std loss normalization | Loss reformulation | -8% to -18% | wip (sent back round 1 — needs rebase + sw sweep on merged schedule; first round Run B at sw=5 gave clean −7.9% val on apples-to-apples pre-merge baseline) |
| #348 | tanjiro | H3: Smooth L1 (Huber) on surface pressure | Loss reformulation | -2% to -6% | wip |
| #468 | fern | H9: surface-arc pressure-gradient penalty | Physics-aware | -2% to -5% | wip |
| #650 | askeladd | H19: Lion optimizer | Optimization | -1% to -5% | wip |
| #654 | frieren | H20: random Re-jitter augmentation on log(Re) input | Regularization | -2% to -5% | wip |
| #662 | edward | H21: exclude 1-D parameters from weight decay | Optimization | -0.5% to -2% | wip |
| #693 | thorfinn | H22: torch.compile mode=reduce-overhead with fixed N_max padding | Throughput | -2% to -5% | wip |
| #700 | nezuko | H23: RMSNorm replacement for LayerNorm | Architecture | -1% to -3% | wip |

## Resolved this round

| PR | Student | Hypothesis | Outcome | val_avg/mae_surf_p |
|----|---------|------------|---------|--------------------|
| #344 | edward | H2: warmup + corrected cosine | **merged** | 120.97 (Run C, –3.4% vs Run A) |
| #346 | frieren | H7: z-mirror augmentation | closed (strict regression) | +231% at p=1.0 |
| #349 | thorfinn | H8: slice_num scaling matrix | closed (regression vs baseline) | 148.65 (+23% vs baseline) |
| #345 | fern | H4: surface-only norm + distance feature | closed (cruise OOD structural regression) | 129.13 (+6.7% vs baseline) |
| #406 | frieren | H10: surf_weight ramp curriculum | closed (effect below seed-variance floor) | 122.90 (+1.6% vs baseline) |
| #490 | frieren | H13: stochastic depth (DropPath) | closed (B-vs-A signature matched but absolute below noise floor) | 120.57 (+1.0% vs baseline) |
| #347 | nezuko | H5: Fourier features (× FiLM in round 3) | closed (Fourier × FiLM antagonistic) | 129.49 (+8.5% vs PR #404 baseline) |
| #523 | edward | H14: 5-D FiLM conditioner | closed (B-C seed spread > 8%; Mean(B,C) − A ≈ 0) | mean 120.76 (+1.2% vs PR #404) |
| #404 | edward | H11: Re-conditional FiLM modulation | **merged** (after disentanglement) | **119.36** (−1.3% vs PR #344) |
| #442 | thorfinn | H12: EMA decay=0.99 × FiLM | **merged** (after compound test) | **val_ema=109.19** (−8.5% vs PR #404) |
| #343 | askeladd | H6: bf16+compile × FiLM × EMA | **merged** (after compound test) | **val=80.91** (−25.7% vs PR #442) |
| #561 | frieren | H15: test-time z-mirror augmentation (TTA) | closed (decisively rejected at +137% regression) | 283.11 (+137% vs PR #404) |
| #602 | edward | H17: layer-wise lr decay | closed (Mean(B,C) − A in [−2%, +2%] band) | 119.65 (mean B,C) |
| #611 | thorfinn | H18: wider Transolver | closed (under-convergence not capacity regression) | mean 132.30 (+21.2%) |
| #576 | nezuko | H16: arcsinh × bf16+compile × FiLM × EMA | **merged** (super-additive compound) | **val_ema=59.31** (−26.7% vs PR #343; **largest cumulative effect of round 0**) |

## Held in reserve / promising follow-ups

- **3-seed nail-down of the merged baseline (PR #576 Run E config)** — would give real error bars on the val=59.31 / test=52.98 floor. Useful for paper-confidence numbers.
- **Fine arcsinh scale sweep** (200, 350, 500, 700, 1000) — likely yields 1-3% additional. Was nezuko's #1 follow-up.
- **Compound H1 (alphonse) with merged arcsinh** — orthogonal mechanisms (loss-space vs target-space reweighting). Should compound.
- **Re-tune EMA for longer-budget regime** (decay=0.998) — may help post-#343 where raw converges well.
- **Re-tuned bs=8 with proper LR** — still untested.
- **Re-test wider model on post-#343 path** — H18 closed because of under-convergence; with bf16+compile's 2.4× throughput, wider would have 26 epochs vs the 9 it got.
- **Frieren's domain-conditional augmentation** — restrict mirroring to cruise-only with proper gap sign-flip.
- **Fern's C2-Lite ablation + multi-scale distance feature** — milder loss-rebalancing variants.
- **FiLM hidden=32** — halve FiLM head params.
- **Concat-Re instead of FiLM** — cheaper alternative.
- **Stack the next round-0 winner + the merged stack** — once H1, H3, H9, H19, H20, H21, H22, or H23 lands, run a combined-best PR.

## Open methodological note

Single-run noise floor on this branch is **~6–8% peak-to-peak** but at fixed config the seed-pair spread is much tighter (often 1-3%). The 3-cell pair-with-control protocol (Run A control + Run B canonical at seed=123 + Run C variance at seed=124) detects sub-3% effects when within-config variance is tight. Edward's H17 close used this protocol to detect a 1.25% test regression below the 6% noise floor.

The `--seed 123` reproducibility protocol has been demonstrated **eight times now** with clean 4-decimal baseline reproduction (PR #442 Run F, #523 Run A, #576 Run A, #343 Run G, #561 Run A, #602 Run A, #611 Run A, and PR #576 round 3 Run A indirectly). The seed-controlled comparison protocol is rock-solid at the program level.

## Potential next research directions (post round 0)

- **Once H1 (alphonse) lands,** combine with the merged arcsinh stack — both attack heavy-tailed pressure from different angles.
- **If H22 (thorfinn, reduce-overhead) lands,** revisit deeper / wider architectures with the additional throughput.
- **If H9 (fern, surface-arc gradient) lands,** combine with merged arcsinh — physics-aware regularization on top of target-space transformation.
- **If geometry-OOD splits remain stubborn,** escalate to graph/edge-aware mesh modules or coordinate-network heads.
- **If Re-OOD splits remain stubborn,** explore Re-conditional separate models or hierarchical heads.
- **Best-checkpoint multi-seed reproducibility** — at some point, lock in the best-known recipe and run a 5-seed reproducer to estimate variance for paper-ready numbers.
