# SENPAI Research State

- **Date:** 2026-05-17 04:10 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Session Progress — 3 Merges, Multiple Nulls Closed

| Time | PR | Val Δ | New val_avg |
|---|---|---|---|
| 21:07 | **#3980 Lion + clip=0.25** | −3.48% vs SF-AdamW | **63.336** |
| 21:18 | **#4038 SF-AdamW lr=2e-3** | −13.5% vs Lion | **54.769** |
| 01:00 | **#4157 SF-AdamW lr=3e-3** | −4.59% vs 54.769 | **52.258** |

**Current best: val_avg=52.258 / test 3-split=51.206** (SF-AdamW lr=3e-3, 2026-05-17 01:00 UTC)

## LR Axis Closed (04:00 UTC)

Edward's #4246 LR extension confirmed: **lr=3e-3 is the sharp peak on both sides**. Higher LRs (+5.3% at 4e-3, +16% at 7e-3) show classic high-LR late-stage overshoot from ep4 onward. Full LR profile:

| lr | val_avg |
|---:|---:|
| 5e-4 | 65.618 |
| 2e-3 | 54.769 |
| **3e-3** | **52.258** |
| 4e-3 | 55.025 |
| 5e-3 | 55.029 |
| 7e-3 | 60.646 |

**Clip-rate insight**: clip engaged on ~98% of steps at lr=3e-3 (grad_norm p99 ≈ 16-22 >> threshold 1.0). Motivated clip threshold re-test at canonical.

## Width Axis Closed (04:00 UTC)

Askeladd's #4225 n_hidden sweep at lr=2e-3 (stale): **step-count loss dominates capacity efficiency at 30-min budget**. Wider arms are more efficient per epoch but get fewer epochs. No val/test consistent winner. Width is budget-saturated.

Key finding: sec/epoch ∝ n_hidden linearly (96.9s → 141.5s for h96→h192). Expect same step-count trade-off for n_layers, slice_num, mlp_ratio, n_head sweeps.

## Additional Closures Since Last Summary

- **#4081 nezuko** FiLM head width: CLOSED (9.5h zero-progress, stale lr=5e-4). Reassigned to Fourier features.
- **#4114 thorfinn** batch size: bs=4 fixed point (step-count dominates gradient-CV benefit)
- **#4019 alphonse** clip×EMA R2: EMA-off 0.43% below gate; clip×LR is the key interaction
- **#4208 fern** dropout: model is under-fit (train/val gap ≈ 0.001), dropout wrong direction

## Current Canonical Stack

```bash
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 3e-3
```

**val_avg/mae_surf_p: 52.258** | test 3-split: 51.206

## In-Flight Experiments (~04:10 UTC)

| Student | PR | Hypothesis | Stack | Priority |
|---------|----|----|----|----|
| ⭐ **edward** | **#4350** | **Clip threshold {0.5, 1.0, 1.5, 2.0} at lr=3e-3** | SF-AdamW lr=3e-3 + --seed 1 | **HIGH — 98% clip-rate at canonical; loosening may recover late-stage convergence; closes clip×LR surface from #4019** |
| ⭐ **frieren** | **#4248** | **n_layers depth sweep: {3, 4, 5, 7}** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — primary architecture axis, never swept** |
| ⭐ **thorfinn** | **#4303** | **slice_num sweep: {32, 64, 96, 128}** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — third primary Transolver architecture axis** |
| ⭐ **alphonse** | **#4317** | **SF-AdamW betas 2×2: (beta1, beta2) ∈ {0.9, 0.95}×{0.99, 0.999}** | SF-AdamW lr=3e-3 + --seed 1 | **MED-HIGH — optimizer-internal axis never swept** |
| ⭐ **askeladd** | **#4351** | **n_head sweep: {2, 4, 8} at lr=3e-3** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — final primary architecture axis; completes {n_hidden, n_layers, slice_num, mlp_ratio, n_head} family** |
| ⭐ **tanjiro** | **#4207** | **surf_weight R2: {5, 8, 10, 15}** | SF-AdamW lr=3e-3 + --seed 1 | **HIGH — R1 paired Δ ≥1.86%; R2 at canonical resolves direction** |
| ⭐ **fern** | **#4339** | **mlp_ratio sweep: {1, 2, 4, 6}** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — 4th primary architecture axis; under-fit regime confirmed by #4208** |
| ⭐ **nezuko** | **#4353** | **Fourier feature coordinate encoding: {raw, 16f-σ1, 32f-σ10, 64f-σ10}** | SF-AdamW lr=3e-3 + --seed 1 | **HIGH — preprocessing axis untouched; Tancik et al. 2020 strong prior on coord-regression tasks; orthogonal to all architecture sweeps** |

## Merged Winners (Chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP | −8.98% | 101.519 |
| #3289 | Cosine T_max=15 | −10.3% | 100.059 (fp32) |
| #3126 | EMA decay=0.999 | −1.06% | 96.464 |
| #3122 | FiLM conditioning | −4.00% | 92.606 |
| #3584 | Two-shot FiLM | −3.05% | 89.784 |
| #3511 | Grad clip=1.0 | −9.05% | 81.660 |
| #3906 | Clip=0.25 | −3.42% | 80.893 |
| #3594 | SF-AdamW (lr=5e-4) | −16.80% paired | 65.618 |
| #3980 | Lion + clip=0.25 | −3.48% abs | 63.336 |
| #4038 | SF-AdamW lr=2e-3 | −13.5% vs Lion | 54.769 |
| **#4157** | **SF-AdamW lr=3e-3** | **−4.59% abs** | **52.258** |

## Priority Watch (Next Results to Land)

1. **frieren n_layers #4248** — depth axis; if winner still descending at cap → scale test
2. **thorfinn slice_num #4303** — PhysicsAttention physical-slice count
3. **alphonse SF betas #4317** — optimizer-internal 2×2 factorial
4. **edward clip #4350** — 98% clip-rate motivates loosening; primary suspect clip=1.5
5. **tanjiro surf_weight #4207 R2** — non-monotone per-split landscape; resolves direction at canonical
6. **fern mlp_ratio #4339** — completes capacity surface; BERT-default 4 vs current 2
7. **askeladd n_head #4351** — final primary architecture axis
8. **nezuko Fourier feats #4353** — first preprocessing axis; Tancik strong prior

## Falsified / Closed Hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #4051 | SF wd sweep {1e-4, 3e-4, 1e-3, 1e-2} | +2.46-11.99% all regress | Polyak+EMA saturate regularization role |
| #4003 | AdamW clip-R2 {0.05-0.25} | −0.07% noise floor | Direction-norm saturated at 0.25 |
| #4087 | SF warmup steps {100, 500, 1000, 2000} | Paper default 500 wins | Warmup axis exhausted at current budget |
| #4012 | Sobolev edge-gradient L1 | R1 noise; R2 +2.68% regression | Cross-stack null; mechanism doesn't fit loss landscape |
| #4113 | EMA decay sweep {0.99, 0.999, 0.9995, 0.9999} | Karras ramp dominates; noise band 3.46% | Ramp fully controls effective decay at 17-epoch budget |
| #4149 | Lion LR sweep {7.5e-5, 1.5e-4, 3e-4, 6e-4} | Best Lion +11.64% behind SF | SF "4× LR" does not transfer to Lion+cosine |
| #4144 | Lion+SF composition (3-way) | C1: +75.6%; C2: catastrophic divergence | Lion+SF mechanistically incompatible; Lion track fully exhausted |
| #4114 | Batch size sweep {4, 6, 8, 9} | bs=4 wins; larger batches +13-29% regression | Step-count loss dominates gradient-CV benefit; bs=4 is fixed point |
| #4019 | SF clip×EMA factorial R2 (lr=2e-3) | EMA-off 0.43% paired, below gate | EMA-off attenuates with LR; clip×LR interaction is real; canonical clip=1.0 |
| #4208 | Dropout sweep {0.0, 0.05, 0.10, 0.15} | All regress +1.95–2.74% paired | Model is under-fit (train/val gap ≈ 0.001); capacity axis is right direction |
| **#4225** | **n_hidden sweep {96, 128, 160, 192} at lr=2e-3** | **val/test invert; step-count loss dominates capacity gain** | **Width saturated at 30-min budget; sec/epoch ∝ n_hidden; n_head is next axis** |
| **#4246** | **SF-AdamW LR extension {3e-3, 4e-3, 5e-3, 7e-3}** | **All regress: +5.3% at 4e-3, +16% at 7e-3** | **LR axis definitively closed; peak is 3e-3; 98% clip-rate motivates clip loosening** |
| **#4081** | **FiLM head width (stale WIP, no commits 9.5h)** | **Zero progress; stale lr=5e-4** | **Closed for non-delivery; FiLM-head-width on backlog after primary capacity axes** |
