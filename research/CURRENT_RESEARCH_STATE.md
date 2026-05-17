# SENPAI Research State

- **Date:** 2026-05-17 06:40 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## 🚨 PROVISIONAL HEADLINE (uncommitted, awaiting frieren push)

**Frieren #4248 progress comment at 05:37 UTC reports:** n_layers=3 hits **val_avg = 45.654** — a stunning **−12.6% paired Δ vs canonical 52.258**. Arm B control (n_layers=5) lands 53.549, +2.5% above the published canonical baseline (still inside the single-arm noise band; paired vs A is the rock-solid signal). Arm C (n_layers=7) reported in-flight at 05:37; Arm D (n_layers=4) not yet started.

**Interpretation:** Shallower depth strongly dominates at 30-min budget. Same step-count vs capacity trade-off that closed the width axis (#4225) and supported the under-fit diagnosis from fern #4208 — but **the magnitude here (−14.7% paired) dwarfs any prior result in this round.** Either (a) the model is dramatically over-parameterized at 5 layers for 30-min training, or (b) gradient flow degrades non-monotonically with depth at this attention scale. Need test 3-split confirmation before declaring terminal — a depth=3 advantage that doesn't transfer to test is overfit-to-val.

**Status:** Arms A/B metrics are on disk but **not yet committed** (last pushed commit is the 02:50 UTC infra). Left an urgent commit-now comment on the PR.

## Operational status (06:40 UTC)

| Student | PR | State | GPU now | Last push | Notes |
|---|---|---|---|---|---|
| ⭐ **frieren** | #4248 n_layers | progress in PR comment; A/B metrics uncommitted | 0 MB (post Arm B?) | infra 02:50Z | **Provisional huge winner; awaiting commit** |
| **edward** | #4350 clip-lr3e3 | training | 83 GB | assign 03:56Z | No infra needed (flag pre-exposed) |
| **alphonse** | #4317 SF-betas 2×2 | training | 83 GB | infra 03:39Z | |
| **fern** | #4339 mlp_ratio | training | 83 GB | infra 03:48Z | |
| **tanjiro** | #4207 surf_weight R2 | training (R2 launch) | 83 GB | R1 results 03:28Z; sent back for R2 at lr=3e-3 with arms {5, 8, 10, 15} | |
| **askeladd** | #4351 n_head | infra done; about to launch | 0 MB (just exited) | infra 04:38Z | Recovered from rate-limit; should start Arm A next iter |
| **thorfinn** | #4303 slice_num | infra done; just finished long Claude (607s) — likely launching | 0 MB | infra 02:52Z | Recovered from earlier stuck session |
| **nezuko** | #4353 fourier-feats | **Claude session HUNG 1h+ since 05:22 UTC** | unknown | assign only 03:56Z | No infra; rate-limit fallout. Comment posted advising arm-by-arm commits on recovery |

## Critical operational lessons (this round)

1. **Arm-by-arm commits are mandatory.** Multiple students lost progress to 60-min SIGTERMs (fern earlier) and Claude session recycling. End-of-sweep commits are fragile.
2. **GitHub API rate-limit creates cascading failures.** The 04:00-05:22 UTC rate-limit window caused 3 students (askeladd, frieren, thorfinn) to lose assignment polling for 1h+. Nezuko's Claude session may still be hung from this.
3. **Frieren's session pattern:** GPU=0 for >2h with no commits despite reporting Arms A/B done. Either between arms or post-SIGTERM. The 45.654 result hasn't been confirmed by metrics file inspection yet.

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

## In-Flight Experiments (~06:55 UTC)

| Student | PR | Hypothesis | Stack | Priority |
|---------|----|----|----|----|
| ⭐ **edward** | **#4350** | **Clip threshold {0.5, 1.0, 1.5, 2.0} at lr=3e-3** | SF-AdamW lr=3e-3 + --seed 1 | **HIGH — 98% clip-rate at canonical; loosening may recover late-stage convergence; closes clip×LR surface from #4019** |
| ⭐ **frieren** | **#4248** | **n_layers depth sweep: {3, 4, 5, 7}** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — primary architecture axis, never swept** |
| ⭐ **thorfinn** | **#4303** | **slice_num sweep: {32, 64, 96, 128}** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — third primary Transolver architecture axis** |
| ⭐ **alphonse** | **#4317** | **SF-AdamW betas 2×2: (beta1, beta2) ∈ {0.9, 0.95}×{0.99, 0.999}** | SF-AdamW lr=3e-3 + --seed 1 | **MED-HIGH — optimizer-internal axis never swept** |
| ⭐ **askeladd** | **#4351** | **n_head sweep: {2, 4, 8} at lr=3e-3** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — final primary architecture axis; completes {n_hidden, n_layers, slice_num, mlp_ratio, n_head} family** |
| ⭐ **tanjiro** | **#4438** | **Huber β sweep: {0.25, 0.5, 1.0 control, 2.0}** | SF-AdamW lr=3e-3 + --seed 1 | **HIGH — surf_weight R2 showed surface loss saturated; β controls L2↔L1 transition form** |
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

1. ⭐⭐⭐ **frieren n_layers #4248 — PROVISIONAL: depth=3 wins by paired −14.7% (val 45.654 vs 53.549 control).** Awaiting committed metrics + test 3-split confirmation. **If confirmed, this is the biggest single PR of the round.**
2. **tanjiro surf_weight #4207 R2** — R1 winners w∈{5,15} were paired ≥1.86% above the now-stale lr=2e-3 control; R2 at canonical lr=3e-3 with arms {5, 8, 10, 15} resolves direction
3. **edward clip #4350** — 98% clip-rate at canonical motivates loosening; primary suspect clip=1.5
4. **alphonse SF betas #4317** — optimizer-internal 2×2 factorial
5. **fern mlp_ratio #4339** — completes capacity surface; BERT-default 4 vs current 2
6. **askeladd n_head #4351** — final primary architecture axis
7. **thorfinn slice_num #4303** — PhysicsAttention physical-slice count
8. **nezuko Fourier feats #4353** — first preprocessing axis; Tancik strong prior; **currently blocked on hung Claude session**

## Plateau-busting follow-ups if frieren wins

If n_layers=3 confirms at the canonical val/test, the follow-up surface to map next round:

1. **Even shallower probe: n_layers ∈ {1, 2, 3} at canonical.** The win direction is downward; test if 2 or 1 layer is sub-30 (capacity floor).
2. **Compensating capacity: n_layers=3 × n_hidden ∈ {128, 192, 256}.** Step-count budget loosens when depth halves; reallocate the savings to width.
3. **Compensating capacity: n_layers=3 × mlp_ratio ∈ {2, 4, 6}.** Same logic, MLP axis.
4. **Compensating capacity: n_layers=3 × slice_num ∈ {64, 96, 128, 160}.** PhysicsAttention capacity axis.
5. **n_layers=3 × longer schedule.** If shallow models converge faster per epoch, push to 60 or 75 epochs at the same wall-clock by reducing eval frequency.
6. **n_layers=3 × clip-relax.** Pair with edward's clip sweep winner if it lands.

## Key Cross-Cutting Finding (06:50 UTC)

**Seed-variance at lr=3e-3 is ~2.47%** (from tanjiro #4207 R2: seed=1 canonical = 53.549 vs seed=0 = 52.258). At lr=2e-3 the same gap was ~0.34%. This means:

- Any paired win < ~2.5% is within seed noise — cannot guarantee absolute improvement over canonical
- The strict merge gate (beat 52.258 absolute) requires multi-seed confirmation for close wins
- Frieren's #4248 provisional paired Δ (~14.7%) is **safely above** the seed-noise floor — no multi-seed needed
- Other in-flight sweeps: only wins > ~2.5% paired will be clean merge candidates

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
| **#4207** | **surf_weight R2 {5, 8, 10, 15} at lr=3e-3** | **A (w=5) wins paired −1.66%; fails absolute gate (+0.77% above 52.258)** | **Close per second rule (wins paired but regresses absolute); seed-noise floor at lr=3e-3 is ~2.47% — critical for all in-flight paired sweeps** |
