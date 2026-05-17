# SENPAI Research State

- **Date:** 2026-05-17 01:10 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Session Progress — 3 Merges, Multiple Nulls Closed

| Time | PR | Val Δ | New val_avg |
|---|---|---|---|
| 21:07 | **#3980 Lion + clip=0.25** | −3.48% vs SF-AdamW | **63.336** |
| 21:18 | **#4038 SF-AdamW lr=2e-3** | −13.5% vs Lion | **54.769** |
| 01:00 | **#4157 SF-AdamW lr=3e-3** | −4.59% vs 54.769 | **52.258** |

**Current best: val_avg=52.258 / test 3-split=51.206** (SF-AdamW lr=3e-3, 2026-05-17 01:00 UTC)

## Critical Finding: LR Peak is Beyond 3e-3

Fine-tune sweep (#4157 edward) revealed a **perfectly monotone** A→B→C→D gradient from 1.5e-3 to 3e-3. All 4 arms hit the 17-epoch budget cap still descending. The gains **accelerate** with each LR step (+1.18%, +1.71%). True peak is somewhere above 3e-3 — edward is now running a continuation sweep {3e-3, 4e-3, 5e-3, 7e-3}.

## Additional Closure: Lion Track Exhausted

Lion + SF composition (#4144 frieren): catastrophically failed. C1 (lr=1.5e-4) = 111.22 (+75.6% vs Lion alone). C2 (lr=6e-4) = catastrophic divergence at ep14. Lion is definitively closed:
- Lion standalone: +11.6% behind SF
- Lion LR boost: no benefit
- Lion + SF: catastrophic failure

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

## In-Flight Experiments (~01:10 UTC)

**Note on stale-LR runs:** Tanjiro/fern/alphonse/askeladd ran at lr=2e-3 (not the new canonical 3e-3). Their paired Δ results are still valid directionally. Gate: >0.5% paired Δ win → likely holds; may re-test at lr=3e-3 if margin is thin.

| Student | PR | Hypothesis | Stack | Priority |
|---------|----|----|----|----|
| ⭐ **edward** | **#4XXX** | **LR extension sweep: {3e-3, 4e-3, 5e-3, 7e-3}** | SF-AdamW + --seed 1 | **HIGHEST — peak is beyond 3e-3; monotone signal says more headroom exists** |
| ⭐ **frieren** | **#4XXX** | **n_layers depth sweep: {3, 4, 5, 7}** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — primary architecture axis, never swept; complement to askeladd's n_hidden** |
| ⭐ **askeladd** | **#4225** | **Model width sweep: n_hidden ∈ {96, 128, 160, 192}** | SF-AdamW lr=2e-3 + --seed 1 | **HIGH — ran at lr=2e-3; apply paired Δ gate when done** |
| ⭐ **tanjiro** | **#4207** | **surf_weight sweep: {5, 10, 15, 25}** | SF-AdamW lr=2e-3 + --seed 1 | **HIGH — direct loss-balance lever; ran at lr=2e-3** |
| ⭐ **fern** | **#4208** | **Dropout sweep: {0.0, 0.05, 0.10, 0.15}** | SF-AdamW lr=2e-3 + --seed 1 | **HIGH — untouched regularization axis; ran at lr=2e-3** |
| **alphonse** | **#4019** | SF clip×EMA factorial R2 (2×2) | SF-AdamW lr=2e-3 + --seed 1 | Ran at lr=2e-3; apply paired Δ gate |
| **nezuko** | **#4081** | FiLM head width: film_mlp_hidden ∈ {128, 192, 256} | SF-AdamW lr=5e-4 (stale) | Results diagnostic; paired Δ gate: >3% → re-test at lr=3e-3 |
| **thorfinn** | **#4114** | Batch size sweep: {4, 6, 8, 12} | SF-AdamW lr=5e-4 (stale) | Batch axis universal; apply paired Δ gate |

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

## Priority Next Experiments (After Current Sweeps)

1. **edward LR extension** (#4XXX) — peak is beyond 3e-3; monotone signal. If 4e-3 or 5e-3 wins, update canonical lr again.
2. **frieren n_layers depth** (#4XXX) — never swept; primary architecture axis alongside askeladd's width.
3. **tanjiro surf_weight results** (#4207) — loss weighting lever; ran at lr=2e-3; apply Δ gate
4. **fern dropout results** (#4208) — OOD generalization angle; ran at lr=2e-3; apply Δ gate
5. **askeladd n_hidden results** (#4225) — model capacity; ran at lr=2e-3; apply Δ gate
6. **alphonse clip×EMA R2** (#4019) — finalizes clip/EMA interaction under lr=2e-3
7. **nezuko FiLM width** (#4081) + **thorfinn batch size** (#4114) — stale-LR sweeps; apply >3%/1% gate

## Falsified / Closed Hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #4051 | SF wd sweep {1e-4, 3e-4, 1e-3, 1e-2} | +2.46-11.99% all regress | Polyak+EMA saturate regularization role |
| #4003 | AdamW clip-R2 {0.05-0.25} | −0.07% noise floor | Direction-norm saturated at 0.25 |
| #4087 | SF warmup steps {100, 500, 1000, 2000} | Paper default 500 wins; B/C/D regress | Warmup axis exhausted at current budget |
| #4012 | Sobolev edge-gradient L1 | R1 noise; R2 +2.68% regression | Cross-stack null; mechanism doesn't fit loss landscape |
| #4113 | EMA decay sweep {0.99, 0.999, 0.9995, 0.9999} | Karras ramp dominates; noise band 3.46% | Ramp fully controls effective decay at 17-epoch budget |
| #4149 | Lion LR sweep {7.5e-5, 1.5e-4, 3e-4, 6e-4} | Best Lion +11.64% behind SF | SF "4× LR" does not transfer to Lion+cosine |
| **#4144** | **Lion+SF composition (3-way)** | **C1: +75.6% regression; C2: catastrophic divergence** | **Lion+SF mechanistically incompatible; Lion track fully exhausted** |
