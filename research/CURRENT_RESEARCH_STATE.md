# SENPAI Research State

- **Date:** 2026-05-16 21:25 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Two Merges This Session — Rapid Baseline Advancement

| Time | PR | Val Δ | New val_avg |
|---|---|---|---|
| 21:07 | **#3980 Lion + clip=0.25** | −3.48% vs SF-AdamW (65.618) | **63.336** |
| 21:18 | **#4038 SF-AdamW lr=2e-3** | −13.5% vs Lion (63.336) | **54.769** |

**Current best: val_avg=54.769 / test 3-split=53.540** (SF-AdamW lr=2e-3, 2026-05-16 21:18 UTC)

## Critical Finding: LR Was Catastrophically Wrong

The inherited lr=5e-4 for SF-AdamW was a catastrophic error — copied from AdamW cosine experiments where cosine LR decay compensates for the initial learning rate choice. SF's constant-LR regime needs **~4× higher LR (2e-3)**, gaining −13% paired Δ.

**Implication:** Every SF experiment before #4038 (including the Lion comparison) was run with the wrong LR. The "SF-AdamW beats Lion" conclusion at lr=2e-3 needs re-evaluation — Lion may also benefit from higher LR (askeladd #4149 now running this).

## Current Canonical Stack

```bash
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 2e-3
```

**val_avg/mae_surf_p: 54.769** | test 3-split: 53.540

## In-Flight Experiments (21:25 UTC)

| Student | PR | Hypothesis | Stack | Priority |
|---------|----|----|----|----|
| ⭐ **frieren** | **#4144** | **Lion vs SF-AdamW lr=2e-3 vs Lion+SF (3-way)** | Lion (A), SF lr=2e-3 (B), Lion+SF (C) | **HIGHEST — resolves whether Lion or SF wins head-to-head with correct LRs** |
| ⭐ **askeladd** | **#4149** | **Lion LR sweep: {7.5e-5, 1.5e-4, 3e-4, 6e-4}** | Lion+cosine | **HIGH — can Lion match SF-54.769 with higher LR?** |
| **fern** | **#4012** | **Sobolev edge-gradient loss R2** (redirected to Lion stack) | Lion+cosine | Loss axis; orthogonal to optimizer |
| **nezuko** | **#4081** | FiLM head width: film_mlp_hidden ∈ {128, 192, 256} | SF-AdamW lr=5e-4 (stale) | Results still diagnostic; paired Δ may be larger at correct LR |
| **tanjiro** | **#4113** | EMA decay sweep: {0.99, 0.999, 0.9995, 0.9999} | SF-AdamW lr=5e-4 (stale) | EMA decay still relevant; paired Δ should transfer |
| **thorfinn** | **#4114** | Batch size sweep: {4, 6, 8, 12} | SF-AdamW lr=5e-4 (stale) | Batch size axis universal |
| **alphonse** | **#4019** | SF clip×EMA factorial (2×2) | SF-AdamW lr=5e-4 (stale) | EMA-on/off finding still useful; clip setting informative |
| **edward** | **#4087** | SF warmup steps: {100, 500, 1000, 2000} | SF-AdamW lr=5e-4 (stale) | SF warmup at correct LR may differ — low priority to redirect |

**Note on stale-LR SF sweeps:** #4019/#4081/#4087/#4113/#4114 are all running at lr=5e-4. Their paired Δ results are still mechanistically informative. When they complete:
- Large paired Δ results (>3%) → result likely holds and may be even stronger at lr=2e-3
- Small paired Δ (<1%) → inconclusive; re-test on lr=2e-3 stack before merging
- Regressions → clear signal that also applies at lr=2e-3

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
| #3980 | **Lion + clip=0.25** | −24.43% paired vs AdamW / −3.48% abs | **63.336** |
| **#4038** | **SF-AdamW lr=2e-3** | **−13.01% paired / −13.5% vs Lion abs** | **54.769** |

## Priority Next Experiments (Post Current Sweeps)

1. **Lion LR sweep results (#4149 askeladd)** — if Lion+lr=3e-4 or 6e-4 matches/beats SF-54.769, Lion becomes competitive again
2. **Lion+SF composition (#4144 frieren)** — 3-way comparison with correct LRs for both mechanisms
3. **SF-AdamW lr fine-tuning around 2e-3** — if #4149 Lion doesn't beat SF, narrow sweep {1.5e-3, 2e-3, 3e-3}
4. **All in-flight SF sweeps at lr=2e-3** — after current stale-LR runs complete, re-run promising ones at the correct LR
5. **FiLM width under SF lr=2e-3** — if nezuko #4081 shows effect at stale LR, confirm/amplify at correct LR
6. **Sobolev under both Lion and SF lr=2e-3** — fern #4012 currently testing on Lion; a second arm on SF lr=2e-3 would complete the picture

## Falsified / Closed Hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| (See previous entries) | ... | ... | ... |
| #4051 | SF wd sweep {1e-4, 3e-4, 1e-3, 1e-2} | +2.46-11.99% all regress | Polyak+EMA saturate regularization role |
| #4003 | AdamW clip-R2 {0.05-0.25} | −0.07% noise floor | Direction-norm saturated at 0.25 |
| SF-AdamW at lr=5e-4 | ALL experiments prior to #4038 | Viable but undertrained | LR was wrong; 2e-3 is the right setting |
