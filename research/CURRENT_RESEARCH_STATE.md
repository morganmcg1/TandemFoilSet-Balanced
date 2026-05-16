# SENPAI Research State

- **Date:** 2026-05-17 00:00 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
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

## In-Flight Experiments (21:55 UTC)

| Student | PR | Hypothesis | Stack | Priority |
|---------|----|----|----|----|
| ⭐ **frieren** | **#4144** | **Lion vs SF-AdamW lr=2e-3 vs Lion+SF (3-way)** | Lion (A), SF lr=2e-3 (B), Lion+SF (C) | **HIGHEST — resolves whether Lion or SF wins head-to-head with correct LRs** |
| ⭐ **askeladd** | **#4225** | **Model width sweep: n_hidden ∈ {96, 128, 160, 192}** | SF-AdamW lr=2e-3 + --seed 1 (requires Config edit) | **HIGH — primary capacity axis, untested at correct LR** |
| ⭐ **edward** | **#4157** | **SF-AdamW LR fine-tune: {1.5e-3, 2e-3, 2.5e-3, 3e-3}** | SF-AdamW lr=2e-3 base | **HIGH — localizes the peak from #4038's coarse sweep; +1-3% EV if peak is off-grid** |
| ⭐ **tanjiro** | **#4207** | **surf_weight sweep at lr=2e-3: {5, 10, 15, 25}** | SF-AdamW lr=2e-3 + --seed 1 | **HIGH — directly modulates primary metric loss; untested at correct LR** |
| ⭐ **fern** | **#4208** | **Dropout sweep at lr=2e-3: {0.0, 0.05, 0.10, 0.15}** | SF-AdamW lr=2e-3 + --seed 1 (requires Config edit) | **HIGH — untouched regularization axis; OOD generalization angle** |
| **alphonse** | **#4019** | SF clip×EMA factorial R2 (2×2 at lr=2e-3) | SF-AdamW lr=2e-3 | R1 sent back — EMA-off won by 0.610% (below 0.97% noise floor); re-test at correct LR |
| **nezuko** | **#4081** | FiLM head width: film_mlp_hidden ∈ {128, 192, 256} | SF-AdamW lr=5e-4 (stale) | Results still diagnostic; paired Δ may be larger at correct LR |
| **thorfinn** | **#4114** | Batch size sweep: {4, 6, 8, 12} | SF-AdamW lr=5e-4 (stale) | Batch size axis universal |

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

1. **Review #4157 edward LR fine-tune** — if any arm in {1.5e-3, 2.5e-3, 3e-3} beats 2e-3, update canonical LR immediately
2. **Lion+SF composition (#4144 frieren)** — Arms A (Lion-cosine) and B (SF lr=2e-3) reproduce closed results; Arm C (Lion+SF) is the only remaining Lion-related question
3. **Surf_weight sweep results (#4207 tanjiro)** — direct loss-balance lever; +1-3% EV
4. **Dropout sweep results (#4208 fern)** — untouched regularization axis; OOD generalization angle
5. **n_hidden sweep results (#4225 askeladd)** — primary model capacity axis; likely either a clean win or a clean null
6. **alphonse #4019 R2 (clip×EMA at lr=2e-3)** — finalize EMA-redundancy question under correct LR
7. **Stale-LR SF sweeps** (nezuko #4081, thorfinn #4114) — apply paired Δ gate when results land: >3% → re-test at lr=2e-3; <1% → close

## Falsified / Closed Hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| (See previous entries) | ... | ... | ... |
| #4051 | SF wd sweep {1e-4, 3e-4, 1e-3, 1e-2} | +2.46-11.99% all regress | Polyak+EMA saturate regularization role |
| #4003 | AdamW clip-R2 {0.05-0.25} | −0.07% noise floor | Direction-norm saturated at 0.25 |
| SF-AdamW at lr=5e-4 | ALL experiments prior to #4038 | Viable but undertrained | LR was wrong; 2e-3 is the right setting |
| **#4087** | **SF warmup steps {100, 500, 1000, 2000}** | **Paper default 500 wins; B/C/D regress +7.15%, +3.81%, +1.64%** | **Warmup axis exhausted; 500 is optimal at 30-min/17-epoch budget** |
| **#4012** | **Sobolev edge-gradient L1 supervision (R1 AdamW, R2 Lion)** | **R1 within seed variance; R2 paired Δ +2.68% val / +2.28% test regression across all splits** | **Cross-stack null. Sobolev penalty fires but doesn't help surface MAE — mechanism doesn't fit loss landscape** |
| **#4113** | **EMA decay sweep {0.99, 0.999, 0.9995, 0.9999}** | **B (0.99) Δ=−3.72% but cross-arm noise band ~3.46% (no --seed); A/C/D theoretically identical under Karras ramp** | **Karras ramp dominates target ema_decay at 17-epoch budget; paired methodology > absolute decay tuning** |
| **#4149** | **Lion LR sweep {7.5e-5, 1.5e-4, 3e-4, 6e-4}** | **A (1.5e-4) optimal; B/D regress; best Lion 61.146 is +11.64% behind SF-canonical 54.769** | **SF \"4× LR\" insight does not transfer to Lion+cosine; Lion as standalone optimizer behind SF — only Lion+SF (frieren #4144 Arm C) still pending** |
