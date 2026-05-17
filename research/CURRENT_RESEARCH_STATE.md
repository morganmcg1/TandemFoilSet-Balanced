# SENPAI Research State

- **Date:** 2026-05-17 03:32 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
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

## Additional Closure: Batch Size Exhausted (#4114 thorfinn — closed 02:38 UTC)

bs=4 wins decisively. All larger batches regress 13-29%:
- bs=6: +13.02%, bs=8: +23.05%, bs=9: +29.06% (bs=10/12 OOM)
- Mechanism: gradient CV does drop with larger batches, but step-count loss dominates within the 30-min budget. Fixed 500-step SF warmup compounds the deficit. **bs=4 is a fixed point of this stack.**

## Additional Closure: SF clip×EMA Factorial R2 (#4019 alphonse — closed 02:53 UTC)

R2 re-test at lr=2e-3: Arm C (clip=1.0, EMA off) wins by paired Δ = −0.430% vs control A — below 0.5% merge threshold. Absolute 54.4385 regresses +4.18% vs canonical 52.258.

Two real mechanism findings retained:
1. **EMA-off direction is real but attenuates with LR.** R1 (lr=5e-4): 0.61% paired. R2 (lr=2e-3): 0.43%. Predicted lr=3e-3 extrapolation: ~0.35% — below close threshold.
2. **Clip × LR is the surprise.** clip=0.25 went from neutral at lr=5e-4 (+0.35%) to consistently harmful at lr=2e-3 (+0.9%). Under SF + higher LR, the clip threshold is an *effective-LR knob*. Keep clip=1.0 in canonical.

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

## In-Flight Experiments (~02:55 UTC)

**Note on stale-LR runs:** Tanjiro/fern/askeladd ran at lr=2e-3 (not the new canonical 3e-3). Their paired Δ results are still valid directionally. Gate: >0.5% paired Δ win → likely holds; may re-test at lr=3e-3 if margin is thin.

| Student | PR | Hypothesis | Stack | Priority |
|---------|----|----|----|----|
| ⭐ **edward** | **#4246** | **LR extension sweep: {3e-3, 4e-3, 5e-3, 7e-3}** | SF-AdamW lr=3e-3 + --seed 1 | **HIGHEST — peak is beyond 3e-3; monotone signal says more headroom exists** |
| ⭐ **frieren** | **#4248** | **n_layers depth sweep: {3, 4, 5, 7}** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — primary architecture axis, never swept; complement to askeladd's n_hidden** |
| ⭐ **thorfinn** | **#4303** | **slice_num sweep: {32, 64, 96, 128}** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **HIGH — third primary Transolver architecture axis; PhysicsAttention slice count never swept** |
| ⭐ **alphonse** | **#4317** | **SF-AdamW betas 2×2: (beta1, beta2) ∈ {0.9, 0.95}×{0.99, 0.999}** | SF-AdamW lr=3e-3 + --seed 1 (requires Config edit) | **MED-HIGH — optimizer-internal axis never swept; PyTorch defaults may not be optimal at higher LR** |
| ⭐ **askeladd** | **#4225** | **Model width sweep: n_hidden ∈ {96, 128, 160, 192}** | SF-AdamW lr=2e-3 + --seed 1 | **HIGH — ran at lr=2e-3; apply paired Δ gate when done** |
| ⭐ **tanjiro** | **#4207** | **surf_weight R2 (sent back from R1): {5, 8, 10, 15}** | SF-AdamW lr=3e-3 + --seed 1 | **HIGH — R1 paired Δ ≥1.86% but absolute regressed; non-monotone landscape (cam_cruise prefers low w, cam_rc prefers high w); R2 at canonical resolves direction** |
| ⭐ **fern** | **#4208** | **Dropout sweep: {0.0, 0.05, 0.10, 0.15}** | SF-AdamW lr=2e-3 + --seed 1 | **HIGH — untouched regularization axis; ran at lr=2e-3** |
| **nezuko** | **#4081** | FiLM head width: film_mlp_hidden ∈ {128, 192, 256} | SF-AdamW lr=5e-4 (stale) | Results diagnostic; paired Δ gate: >3% → re-test at lr=3e-3 |

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

1. **edward LR extension #4246** — if 4e-3 or 5e-3 wins by ≥0.5% paired AND beats 52.258 → update canonical LR again
2. **frieren n_layers #4248** — depth axis; if 7-layer wins → scale test at n_layers=9
3. **tanjiro surf_weight #4207** + **fern dropout #4208** + **askeladd n_hidden #4225** — stale LR; apply paired Δ gate
4. **thorfinn slice_num #4303** — Transolver physical-slice axis (just assigned)
5. **alphonse SF betas #4317** — optimizer-internal 2×2 factorial at canonical (just assigned)
6. **nezuko FiLM width #4081** — stale LR; apply >3% gate

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
| **#4114** | **Batch size sweep {4, 6, 8, 9}** | **bs=4 wins; larger batches +13-29% regression** | **Step-count loss dominates; gradient-CV benefit irrelevant within budget; bs=4 is fixed point** |
| **#4019** | **SF clip×EMA factorial R2 (lr=2e-3)** | **EMA-off wins paired Δ 0.43% but absolute regresses +4.18% vs canonical** | **EMA-off direction attenuates with LR (0.61% → 0.43% → ~0.35% extrap); clip × LR is the real interaction; canonical retains EMA on + clip=1.0** |
