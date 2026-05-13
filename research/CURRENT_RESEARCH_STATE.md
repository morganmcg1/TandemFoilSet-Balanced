# SENPAI Research State

- **Date:** 2026-05-13 ~04:00
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 64.918** (GeGLU+Lion, PR #1769, epoch 13)
**`test_avg/mae_surf_p` = 58.171**

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 72.021 | 64.947 |
| geom_camber_rc | 89.234 | 80.467 |
| geom_camber_cruise | 37.058 | 32.329 |
| re_rand | 61.359 | 54.939 |
| **avg** | **64.918** | **58.171** |

## What we've learned

### Big wins (merged)
1. **L1 loss**: −20.5% (PR #1358)
2. **GeGLU+Lion compound**: −25.3% (PR #1769) ← NOW THE BIGGEST SINGLE WIN
3. **Lion optimizer lr=1e-4**: −14.3% (PR #1725)
4. **n_layers=6**: −9.4% (PR #1392)
5. **mlp_ratio=4**: −5% (PR #1408)
6. **bf16 mixed precision**: −0.34% (PR #1724) ← infrastructure win, +1-2 epochs/run

### Current stack (all defaults in train.py)
- L1 (MAE) loss in normalized space, surf_weight=10
- n_layers=6, **mlp_ratio=4, GeGLU activation** (PR #1769)
- n_hidden=128, n_head=4, slice_num=64
- Lion optimizer lr=1e-4, weight_decay=1e-4
- CosineAnnealingLR T_max=50
- bf16 mixed precision (autocast)
- ~13 epochs in 30 min (GeGLU adds minimal overhead with bf16, ~143s/epoch)

### Dead ends
- **AdamW hyperparameter space fully exhausted:** WD (0, 1e-4 optimal, 5e-4), LR (5e-4 only), betas (0.85/0.9/0.95 for β1, 0.99/0.999 for β2), eps (1e-8, 1e-4), schedule (T_max=14/50, warmup, cosine restarts)
- Dropout=0.1: +11.8% (model is underfitting — regularizing makes it worse)
- Gradient clipping max_norm=1 and max_norm=10: both worse (oscillations = useful search)
- Huber loss β=1.0: +15.7% (mostly-MSE; L1's constant gradient is the key)
- **Channel-weighted L1 [0.03,0.03,1.0] on GeGLU+Lion**: +11.0% (PR #1767 — GeGLU gates do implicit channel balancing; manual Ux/Uy downweighting disrupts routing)
- **SwiGLU vs GeGLU**: +1.6% (PR #1824 — LLM finding doesn't transfer; GELU's slightly negative gate range may benefit CFD pressure-gradient features)
- n_head=8: +43% per-epoch cost, +15.7% worse
- slice_num=128: +12% per-epoch cost, +17.8% worse
- n_layers=7: +51% worse, too slow (~205s/epoch)
- EMA decay=0.999: cold-start drag (+41% worse)
- Batch=8 (accum_steps=2): +23.6% worse (step-count limited)
- Fourier L=4: +5.6% worse (doesn't compound with L1)
- Width n_hidden=192: too slow/epoch
- GeGLU with AdamW: essentially baseline (+0.04%), but too slow (210s/epoch); reassigned to Lion

### Key insights
1. **Lion is structurally complementary to L1**: Both operate via sign direction; combined signal is clean
2. **Budget is the constraint**: 30 min → ~13 epochs with GeGLU+Lion (~143s/epoch).
3. **L1 loss in normalized space is validated**: channel-weighted loss hurts on GeGLU+Lion (+11%); GeGLU gates do implicit channel balancing — the gradient channel weights must stay equal.
4. **AdamW hyperparameter space is exhausted**: All optimizer knobs tested. Lion is the new baseline optimizer.
5. **Lion warmup (+2-epoch linear) confirmed on Lion+GELU (−9.9%)**: awaiting retest on GeGLU+Lion baseline.
6. **Lion WD=1e-2 confirmed on Lion+GELU (−10.4%)**: awaiting retest on GeGLU+Lion baseline.
7. **geom_camber_rc (89.2) and single_in_dist (72.0)** are now the hardest splits — reduced but still dominant

## Active experiments (Round 8/9 — all on GeGLU+Lion baseline)

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1765 | Lion lr=2e-4 with lr=cfg.lr bug fix | WIP (rerun on GeGLU+Lion) |
| askeladd | #1766 | Lion WD=1e-2 on GeGLU+Lion (compound test) | SENT BACK (rerun on GeGLU+Lion) |
| edward | #1859 | SmoothL1 β=0.1 on GeGLU+Lion (remove L1 gradient discontinuity at zero) | NEW |
| tanjiro | #1872 | mlp_ratio=8 + GeGLU: recover fc2 capacity halved by gating split | NEW |
| fern | #1790 | Lion + 2-epoch cosine warmup on GeGLU+Lion | SENT BACK (rerun on GeGLU+Lion) |
| nezuko | #1793 | Lion + T_max=12 aligned to budget | WIP (on pre-GeGLU; check upon completion) |
| thorfinn | #1836 | surf_weight 10 → 5 on GeGLU+Lion | WIP |
| frieren | #1837 | RMSNorm replaces LayerNorm on GeGLU+Lion | WIP |

**Recently closed:**
- edward #1767: channel-weighted L1 on GeGLU+Lion (+11%) — GeGLU gates do implicit channel balancing; manual reweighting disrupts routing
- tanjiro #1824: SwiGLU vs GeGLU (+1.6%) — GELU's slightly negative gate range better for CFD pressure features; LLM finding doesn't transfer

## Critical infra issue: train.py:440 LR hardcoding bug

`optimizer = Lion(model.parameters(), lr=1e-4, ...)` hardcodes lr=1e-4, ignoring `cfg.lr`. Discovered by askeladd in PR #1766; fix in flight (`lr=cfg.lr`). Until merged, any LR experiment with `--lr != 1e-4` is silently broken. alphonse's lr=2e-4 in #1765 actually ran at 1e-4 on the old baseline.

> Note: #1793 (nezuko T_max=12) is still on the pre-GeGLU Lion+GELU baseline. When it lands, evaluate relative to 86.938; if positive, retest on 64.918.

## Round 8/9 priorities (GeGLU+Lion baseline)

**Tier 1 (directly on new baseline):**
1. **mlp_ratio=8 + GeGLU** (tanjiro #1872): recover fc2 capacity halved by GeGLU split; tests "gating alone vs gating+capacity"
2. **SmoothL1 β=0.1 + GeGLU+Lion** (edward #1859): remove L1 gradient discontinuity at zero for Lion sign updates.

**Tier 2 (mechanism confirmation on GeGLU+Lion):**
3. **Lion WD=1e-2** (askeladd #1766 rerun): confirmed −10.4% on Lion+GELU; testing if it compounds with GeGLU.
4. **Lion + 2-epoch warmup** (fern #1790 rerun): confirmed −9.9% on Lion+GELU; testing on GeGLU stack.
5. **Lion lr=2e-4 + bug fix** (alphonse #1765): previous run silently used 1e-4; now re-running with actual 2e-4.

**Tier 3 (pre-GeGLU, check upon completion):**
6. **Lion + T_max=12** (nezuko #1793): still on Lion+GELU; if positive → retest on 64.918.

**Queued ideas for next idle students:**
- **RMSNorm + GeGLU + Lion**: frieren testing RMSNorm (#1837); if positive, it's already on GeGLU+Lion
- **surf_weight 5** (thorfinn #1836): on GeGLU+Lion; may help if pressure signal too noisy with high surf_weight
- **GeGLU + CosineAnnealingLR eta_min=1e-5**: fern suggested Lion benefits from non-zero final LR; needs testing
- **Lion WD=3e-2 or 1e-1**: askeladd noted 1e-2 was low end of paper's recommended range; more WD may be better
- **n_hidden widening 128→160**: capacity increase (slower than ratio; ~30% more params)
- **n_layers=7 with GeGLU+Lion**: depth was "too slow" on AdamW (~205s); GeGLU+Lion+bf16 at 143s may now fit (~167s/epoch → 10 epochs)

## Key constraints

- 30 min / run cap: Lion → ~11 epochs (~165s/epoch estimated with bf16+Lion)
- Per-epoch time eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~205s/epoch)
- EMA: cold-start drag, incompatible with short budget
- Batch increase: always worse (step-count limited)
- Gradient clipping: always worse (oscillations are useful)
- Dropout: always worse (model is underfitting at 11 epochs)
