# SENPAI Research State

- **Date:** 2026-05-13 ~02:50
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
2. **Budget is the constraint**: 30 min → ~11 epochs with Lion. Any deeper/wider arch needs batching change.
3. **L1 loss in normalized space is validated**: Physical-space L1 untested (may align gradient signal better with metric)
4. **AdamW hyperparameter space is exhausted**: All optimizer knobs tested. Lion is the new baseline optimizer.
5. **Lion was still improving at epoch 11 cutoff**: LR tuning and warmup may extract more performance
6. **geom_camber_rc (104.7) and single_in_dist (99.0)** are now the hardest splits to improve

## Active experiments (Round 7 — GeGLU+Lion baseline, compounding)

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1765 | Lion lr=2e-4: upper LR bound | WIP (pre-GeGLU baseline) |
| askeladd | #1766 | Lion WD=1e-2: paper-recommended WD | WIP (pre-GeGLU baseline) |
| edward | #1767 | Pressure-weighted normalized loss (rebase + explicit wt) | SENT BACK |
| tanjiro | #1824 | SwiGLU vs GeGLU: SiLU gate comparison | NEW |
| fern | #1790 | Lion + 2-epoch cosine warmup | WIP (pre-GeGLU baseline) |
| nezuko | #1793 | Lion + T_max=12 aligned to budget | WIP (pre-GeGLU baseline) |

**Round 6 stragglers (AdamW, compare to new 64.918 baseline):**
- frieren #1729: RMSNorm — may still provide architectural insight
- thorfinn #1737: surf_weight 10 → 5

> Note: Experiments #1765, #1766, #1790, #1793 are running on the GeGLU+Lion baseline's PREDECESSOR (Lion+GELU). When they land, results should be evaluated relative to their own training config — if they beat 86.938 (Lion+GELU baseline), the change is positive but needs retesting on the new GeGLU+Lion stack. If they beat 64.918 (new baseline), they compound.

## Round 8 priorities (GeGLU+Lion baseline)

**Tier 1 (directly on new baseline):**
1. **SwiGLU + Lion** (tanjiro #1824): A/B test SiLU gate vs GELU gate. Clean single-change test.
2. **Pressure-weighted loss** (edward #1767 rebased): explicit channel weighting `loss = mae_p + 0.03*(mae_Ux + mae_Uy)`. Student's own analysis shows this is the mechanism behind physical-space L1.

**Tier 2 (may compound if WD/LR/warmup/schedule help):**
3. **Lion WD=1e-2** (askeladd #1766 — running on old baseline): if positive on 86.938, retest on 64.918
4. **Lion lr=2e-4** (alphonse #1765 — running): same caveat
5. **Lion + T_max=12** (nezuko #1793): proper schedule decay
6. **Lion + warmup** (fern #1790): init stability

**Queued ideas for next idle students:**
- **n_hidden widening 128→160**: tanjiro's GeGLU fc2 halved effective width; widening recovers params
- **RMSNorm + GeGLU + Lion**: if frieren's RMSNorm lands positive, compound with new stack
- **GeGLU + T_max=13**: nezuko will test T_max=12 for Lion+GELU; if positive, also test with GeGLU
- **GeGLU surf_weight tuning**: with cruise now at 37 val, surface weighting may need rebalancing
- **n_layers=7 + GeGLU + Lion**: GeGLU runs 13 epochs/30min; maybe depth now viable?
- **mlp_ratio=8 + GeGLU**: push feedforward capacity with gated MLP

## Key constraints

- 30 min / run cap: Lion → ~11 epochs (~165s/epoch estimated with bf16+Lion)
- Per-epoch time eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~205s/epoch)
- EMA: cold-start drag, incompatible with short budget
- Batch increase: always worse (step-count limited)
- Gradient clipping: always worse (oscillations are useful)
- Dropout: always worse (model is underfitting at 11 epochs)
