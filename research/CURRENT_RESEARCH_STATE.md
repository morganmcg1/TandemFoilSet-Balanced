# SENPAI Research State

- **Date:** 2026-05-13 ~02:15
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion optimizer broke the plateau decisively (−14.3%).

## Current baseline

**`val_avg/mae_surf_p` = 86.938** (Lion optimizer lr=1e-4, PR #1725, epoch 11)
**`test_avg/mae_surf_p` = 77.990**

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 98.979 | 91.606 |
| geom_camber_rc | 104.737 | 92.561 |
| geom_camber_cruise | 62.041 | 52.841 |
| re_rand | 81.995 | 74.952 |
| **avg** | **86.938** | **77.990** |

## What we've learned

### Big wins (merged)
1. **L1 loss**: −20.5% (PR #1358) ← dominant factor
2. **Lion optimizer lr=1e-4**: −14.3% (PR #1725) ← second biggest win
3. **n_layers=6**: −9.4% (PR #1392)
4. **mlp_ratio=4**: −5% (PR #1408)
5. **bf16 mixed precision**: −0.34% (PR #1724) ← infrastructure win, +1 epoch/run

### Current stack (all defaults in train.py)
- L1 (MAE) loss in normalized space, surf_weight=10
- n_layers=6, mlp_ratio=4, n_hidden=128, n_head=4, slice_num=64
- Lion optimizer lr=1e-4, weight_decay=1e-4
- CosineAnnealingLR T_max=50
- bf16 mixed precision (autocast)
- ~11 epochs in 30 min (Lion converges slightly fewer epochs than AdamW at same wall-clock)

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

## Active experiments (Round 7 — Lion-compounding hypotheses)

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1765 | Lion lr=2e-4: upper LR bound (2× above current 1e-4) | NEW |
| askeladd | #1766 | Lion WD=1e-2: paper-recommended WD (100× higher than current) | NEW |
| edward | #1767 | Physical-space L1 + Lion: loss in Pa units aligns with metric | NEW |
| tanjiro | #1769 | GeGLU + Lion: gated activation with sign-based optimizer | NEW |
| fern | #1790 | Lion + 2-epoch cosine warmup: stabilize sign-update init | NEW |
| nezuko | #1793 | Lion + T_max=12 aligned: proper cosine decay over actual budget | NEW |

**Still WIP from Round 6 (running under AdamW — results will compare to 86.938):**
- frieren #1729: RMSNorm (architectural change — may compound with Lion if positive)
- thorfinn #1737: surf_weight 10 → 5 (loss weighting)

**Recently closed:**
- fern #1726 SWA: +7.9% worse (premature start, only 3 avg epochs, SWALR cut LR 10×)
- nezuko #1678 lr=7e-4: stale 3h, obsolete after Lion merge

> Note: The remaining WIP AdamW experiments may still produce useful insights. RMSNorm in particular may transfer to Lion. surf_weight is loss-formulation orthogonal to optimizer.

## Round 7 hypotheses and priorities

**Tier 1 (highest priority — pure Lion tuning):**
1. **Lion WD=1e-2** (askeladd #1766): Lion paper explicitly recommends 10-100× higher WD than Adam. This is likely another significant win.
2. **Lion lr=2e-4** (alphonse #1765): Model still improving at epoch 11; faster convergence might help in 30-min window.

**Tier 2 (architecture/loss changes):**
3. **Physical-space L1** (edward #1767): Computing loss in actual Pa units may align gradient signal with the evaluation metric.
4. **GeGLU + Lion** (tanjiro #1769): GeGLU's gating may interact differently with Lion's sign-based updates.

**Queued ideas (if current batch fails or for additional idle students):**
- **Lion LR schedule alignment**: T_max aligned to actual epochs reached (~11-12), not 50
- **RMSNorm + Lion**: If frieren's RMSNorm shows any signal, compound with Lion
- **SwiGLU + Lion**: Alternative gated activation (SiLU gate instead of GELU)
- **Lion lr=5e-5**: Step down below current 1e-4 to explore lower bound
- **n_layers=7 + Lion**: Lion's faster convergence might allow deeper arch within budget
- **Data augmentation**: AoA jitter, mesh coarsening for OOD robustness
- **Cosine warmup for Lion**: 2-epoch linear warmup before cosine decay (Lion is warmup-sensitive)

## Key constraints

- 30 min / run cap: Lion → ~11 epochs (~165s/epoch estimated with bf16+Lion)
- Per-epoch time eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~205s/epoch)
- EMA: cold-start drag, incompatible with short budget
- Batch increase: always worse (step-count limited)
- Gradient clipping: always worse (oscillations are useful)
- Dropout: always worse (model is underfitting at 11 epochs)
