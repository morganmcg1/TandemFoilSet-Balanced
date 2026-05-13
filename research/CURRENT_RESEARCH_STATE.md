# SENPAI Research State

- **Date:** 2026-05-13 ~04:35
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 63.017** (RMSNorm+GeGLU+Lion, PR #1837, epoch 13)
**`test_avg/mae_surf_p` = 54.731**

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 76.710 | 67.384 |
| geom_camber_rc | 73.930 | 64.508 |
| geom_camber_cruise | 40.746 | 34.707 |
| re_rand | 60.683 | 52.327 |
| **avg** | **63.017** | **54.731** |

## What we've learned

### Big wins (merged)
1. **L1 loss**: −20.5% (PR #1358)
2. **GeGLU+Lion compound**: −25.3% (PR #1769)
3. **Lion optimizer lr=1e-4**: −14.3% (PR #1725)
4. **n_layers=6**: −9.4% (PR #1392)
5. **mlp_ratio=4**: −5% (PR #1408)
6. **RMSNorm**: −2.9% val / −5.9% test (PR #1837) ← geom_camber_rc −17.2%
7. **bf16 mixed precision**: −0.34% (PR #1724) ← infrastructure win, +1-2 epochs/run

### Current stack (all defaults in train.py)
- L1 (MAE) loss in normalized space, surf_weight=10
- n_layers=6, **mlp_ratio=4, GeGLU activation** (PR #1769)
- **RMSNorm** (PR #1837, replaces LayerNorm)
- n_hidden=128, n_head=4, slice_num=64
- Lion optimizer lr=1e-4, weight_decay=1e-4
- CosineAnnealingLR T_max=50
- bf16 mixed precision (autocast)
- ~14 epochs in 30 min (RMSNorm ~138s/epoch vs 143s before)

### Dead ends
- **AdamW hyperparameter space fully exhausted:** WD (0, 1e-4 optimal, 5e-4), LR (5e-4 only), betas (0.85/0.9/0.95 for β1, 0.99/0.999 for β2), eps (1e-8, 1e-4), schedule (T_max=14/50, warmup, cosine restarts)
- Dropout=0.1: +11.8% (model is underfitting — regularizing makes it worse)
- Gradient clipping max_norm=1 and max_norm=10: both worse (oscillations = useful search)
- Huber loss β=1.0: +15.7% (mostly-MSE; L1's constant gradient is the key)
- **Channel-weighted L1 [0.03,0.03,1.0] on GeGLU+Lion**: +11.0% (PR #1767 — GeGLU gates do implicit channel balancing; manual Ux/Uy downweighting disrupts routing)
- **SwiGLU vs GeGLU**: +1.6% (PR #1824 — LLM finding doesn't transfer; GELU's slightly negative gate range benefits CFD pressure-gradient features)
- **SmoothL1 β=0.1 on GeGLU+Lion**: +7.1% (PR #1859 — all loss modifications exhausted; pure L1 optimal for Lion)
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

## Active experiments (Round 9 — all on RMSNorm+GeGLU+Lion baseline)

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| alphonse | #1765 | Lion lr=2e-4 (bug fix landed): rerun on RMSNorm+GeGLU+Lion | SENT BACK (rerun needed) |
| askeladd | #1766 | Lion WD=1e-2 on RMSNorm+GeGLU+Lion | SENT BACK (rerun needed) |
| edward | #1889 | Lion WD=1e-1: upper end of paper-recommended range | NEW |
| tanjiro | #1872 | mlp_ratio=8 + GeGLU+Lion: recover fc2 capacity | WIP |
| fern | #1790 | Lion + 2-epoch cosine warmup on RMSNorm+GeGLU+Lion | SENT BACK (rerun needed) |
| nezuko | #1793 | Lion + T_max=12 aligned to budget on RMSNorm+GeGLU+Lion | SENT BACK (rerun needed) |
| thorfinn | #1836 | surf_weight 5 on RMSNorm+GeGLU+Lion (rebase needed) | SENT BACK (rerun needed) |
| frieren | #1890 | n_layers=7 + RMSNorm+GeGLU+Lion: depth re-test with faster norm | NEW |

**Recently merged:**
- frieren #1837: RMSNorm on GeGLU+Lion (−2.9% val / −5.9% test) ← new baseline 63.017

**Recently closed:**
- edward #1859: SmoothL1 β=0.1 (+7.1%) — all loss modifications exhausted; pure L1 optimal for Lion
- edward #1767: channel-weighted L1 (+11%) — GeGLU gates do implicit channel balancing
- tanjiro #1824: SwiGLU (+1.6%) — GELU's negative gate range benefits CFD features

## Critical infra issue: train.py:440 LR hardcoding bug

Discovered by askeladd in #1766; alphonse's #1765 also contains the same fix (`lr=cfg.lr`, plus `Config.lr` default updated to 1e-4). Once either PR rebases cleanly onto the new baseline and is merged, the bug is resolved. Until then, any LR experiment with `--lr != 1e-4` is silently broken.

> Note: This is the first round where 0 students are running on the Lion+GELU pre-GeGLU baseline. All experiments now target the GeGLU+Lion 64.918 baseline.

## Round 8/9 priorities (GeGLU+Lion baseline)

**Tier 1 (directly on new RMSNorm baseline):**
1. **n_layers=7** (frieren #1890): previous failure was at 205s/epoch on AdamW; RMSNorm+Lion+GeGLU+bf16 now ~138s → 11 epochs at 7 layers. Re-test under much better conditions.
2. **mlp_ratio=8 + GeGLU** (tanjiro #1872): recover fc2 capacity halved by GeGLU split.
3. **Lion WD=1e-1** (edward #1889): upper end of paper recommendation; brackets optimum vs askeladd's concurrent WD=1e-2.

**Tier 2 (mechanism confirmation — confirmed on pre-GeGLU/pre-RMSNorm; need rebase+rerun):**
4. **Lion WD=1e-2** (askeladd #1766): confirmed −10.4% on Lion+GELU; now rerun on full RMSNorm+GeGLU+Lion stack.
5. **Lion + 2-epoch warmup** (fern #1790): confirmed −9.9% on Lion+GELU; now rerun on full stack.
6. **Lion lr=2e-4** (alphonse #1765): confirmed −7.8% on Lion+GELU; now rerun on full stack.
7. **Lion + T_max=12** (nezuko #1793): confirmed −9.18% on Lion+GELU; now rerun on full stack.
8. **surf_weight=5** (thorfinn #1836): confirmed −2.74% on GeGLU+Lion (just missed RMSNorm baseline); rerun on RMSNorm stack.

**Queued ideas for next idle students (after current round lands):**
- **CosineAnnealingLR eta_min=1e-5**: fern suggested Lion benefits from non-zero final LR; not yet tested
- **Lion WD=3e-2**: bracket WD optimum between 1e-2 and 1e-1 after both land
- **RMSNorm + surf_weight=5**: if thorfinn's rerun lands positive, compound already confirmed
- **n_hidden widening 128→160**: capacity without restructuring MLP; ~225s/epoch is tight
- **PhysicsAttention slice_num=48**: slight reduction for faster epochs; might help convergence

## Key constraints

- 30 min / run cap: Lion → ~11 epochs (~165s/epoch estimated with bf16+Lion)
- Per-epoch time eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~205s/epoch)
- EMA: cold-start drag, incompatible with short budget
- Batch increase: always worse (step-count limited)
- Gradient clipping: always worse (oscillations are useful)
- Dropout: always worse (model is underfitting at 11 epochs)
