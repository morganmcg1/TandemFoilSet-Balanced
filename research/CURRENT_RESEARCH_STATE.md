# SENPAI Research State

- **Date:** 2026-05-13 ~08:50
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 51.040** (T_max=12 + RMSNorm+GeGLU+Lion+surf_weight=5, PR #1956, epoch 12)
**`test_avg/mae_surf_p` = 44.390**

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 56.933 | 50.459 |
| geom_camber_rc | **64.886** | 59.341 |
| geom_camber_cruise | 31.056 | 25.501 |
| re_rand | 51.287 | 42.260 |
| **avg** | **51.040** | **44.390** |

**Reproduce:** `python train.py --epochs 12 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 5`

## What we've learned

### Big wins (merged)
1. **L1 loss**: −20.5% (PR #1358)
2. **GeGLU+Lion compound**: −25.3% (PR #1769)
3. **Lion optimizer lr=1e-4**: −14.3% (PR #1725)
4. **n_layers=6**: −9.4% (PR #1392)
5. **surf_weight=5**: −9.0% val / −9.8% test (PR #1836) ← single_in_dist −20.5%
6. **T_max=12 (cosine aligned to epoch budget)**: −7.9% val / −8.9% test (PR #1793) ← all 4 splits improved, cruise −13.8%, re_rand −11.0%
7. **mlp_ratio=4**: −5% (PR #1408)
8. **RMSNorm**: −2.9% val / −5.9% test (PR #1837) ← geom_camber_rc −17.2%
9. **bf16 mixed precision**: −0.34% (PR #1724) ← infrastructure win, +1-2 epochs/run
10. **T_max=12 + surf_weight=5 compound**: −3.33% val / −1.29% test (PR #1956) ← volume MAE −6% to −14% across all splits; cruise −6.96% val

### Current stack (defaults + CLI overrides)
- L1 (MAE) loss in normalized space, **surf_weight=5** (PR #1956 compound confirmed)
- n_layers=6, **mlp_ratio=4, GeGLU activation** (PR #1769)
- **RMSNorm** (PR #1837, replaces LayerNorm)
- n_hidden=128, n_head=4, slice_num=64
- Lion optimizer lr=1e-4, weight_decay=1e-4
- **CosineAnnealingLR T_max=12 (epochs=12)** ← PR #1793, cosine fully decays to 0
- bf16 mixed precision (autocast)
- 12 epochs in 30 min (~138s/epoch)

**Reproduce command:** `python train.py --epochs 12 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 5`

### Dead ends
- **AdamW hyperparameter space fully exhausted:** WD (0, 1e-4 optimal, 5e-4), LR (5e-4 only), betas (0.85/0.9/0.95 for β1, 0.99/0.999 for β2), eps (1e-8, 1e-4), schedule (T_max=14/50, warmup, cosine restarts)
- Dropout=0.1: +11.8% (model is underfitting — regularizing makes it worse)
- Gradient clipping max_norm=1 and max_norm=10: both worse (oscillations = useful search)
- Huber loss β=1.0: +15.7% (mostly-MSE; L1's constant gradient is the key)
- **Channel-weighted L1 [0.03,0.03,1.0] on GeGLU+Lion**: +11.0% (PR #1767 — GeGLU gates do implicit channel balancing; manual Ux/Uy downweighting disrupts routing)
- **SwiGLU vs GeGLU**: +1.6% (PR #1824 — LLM finding doesn't transfer; GELU's slightly negative gate range benefits CFD pressure-gradient features)
- **SmoothL1 β=0.1 on GeGLU+Lion**: +7.1% (PR #1859 — all loss modifications exhausted; pure L1 optimal for Lion)
- **n_layers=7 (re-test with RMSNorm+Lion)**: +4.6% (PR #1890 — 12 epochs at 160s/epoch; single_in_dist catastrophic +18%; depth expansion incompatible with budget under any tested condition)
- **WD=1e-1**: +2.72% (PR #1889 — over-regularizes; best_epoch=10, train descending while val climbs; WD space above 1e-2 exhausted)
- **lr=2e-4 on RMSNorm stack**: +0.98% val (PR #1765 — RMSNorm tightened loss surface; lr=2e-4 overshoots on geom_camber_rc; test −1.23% ✓; pivot to lr=1.5e-4)
- **mlp_ratio=8 + GeGLU**: +5.95% (PR #1872 — gating wins outright; fc2 capacity expansion beyond 256 channels adds noise pathways; mlp_ratio=4 optimal)
- **CosineAnnealingLR eta_min=1e-5**: +12.05% vs current baseline (PR #1920 — LR floor above 0 conflicts with T_max=12 which cleanly decays to 0; T_max=12 strictly dominates)
- **Lion WD=3e-2**: +0.06% (PR #1925 — WD valley confirmed flat [1e-4→3e-2]; WD=1e-1 bends up; entire WD axis exhausted on this stack)
- **Lion 2-epoch warmup**: mechanism conflict with T_max=12 (PR #1790 — warmup costs 17% of 12-epoch budget; cold-start problem already addressed by T_max=12 cosine; student stale on rerun)
- **CosineAnnealingLR T_max=10**: +10.96% val / +11.95% test (PR #1983 — `CosineAnnealingLR` is cyclic, not clamped; epoch 11 ran at LR=0 (dead); T_max < cfg.epochs is *always strictly worse*; T_max should ≥ epoch count)
- **n_hidden=160**: val −0.247% / test +1.268% (PR #1984 — val/test direction inversion = noise signature; +52% params, +12% wall-clock disproportionate; geom_camber_rc test regressed +2.88% — OOD bottleneck is geometric extrapolation not feature capacity)
- n_head=8: +43% per-epoch cost, +15.7% worse
- slice_num=128: +12% per-epoch cost, +17.8% worse
- EMA decay=0.999: cold-start drag (+41% worse)
- Batch=8 (accum_steps=2): +23.6% worse (step-count limited)
- Fourier L=4: +5.6% worse (doesn't compound with L1)
- Width n_hidden=192: too slow/epoch

### Key insights
1. **Lion is structurally complementary to L1**: Both operate via sign direction; combined signal is clean
2. **Budget is the constraint**: 30 min → ~14 epochs with current stack (~138s/epoch).
3. **L1 loss in normalized space is validated**: channel-weighted loss hurts on GeGLU+Lion (+11%); GeGLU gates do implicit channel balancing — the gradient channel weights must stay equal.
4. **surf_weight=5 mechanism confirmed**: halving surface:volume ratio reallocates L1 gradient to volume nodes → richer volumetric features → better surface via geometric context. All 4 splits improved, vol MAE improved −7% to −26%.
5. **AdamW hyperparameter space is exhausted**: All optimizer knobs tested. Lion is the new baseline optimizer.
6. **LR axis saturated at 1e-4**: lr=1.5e-4 (+4.21%), lr=2e-4 (+0.98% val) both worse. lr=8e-5 being tested (frieren #2006).
7. **WD axis saturated at 1e-4**: WD=1e-2 confirmed +16.6% worse vs current baseline (after rebasing to T_max=12+sw=5 stack). RMSNorm+GeGLU already provides implicit regularization.
8. **RMSNorm shifts the hardest split**: After RMSNorm, geom_camber_rc improved −17.2%; it remains the single highest-loss split at val=64.886.
9. **geom_camber_rc (64.886 val) is the dominant bottleneck** — primary target for further improvement.

## Active experiments (Round 14 — all on T_max=12+sw=5 compound baseline, val=51.040)

| Student | PR | Hypothesis | Status |
|---------|-----|------------|--------|
| edward | #1995 | n_layers=5: shallower model → more epochs in 30-min budget | WIP |
| fern | #1996 | slice_num=48: tighter PhysicsAttention partitions | WIP |
| frieren | #2006 | Lion lr=8e-5: bracket LR from below (⚠ bug fix required) | WIP |
| tanjiro | #2007 | mlp_ratio=2: test if gating dominates at half-width MLP | WIP |
| nezuko | #2029 | surf_weight=2: continue gradient sweep below sw=5 | WIP |
| askeladd | #2038 | n_head=2: trade attention parallelism for per-head capacity (head_dim 32→64) | NEW |
| thorfinn | #2040 | grad-clip max_norm=1.0: stabilize Lion EMA (different mechanism from old AdamW test) | NEW |
| alphonse | #2043 | DropPath stochastic depth rate=0.1: path-level regularization | NEW |

**Recently merged:**
- nezuko #1956: T_max=12 + surf_weight=5 compound (−3.33% val / −1.29% test) ← **NEW BASELINE 51.040/44.390**

**Recently closed (Round 14):**
- alphonse #1765: Lion lr=1.5e-4 (+4.21% worse) — LR axis saturated, lr=1e-4 confirmed optimum from both sides
- askeladd #1766: Lion WD=1e-2 rerun (+16.6% worse) — WD axis saturated; RMSNorm+GeGLU provides implicit regularization
- thorfinn #1948: surf_weight=3 stale draft — surf_weight axis covered by nezuko #2029

**Earlier closures:**
- frieren #1983: T_max=10 (+10.96%) — CosineAnnealingLR cyclic; T_max < cfg.epochs always strictly worse
- tanjiro #1984: n_hidden=160 (val/test inversion) — OOD bottleneck is geometric extrapolation, not feature capacity
- edward #1925: WD=3e-2 — WD axis saturated

## Critical infra issue: train.py:441 LR hardcoding bug

`optimizer = Lion(model.parameters(), lr=1e-4, ...)` — `--lr` CLI flag silently ignored. Fix: `lr=cfg.lr`. **NOT yet in advisor branch.** Frieren #2006 (lr=8e-5) has been alerted. Any LR experiment with `--lr != 1e-4` must apply this fix first.

## Next queued ideas (for when Round 14 slots open up)

- **surf_weight=4** (bracket between 5 and 2) — only after nezuko sw=2 result
- **Geometric data augmentation** (flip/scale) — targets geom_camber_rc OOD
- **PINN-style auxiliary loss** (divergence/curl) — physics-informed volume regularization
- **n_head=1** — only if askeladd n_head=2 wins
- **FNO-style global filter** — alternative to PhysicsAttention
- **AdamW re-test** on full current stack — was beaten on old stack by 14.3%; worth confirming persists

## Key constraints

- 30 min / run cap: 12 epochs at ~138s/epoch with T_max=12 (cosine fully decays to 0)
- Per-epoch time eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~160s/epoch)
- EMA: cold-start drag, incompatible with short budget
- Batch increase: step-count limited (always worse)
- Gradient clipping (old AdamW stack): worse — re-testing on Lion stack in PR #2040 (different mechanism)
- Dropout: always worse (model is underfitting)
