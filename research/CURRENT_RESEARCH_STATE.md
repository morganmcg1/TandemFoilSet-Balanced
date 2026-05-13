# SENPAI Research State

- **Date:** 2026-05-13 ~11:00
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 46.847** (slice_num=48 + T_max=15, PR #1996 result; merged into n_layers=5 advisor code)
**`test_avg/mae_surf_p` = 40.837**

⚠ **Compound code/measurement mismatch:** PR #1996 result was on n_layers=6 + slice_num=48 + T_max=15. The advisor code now has n_layers=5 + slice_num=48. fern #2062 is verifying the actual compound.

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 50.491 | 45.728 |
| geom_camber_rc | **60.364** | 55.146 |
| geom_camber_cruise | 29.835 | 24.157 |
| re_rand | 46.699 | 38.317 |
| **avg** | **46.847** | **40.837** |

**Reproduce (what was actually measured):** `python train.py --epochs 15 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10`  ← n_layers=6 in fern's config; advisor code is now n_layers=5

> **Current advisor code:** n_layers=5 + slice_num=48 + T_max=14 (default, may need update to T_max=15 or 16 for new epoch count). surf_weight=5 not yet tested on this stack.

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
10. **T_max=12 + surf_weight=5 compound**: −3.33% val / −1.29% test (PR #1956) ← volume MAE −6% to −14% across all splits
11. **n_layers=5 + T_max=14**: −6.98% val / −6.98% test (PR #1995) ← epoch count was binding constraint; 14 vs 12 epochs in 30-min budget; ALL 4 splits improved; −20% VRAM
12. **slice_num=48 + T_max=15**: −1.33% val / −1.10% test (PR #1996) ← same mechanism; 15 epochs at ~123s each on n_layers=6; compound with n_layers=5 pending (fern #2062)

### Current stack (defaults + CLI overrides)
- L1 (MAE) loss in normalized space, **surf_weight=10** (sw=5 compound pending edward #2048)
- **n_layers=5** (PR #1995) ← shallower enables more epochs in 30-min budget
- **slice_num=48** (PR #1996) ← updated; fewer slices → further per-epoch speedup
- **mlp_ratio=4, GeGLU activation** (PR #1769)
- **RMSNorm** (PR #1837, replaces LayerNorm)
- n_hidden=128, n_head=4
- Lion optimizer lr=1e-4, weight_decay=1e-4
- **CosineAnnealingLR T_max=? (pending fern #2062)** — T_max=14 default; compound ~100-108s/epoch → 16-17 epochs possible
- bf16 mixed precision (autocast)
- **~14-16 epochs in 30 min** (n_layers=5 + slice_num=48 untimed, pending fern #2062 verification)
- n_params: 976,827 (slice_num doesn't change parameter count meaningfully)

**Reproduce command (when compound verified by fern #2062):** `python train.py --epochs 15 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10`

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
- **mlp_ratio=2**: +9.95% vs current baseline (PR #2007, n_layers=6 stack — 30% param savings, ~10% per-epoch speedup; speedup too small to unlock new epoch budget; mlp_ratio axis now bracketed: 2 worse, 4 optimal, 8 worse)
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

## Active experiments (Round 16/17)

⚠ **Baseline shifting rapidly:** Most in-flight PRs (#2006, #2029, #2038, #2040, #2043) are on OLD n_layers=6 + T_max=12 stack. Their results will likely be val ~50-53, worse than new baseline (46.847). Review strategy: note relative improvement for potential compound-stack re-test; close if they don't beat 46.847 and don't show unusual signal.

| Student | PR | Hypothesis | Base stack |
|---------|-----|------------|-----------|
| edward | #2048 | surf_weight=5 on n_layers=5+T_max=14 | NEW n_layers=5 |
| fern | #2062 | n_layers=5 + slice_num=48 compound verification + T_max tuning | NEW compound |
| tanjiro | #2080 | n_layers=4 + T_max=17 (continue depth sweep) | NEW compound |
| frieren | #2006 | Lion lr=8e-5 (⚠ apply lr=cfg.lr fix) | OLD n_layers=6 |
| nezuko | #2029 | surf_weight=2 | OLD n_layers=6 |
| askeladd | #2038 | n_head=2 | OLD n_layers=6 |
| thorfinn | #2040 | grad-clip max_norm=1.0 | OLD n_layers=6 |
| alphonse | #2043 | DropPath rate=0.1 | OLD n_layers=6 |

**Recently merged:**
- fern #1996: slice_num=48 + T_max=15 (−1.33% val) ← **NEW BASELINE 46.847/40.837** (⚠ measured on n_layers=6)
- edward #1995: n_layers=5 + T_max=14 (−6.98% val) 
- nezuko #1956: T_max=12 + surf_weight=5 compound (−3.33% val)

**Recently closed:**
- tanjiro #2007: mlp_ratio=2 (+9.95% worse vs 46.847) — parameter savings don't buy enough per-epoch speedup to unlock new epoch budget
- alphonse #1765: Lion lr=1.5e-4 (+4.21% worse) — LR axis saturated
- askeladd #1766: Lion WD=1e-2 rerun (+16.6% worse) — WD axis saturated
- thorfinn #1948: surf_weight=3 stale draft

## Critical infra issue: train.py:441 LR hardcoding bug

`optimizer = Lion(model.parameters(), lr=1e-4, ...)` — `--lr` CLI flag silently ignored. Fix: `lr=cfg.lr`. **NOT yet in advisor branch.** Frieren #2006 (lr=8e-5) has been alerted. Any LR experiment with `--lr != 1e-4` must apply this fix first.

## The epoch-count mechanism: key insight and open questions

The dominant win pattern this session has been "make epochs faster → fit more epochs → align T_max → better convergence":
- n_layers=5 → 116s/epoch → 14 epochs (PR #1995, val −6.98%)
- slice_num=48 → 123s/epoch on n_layers=6 → 15 epochs (PR #1996, val −1.33%)
- n_layers=5 + slice_num=48 → ~100-108s/epoch → 16-17 epochs (pending fern #2062)

**Key open questions:**
1. Does n_layers=4 + T_max=~18 continue the pattern? (n_layers=4 → ~90-95s/epoch → 18-19 epochs)
2. Does surf_weight=5 compound onto the new stack? (edward #2048, in flight)
3. Is there a per-epoch floor? At some point adding epochs has diminishing returns even with T_max alignment.

## Next queued ideas (for when Round 17 slots open up)

- **n_layers=3 + T_max=~20**: if n_layers=4 wins, push one step further; if n_layers=4 loses, bracket confirmed at 4–5
- **surf_weight=5 on n_layers=5+slice_num=48**: after fern #2062 verifies compound (edward #2048 already testing on n_layers=5 but pre-slice_num=48)
- **T_max alignment after fern #2062**: if n_layers=5+slice_num=48 gives ~100-108s/epoch, T_max=16 or 17 may be better than 15
- **Any winning axis from old-stack screening** (n_head=2, grad-clip, droppath) re-tested on current compound stack
- **Geometric data augmentation** (flip/scale) — targets geom_camber_rc OOD
- **PINN-style auxiliary loss** (divergence/curl) — physics-informed volume regularization
- **lr bug fix PR**: apply train.py:441 fix (`lr=cfg.lr`) as a standalone change to advisor branch

## Key constraints

- 30 min / run cap: 12 epochs at ~138s/epoch with T_max=12 (cosine fully decays to 0)
- Per-epoch time eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~160s/epoch)
- EMA: cold-start drag, incompatible with short budget
- Batch increase: step-count limited (always worse)
- Gradient clipping (old AdamW stack): worse — re-testing on Lion stack in PR #2040 (different mechanism)
- Dropout: always worse (model is underfitting)
