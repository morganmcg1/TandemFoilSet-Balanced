# SENPAI Research State

- **Date:** 2026-05-13 ~13:20
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 42.709** (n_head=2 + n_layers=4 + slice_num=32 + T_max=21, PR #2149)
**`test_avg/mae_surf_p` = 36.784**

**⚠ Note:** best_epoch=21 STILL DESCENDING. Per-head capacity wins over diversity at n_layers=4+slice_num=32. n_head=2 now a CLI arg.

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 45.089 | 41.257 |
| geom_camber_rc | **57.248** | 50.023 |
| geom_camber_cruise | 25.495 | 21.336 |
| re_rand | 43.004 | 34.519 |
| **avg** | **42.709** | **36.784** |

**Reproduce:** `cd target/ && python train.py --epochs 21 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 4 --slice_num 32 --n_head 2`

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
13. **n_layers=4 + T_max=17**: −1.07% val / −2.17% test (PR #2080) ← same mechanism; 17 epochs at ~94s/epoch; lr=cfg.lr bug fixed; best_epoch=17 STILL DESCENDING
14. **slice_num=32 + T_max=21**: −7.6% val / −7.6% test (PR #2108) ← 74s/epoch, 21 epochs; all 8 per-split metrics improved; val still descending at epoch 21
15. **n_head=2 (head_dim=32→64)**: −0.25% val / −0.31% test (PR #2149) ← per-head capacity > diversity on compact stack; mixed per-split (re_rand −2.38%, rc test −2.06%, single_in_dist slightly regressed)

### Current stack (defaults + CLI overrides)
- L1 (MAE) loss in normalized space, **surf_weight=10**
- **n_layers=4** (PR #2080) ← CLI `--n_layers 4` (default is 5)
- **slice_num=32** (PR #2108) ← CLI `--slice_num 32` (default is 48)
- **n_head=2** (PR #2149) ← CLI `--n_head 2` (default is 4)
- **mlp_ratio=4, GeGLU activation** (PR #1769)
- **RMSNorm** (PR #1837, replaces LayerNorm)
- n_hidden=128
- Lion optimizer, **lr=cfg.lr** (bug fixed in PR #2080), lr=1e-4, weight_decay=1e-4
- **CosineAnnealingLR T_max=21** (=epochs, aligned)
- bf16 mixed precision (autocast)
- **21 epochs in 30 min** (~65s/epoch; best_epoch=21 STILL DESCENDING)
- n_params: 708,875 (+6.3% vs PR #2108 667,923)

**Reproduce command:** `cd target/ && python train.py --epochs 21 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 4 --slice_num 32 --n_head 2`

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
- **mlp_ratio=2**: +9.95% vs PR #1996 baseline (PR #2007, n_layers=6 stack — 30% param savings, ~10% per-epoch speedup; speedup too small to unlock new epoch budget; mlp_ratio axis now bracketed: 2 worse, 4 optimal, 8 worse)
- **grad-clip max_norm=1.0**: +14.1% vs current baseline (PR #2040 — grad norms 20–140, max_norm=1 fires 100% of batches; too aggressive; Lion sign-update already handles magnitude; dead end for this stack)
- **DropPath rate=0.1**: +25.2% vs current baseline (PR #2043 — model is underfitting at 12-17 epoch budgets; DropPath needs 100-300 epochs to extract ensemble benefit; OOD suffered MORE not less because reg only helps OOD post-convergence)
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
2. **Budget is the constraint**: 30 min → 17 epochs with current n_layers=4 stack (~94s/epoch). best_epoch=17 was STILL DESCENDING — model not saturated yet, schedule expires first.
3. **L1 loss in normalized space is validated**: channel-weighted loss hurts on GeGLU+Lion (+11%); GeGLU gates do implicit channel balancing — the gradient channel weights must stay equal.
4. **surf_weight=5 mechanism confirmed**: halving surface:volume ratio reallocates L1 gradient to volume nodes → richer volumetric features → better surface via geometric context. All 4 splits improved, vol MAE improved −7% to −26%.
5. **AdamW hyperparameter space is exhausted**: All optimizer knobs tested. Lion is the new baseline optimizer.
6. **LR axis saturated at 1e-4**: lr=1.5e-4 (+4.21%), lr=2e-4 (+0.98% val) both worse. lr=8e-5 being tested (frieren #2006).
7. **WD axis saturated at 1e-4**: WD=1e-2 confirmed +16.6% worse vs current baseline (after rebasing to T_max=12+sw=5 stack). RMSNorm+GeGLU already provides implicit regularization.
8. **RMSNorm shifts the hardest split**: After RMSNorm, geom_camber_rc improved −17.2%; it remains the single highest-loss split at val=64.886.
9. **geom_camber_rc (64.886 val) is the dominant bottleneck** — primary target for further improvement.

## Active experiments (Round 18)

⚠ **Baseline now at 42.815 (PR #2108 slice_num=32).** Two PRs reviewed this turn — both sent back as compound tests against the new baseline. All running students now target the new compact stack.

| Student | PR | Hypothesis | Base stack |
|---------|-----|------------|-----------|
| thorfinn | #2151 | slice_num=24 + n_layers=4 (slice sweep step 3) | NEW slice_num=32 stack |
| askeladd | #2193 | n_head=1 on NEW stack (bracket per-head capacity axis) | NEW n_head=2 stack |
| frieren | #2150 | lr=8e-5 on NEW stack (LR bracket below default) | NEW slice_num=32 stack |
| fern | #2172 | epochs=24 + T_max=24 on slice_num=32 (squeeze epoch budget) | NEW slice_num=32 stack |
| tanjiro | #2107 (sent back) | **n_layers=3 + slice_num=32 + T_max=27** (compound depth+slice) | NEW slice_num=32 stack |
| alphonse | #2134 (sent back) | **lr=1.5e-4 + slice_num=32 + n_layers=4** (compound LR+slice) | NEW slice_num=32 stack |
| nezuko | #2109 | surf_weight=2 + n_layers=4 (still on OLD slice_num=48) | OLD slice_num=48 stack |
| edward | #2185 | mlp_ratio=6 + slice_num=32 (MLP capacity midpoint) | NEW slice_num=32 stack |

**Recently merged:**
- askeladd #2149: n_head=2 on n_layers=4+slice_num=32 (−0.25%/−0.31%) ← **NEW BASELINE 42.709/36.784** (n_head axis: per-head capacity > diversity on compact stack; n_head CLI arg now plumbed)

**Recently reviewed:**
- edward #2143: sw=15 gave essentially flat val (−0.44% vs OLD), slightly worse test (+1.14% OLD); +7.77% vs NEW baseline. CLOSED. sw axis bracket on n_layers=4 nearly complete (sw=2 nezuko in flight remains). Reassigned to mlp_ratio=6 on new stack.
- alphonse #2134: lr=1.5e-4 gave −2.24% vs OLD baseline, but +5.81% vs NEW. SENT BACK for compound lr=1.5e-4 + slice_num=32 test.
- tanjiro #2107: n_layers=3 gave −7.16% vs OLD baseline, +0.49% vs NEW (extraordinarily close). SENT BACK for compound n_layers=3 + slice_num=32 test.
- fern #2062: stale 4h WIP on superseded stack. CLOSED; reassigned to epochs=24 on new stack (#2172).

**Recently merged:**
- thorfinn #2108: slice_num=32 + n_layers=4 (−7.6% val) ← **NEW BASELINE 42.815/36.899** (best_epoch=21 STILL DESCENDING)
- tanjiro #2080: n_layers=4 + T_max=17 (−1.07% val, lr bug fixed)
- fern #1996: slice_num=48 + T_max=15 (−1.33% val)

**Recently closed:**
- edward #2143: surf_weight=15 on n_layers=4 (+7.77% vs current) — neutral on OLD baseline, surf↔vol trade saturated on n_layers=4
- fern #2062: n_layers=5 + slice_num=48 (stale, superseded by #2080 + #2108)
- edward #2048: surf_weight=5 on n_layers=5 (+3.16% vs current) — vol-gradient mechanism active but stack-depth dependent
- askeladd #2038: n_head=2 on old n_layers=6 stack (+12.4% vs current) — draft never readied; retesting on compact stack
- frieren #2006: lr=8e-5 never started (old stack, lr bug present) — clean retest on new stack assigned
- alphonse #2043: DropPath rate=0.1 (+25.2% vs current) — DropPath needs 100-300 epoch budgets
- thorfinn #2040: grad-clip max_norm=1.0 (+14.1% vs current) — max_norm=1 wrong regime
- nezuko #2029: surf_weight=2 on old n_layers=6 stack (+6.32% vs current) — direction strong, retesting on new stack

## Infrastructure notes

- **lr=cfg.lr bug FIXED** (PR #2080): Lion now uses `cfg.lr`; `--lr` flag works correctly.
- **slice_num plumbed as CLI arg** (PR #2108): `--slice_num` now accepted in Config dataclass; model reads `slice_num=cfg.slice_num`.

## The epoch-count mechanism: trajectory so far

"Make epochs faster → fit more epochs → align T_max → better convergence":
- n_layers=6→5: 116s/epoch → 14 epochs (PR #1995, val −6.98%)
- slice_num=64→48: 123s→ ~100s/epoch → 15-16 epochs (PR #1996, val −1.33%)
- n_layers=5→4: 94s/epoch → 17 epochs (PR #2080, val −1.07%) — best_epoch=17 STILL DESCENDING
- slice_num=48→32: 94s→74s/epoch → 21 epochs (PR #2108, val **−7.6%**) — best_epoch=21 STILL DESCENDING
- slice_num=32→24: ~74s→~65s/epoch → ~26 epochs (PR #2151, in flight)
- n_layers=4→3: ~75s/epoch → ~22 epochs (PR #2107, in flight; but at OLD slice_num=48 stack)

**Gains are NOT diminishing** — PR #2108 at −7.6% reversed the trend. The slice_num axis appears orthogonal to the depth axis and has substantial headroom. Both axes produce wins via the same epoch-budget mechanism.

**Critical open questions:**
1. Does slice_num=24 continue the trend or do we hit the PhysicsAttention granularity floor? (thorfinn #2151)
2. Does n_layers=3 break the model (too shallow) or compound with slice_num=32? (tanjiro #2107 on old stack)
3. Does lr=8e-5 help on the new compact stack? (frieren #2150)
4. Does n_head=2 benefit the compact stack (less per-head capacity per 32-slice partition)? (askeladd #2149)

## Next queued ideas (for when Round 18 slots open up)

- **slice_num=16**: if #2151 wins, continue the floor-finding sweep
- **n_layers=3 + slice_num=32**: compound both depth and slice axes if both win independently
- **surf_weight=2 on new slice_num=32 stack**: nezuko #2109 ran on OLD stack; if direction holds, retest on new
- **surf_weight=1**: next step in gradient sweep if nezuko wins
- **lr combinations**: once #2150 (frieren lr=8e-5) and #2134 (alphonse lr=1.5e-4) land, narrow in on optimal
- **Geometric data augmentation** (flip/scale) — targets geom_camber_rc OOD bottleneck (~56.7 val on new baseline)
- **PINN-style auxiliary loss** (divergence/curl regularization) — physics-informed volume constraint

## Key constraints

- 30 min / run cap: 12 epochs at ~138s/epoch with T_max=12 (cosine fully decays to 0)
- Per-epoch time eliminates: n_head=8 (+43%), slice_num=128 (+12%), n_layers=7 (~160s/epoch)
- EMA: cold-start drag, incompatible with short budget
- Batch increase: step-count limited (always worse)
- Gradient clipping (old AdamW stack): worse — re-testing on Lion stack in PR #2040 (different mechanism)
- Dropout: always worse (model is underfitting)
