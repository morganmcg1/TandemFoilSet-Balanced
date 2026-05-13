# SENPAI Research State

- **Date:** 2026-05-13 ~15:00
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 38.270** (n_head=4 + n_layers=3 + slice_num=32 + epochs=30, PR #2228)
**`test_avg/mae_surf_p` = 32.470**

> **Capacity floor identified:** n_layers=2 lost (PR #2230, +0.94% val). The single_in_dist and geom_camber_rc splits regressed — n_layers=3 is the depth floor. Stop pushing n_layers lower.

> **Config note:** `train.py` default is `n_head=4` (line 392). PR #2149 plumbed `--n_head` as CLI arg but did NOT change the default. All runs that did NOT pass `--n_head` explicitly have been running with n_head=4.

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 40.481 | 36.568 |
| geom_camber_rc | **52.042** | 46.624 |
| geom_camber_cruise | 20.785 | 16.956 |
| re_rand | 39.772 | 29.734 |
| **avg** | **38.270** | **32.470** |

**Reproduce:** `cd target/ && python train.py --epochs 30 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 3 --slice_num 32`

**best_epoch=30/30 STILL DESCENDING** at 58s/epoch — 28.5 min used, only 1.5 min margin. The epoch axis is now tight; further extensions require per-epoch speedups (slice_num=24 in flight, code-change axes available).

## What we've learned

### Big wins (merged)
1. **L1 loss**: −20.5% (PR #1358)
2. **GeGLU+Lion compound**: −25.3% (PR #1769)
3. **Lion optimizer lr=1e-4**: −14.3% (PR #1725)
4. **n_layers=6**: −9.4% (PR #1392)
5. **surf_weight=5**: −9.0% val / −9.8% test (PR #1836)
6. **T_max=12 (cosine aligned to epoch budget)**: −7.9% val / −8.9% test (PR #1793)
7. **mlp_ratio=4**: −5% (PR #1408)
8. **RMSNorm**: −2.9% val / −5.9% test (PR #1837) ← geom_camber_rc −17.2%
9. **bf16 mixed precision**: −0.34% (PR #1724)
10. **T_max=12 + surf_weight=5 compound**: −3.33% val (PR #1956)
11. **n_layers=5 + T_max=14**: −6.98% val / −6.98% test (PR #1995)
12. **slice_num=48 + T_max=15**: −1.33% val / −1.10% test (PR #1996)
13. **n_layers=4 + T_max=17**: −1.07% val / −2.17% test (PR #2080) ← lr=cfg.lr bug fixed
14. **slice_num=32 + T_max=21**: −7.6% val / −7.6% test (PR #2108)
15. **n_head=2 (head_dim=32→64)**: −0.25% val (PR #2149) ← plumbed --n_head CLI arg; n_head=4 is still the train.py default
16. **epochs=24 (3 extra cosine epochs)**: −6.21% val / −5.41% test (PR #2172) ← n_head=4 default at run time
17. **n_layers=3 + slice_num=32 + epochs=27**: −8.58% val / −9.02% test (PR #2107) ← BIGGEST SINGLE STEP; best_epoch=27 STILL DESCENDING

### Current stack (defaults + CLI overrides)
- L1 (MAE) loss in normalized space, **surf_weight=10**
- **n_layers=3** (PR #2107) ← CLI `--n_layers 3` (default is 5)
- **slice_num=32** (PR #2108) ← CLI `--slice_num 32` (default is 48)
- **n_head=4** — this is the actual train.py default (line 392). PR #2149 plumbed CLI arg but did NOT change the default.
- **epochs=27** (PR #2107) ← CLI `--epochs 27`
- **mlp_ratio=4, GeGLU activation** (PR #1769) — hardcoded in model_config at line 435
- **RMSNorm** (PR #1837)
- n_hidden=128
- Lion optimizer, lr=1e-4, weight_decay=1e-4
- **CosineAnnealingLR T_max=27** (auto-aligns: `T_max=MAX_EPOCHS=cfg.epochs`)
- bf16 mixed precision
- **~515K params** at n_layers=3 (down from 667K at n_layers=4)
- **~57s/epoch**, 27 epochs = 25.6 min total

## The epoch-count mechanism: trajectory so far

"Make epochs faster → fit more epochs → align T_max → better convergence":

| Step | Change | Per-epoch | Epochs | val Δ |
|---|---|---|---|---|
| PR #1995 | n_layers=6→5 | 116s→94s | 14 | −6.98% |
| PR #1996 | slice_num=64→48 | 94s→~100s | 15 | −1.33% |
| PR #2080 | n_layers=5→4 | ~94s | 17 | −1.07% |
| PR #2108 | slice_num=48→32 | 94s→74s | 21 | **−7.6%** |
| PR #2172 | epochs=21→24 | ~74s | 24 | **−6.21%** |
| PR #2107 | n_layers=4→3 + compound | 74s→57s | 27 | **−8.58%** |

**Gains are NOT diminishing.** The n_layers=3 step was the biggest single gain yet. best_epoch=final has held for 8+ consecutive experiments. The model is always budget-limited, never capacity-saturated.

**Current per-epoch timing at n_layers=3+slice_num=32: ~57s → 30 epochs = 28.5 min (fits in 30-min cap)**

## Active experiments (Round 22)

**New baseline: val=38.270 (PR #2228), test=32.470**

| Student | PR | Hypothesis | Stack |
|---------|-----|------------|-----------|
| tanjiro | #2273 | **Linear warmup (2 ep) + cosine** on n_layers=3+epochs=30 (code change) | n_layers=3 |
| alphonse | #2229 | slice_num=24 on n_layers=3+epochs=33 (in flight) | n_layers=3 |
| frieren | #2274 | weight_decay=0 on n_layers=3+epochs=30 (test if compact model needs reg) | n_layers=3 |
| fern | #2245 | surf_weight=5 on n_layers=3+epochs=27 (vol-gradient mechanism) | n_layers=3 |
| askeladd | #2248 | surf_weight=2 on n_layers=3+epochs=27 (bracket vol-gradient axis below) | n_layers=3 |
| nezuko | #2214 | surf_weight=5 on n_layers=4+slice_num=32+epochs=24 (n_head=4 confirmed) | n_layers=4 |
| edward | #2185 | mlp_ratio=6 on n_layers=4+slice_num=32+epochs=24 (legacy stack) | n_layers=4 |
| thorfinn | #2151 | slice_num=24 on n_layers=4 (legacy stack) | n_layers=4 |

**Round summary:** 5/8 students on the n_layers=3 stack testing orthogonal axes (warmup, slice_num, WD, surf_weight×2). 3/8 on legacy n_layers=4 stack — will need to beat 38.270 to win.

**Closed this turn:** #2213 (fern n_head=2+epochs=24, +2.01% val), #2193 (askeladd n_head=1, +12.4% val), #2230 (frieren n_layers=2, +0.94% val capacity floor identified).

## Next priority hypotheses (when slots open)

**Highest EV (direct extensions of current best):**
1. **surf_weight=5 on n_layers=3+slice_num=32**: nezuko is testing sw=5 on n_layers=4 — retest on new depth after that completes
2. **slice_num=16 on n_layers=3**: if alphonse #2229 (slice_num=24) wins, continue sweep
3. **n_layers=2 + slice_num=24**: compound if both individual wins land
4. **n_head=2 on n_layers=3+slice_num=32**: fern's test is on n_layers=4; needs retest at new depth
5. **mlp_ratio=6 on n_layers=3**: edward's test is on n_layers=4; needs retest at new depth

**Medium priority (orthogonal axes not yet fully explored):**
6. **surf_weight=2 on n_layers=3**: sw axis not fully swept at new depth
7. **lr=1.5e-4 on n_layers=3+slice_num=32**: prior test was neutral at n_layers=4 — may differ at n_layers=3 (different capacity/speed tradeoff)
8. **n_hidden reduction** (e.g., n_hidden=96): if n_layers=2 loses on capacity grounds, trading params for width might recover

**Research frontier ideas:**
- PINN-style auxiliary loss (divergence/curl regularization) — physics-informed volume constraint
- Geometric data augmentation (flip/scale) — targets geom_camber_rc OOD bottleneck
- Attention mechanism investigation: PhysicsAttention slice granularity interacts with slice_num — does the attention quality degrade below slice_num=16?

## Dead ends

- **LR axis fully saturated at 1e-4**: lr=1.5e-4 neutral on n_layers=4+slice_num=32, lr=8e-5 regressed +12.4% vs new baseline
- **surf_weight=15**: neutral on n_layers=4 — sw=10 near optimum in high direction
- **n_head=8**: +43% per-epoch, +15.7% worse
- **n_layers=7**: +4.6% (epoch budget dominates; 160s/epoch too slow)
- **mlp_ratio=2**: +9.95%; mlp_ratio=8: +5.95% (both worse, 4 optimal)
- **grad-clip**: worse for Lion (sign-update already handles magnitude)
- **DropPath**: needs 100-300 epoch budgets; useless at 20-30 epoch budgets
- **Dropout**: always worse (model is underfitting)
- **n_head=2 on n_layers=4**: marginal win (+0.25%), but current best is n_layers=3+n_head=4 anyway
- See full dead-ends list in older entries above for complete history

## Key constraints

- 30 min / run cap: ~57s/epoch at n_layers=3+slice_num=32 → max ~31 epochs
- n_layers=2 estimate: ~48s/epoch → max ~35 epochs
- slice_num=24 estimate: ~50s/epoch → max ~33 epochs
- n_head, mlp_ratio do not dramatically change per-epoch time at current config
- Batch increase: step-count limited (always worse)
- EMA: cold-start drag, incompatible with short budgets

## Infrastructure notes

- **lr=cfg.lr bug FIXED** (PR #2080): Lion uses `cfg.lr`; `--lr` flag works correctly.
- **slice_num CLI arg** (PR #2108): `--slice_num` accepted in Config.
- **n_head CLI arg** (PR #2149): `--n_head` accepted in Config. **Default is still 4** (train.py line 392: `n_head: int = 4`).
- **mlp_ratio=4 hardcoded** at train.py line 435: `mlp_ratio=4,` — not in Config dataclass; requires code change to vary.
