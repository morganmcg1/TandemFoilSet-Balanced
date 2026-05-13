# SENPAI Research State

- **Date:** 2026-05-13 15:38
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 35.969** (n_head=4 + n_layers=3 + slice_num=12 + epochs=36, PR #2351)
**`test_avg/mae_surf_p` = 30.265**

> **Capacity floor NOT found at slice_num=12.** The floor probe won decisively (−3.74%). Partition-sweep mechanism still dominant.

> **Config note:** `train.py` default is `n_head=4` (line 392). All runs that did NOT pass `--n_head` explicitly have been running with n_head=4.

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 36.308 | 33.241 |
| geom_camber_rc | **49.521** | 43.631 |
| geom_camber_cruise | 19.576 | 15.969 |
| re_rand | 38.470 | 28.220 |
| **avg** | **35.969** | **30.265** |

**Reproduce:** `cd target/ && python train.py --epochs 36 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 3 --slice_num 12`

**best_epoch=36/36 STILL DESCENDING** at 50.3s/epoch — 30.2 min used, HIT CAP. Partition sweep continuing to slice_num=8.

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
18. **epochs=30 on n_layers=3+slice_num=32**: −2.20% val (PR #2228) ← best_epoch=30 STILL DESCENDING
19. **slice_num=24 + epochs=33 on n_layers=3**: −2.36% val / −3.38% test (PR #2229) ← ~54s/epoch, 33 epochs, STILL DESCENDING
20. **slice_num=12 + epochs=36 on n_layers=3**: −3.74% val / −3.53% test (PR #2351) ← ~50s/epoch, 36 epochs, HIT CAP STILL DESCENDING

### Current stack (defaults + CLI overrides)
- L1 (MAE) loss in normalized space, **surf_weight=10**
- **n_layers=3** (PR #2107) ← CLI `--n_layers 3` (default is 5)
- **slice_num=24** (PR #2229) ← CLI `--slice_num 24` (default is 48)
- **n_head=4** — this is the actual train.py default (line 392). PR #2149 plumbed CLI arg but did NOT change the default.
- **epochs=33** (PR #2229) ← CLI `--epochs 33`
- **mlp_ratio=4, GeGLU activation** (PR #1769) — hardcoded in model_config at line 435
- **RMSNorm** (PR #1837)
- n_hidden=128
- Lion optimizer, lr=1e-4, weight_decay=1e-4
- **CosineAnnealingLR T_max=33** (auto-aligns: `T_max=MAX_EPOCHS=cfg.epochs`)
- bf16 mixed precision
- **~514K params** at n_layers=3+slice_num=24
- **~53.7s/epoch**, 33 epochs = 29.5 min total

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

## Active experiments (Round 24)

**NEW Baseline: val=37.366 (PR #2229), test=31.371**

| Student | PR | Hypothesis | Stack |
|---------|-----|------------|-----------|
| alphonse | #2348 | slice_num=16 on n_layers=3+epochs=36 (partition sweep — expected ~36.5, CANNOT beat new baseline) | n_layers=3 |
| askeladd | #2375 | slice_num=20 on n_layers=3+epochs=34 (partition sweep — expected ~36.9, CANNOT beat new baseline) | n_layers=3 |
| tanjiro | #2408 | **slice_num=8 on n_layers=3+epochs=38** (next partition floor probe) | n_layers=3 |
| fern | #2409 | **lr=1.5e-4 on n_layers=3+slice_num=12+epochs=36** (LR at NEW baseline stack) | n_layers=3 |
| edward | #2383 | n_head=2 on n_layers=3+slice_num=24+epochs=33 (stale stack — informative for axis, cannot win) | n_layers=3 |
| thorfinn | #2417 | **n_head=2 on n_layers=3+slice_num=12+epochs=36** (attention-head axis at NEW stack) | n_layers=3 |
| frieren | #2402 | lr=5e-5 on n_layers=3+slice_num=24+epochs=33 (stale stack — informative for axis, cannot win) | n_layers=3 |
| nezuko | #2404 | n_head=1 on n_layers=3+slice_num=24+epochs=33 (stale stack — informative for axis, cannot win) | n_layers=3 |

**⚠️ NOTE:** alphonse/askeladd/edward/thorfinn/frieren/nezuko all testing at slice_num=24 or older stacks. Cannot beat new baseline 35.969. Will confirm axes when results land but will be closed.

**Merged round 27:** #2351 (tanjiro slice_num=12, val=35.969, −3.74%) — new baseline
**Closed this turn:** #2353 (thorfinn lr=1.5e-4 at old stack — won vs old, lost vs new; LR signal recorded), #2301 (fern lr=1.5e-4 older stack)

**Key signal from #2353:** lr=1.5e-4 beats lr=1e-4 at slice_num=24 (−1.41% val). This validates fern's in-flight #2409 (lr=1.5e-4 at slice_num=12). If the signal carries over to the new stack, fern's experiment could yield val ~35.4.

**Complete baseline trajectory:** 40.158 → 39.143 → 38.270 → 37.366 → **35.969** (−10.5% total from round start)

**Variance note (from #2274 student diagnosis):** Inter-run variance on identical configs is ~1.7 val units. Recent ~0.5 val improvements are at or near this noise floor — single-run signals require corroboration.

**Partition sweep ladder (Round 24 results pending):**
- slice_num=32 → val=39.143 (PR #2107, OLD baseline)
- slice_num=24 → val=37.366 (PR #2229, CURRENT baseline)
- slice_num=20 → askeladd #2375 (in flight)
- slice_num=16 → alphonse #2348 (in flight, body repaired this turn)
- slice_num=12 → tanjiro #2351 (in flight, floor probe)

## Confirmed mechanisms

**Epoch-budget (dominant lever):** Faster per-epoch → more epochs → better convergence. **13 consecutive best_epoch=final wins.** Partition sweep: 32→24→12 each a large win. Mechanism shows NO sign of saturating.

**Dead at n_layers=3:**
- mlp_ratio=6 (+5.4%) — attention is the bottleneck, not FFN capacity
- Linear warmup (+1.66%) — compresses cosine tail
- sw=5/vol-gradient (+2.57%) — depth-sensitive, doesn't compound at compact stack

## Next priority hypotheses

**Highest EV (immediate):**
1. **slice_num=8**: IN FLIGHT (tanjiro #2408) — continue partition sweep
2. **slice_num=4 probe**: if slice_num=8 wins, try slice_num=4 (will likely fail on capacity but worth checking)
3. **LR at slice_num=12**: fern #2409 (lr=1.5e-4) in flight — LR axis at NEW stack
4. **Compound slice_num=12 + winning LR**: once fern lands, compound best-LR with current best partition

**Medium priority (after stale-stack tests land):**
5. **n_head=1/2 at slice_num=12**: once edward/nezuko results land at slice_num=24, re-run winning head count at new stack
6. **lr=5e-5 at slice_num=12**: retest frieren's probe at new stack if lr=5e-5 shows any signal at slice_num=24

**Partition sweep state:**
- slice_num=32: val=39.143
- slice_num=24: val=37.366 (BASELINE −2.36%)
- slice_num=20: askeladd #2375 (in flight, expected ~36.9, cannot win vs 35.969)
- slice_num=16: alphonse #2348 (in flight, expected ~36.5, cannot win vs 35.969)
- slice_num=12: **35.969 (CURRENT BASELINE)**
- slice_num=8: tanjiro #2408 (IN FLIGHT)
- slice_num=4: next if 8 wins (may fail on capacity)

**LR axis state at slice_num=12 (NEW stack):**
- lr=5e-5 (NOT YET TESTED at slice_num=12)
- lr=1e-4 (BASELINE, val=35.969)
- lr=1.5e-4 (fern #2409, IN FLIGHT)
- lr=2e-4 (tested at slice_num=24, lost +4.4%)

**Research frontier ideas:**
- PINN-style auxiliary loss (divergence/curl regularization) — physics-informed volume constraint
- PhysicsAttention capacity floor: actively probing with slice_num=12/16 — where does attention quality degrade?

## Dead ends

- **Vol-gradient mechanism (surf_weight<10) at n_layers=3**: sw=5 regressed +2.57% vs baseline (PR #2245); mechanism is depth-sensitive and does not compound at compact stack. sw=2/sw=3 still in flight (askeladd #2248 / nezuko #2279).
- **LR axis fully saturated at 1e-4** on n_layers=4: lr=1.5e-4 neutral on n_layers=4+slice_num=32, lr=8e-5 regressed +12.4% vs old baseline. **Retesting lr=1.5e-4 at n_layers=3** (fern #2301 in flight).
- **surf_weight=15**: neutral on n_layers=4 — sw=10 near optimum in high direction
- **n_head=8**: +43% per-epoch, +15.7% worse
- **n_layers=7**: +4.6% (epoch budget dominates; 160s/epoch too slow)
- **mlp_ratio axis at n_layers=3 CLOSED**: mlp_ratio=6 (+5.4%, PR #2278) AND mlp_ratio=2 (+2.3%, PR #2350) both lose. mlp_ratio=4 confirmed optimal at compact stack. FFN width is load-bearing (can't cut) but not the limiting factor (can't expand). The mlp_ratio flag was hardcoded; student added CLI arg `--mlp_ratio`.
- **mlp_ratio axis fully closed**: mlp_ratio=2 +9.95% (old stack), +2.3% (PR #2350 new stack); mlp_ratio=6 +5.4% (PR #2278); mlp_ratio=8 +5.95%. mlp_ratio=4 confirmed optimal across all stacks.
- **Linear epoch warmup (2 ep)**: +1.66% val worse (PR #2273) — compresses cosine T_max by 2 epochs, hurting the high-value late-stage descent; no benefit to Lion+L1 at lr=1e-4
- **mlp_ratio=6 at n_layers=3**: +5.4% val worse (PR #2278) — FFN capacity is not the bottleneck at depth 3
- **weight_decay=0 at compact stack**: +2.2% val worse vs new baseline (PR #2274), BUT confirmed compact stack does NOT overfit at WD=0 → WD axis is effectively FLAT. Future experiments can safely use WD=0 if convenient; not a winning lever.
- **surf_weight axis at n_layers=3 FULLY CLOSED**: sw=2 (+3.13%), sw=3 (+3.87%), sw=5 (+2.57%) all lose vs new baseline. Vol-gradient mechanism confirmed active at all three (mae_vol_p −10 to −19%) but vol→surface coupling too weak to compensate. sw=10 remains optimal at compact stack.
- **lr=2e-4 at compact stack**: +4.4% val worse (#2367). Geometry-OOD splits hurt most. LR ceiling below 2e-4 confirmed.
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
- **mlp_ratio CLI arg added** (PR #2350): `--mlp_ratio` accepted via Config. Default is still 4 (optimal, axis now closed).
