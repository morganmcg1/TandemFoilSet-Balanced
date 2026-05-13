# SENPAI Research State

- **Date:** 2026-05-13 ~17:25
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 37.366** (n_head=4 + n_layers=3 + slice_num=24 + epochs=33, PR #2229)
**`test_avg/mae_surf_p` = 31.371**

> **Capacity floor identified:** n_layers=2 lost (PR #2230, +0.94% val). n_layers=3 is the depth floor.

> **Config note:** `train.py` default is `n_head=4` (line 392). All runs that did NOT pass `--n_head` explicitly have been running with n_head=4.

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 38.082 | 33.836 |
| geom_camber_rc | **51.356** | 45.411 |
| geom_camber_cruise | 20.702 | 16.874 |
| re_rand | 39.325 | 29.365 |
| **avg** | **37.366** | **31.371** |

**Reproduce:** `cd target/ && python train.py --epochs 33 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 3 --slice_num 24`

**best_epoch=33/33 STILL DESCENDING** at 53.7s/epoch — 29.5 min used, very tight margin. Per-epoch further reductions remain the primary lever (slice_num=16/12 in flight).

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
| alphonse | new | **slice_num=16 on n_layers=3+epochs=36** (continue partition sweep) | n_layers=3 |
| tanjiro | new | **slice_num=12 on n_layers=3+epochs=38** (floor probe — where does slice axis saturate?) | n_layers=3 |
| edward | new | **mlp_ratio=2 on n_layers=3+slice_num=24+epochs=33** (test lighter FFN at compact depth) | n_layers=3 |
| thorfinn | new | **lr=1.5e-4 on n_layers=3+slice_num=24+epochs=33** (LR retest at new baseline stack) | n_layers=3 |
| frieren | #2367 | **lr=2e-4 on n_layers=3+slice_num=24+epochs=33** (LR upper bracket at new baseline) | n_layers=3 |
| nezuko | #2279 | surf_weight=3 on n_layers=3+slice_num=32+epochs=27 (sw curve fill) | n_layers=3 |
| fern | #2301 | lr=1.5e-4 on n_layers=3+slice_num=32+epochs=30 (LR retest at old stack) | n_layers=3 |
| askeladd | #2375 | **slice_num=20 on n_layers=3+epochs=34** (fine-grain partition sweep fill) | n_layers=3 |

**Round summary:** All 8 on n_layers=3 stack. Primary focus is the partition sweep (slice_num=16/12) to continue the dominant mechanism. The frieren/nezuko/fern/askeladd PRs (#2274/#2279/#2301/#2248) are on the OLD slice_num=32 stack — they will need to beat the NEW baseline 37.366 to merge. If they test orthogonal axes that win at slice_num=32, those axes are worth compounding with slice_num=24.

**Merged this turn:** #2229 (alphonse slice_num=24, val=37.366, −2.36%) — new baseline
**Closed since baseline shift:** #2278 (edward mlp_ratio=6 +5.4%), #2273 (tanjiro warmup +1.66%), #2151 (thorfinn legacy n_layers=4 superseded), #2274 (frieren WD=0 +2.2%, WD axis flat), #2248 (askeladd sw=2 +3.13%, sw axis closed at n_layers=3, vol-grad mechanism confirmed but doesn't transfer to surface wins)

**Variance note (from #2274 student diagnosis):** Inter-run variance on identical configs is ~1.7 val units. Recent ~0.5 val improvements are at or near this noise floor — single-run signals require corroboration.

**Partition sweep ladder (Round 24 results pending):**
- slice_num=32 → val=39.143 (PR #2107, OLD baseline)
- slice_num=24 → val=37.366 (PR #2229, CURRENT baseline)
- slice_num=20 → askeladd #2375 (in flight)
- slice_num=16 → alphonse #2348 (in flight, body repaired this turn)
- slice_num=12 → tanjiro #2351 (in flight, floor probe)

## Confirmed mechanisms

**Epoch-budget (dominant lever):** Faster per-epoch → more epochs → better convergence. 11 consecutive best_epoch=final wins. PR #2229 confirms at slice_num=24: 33 epochs in 29.5 min, still descending.

**Dead at n_layers=3:**
- mlp_ratio=6 (+5.4%) — attention is the bottleneck, not FFN capacity
- Linear warmup (+1.66%) — compresses cosine tail
- sw=5/vol-gradient (+2.57%) — depth-sensitive, doesn't compound at compact stack

## Next priority hypotheses (when slots open after Round 24)

**Highest EV:**
1. **slice_num=8 floor probe** — if slice_num=12 wins, probe the hard floor
2. **Compound slice_num=16 + any winning orthogonal** — stack mechanisms once slice axis is resolved
3. **n_hidden=96 on compact stack** — trade width for epoch budget (~40s/epoch → ~44 epochs); untested

**Medium priority:**
4. **WD and LR on slice_num=24** — frieren/fern/askeladd/nezuko tests are at slice_num=32; if they show signal, retest at new baseline
5. **mlp_ratio=2 follow-up**: if edward's run confirms lighter FFN helps, try mlp_ratio=1 (pure attention, no FFN expansion)

**Research frontier ideas:**
- PINN-style auxiliary loss (divergence/curl regularization) — physics-informed volume constraint
- PhysicsAttention capacity floor: actively probing with slice_num=12/16 — where does attention quality degrade?

## Dead ends

- **Vol-gradient mechanism (surf_weight<10) at n_layers=3**: sw=5 regressed +2.57% vs baseline (PR #2245); mechanism is depth-sensitive and does not compound at compact stack. sw=2/sw=3 still in flight (askeladd #2248 / nezuko #2279).
- **LR axis fully saturated at 1e-4** on n_layers=4: lr=1.5e-4 neutral on n_layers=4+slice_num=32, lr=8e-5 regressed +12.4% vs old baseline. **Retesting lr=1.5e-4 at n_layers=3** (fern #2301 in flight).
- **surf_weight=15**: neutral on n_layers=4 — sw=10 near optimum in high direction
- **n_head=8**: +43% per-epoch, +15.7% worse
- **n_layers=7**: +4.6% (epoch budget dominates; 160s/epoch too slow)
- **mlp_ratio=6 at n_layers=3**: +5.4% worse (PR #2278) — attention bottleneck, not FFN; mlp_ratio=4 optimal at n_layers=3; mlp_ratio=2 now being tested (edward #2350)
- **mlp_ratio=2 at older stack**: +9.95%; mlp_ratio=8: +5.95% (both worse on old stack; mlp_ratio=4 optimal there too)
- **Linear epoch warmup (2 ep)**: +1.66% val worse (PR #2273) — compresses cosine T_max by 2 epochs, hurting the high-value late-stage descent; no benefit to Lion+L1 at lr=1e-4
- **mlp_ratio=6 at n_layers=3**: +5.4% val worse (PR #2278) — FFN capacity is not the bottleneck at depth 3
- **weight_decay=0 at compact stack**: +2.2% val worse vs new baseline (PR #2274), BUT confirmed compact stack does NOT overfit at WD=0 → WD axis is effectively FLAT. Future experiments can safely use WD=0 if convenient; not a winning lever.
- **surf_weight axis at n_layers=3**: sw=2 (+3.13%, PR #2248), sw=5 (+2.57%, PR #2245) both lose vs new baseline. Vol-gradient mechanism is confirmed active (val mae_vol_p −18%, test mae_vol_p −19% at sw=2) but vol→surface coupling can't compensate for the deeper-budget baseline advantage. sw=10 remains optimal at compact stack.
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
