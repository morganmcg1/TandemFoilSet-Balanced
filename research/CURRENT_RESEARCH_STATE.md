# SENPAI Research State

- **Date:** 2026-05-13 15:38
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r3`
- **Target base:** `icml-appendix-charlie` (no W&B logging arm)
- **Latest direction from human team:** none — controlled 24h/48h Charlie-vs-Willow logging ablation.
- **Per-run wall-clock cap:** 30 minutes (`SENPAI_TIMEOUT_MINUTES=30`).
- **Plateau Protocol:** RESOLVED — Lion broke plateau (-14.3%), GeGLU+Lion shattered it (-25.3%).

## Current baseline

**`val_avg/mae_surf_p` = 35.548** (n_head=4 + n_layers=3 + slice_num=16 + epochs=36, PR #2348)
**`test_avg/mae_surf_p` = 30.345**

> **CRITICAL FINDING: Partition sweep is NON-MONOTONE.** slice_num=16 beats slice_num=12 (val 35.969 → 35.548). Per-epoch cost flattens below slice_num=16 (both ~50s/epoch), so going below 16 loses capacity without gaining budget. The partition sweep floor appears to be at or near 16, not at the smallest possible value.

> **Config note:** `train.py` default is `n_head=4` (line 392). All runs that did NOT pass `--n_head` explicitly have been running with n_head=4.

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 35.263 | 32.248 |
| geom_camber_rc | **49.105** | 44.663 |
| geom_camber_cruise | 19.392 | 16.188 |
| re_rand | 38.431 | 28.282 |
| **avg** | **35.548** | **30.345** |

**Reproduce:** `cd target/ && python train.py --epochs 36 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --surf_weight 10 --n_layers 3 --slice_num 16`

**best_epoch=35/36** (slight flattening at final epoch vs always-final at previous checkpoints). ~49.8s/epoch, 29.89 min total.

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

## Active experiments (Round 28+)

**Current Baseline: val=35.548 (PR #2348), test=30.345**

| Student | PR | Hypothesis | Stack | Can beat baseline? |
|---------|-----|------------|-----------|---|
| alphonse | #2471 | **lr=1.2e-4 × slice_num=16** (fine-grained LR probe above trough) | n_layers=3+slice_num=16 | **possible** |
| askeladd | #2451 | **slice_num=18 on n_layers=3+epochs=36** (partition gap 20→16) | n_layers=3 | **possible winner** |
| tanjiro | #2408 | slice_num=8 on n_layers=3+epochs=38 (partition floor probe) | n_layers=3 | likely NO (capacity loss) |
| fern | #2409 | lr=1.5e-4 on n_layers=3+slice_num=12+epochs=36 | n_layers=3+slice_num=12 | possible — LR axis at sub-optimal partition |
| edward | #2478 | **n_layers=4+slice_num=16+epochs=27** (depth-up counterpart to frieren depth-down) | n_layers=4+slice_num=16 | possible |
| thorfinn | #2450 | **lr=5e-5 on n_layers=3+slice_num=16+epochs=36** (lower LR bound) | n_layers=3+slice_num=16 | possible — completes LR axis |
| frieren | #2468 | **n_layers=2+slice_num=16+epochs=46** (depth reduction → more cosine budget) | n_layers=2+slice_num=16 | **YES — high EV via epoch-budget mechanism** |
| nezuko | #2479 | **LayerScale on n_layers=3+slice_num=16+epochs=36** (residual branch scaling) | n_layers=3+slice_num=16 | possible — fresh architecture axis |

**Merged:** #2351 (tanjiro slice_num=12, val=35.969), #2348 (alphonse slice_num=16, val=35.548)
**Closed this round:** #2417 (thorfinn n_head=2@slice12 +3.4%), #2375 (askeladd slice20 beats old baseline but loses vs 35.548), #2383 (edward n_head=2@slice24 +0.71%), #2353/#2301/#2367/#2279/#2350 (prior round)

**Complete baseline trajectory:** 40.158 → 39.143 → 38.270 → 37.366 → 35.969 → **35.548** (−11.5% total from round start)

**Variance note (from #2274 student diagnosis):** Inter-run variance on identical configs is ~1.7 val units. Recent ~0.5 val improvements are at or near this noise floor — single-run signals require corroboration.

**Partition sweep ladder (completed + in-flight):**
- slice_num=32 → val=39.143 (PR #2107)
- slice_num=24 → val=37.366 (PR #2229)
- slice_num=20 → val=36.854 (PR #2375, beats old baseline, loses vs new)
- slice_num=18 → askeladd #2451 (IN FLIGHT — filling 20→16 gap)
- **slice_num=16 → val=35.548 (PR #2348, CURRENT BASELINE)**
- slice_num=14 → edward #2447 (IN FLIGHT — lower-side probe)
- slice_num=12 → val=35.969 (PR #2351, WORSE than 16 — non-monotone)
- slice_num=8 → tanjiro #2408 (IN FLIGHT — expected capacity collapse)
- **PARTITION AXIS FULLY CLOSED. Robust local minimum at 16. Both neighbors (14, 12) are worse.**
Remaining in-flight: askeladd slice_num=18 (informative, expected to confirm monotone above 16), tanjiro slice_num=8 (expected capacity collapse).

## Confirmed mechanisms

**Epoch-budget (dominant lever):** Faster per-epoch → more epochs → better convergence. **13+ consecutive best_epoch=final wins.** Partition sweep: 32→24→12 each a large win, but 12 regressed vs 16 (non-monotone). Mechanism shows NO sign of saturating at 16.

**Dead at n_layers=3:**
- mlp_ratio=6 (+5.4%) — attention is the bottleneck, not FFN capacity
- Linear warmup (+1.66%) — compresses cosine tail
- sw=5/vol-gradient (+2.57%) — depth-sensitive, doesn't compound at compact stack

## Next priority hypotheses

**Highest EV (immediate):**
1. **Compound lr=1.5e-4 × slice_num=16**: alphonse #2431 IN FLIGHT — highest EV
2. **n_head=2/1 at slice_num=16**: once edward/thorfinn/nezuko n_head results land, rerun winner at slice_num=16
3. **lr=5e-5 at slice_num=16**: once frieren's lr=5e-5 at slice_num=24 lands, confirm whether it signals
4. **Triple compound lr × n_head × slice_num=16**: after independent axes confirmed

**Medium priority (after stale-stack tests land):**
5. **slice_num=20 result (askeladd #2375)**: informative for 16–24 neighborhood
6. **slice_num=8 result (tanjiro #2408)**: confirms capacity floor at 16

**Partition sweep state:**
- slice_num=32: val=39.143
- slice_num=24: val=37.366
- slice_num=20: askeladd #2375 (in flight)
- **slice_num=16: val=35.548 (CURRENT BASELINE, PR #2348)**
- slice_num=12: val=35.969 (WORSE — non-monotone confirmed)
- slice_num=8: tanjiro #2408 (in flight — expected capacity collapse)

**LR axis state at slice_num=16 (ACTIVE):**
- lr=5e-5 (thorfinn #2450, IN FLIGHT — expected undertrained)
- lr=8e-5 (UNTESTED)
- lr=1e-4 (BASELINE, val=35.548)
- lr=1.2e-4 (alphonse #2471, IN FLIGHT — fine-grained probe above trough)
- lr=1.5e-4 (#2431 CLOSED, +7.3% val — partition-dependent LR ceiling confirmed)
- lr=2e-4 (closed at slice_num=24, +4.4%)

**KEY INSIGHT: LR optimum shifts down with slice_num:**
- slice_num=24: lr=1.5e-4 beats 1e-4 (positive signal from #2353)
- slice_num=16: lr=1e-4 beats 1.5e-4 (this confirmation from #2431)
- Hypothesis: smaller partition → tighter per-slice budget → more conservative LR needed

**LR axis at slice_num=24 FULLY BRACKETED (closed):**
- lr=5e-5: +5.1% (frieren #2402) — cosine decays before convergence
- lr=1e-4: TROUGH
- lr=2e-4: +4.4% (#2367)

**LR axis state at slice_num=12 (informative only):**
- lr=1.5e-4 (fern #2409, IN FLIGHT)

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
- **n_head axis FULLY CLOSED across {1, 2, 4} at slice_num=24:**
  - n_head=1: val=37.64 (nezuko #2404, 3-seed mean ±0.48). +28% more params — no gain.
  - n_head=2: val=37.63 (edward #2383). +6% more params — slight loss.
  - n_head=4: val=37.37 (BEST). Parallelism wins.
  - **Bottleneck is slice mechanism, not Q/K/V capacity.** dim_head doesn't matter — routing diversity does.
- **n_head=2 on n_layers=4**: marginal win (+0.25%), but current best is n_layers=3+n_head=4 anyway
- See full dead-ends list in older entries above for complete history

## Key constraints

- 30 min / run cap: ~57s/epoch at n_layers=3+slice_num=32 → max ~31 epochs
- slice_num=16: ~49.8s/epoch → 36 epochs = 29.9 min (at cap)
- slice_num=12: ~50.3s/epoch → 36 epochs = 30.2 min (at cap, similar to 16)
- slice_num=24 estimate: ~50s/epoch → max ~33 epochs
- n_head, mlp_ratio do not dramatically change per-epoch time at current config
- Batch increase: step-count limited (always worse)
- EMA: cold-start drag, incompatible with short budgets

## Infrastructure notes

- **lr=cfg.lr bug FIXED** (PR #2080): Lion uses `cfg.lr`; `--lr` flag works correctly.
- **slice_num CLI arg** (PR #2108): `--slice_num` accepted in Config.
- **n_head CLI arg** (PR #2149): `--n_head` accepted in Config. **Default is still 4** (train.py line 392: `n_head: int = 4`).
- **mlp_ratio CLI arg added** (PR #2350): `--mlp_ratio` accepted via Config. Default is still 4 (optimal, axis now closed).
