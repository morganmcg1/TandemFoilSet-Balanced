# SENPAI Research State

- **Last updated:** 2026-05-13 ~11:20 (closed #1813 frieren warmup-5 +0.85% dead end on RFF base, warmup axis locked at 4; assigned frieren #2197 rff-nfeatures-64 capacity probe)
- **Advisor branch:** `icml-appendix-charlie-pai2g-48h-r2`
- **Launch context:** Charlie no-W&B logging ablation, 48h fleet wall-clock, 30 min cap per training execution, local JSONL metrics only
- **Most recent human research directive:** none received

## Current baseline

**`val_avg/mae_surf_p = 65.3304`** — PR #1657 (RFF σ=3.0 positional encoding + asinh + warmup-4 + lr=1.5e-3 + β2=0.99), epoch 14/14, 678K param Transolver.

Per-split: val_single=72.691, val_rc=78.833, val_cruise=44.439, val_re_rand=65.359.  
Test: test_avg=56.9425 (test_single=64.577, test_rc=71.531, test_cruise=36.392, test_re_rand=55.269).

**Historical trajectory:**
- 122.64 (#1418 channel_weights=[1,1,3])
- 102.85 (#1424 LR warmup 7e-4 + grad_clip)
- 95.336 (#1414 Smooth L1 β=0.1 + NaN-skip)
- 84.562 (#1684 T_max=14 alignment)
- 83.230 (#1682 pure-L1 loss)
- 80.7014 (#1776 4-epoch warmup)
- 79.8623 (#1777 asinh pressure compression GAIN=1.0)
- 77.1419 (#1814 lr=1e-3 + asinh super-additive stack)
- 74.2082 (#1895 lr=1.5e-3 ceiling probe)
- 73.9964 (#2004 adamw-β2=0.99)
- **65.3304** (#1657 RFF σ=3.0 positional encoding) — **current** (−11.71%, LARGEST SINGLE JUMP)

**Canonical config on advisor HEAD:**
- `F.l1_loss(reduction='none')` × channel_weights[1,1,3] / 5 in asinh-compressed target space
- **Asinh pressure compression**: ASINH_GAIN=1.0 (pressure channel only)
- **RFF positional encoding**: σ=3.0, 64-dim [cos, sin] of (x,z) coordinates, preprocess MLP input 24→86
- AdamW **lr=1.5e-3**, **4-epoch linear warmup**, CosineAnnealingLR(T_max=10), grad_clip=1.0, **betas=(0.9, 0.99)**
- batch_size=4, surf_weight=10, NaN-skip in evaluate_split

## In-flight PRs

| PR | Student | Slug | Axis | Epoch setting | vs. Baseline |
|----|---------|------|------|------|---|
| #2158 | fern | `rff-sigma5` | RFF bandwidth sweep: σ=3.0 → 5.0 (monotone test) | **--epochs 14** ✓ | WIP — just assigned |
| #2197 | frieren | `rff-nfeatures-64` | RFF capacity probe: 32→64 frequencies at σ=3.0 (output 64→128 dim) | **--epochs 14** ✓ | WIP — just assigned |
| #2184 | alphonse | `lr-2e-3-rff` | LR ceiling retest on RFF base: 1.5e-3 → 2e-3 (was +2.99% on pre-RFF; RFF may shift ceiling) | **--epochs 14** ✓ | WIP — just assigned |
| #1815 | askeladd | `node-dropout-0.9` | Node dropout 0.9 (needs rebase on RFF base) | **--epochs 14** ✓ | WIP — rebase requested |
| #1817 | tanjiro | `charbonnier-eps-1e-3` | Charbonnier loss eps=1e-3 (needs rebase on RFF base) | **--epochs 14** ✓ | WIP — rebase requested |
| #1820 | thorfinn | `weight-decay-5e-3` | Weight decay 1e-4→5e-3 (needs rebase on RFF base) | **--epochs 14** ✓ | WIP — rebase requested |
| #2130 | nezuko | `adamw-eps-1e-6` | AdamW ε=1e-6 (needs rebase on RFF base) | **--epochs 14** ✓ | WIP — rebase requested |
| #1421 | edward | `surf-weight-25` | Decoupled surf/vol channel weighting (needs rebase on RFF base) | **--epochs 14** ✓ | WIP — rebase requested |

### Merged PRs (this round — new bests)
- #1657 fern rff-pos-encoding σ=3.0: **−11.71%** val (73.9964 → 65.3304), −11.65% test — LARGEST SINGLE JUMP
- #2004 nezuko adamw-beta2-0.99: **−0.29%** val (74.2082 → 73.9964)
- #1895 alphonse lr-1.5e-3: **−3.80%** (77.1419 → 74.2082)

### Closed as dead ends (this round)
- #1813 frieren warmup-5 on RFF: +0.85% vs 65.3304 (pre-RFF win −0.52% did NOT stack on RFF; warmup=4 confirmed optimum on RFF base)
- #2045 alphonse lr-1.75e-3: +4.13% vs 74.2082 (LR axis fully mapped on pre-RFF: {1e-3: +3.96%, **1.5e-3 sweet spot**, 1.75e-3: +4.13%, 2e-3: +2.99%}; non-parabolic floor)
- #2054 nezuko adamw-beta2-0.95: +1.12% (β2 axis mapped; non-monotone, 0.99 is sweet spot)
- #1942 alphonse lr-2e-3: +2.99% vs 74.2082 (stable but optimization quality degraded)
- #1911 nezuko warmup-3-epochs: +1.56% vs 77.1419
- #1970 nezuko drop-path-0.1: +6.99% vs 74.2082 (capacity-reducing at 14-epoch budget)
- #1941 nezuko asinh-all-channels: +2.75% vs 74.2082 (mechanism pressure-specific)

## Current research focus

1. **RFF bandwidth axis — sweep σ upward:**
   - **#2158 fern rff-sigma5**: σ=3.0 (winner) → σ=5.0. Monotone in {1, 3}; does gain continue? If yes: push higher. If neutral/regresses: σ=3.0 is the bandwidth optimum, move to RFF capacity axis.

2. **RFF capacity axis — n_features probe:**
   - **#2197 frieren rff-nfeatures-64**: At fixed σ=3.0, double frequency count 32→64 (output 64→128 dim). Tests kernel-approx quality axis: with d=32, per-pair error ~18%; d=64 cuts to ~12.5%. Tiny param increase (~8K). If wins → capacity-limited; probe 128 next.

3. **LR axis — retest ceiling on RFF base:**
   - **#2184 alphonse lr-2e-3-rff**: Pre-RFF mapping closed (sweet spot 1.5e-3, non-parabolic floor). RFF transforms input geometry (24→86 dim), may shift LR ceiling upward. Retest lr=2e-3 directly on RFF base. If <65.33 → RFF shifts ceiling; if regresses → ceiling locked at 1.5e-3 across bases.

4. **AdamW ε axis:**
   - **#2130 nezuko adamw-eps-1e-6**: Notified of RFF baseline, rebase needed.

5. **Orthogonal axes rebasing on RFF:**
   - All other in-flight PRs (#1815 askeladd, #1817 tanjiro, #1820 thorfinn, #1421 edward) notified to rebase on RFF base.

## Key research insights so far

- **Loss shape wins (exhausted):** pure-L1 is the global minimum of Smooth-L1 family for MAE criterion.
- **Channel weighting [1,1,3]:** −9.5% standalone. Canonical.
- **LR warmup + grad clip:** −16.1% on MSE baseline.
- **T_max alignment (--epochs 14):** −11.3% — critical!
- **4-epoch warmup:** −3.04% on pure-L1 base. Canonical.
- **Asinh pressure compression (GAIN=1.0):** −1.04%. PRESSURE-SPECIFIC.
- **lr=1e-3 + asinh SUPER-ADDITIVE:** −4.41% vs old base.
- **lr=1.5e-3 + asinh:** −3.80% further gain.
- **β2=0.99:** −0.29% val, −1.03% test. Epoch-5 spike collapsed. β2 axis MAPPED.
- **RFF σ=3.0 BREAKTHROUGH:** −11.71% val / −11.65% test. ALL FOUR SPLITS improve. Largest single gain in programme. σ axis: {1.0: −6.4%, 3.0: −11.7%} — monotone, σ=5.0 next.
- **Warmup axis CLOSED on RFF base:** warmup=3 (dead end pre-RFF), warmup=4 (canonical optimum), warmup=5 (+0.85% regression on RFF — pre-RFF win DID NOT stack; RFF changes gradient dynamics, extra warmup epoch loses cosine tail).
- **14-epoch budget constraint:** Capacity-reducing regularizers fail. Confirmed by DropPath, SWA failures.
- **Architecture/capacity axes: EXHAUSTED** (pre-RFF). RFF opens new spatial-encoding axis.

## Next research directions (when new slots open)

1. **RFF capacity probe** (n_features 32→64: double 64-dim output to 128-dim) — if σ=5.0 plateau, test wider features at σ=3.0
2. **RFF on surface normals** (add surface normal vector to RFF input, not just (x,z)) — may help cruise/rc further
3. **Learned RFF frequencies** (train B jointly rather than fixing it) — higher capacity but adds params
4. **lr=1.75e-3 on RFF base** (if alphonse shows ceiling above 1.5e-3 on old base)
5. **Longer training (--epochs 18)** — best_epoch=14/14 multiple times, RFF might benefit more from longer cosine tail
6. **Foil mirroring augmentation** (z-coord flip + AoA sign flip + Uy sign flip) — doubled effective data

## Epoch budget arithmetic

- Epoch time: ~131s (678K params, RFF negligible overhead)
- 30-min cap: **14 epochs max** (confirmed)
- **Canonical schedule: --epochs 14, warmup_epochs=4, T_max=10 (cosine) after warmup**
