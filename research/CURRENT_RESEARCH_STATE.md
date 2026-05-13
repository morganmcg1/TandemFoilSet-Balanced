# SENPAI Research State

- **Last updated:** 2026-05-13 ~12:20 (closed #2197 frieren rff-nfeatures-64 +1.78% dead end, repeated per-split signature across 3 RFF mods: cruise gains, rc/single regress; assigned frieren #2257 foil-mirror-aug, switching to data augmentation lever)
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
- AdamW **lr=1.5e-3**, **4-epoch linear warmup**, CosineAnnealingLR(T_max=10, **eta_min=0**), grad_clip=1.0, **betas=(0.9, 0.99)**
- batch_size=4, surf_weight=10, NaN-skip in evaluate_split

## In-flight PRs

| PR | Student | Slug | Axis | Epoch setting | Status |
|----|---------|------|------|------|---|
| #2184 | alphonse | `lr-2e-3-rff` | LR ceiling retest on RFF base: 1.5e-3 → 2e-3 | **--epochs 14** ✓ | WIP |
| #2257 | frieren | `foil-mirror-aug` | Foil z-axis mirroring augmentation (doubles effective training data via reflection symmetry) | **--epochs 14** ✓ | WIP — just assigned |
| #2238 | fern | `rff-trainable-b` | Trainable RFF B matrix: learnable frequencies (requires_grad=True, +64 params) | **--epochs 14** ✓ | WIP — just assigned |
| #2207 | nezuko | `cosine-eta-min-1e-4` | Cosine eta_min=1e-4: sustain learning at epoch 14 (best_epoch=14/14 consistently) | **--epochs 14** ✓ | WIP — just assigned |
| #1815 | askeladd | `node-dropout-0.9` | Node dropout 0.9 (rebasing on RFF base) | **--epochs 14** ✓ | WIP — rebase requested |
| #1817 | tanjiro | `charbonnier-eps-1e-3` | Charbonnier loss eps=1e-3 (rebasing on RFF base) | **--epochs 14** ✓ | WIP — rebase requested |
| #1820 | thorfinn | `weight-decay-5e-3` | Weight decay 1e-4→5e-3 (rebasing on RFF base) | **--epochs 14** ✓ | WIP — rebase requested |
| #1421 | edward | `surf-weight-25` | Decoupled surf/vol weighting (rebasing on RFF base) | **--epochs 14** ✓ | WIP — rebase requested |

### Merged PRs (this round — new bests)
- #1657 fern rff-pos-encoding σ=3.0: **−11.71%** val (73.9964 → 65.3304), −11.65% test — LARGEST SINGLE JUMP
- #2004 nezuko adamw-beta2-0.99: **−0.29%** val (74.2082 → 73.9964)
- #1895 alphonse lr-1.5e-3: **−3.80%** (77.1419 → 74.2082)

### Closed as dead ends (this round)
- #2197 frieren rff-nfeatures-64: +1.78% val regression; same per-split signature as #2206 (cruise gains, rc/single regress). Capacity axis CLOSED at d=32.
- #2206 fern rff-anisotropic-sx3-sz1p5: +0.51% val (regression), **−0.81% test** (mixed); per-split tradeoff: cruise/re_rand improve, rc/single regress on both val and test. Anisotropy axis CLOSED; isotropic σ=3.0 optimal.
- #2158 fern rff-sigma5: +2.75% vs 65.3304 (σ axis non-monotone; {1.0: −6.4%, 3.0: −11.71%, 5.0: +2.75%}; σ=3.0 is sweet spot, bandwidth axis CLOSED)
- #2130 nezuko adamw-eps-1e-6: +0.40% val (tie); **CRITICAL DIAGNOSTIC: epoch-5 spike +91 units on RFF base** (vs +3.2 pre-RFF); ε axis CLOSED on RFF base
- #1813 frieren warmup-5 on RFF: +0.85% (pre-RFF win −0.52% DID NOT stack; warmup axis CLOSED at 4 on RFF base)
- #2045 alphonse lr-1.75e-3: +4.13% vs 74.2082 (LR axis mapped pre-RFF; non-parabolic floor at 1.5e-3)
- #2054 nezuko adamw-beta2-0.95: +1.12% (β2 non-monotone, 0.99 is sweet spot; β2 axis CLOSED)
- #1942 alphonse lr-2e-3: +2.99% vs 74.2082 (stable but degraded pre-RFF)
- #1911 nezuko warmup-3-epochs: +1.56% vs 77.1419
- #1970 nezuko drop-path-0.1: +6.99% vs 74.2082 (capacity-reducing at 14-epoch budget)
- #1941 nezuko asinh-all-channels: +2.75% vs 74.2082 (asinh mechanism pressure-specific)

## Current research focus

1. **RFF sub-axes (3 independent experiments):**
   - **#2257 frieren foil-mirror-aug**: Z-axis reflection symmetry augmentation (z→−z, n_z→−n_z, AoA→−AoA, Uy→−Uy, target_Uy→−target_Uy, 50% prob during training). Doubles effective training data. Fundamentally different lever from RFF mods. Targets weak rc/single splits.
   - **#2238 fern rff-trainable-b**: B initialized at σ=3.0, set `requires_grad=True`. Lets gradient descent find optimal per-frequency bandwidth. Adds only 64 params (0.009% of model). Motivated by per-split bandwidth tradeoff observed in #2206 anisotropic result.
   - **#2184 alphonse lr-2e-3-rff**: Does the RFF input expansion shift the LR ceiling? Pre-RFF: 1.5e-3 sweet spot, 2e-3 regressed +2.99%. RFF gradients larger (+91-unit spike), ceiling may shift. Informative either way.

2. **LR schedule sub-axes:**
   - **#2207 nezuko cosine-eta-min-1e-4**: best_epoch=14/14 in all recent runs; current eta_min=0 freezes model at epoch 14 (~3.5e-5 LR). Non-zero floor (1e-4) may sustain learning. Motivated by the model not having converged within budget.

3. **Orthogonal axes rebasing on RFF** (askeladd, tanjiro, thorfinn, edward):
   - All notified to rebase on RFF base and compare against 65.3304.
   - These are low-signal hypotheses on the old base; results on RFF base may differ.

## Key research insights so far

- **Loss shape wins (exhausted):** pure-L1 is the global minimum of Smooth-L1 family for MAE criterion.
- **Channel weighting [1,1,3]:** −9.5% standalone. Canonical.
- **LR warmup + grad clip:** −16.1% on MSE baseline.
- **T_max alignment (--epochs 14):** −11.3% — critical!
- **4-epoch warmup:** −3.04% on pure-L1 base. Canonical.
- **Asinh pressure compression (GAIN=1.0):** −1.04%. PRESSURE-SPECIFIC.
- **lr=1e-3 + asinh SUPER-ADDITIVE:** −4.41% vs old base.
- **lr=1.5e-3 + asinh:** −3.80% further gain.
- **β2=0.99:** −0.29% val, −1.03% test. Epoch-5 spike collapsed pre-RFF. β2 axis CLOSED.
- **RFF σ=3.0 BREAKTHROUGH:** −11.71% val / −11.65% test. ALL FOUR SPLITS improve. Largest single gain in programme.
- **σ axis CLOSED at σ=3.0:** {1.0: −6.4%, 3.0: −11.71%, 5.0: +2.75%} — non-monotone, σ=3.0 sweet spot.
- **Anisotropy axis CLOSED at isotropic:** σ_z=1.5 (#2206) gave +0.51% val regression but −0.81% test improvement. Per-split tradeoff: cruise/re_rand benefit from lower z-bandwidth (smooth freestream pressure), rc/single need full σ_z=3.0 (ground-effect asymmetric loading). Isotropic σ=3.0 is the right COMPROMISE.
- **RFF capacity axis CLOSED at d=32:** n_features=64 (#2197) gave +1.78% val regression; same per-split signature (cruise −6.86%, rc/single +5.30%/+4.12%). Adding RFF frequency directions at fixed σ=3.0 lets the model overfit on geometry-OOD splits within 14 epochs.
- **REPEATED PER-SPLIT SIGNATURE across 3 RFF mods (σ=5.0, anisotropic, capacity):** more flexible RFF = cruise improves, rc/single regress. The model's optimal RFF complexity is split-dependent. Need a different mechanism (data augmentation, loss reformulation) to capture the cruise potential without rc/single regressing.
- **Warmup axis CLOSED at warmup=4 on RFF:** warmup=5 −0.52% on pre-RFF but +0.85% on RFF. Stacking does not hold across base shifts.
- **ε axis CLOSED on RFF base.** ε=1e-6 is mechanistically irrelevant when epoch-5 spike is gradient-magnitude dominated.
- **EPOCH-5 SPIKE DIAGNOSTIC:** RFF base has +91-unit spike (vs +3.2 pre-RFF). Root cause: large gradient magnitudes from 86-dim RFF input × peak LR. Not addressable by ε or warmup-duration alone.
- **14-epoch budget:** best_epoch=14/14 consistently — model never fully converged within budget. eta_min=0 freezes at ~3.5e-5 LR, leaving late-epoch learning potential on the table.
- **Pre-RFF validated improvements may NOT stack on RFF base** — must re-validate each axis independently.

## Next research directions (when new slots open)

1. **Cosine T_max shorter** — if eta_min=1e-4 (nezuko #2207) wins, the follow-up is probing T_max=8 (keeps LR higher longer, more high-LR epochs for RFF-input optimization)
2. **Anisotropic σ follow-up** — if fern #2206 wins (z smaller), probe σ_z=1.0 or σ_z=0.5; if ties, probe σ_x=4.0 (chord benefits from even higher bandwidth)
3. **RFF capacity higher** — if frieren #2197 (32→64) wins, probe n_features=128 (output 256 dim)
4. **Foil mirroring augmentation** (z-coord flip + AoA sign flip + Uy sign flip) — doubled effective data, likely orthogonal to all current axes
5. **Lower LR on RFF base (lr=1e-3 retest)** — if alphonse #2184 shows lr=2e-3 regresses AND epoch-5 spike remains, lower LR may be the remedy
6. **RFF on additional features** — apply RFF to surface normals (n_x, n_z) in addition to (x, z)

## Epoch budget arithmetic

- Epoch time: ~131–133s (678K params, RFF overhead ~1s/epoch)
- 30-min cap: **14 epochs max** (confirmed)
- **Canonical schedule: --epochs 14, warmup_epochs=4, T_max=10 (cosine) after warmup**
- best_epoch=14/14 consistently → model not converged; consider eta_min or longer T_max in future
