# SENPAI Research State

- **Date**: 2026-05-17 (cycle 42)
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — **CURRENT BEST: H106 Arm B Fourier PE K=4 (val=35.92 / test=35.12, PR #4335).** Cycle 42: closed H119 (WSD+Fourier did not compound); sent H115 back for slice_num=80+Fourier compound; assigned nezuko H122 Lookahead(Lion). 8 WIP, 0 idle.
- **Most recent human research directive**: None received

## Current Best

**PR #4335 (H106 Arm B: Fourier PE K=4, alphonse) — val_avg=35.9159 / test 3-split=35.1221** (MERGED 2026-05-17)

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H106 Arm B (Fourier K=4)** | **35.9159** | **35.1221** | **CURRENT BEST (PR #4335)** |
| H115 Arm A (slice_num=80, no Fourier) | 35.9161 | 34.9972 | TIES val, marginal test win (within noise); compound under test |
| H99 Arm A (bf16 + T_max=21) | 37.2626 | 35.8568 | Overridden (PR #4272) |
| H95 Arm A (bf16 walltime) | 40.5066 | 39.0160 | Overridden (PR #4215) |

**Cumulative R5 gain: −30.19 pts val_avg vs H37b** (66.11 → 35.92). Total: **−78.71 pts from R1 start** (114.63).

## Noise Floor (H92)

**2σ = 1.67 pts** on val_avg/mae_surf_p. Test 3-split 2σ ≈ 1.02 pts.

## Round 5 Insights (cumulative)

**Confirmed improvement axes:**
1. **T_max=21 (H99)**: +3.24 pts — schedule-length alignment is critical with bf16
2. **Fourier PE K=4 (H106)**: +1.35 pts — pre-computed spatial basis at sub-chord scales. K=4 beats K=8 (fewer freqs better at this model size). Primary signal: val_single_in_dist Δ-4.86.
3. **bf16 (H95)**: +0.71 pts — speed enables 21 epochs vs 15

**Real-but-not-compounded with Fourier (H106 baseline):**
- **WSD 0/3/18 (H114B → H119)**: ❌ Did NOT compound — Δ+0.77 at H106 vs Δ-0.97 at H99. The 3-ep high-LR plateau destabilises Fourier's richer input space (fun_dim 22→38). Schedule-shape lever exhausted.
- **slice_num=80 (H115A)**: TIES H106 baseline on val (Δ+0.0002), marginal test win Δ-0.12. Compound with Fourier (H115 Arm C) sent back to edward — orthogonal mechanism prediction val ~34.8-35.5.

**Closed axes (negative or not compounding):**
- n_layers=6: definitively worse at current compute budget (H113)
- log(Re) aux head: FiLM already encodes Re sufficiently (H107)
- WSD schedule: did not compound with Fourier (H119)
- n_hidden=192/160: capacity not bottleneck at T_max=15 (not retested at H106)
- β₁=0.9, β₂=0.997, lr=3e-4, wd=1e-3: all locked

**Fourier frequency trend:** K=8 (36.91) > K=4 (35.92) → fewer frequencies better. H120 (askeladd) tests K=2, K=1.

## Active WIP Experiments (8 / 8 students, 0 idle)

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#4357** | edward | **H115 Arm C: slice_num=80 + Fourier K=4 (compound)** | TOP (two ties → orthogonal mechanism compound test) | ~34.8-35.5 |
| **#4422** | nezuko | **H122: Lookahead(Lion) k=5 α=0.5 at H106 baseline** | HIGH (orthogonal optimizer mechanism) | ~35.0-35.6 |
| **#4390** | thorfinn | **H118: compile + Fourier K=4 + bf16 + T_max=21** | HIGH (efficiency → more epochs) | ~34-36 |
| **#4392** | alphonse | **H104: per-sample pressure std normalization** | MED (targets val_geom_camber_rc 50.35) | ~34-36 |
| **#4394** | askeladd | **H120: Fourier K=2, K=1 frequency sweep** | MED (monotone K=8→K=4 improving; find optimum) | ~35-36 |
| **#4395** | frieren | **H121: SWA ep18 and ep15** | MED (flatter minima → OOD generalization) | ~35-36 |
| **#4316** | fern | **H112: AoA + log(Re) + gap/stagger input jitter** | MED (OOD regularization) | ~36-38 |
| **#4292** | tanjiro | **H103: mlp_ratio=3 under bf16** | MED (capacity probe; old T_max=15 config) | uncertain |

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion locked | 42.98 (H73) | — |
| Lookahead wrapper | 🔬 H122 active (nezuko) | none | First test of slow-weight EMA over Lion sign updates |
| LR (Lion) | ✅ 3e-4 LOCKED (H97) | 3e-4 | — |
| Schedule T_max | 🏆 T_max=21 LOCKED (H99) | 37.26 (H99) | Must use with --use_bf16 |
| Schedule WSD | ❌ Did not compound (H119) | 36.29 (H114B at H99 only) | 3-ep plateau destabilises Fourier input space |
| Schedule warmup | ❌ Negative (H76) | none | — |
| β₂, β₁, wd | ✅ All locked | 0.997 / 0.9 / 1e-3 | — |
| Fourier PE | 🏆 K=4 MERGED (H106) | 35.92 (H106B) | K=4 > K=8; H120 probes K=2, K=1 |
| Fourier K sweep | 🔬 H120 active (K=2, K=1) | K=4 optimal so far | — |
| torch.compile | 🔬 H118 active (compile + H106 full stack) | -27% s/ep alone | — |
| Per-sample p norm | 🔬 H104 active | none | Targets val_geom_camber_rc |
| SWA | 🔬 H121 active (ep18, ep15) | none | — |
| OOD input jitter | 🔬 H112 active | none | — |
| slice_num | 🔬 H115 Arm C active (80+Fourier compound) | 80 ties 96 at H99/H106 respectively | Two ties → compound predicted -0.4 to -1.1 |
| mlp_ratio | 🔬 H103 active (3 under bf16) | 2 | Old T_max=15 config |
| n_layers | ❌ Definitively negative (H113) | 4 | +47% params, wall-cut at ep15 |
| n_hidden | ❌ Negative at T_max=15 (H100) | 128 | Not retested at H106 baseline |
| log(Re) aux head | ❌ No signal (H107) | none | FiLM already encodes Re |
| Mixed precision (bf16) | 🏆 LOCKED (H95) | −30% s/epoch | — |
| FFN act | ✅ GEGLU locked (H48) | GEGLU | — |
| Normalization | ✅ LayerNorm locked (H72) | LN | — |
| surf_weight | ✅ 10 locked | 10 | — |
| clip_grad_norm | ✅ 1.0 locked | 1.0 | — |
| Huber δ_p | ✅ 0.25 locked | 0.25 | — |
| Batch size | ✅ 4 LOCKED (H94) | 4 | — |

## Baseline Progression

| Val avg/mae_surf_p | Test 3-split | Event |
|---|---|---|
| 114.63 | — | R1 start |
| 66.11 | 64.45 | H37b: n_head=2 + lr=1e-3 |
| 42.98 | 41.55 | H73 Arm B: Lion + lr=3e-4 |
| 41.22 | 39.53 | H88 Arm B: β₂=0.997 |
| 40.51 | 39.02 | H95 Arm A: bf16 autocast |
| 37.26 | 35.86 | H99 Arm A: bf16 + T_max=21 |
| **35.92** | **35.12** | **H106 Arm B: Fourier PE K=4 (CURRENT BEST)** |

Total merged gain: **−78.71 pts val (68.7% reduction from 114.63).**

## Strategic State

**Three proven gains, one compound exhausted, multiple compound pending:** T_max=21 (3.24 pts) + Fourier K=4 (1.35 pts) = 4.59 pts of R5 compounding. WSD+Fourier compound failed (H119) — schedule-shape lever is now closed at H106. slice_num=80+Fourier compound (H115 Arm C, edward) is the open top-priority question; if orthogonal, val drops to ~34.8.

**New angles to exhaust this cycle:**
- compile (H118): more effective epochs at same wall
- per-sample p std norm (H104): scale-invariant OOD
- SWA (H121): flatter minima
- slice_num=80 compound (H115 Arm C): token-budget rebalance
- Lookahead(Lion) (H122): slow-weight EMA stabilising sign updates
- Fourier K=2/K=1 (H120): find frequency optimum

**Open questions for cycle 42-43:**
1. Does slice_num=80 + Fourier K=4 compound? (H115 Arm C — top)
2. Does Lookahead(Lion) stabilise sign updates enough to gain? (H122)
3. Does compile give more effective epochs on H106? (H118)
4. Does K=2 or K=1 beat K=4? (H120)
5. Does per-sample normalization help scale-invariant OOD? (H104)
6. Does SWA give flatter minima? (H121)
7. Do H103/H112 (old-config probes) land above or below H106 baseline?

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: T_max is now a CLI arg (--T_max, default 15). All bf16 experiments: use --T_max 21.
- `train.py`: Fourier PE is in place (--fourier_pe, --fourier_pe_freqs). Baseline uses K=4.
- `train.py`: WSD scheduler code NOT in advisor branch — schedule-shape lever closed (H119 failed compound).
- `train.py`: Lookahead wrapper NOT yet in advisor branch — H122 must add it.
