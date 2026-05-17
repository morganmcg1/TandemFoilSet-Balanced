# SENPAI Research State

- **Date**: 2026-05-17 (cycle 41)
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — **CURRENT BEST: H106 Arm B Fourier PE K=4 (val=35.92 / test=35.12, PR #4335).** Cycle 41: merged H106 (Δ-1.35 vs H99); closed H113/H114/H107/H118; assigned H119/H118/H104/H120/H121 to 5 idle students. 8 WIP, 0 idle.
- **Most recent human research directive**: None received

## Current Best

**PR #4335 (H106 Arm B: Fourier PE K=4, alphonse) — val_avg=35.9159 / test 3-split=35.1221** (MERGED 2026-05-17)

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H106 Arm B (Fourier K=4)** | **35.9159** | **35.1221** | **CURRENT BEST (PR #4335)** |
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

**Real but not yet compounded:**
- WSD 0/3/18 long-decay tail (H114 Arm B): Δ-0.97 vs H99 → H119 tests on H106 baseline

**Closed axes (negative or not compounding):**
- n_layers=6: definitively worse at current compute budget
- log(Re) aux head: FiLM already encodes Re sufficiently
- n_hidden=192/160: capacity not bottleneck at T_max=15 (not retested at H106 baseline yet)
- β₁=0.9, β₂=0.997, lr=3e-4, wd=1e-3: all locked

**Fourier frequency trend:** K=8 (36.91) > K=4 (35.92) → fewer frequencies better. H120 tests K=2, K=1 to find optimum.

## Active WIP Experiments (8 / 8 students, 0 idle)

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#4389** | nezuko | **H119: WSD 0/3/18 + Fourier K=4 compound** | TOP (orthogonal mechanisms; predicted ~34.9-35.5) | ~34-35 |
| **#4390** | thorfinn | **H118: compile + Fourier K=4 + bf16 + T_max=21** | HIGH (efficiency → more epochs) | ~34-36 |
| **#4392** | alphonse | **H104: per-sample pressure std normalization** | MED (OOD scale variance; targets val_geom_camber_rc 50.35) | ~34-36 |
| **#4394** | askeladd | **H120: Fourier K=2, K=1 frequency sweep** | MED (monotone K=8→K=4 improving; find optimum) | ~35-36 |
| **#4395** | frieren | **H121: SWA ep18 and ep15** | MED (flatter minima → OOD generalization) | ~35-36 |
| **#4316** | fern | **H112: AoA + log(Re) + gap/stagger input jitter** | MED (OOD regularization) | ~36-38 |
| **#4291** | edward | **H115: slice_num=80,64 on H99 stack** | MED (sub-96 slice probe; old T_max=15 baseline) | uncertain |
| **#4292** | tanjiro | **H103: mlp_ratio=3 under bf16** | MED (capacity probe; old T_max=15 config) | uncertain |

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion locked | 42.98 (H73) | — |
| LR (Lion) | ✅ 3e-4 LOCKED (H97) | 3e-4 | — |
| Schedule T_max | 🏆 T_max=21 LOCKED (H99) | 37.26 (H99) | Must use with --use_bf16 |
| Schedule WSD | 🔬 H119 active (WSD 0/3/18 + Fourier K=4 compound) | 36.29 (H114B, old base) | Does WSD compound with Fourier? |
| Schedule warmup | ❌ Negative (H76) | none | — |
| β₂, β₁, wd | ✅ All locked | 0.997 / 0.9 / 1e-3 | — |
| Fourier PE | 🏆 K=4 MERGED (H106) | 35.92 (H106B) | K=4 > K=8; H120 probes K=2, K=1 |
| Fourier K sweep | 🔬 H120 active (K=2, K=1) | K=4 optimal so far | — |
| torch.compile | 🔬 H118 active (compile + H106 full stack) | -27% s/ep alone | — |
| Per-sample p norm | 🔬 H104 active | none | Targets val_geom_camber_rc |
| SWA | 🔬 H121 active (ep18, ep15) | none | — |
| OOD input jitter | 🔬 H112 active | none | — |
| slice_num | 🔬 H115 active (80, 64) | 96 (H73) | Testing sub-96 on H99 baseline |
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

**Three proven gains, one open compound:** T_max=21 (3.24 pts) + Fourier K=4 (1.35 pts) = 4.59 pts of R5 compounding. WSD 0/3/18 showed real signal (0.97 pts vs H99) but hasn't been tested on top of Fourier. H119 tests this compound directly.

**Fourier frequency optimum:** K=8→K=4 monotone improvement suggests even fewer frequencies may help. H120 (K=2, K=1) completes the sweep.

**New axes to exhaust:** compile (H118), per-sample p std norm (H104), SWA (H121). These are all orthogonal to the T_max/Fourier/WSD stack.

**Open questions:**
1. Does WSD + Fourier compound? (H119 — top)
2. Does compile give more effective epochs on H106 baseline? (H118)
3. Does K=2 or K=1 beat K=4? (H120)
4. Does per-sample normalization help scale-invariant OOD? (H104)
5. Does SWA give flatter minima? (H121)
6. Do H103/H115/H112 capacity probes land above or below H106 baseline?

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: T_max is now a CLI arg (--T_max, default 15). All bf16 experiments: use --T_max 21.
- `train.py`: Fourier PE is in place (--fourier_pe, --fourier_pe_freqs). Baseline uses K=4.
- `train.py`: WSD scheduler code NOT in advisor branch — H114 was closed without merge. H119 must re-port.
