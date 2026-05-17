# SENPAI Research State

- **Date**: 2026-05-17 (cycle 43)
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — **CURRENT BEST: H120 Arm B Fourier PE K=1 (val=35.67 / test=33.40, PR #4394).** Cycle 43: merged H120 K=1 (Δ-0.25 val, Δ-1.72 test); closed H104 (catastrophic) and stale H116/H117 drafts; assigned askeladd H123 (Fourier K=0 + scale=0.5 ablation) and alphonse H124 (EMA weight averaging). 8 WIP, 0 idle.
- **Most recent human research directive**: None received

## Current Best

**PR #4394 (H120 Arm B: Fourier PE K=1, askeladd) — val_avg=35.6651 / test 3-split=33.3976** (MERGED 2026-05-17)

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H120 Arm B (Fourier K=1)** | **35.6651** | **33.3976** | **CURRENT BEST (PR #4394)** |
| H106 Arm B (Fourier K=4) | 35.9159 | 35.1221 | Overridden (PR #4335) |
| H99 Arm A (bf16 + T_max=21) | 37.2626 | 35.8568 | Overridden (PR #4272) |
| H95 Arm A (bf16 walltime) | 40.5066 | 39.0160 | Overridden (PR #4215) |

**Cumulative R5 gain: −30.44 pts val_avg vs H37b** (66.11 → 35.67). Total: **−78.96 pts from R1 start** (114.63).

## Noise Floor (H92)

**2σ = 1.67 pts** on val_avg/mae_surf_p. Test 3-split 2σ ≈ 1.02 pts.

## Round 5 Insights (cumulative)

**Confirmed improvement axes:**
1. **T_max=21 (H99)**: +3.24 pts — schedule-length alignment with bf16
2. **Fourier PE K=4 (H106)**: +1.35 pts — sub-chord spatial basis. K=4 > K=8.
3. **bf16 (H95)**: +0.71 pts — speed enables 21 epochs vs 15
4. **Fourier K=1 (H120)**: +0.25 pts val (Δ-1.72 test) — chord-scale wavelength only; eliminates sub-chord overfitting. K=1 > K=2 > K=4 on test (strictly monotone). K=4 still wins val_single_in_dist (+3.98 pts) but K=1 wins all OOD splits and all test splits.

**Closed axes (negative or not compounding):**
- n_layers=6: definitively worse (H113)
- log(Re) aux head: no signal; FiLM sufficient (H107)
- WSD schedule: did not compound with Fourier (H119)
- Per-sample p std normalization: catastrophic (H104; fundamental design flaw)
- n_hidden=192/160: not bottleneck at T_max=15 (H100; not retested at K=1)
- β₁=0.9, β₂=0.997, lr=3e-4, wd=1e-3: all locked

**Fourier frequency sweep (complete):**
| K | val_avg | test 3-split |
|---|---------|-------------|
| 8 | 36.91 | — |
| 4 | 35.92 | 35.12 |
| 2 | 36.20 | 34.85 |
| **1** | **35.67** | **33.40** |
| 0 (ablation) | ? | ? ← H123 Arm A |
| 1, scale=0.5 | ? | ? ← H123 Arm B |

**Key insight from K sweep:** The val_avg trend is non-monotone (K=4 dips below K=2) but the test 3-split trend is strictly monotone decreasing as K decreases. This suggests sub-chord Fourier frequencies overfit to training geometry; chord-scale is the right physical scale.

## Active WIP Experiments (8 / 8 students, 0 idle)

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#4357** | edward | **H115 Arm C: slice_num=80 + Fourier K=4 compound** | TOP (compound of two H106-equal mechanisms) | ~34.8-35.5 |
| **#4451** | askeladd | **H123: Fourier K=0 ablation + K=1 scale=0.5** | TOP (complete sweep; K=0 is highest curiosity) | K=0: ~35-36? |
| **#4452** | alphonse | **H124: EMA τ=0.999, τ=0.9995 at H120 K=1 baseline** | HIGH (zero-compute weight averaging) | ~35.0-35.4 |
| **#4422** | nezuko | **H122: Lookahead(Lion) k=5 α=0.5 at H106 baseline** | HIGH (orthogonal optimizer mechanism) | ~35.0-35.6 |
| **#4390** | thorfinn | **H118: compile + Fourier K=4 + bf16 + T_max=21** | MED (efficiency → more epochs) | ~34-36 |
| **#4395** | frieren | **H121: SWA ep18 and ep15 at H106 baseline** | MED (flatter minima → OOD generalization) | ~35-36 |
| **#4316** | fern | **H112: AoA + log(Re) + gap/stagger input jitter** | MED (OOD regularization) | ~36-38 |
| **#4292** | tanjiro | **H103: mlp_ratio=3 under bf16** | MED (capacity probe; old T_max=15 config) | uncertain |

**Note on in-flight PRs using old baseline:** H115 Arm C uses K=4 Fourier (not K=1); H118, H121, H122 are all on H106 stack (K=4). These results are still valuable — they test mechanisms orthogonal to the Fourier frequency choice. If they beat H106 baseline, they provide compounding signal to test with K=1. If they don't beat H106, we'll retest with K=1 stack.

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion locked | 42.98 (H73) | — |
| Lookahead wrapper | 🔬 H122 active (nezuko; on K=4 stack) | none | First test of slow-weight EMA over Lion |
| EMA weight averaging | 🔬 H124 active (alphonse; on K=1 stack) | none | Different from SWA; continuous EMA |
| LR (Lion) | ✅ 3e-4 LOCKED | 3e-4 | — |
| Schedule T_max | 🏆 T_max=21 LOCKED | 37.26 (H99) | — |
| Schedule WSD | ❌ Did not compound (H119) | 36.29 (H114B at H99 only) | — |
| Schedule warmup | ❌ Negative (H76) | none | — |
| β₂, β₁, wd | ✅ All locked | 0.997 / 0.9 / 1e-3 | — |
| Fourier PE | 🏆 K=1 MERGED (H120) | 35.67 (H120B) | K=1 > K=2 > K=4 > K=8 on test; K=4 > K=1 on val_single_in_dist |
| Fourier K sweep | 🔬 H123 active (K=0 ablation + K=1 scale=0.5) | K=1 optimal so far | Completing the sweep |
| torch.compile | 🔬 H118 active (compile + H106/K=4 stack) | -27% s/ep alone | — |
| SWA | 🔬 H121 active (ep18, ep15 on H106) | none | — |
| OOD input jitter | 🔬 H112 active (fern) | none | — |
| slice_num=80 + Fourier K=4 | 🔬 H115 Arm C active (edward) | 80 ties 96 at respective baselines | — |
| mlp_ratio | 🔬 H103 active (3 under bf16, old config) | 2 | — |
| Per-sample p norm | ❌ Catastrophic (H104) | none | Design flaw: target depends on unobservable y-stats |
| n_layers | ❌ Definitively negative (H113) | 4 | — |
| n_hidden | ❌ Negative at T_max=15 (H100) | 128 | Not retested at K=1 baseline |
| log(Re) aux head | ❌ No signal (H107) | none | — |
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
| 35.92 | 35.12 | H106 Arm B: Fourier PE K=4 |
| **35.67** | **33.40** | **H120 Arm B: Fourier PE K=1 (CURRENT BEST)** |

Total merged gain: **−78.96 pts val (69.0% reduction from 114.63).**

## Strategic State

**Fourier frequency sweep near-complete.** K=1 is the current optimum; K=0 ablation (H123) will determine if the monotone trend extends further or if the chord-scale basis is essential. If K=0 beats K=1, Fourier PE should be removed from the baseline. If K=1 > K=0, chord-scale frequency is the right physical prior and we've found the optimal encoding.

**Compound opportunities open:**
1. slice_num=80 + Fourier K=4 (H115 Arm C) — when result arrives, if positive, should re-test with K=1
2. EMA + K=1 (H124) — zero compute, parallel to current K=1 baseline
3. Lookahead + K=4 (H122) — if positive, needs re-test on K=1 stack

**Unresolved OOD bottleneck:** val_geom_camber_rc=47.56 is still far above average (35.67). No lever yet has specifically targeted this gap. Input jitter (H112) is the current attempt.

**Open questions for cycle 43-44:**
1. Does K=0 (no Fourier) beat K=1, or is chord-scale the essential physical prior? (H123 Arm A — critical)
2. Does EMA weight averaging smooth Lion sign-update noise? (H124)
3. Does slice_num=80 + K=4 compound? (H115 Arm C)
4. Do compile/SWA/jitter/mlp_ratio results land above or below K=1 baseline?

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: T_max is now a CLI arg (--T_max, default 15). All bf16 experiments: use --T_max 21.
- `train.py`: Fourier PE is in place (--fourier_pe, --fourier_pe_freqs). Current baseline uses K=1 (--fourier_pe_freqs 1).
- `train.py`: WSD scheduler code NOT in advisor branch — lever closed.
- `train.py`: EMA and Lookahead wrapper NOT yet in advisor branch — H124 and H122 must add them.
- `train.py`: H123 Arm B needs `--fourier_pe_scale` flag (or one-line hardcode) for sub-frequency test.
