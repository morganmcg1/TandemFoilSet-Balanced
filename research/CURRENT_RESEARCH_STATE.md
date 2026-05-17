# SENPAI Research State

- **Date**: 2026-05-17 (cycle 40)
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — **CURRENT BEST: H99 Arm A bf16 + T_max=21 (val=37.26 / test=35.86, PR #4272). Δ-3.24 vs H95 (40.51).** Cycle 40: closed thorfinn/H96 (val=41.05, below baseline; rebase failures); assigned H118 (torch.compile+bf16+T_max=21 compound) to thorfinn (PR #4380). 8 WIP, 0 idle.
- **Most recent human research directive**: None received

## Current Best

**PR #4272 (H99 Arm A: bf16 + T_max=21, alphonse) — val_avg=37.2626 / test 3-split=35.8568** (MERGED 2026-05-17)

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H99 Arm A (bf16 + T_max=21)** | **37.2626** | **35.8568** | **CURRENT BEST (PR #4272)** |
| H95 Arm A (bf16 walltime) | 40.5066 | 39.0160 | Overridden (PR #4215) |
| H88 Arm B (β₂=0.997) | 41.2153 | 39.5337 | Overridden (PR #4166) |

**Cumulative R5 gain: −28.84 pts val_avg vs H37b** (66.11 → 37.26). Total: **−77.37 pts from R1 start** (114.63).

## Noise Floor (H92)

**2σ = 1.67 pts** on val_avg/mae_surf_p. Test 3-split 2σ ≈ 1.02 pts.

Decision thresholds:
- Δ < 1.7 pts → noise (tie)
- Δ ≥ 2.5 pts → real signal
- Δ ≥ 4.0 pts → strong signal

## Round 5 Insights (cumulative)

**Schedule is the dominant lever:** Two major discoveries:
1. **T_max=21 fix (H99, cycle 38)**: Aligning cosine schedule to 21-epoch bf16 wall budget → Δ-3.24 pts. Eliminates LR-bounce confound at ep15-21. All bf16 experiments going forward must use `--T_max 21`.
2. **WSD 0/5/10 (H93 Arm C, closed cycle 38)**: Long decay tail (10 of 15 epochs decaying) → Δ-1.00 pts vs H95 at old baseline. Now below new H99 baseline. WSD compound on H99 config is the top EV hypothesis (H114).

**bf16 unlock (H95, cycle 34):** ~30% s/epoch speedup. Enables 21 epochs vs 15. The T_max=21 fix (H99) fully realizes this benefit.

**Lion optimizer axis saturated:** β₁=0.9, β₂=0.997, lr=3e-4, wd=1e-3, surf_weight=10 all locked.

**Capacity axis negative at T_max=15:** H100 (width) and H101 (depth) were tested at old T_max=15 — results were noise/ties. H103 (mlp_ratio) and H102 (slice_num) still active at T_max=15 baseline. H113 retests n_layers=6 at T_max=21. Capacity verdict pending the T_max-aligned results.

## Active WIP Experiments (8 / 8 students, 0 idle)

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#4332** | nezuko | **H114: WSD 0/7/14 + 0/3/18 on H99 baseline** | TOP (direct compound; WSD long-decay vs clean T_max=21 cosine) | ~35-37 |
| **#4333** | frieren | **H113: n_layers=6 + bf16 + T_max=21 (retest H101 Arm B undertraining)** | HIGH (was wall-cut at T_max=15; with T_max=21 shape, potential val ~36-37) | ~36-38 |
| **#4335** | alphonse | **H106: Fourier PE of mesh coordinates (K=8, K=4)** | MED (orthogonal axis; multi-scale spatial frequencies) | ~35-38 |
| **#4337** | askeladd | **H107: log(Re) aux head (λ=0.1, λ=0.01)** | MED (multi-task Re regularization; targets val_re_rand) | ~36-38 |
| **#4378** | fern | **H116: log(Re) aux head v2 (λ=0.01, λ=0.05 on H99 stack)** | MED (Re-structured trunk; targets val_geom_camber_rc 49.78 + val_re_rand 39.25) | ~36-38 |
| **#4379** | tanjiro | **H117: Fourier PE of mesh coords (K=8, K=4 on H99 stack)** | MED (multi-scale spatial frequency encoding; orthogonal axis) | ~35-38 |
| **#4380** | thorfinn | **H118: torch.compile + bf16 + T_max=21 compound (Arms A,B: reduce-overhead vs default)** | HIGH (compile may cut s/epoch ~85.7→~70, fitting ~25 epochs; clean H99 baseline test) | ~35-37 |
| **#4357** | edward | **H115: slice_num=80 and 64 on H99 bf16+T_max=21 stack** | MED (sub-96 slice probe; does narrower slice improve generalization?) | uncertain |

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion locked | 42.98 (H73) | — |
| LR (Lion) | ✅ 3e-4 LOCKED (H97) | 3e-4 | — |
| Schedule T_max | 🏆 T_max=21 LOCKED (H99) | 37.26 (H99) | Must use --T_max 21 with --use_bf16. Δ-3.24 pts. |
| Schedule WSD | 🔬 H114 active (WSD 0/7/14 + 0/3/18 on H99 baseline) | 39.51 (H93C, old base) | Does WSD add to T_max=21 cosine or is it already absorbed? |
| Schedule warmup | ❌ Negative (H76) | none | — |
| β₂ (Lion) | ✅ 0.997 LOCKED (H88) | 0.997 | — |
| β₁ (Lion) | ✅ 0.9 LOCKED (H98) | 0.9 | — |
| wd (Lion) | ✅ 1e-3 LOCKED (H79) | 1e-3 | — |
| n_layers | 🔬 H113 active (6 + bf16 + T_max=21) | 4 | H101 Arm B was undertrained; retest pending |
| n_hidden | ❌ Negative at T_max=15 (H100) | 128 | n_hidden=192/160 no signal; capacity not bottleneck at T_max=15 |
| mlp_ratio | 🔬 H103 active (3 under bf16) | 2 (default) | Running at old T_max=15 config |
| slice_num | 🔬 H102 active (128 under bf16) | 96 (H73) | Running at old T_max=15 config |
| FFN act | ✅ GEGLU locked (H48) | GEGLU | — |
| Normalization | ✅ LayerNorm locked (H72, H81) | LN | — |
| Mixed precision (bf16) | 🏆 LOCKED (H95) | −30% s/epoch | — |
| torch.compile | 🔬 H118 active (compile+bf16+T_max=21 compound, Arms A,B; H96 CLOSED val=41.05 below baseline) | -27% s/epoch alone | H96 confounded by missing bf16+T_max=21; H118 retests cleanly |
| Fourier PE | 🔬 H106 active (K=8, K=4) | none | First coordinate-encoding experiment |
| log(Re) aux head | 🔬 H107 active (λ=0.1, λ=0.01) | none | Multi-task Re regularization |
| OOD input jitter | 🔬 H112 active (AoA+log(Re)+gap/stagger) | none | — |
| surf_weight | ✅ 10 locked (H54/H91) | 10 | — |
| clip_grad_norm | ✅ 1.0 locked (H20, H56) | 1.0 | — |
| Huber δ_p | ✅ 0.25 locked (H25/H64) | 0.25 | — |
| Batch size | ✅ 4 LOCKED (H94) | 4 | — |

## Baseline Progression

| Val avg/mae_surf_p | Test 3-split | Event |
|---|---|---|
| 114.63 | — | R1 start |
| 66.11 | 64.45 | H37b: n_head=2 + lr=1e-3 |
| 58.63 | 56.70 | H48 GEGLU |
| 57.58 | 56.46 | H60: n_layers=4 |
| 56.75 | 54.50 | H66: slice_num=96 |
| 42.98 | 41.55 | H73 Arm B: Lion + lr=3e-4 |
| 42.30 | 40.56 | H78 Arm B: β₂=0.995 |
| 41.22 | 39.53 | H88 Arm B: β₂=0.997 |
| 40.51 | 39.02 | H95 Arm A: bf16 autocast |
| **37.26** | **35.86** | **H99 Arm A: bf16 + T_max=21 (CURRENT BEST)** |

Total merged gain: **−77.37 pts val (67.5% reduction from 114.63).**

## Strategic State

**Schedule is now the dominant lever with two active hypotheses:**
1. **H114 (nezuko, TOP):** Does WSD on top of H99's monotone cosine add further headroom? H93 Arm C (0/5/10 at 15 epochs) showed the "long decay tail" signal. Scaling to 21 epochs → 0/7/14 is the natural compound. If WSD adds Δ-1 pts on top of H99 → val ~36.
2. **H113 (frieren, HIGH):** n_layers=6 was undertrained at T_max=15 (ep15, still descending). With T_max=21, the deeper model gets clean monotone decay. If n_layers=6 + T_max=21 → val ~36, this is an architectural gain on top of H99.

**Capacity probe status:**
- H102/H103 still running at old T_max=15 config. These will be compared vs H99 baseline. If they show even tiny gains over H95 at T_max=15, the T_max=21 fix would add ~3 pts on top → potentially competitive.
- H100 (width) confirmed negative at T_max=15. No retest scheduled unless H103 or H102 shows a signal.

**Fresh axes starting:**
- H106 (Fourier PE): coordinate-encoding improvement, orthogonal to all prior axes.
- H107 (log(Re) aux head): multi-task regularization targeting val_re_rand OOD.

**Open strategic questions:**
1. Does WSD compound with T_max=21 cosine? (H114 — top priority)
2. Does n_layers=6 break through with T_max=21? (H113)
3. Does compile+bf16 compound? (H96 active)
4. Do any capacity probes (H102, H103) show signal at old T_max=15 config?
5. Does Fourier PE improve multi-scale spatial encoding? (H106)
6. Does Re auxiliary supervision help val_re_rand? (H107)

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `train.py`: `T_max=15` was hardcoded; now a CLI arg (`--T_max`, default 15) added by H99. All bf16 experiments should use `--T_max 21`.
- `train.py`: WSD scheduler (from H93, closed) was NOT merged — only T_max CLI arg. H114 must re-port the WSD implementation.
