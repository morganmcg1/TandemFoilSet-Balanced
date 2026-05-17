# SENPAI Research State

- **Date**: 2026-05-17 (cycle 49 — H128 MERGED, OOD bottleneck cracked)
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — **CURRENT BEST: H128 Arm A (compile + T_max=24, val=33.4710 / test=32.638, PR #4463).** Cycle 49: H128 merged — first lever to significantly move OOD camber split (val_geom_camber_rc −2.02). 8 WIP incl. new H136 compound test (wd=5e-3 × compile × T_max=24).
- **Most recent human research directive**: None received

## Current Best

**PR #4463 (H128 Arm A: compile + T_max=24, thorfinn) — val_avg=33.4710 / test 3-split=32.638** (MERGED 2026-05-17)

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H128 Arm A (compile + T_max=24)** | **33.4710** | **32.638** | **CURRENT BEST (PR #4463)** |
| H125 Arm A (wd=5e-3) | 34.5532 | 33.0792 | Overridden (PR #4459) |
| H120 Arm B (Fourier K=1) | 35.6651 | 33.3976 | Overridden (PR #4394) |
| H106 Arm B (Fourier K=4) | 35.9159 | 35.1221 | Overridden (PR #4335) |
| H99 Arm A (bf16 + T_max=21) | 37.2626 | 35.8568 | Overridden (PR #4272) |

**Cumulative R5 gain: −32.64 pts val_avg vs H37b** (66.11 → 33.47). Total: **−81.17 pts from R1 start** (114.63 → 33.47, 70.8% reduction).

## OOD Bottleneck — MOVED

**val_geom_camber_rc = 45.76** (was 47.78 at H125 baseline) — **−2.02 pts**, first lever to move the OOD camber split. Gap is now 12.3 pts (vs val_avg=33.47). Mechanism: 1.86× compile speedup → 3 extra polish epochs (T_max=21 → 24). Compounds in the OOD direction more than the in-dist direction. ALL 4 splits improved.

| Split | H125 | H128 | Δ |
|---|---:|---:|---:|
| val_single_in_dist | 21.93 | 21.42 | −0.51 |
| val_geom_camber_rc | 47.78 | 45.76 | **−2.02** |
| val_geom_camber_cruise | NaN | NaN | (scoring bug) |
| val_re_rand | 33.05 | 32.50 | −0.55 |

## Noise Floor

**2σ ≈ 1.67 pts** on val_avg/mae_surf_p. Test 3-split 2σ ≈ 1.02 pts.

H128's −1.08 pts val_avg gain and −0.44 pts test gain are inside the noise band singly but the consistent improvement across ALL 4 splits + the −2.02 OOD camber movement (>2σ) is significant.

## Round 5 Insights (cumulative)

**Confirmed improvement axes (merged):**
1. **T_max=21 (H99)**: +3.24 pts — schedule-length alignment with bf16
2. **Fourier PE K=4 (H106)**: +1.35 pts — sub-chord spatial basis
3. **Fourier K=1 (H120)**: +0.25 pts val (Δ-1.72 test) — chord-scale only; **key anti-overfitting signal**
4. **bf16 (H95)**: +0.71 pts — speed enables 21 epochs vs 15
5. **wd=5e-3 (H125)**: +1.11 pts val — in-dist memorisation regularization
6. **compile + T_max=24 (H128)**: +1.08 pts val (−2.02 on val_geom_camber_rc) — first lever to move OOD camber

**Closed axes (cycle 47 complete):**
- Weight averaging (EMA τ=0.999): schedule-incompat cosine→0 (H124)
- Weight averaging (SWA): schedule-incompat (H121)
- Weight averaging (Lookahead): schedule-incompat (H122)
- Capacity reduction (n_hidden=96,112): BOTH directions worse; n_hidden=128 confirmed optimum (H127)
- Fourier K-sweep: K=0 and scale=0.5 both fail; K=1 scale=1.0 is true optimum (H123)
- Literal Mixup (sample pairs): mesh identity violation + H55 repeat (H129)
- slice_num=80 + Fourier: anti-compound (H115 Arm C)
- Per-sample p norm: catastrophic (H104)
- FiLM cond jitter: washes out conditioning (H112)
- mlp_ratio=3: no signal (H103)
- n_layers=6: definitively worse (H113)
- log(Re) aux head: no signal (H107)
- WSD schedule: incompat with Fourier (H119)

**Fourier frequency sweep (complete):**
| K | scale | val_avg | test 3-split |
|---|-------|---------|-------------|
| 8 | 1.0 | 36.91 | — |
| 4 | 1.0 | 35.92 | 35.12 |
| 2 | 1.0 | 36.20 | 34.85 |
| **1** | **1.0** | **35.67** | **33.40** |
| 0 | — | 35.82 | 34.61 |
| 1 | 0.5 | 36.15 | 33.49 |

## Active WIP Experiments (8 / 8 students, 0 idle)

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#4545** | thorfinn | **H136: compound wd=5e-3 + T_max=24 + compile** | TOP (compound test of two recent wins) | ~32-33.5 |
| **#4526** | askeladd | **H134: GALE geometry cross-attention K=32/K=16 tokens** | TOP (architecture, OOD-targeted) | ~32-34 |
| **#4509** | alphonse | **H132: DSDF Fourier K=1 (combined+ablation)** | HIGH (input repr, OOD-targeted) | ~32-34 |
| **#4480** | fern | **H131: LE+TE dual coord features (4/8 extra dims)** | HIGH (input repr, OOD-targeted) | ~32-34 |
| **#4527** | edward | **H133: spectral norm on in_project_slice** | HIGH (Lipschitz regularization) | ~32.5-34 |
| **#4529** | nezuko | **H135: condition-only Mixup α={0.2,0.5}** | HIGH (camber-axis augmentation) | ~32-34 |
| **#4460** | frieren | **H126: FFN dropout {0.1, 0.2} at K=1** | MED (baseline shifted twice since assignment) | ~33-34.5 |
| **#4466** | tanjiro | **H130: AdamW vs Lion revalidation at K=1** | LOW (sanity check, baseline shifted) | likely confirms Lion |

**Note on H126/H130:** These were started on wd=1e-3 baseline (35.6651) before two improvements landed. Compare to new baseline (33.4710) — if they beat their assignment baseline but not the new one, rerun on updated stack.

**Note on H136:** thorfinn's H128 win used wd=1e-3 (assigned before H125 merge). H136 tests whether wd=5e-3 + T_max=24 + compile compounds. Critical to resolve which weight decay is now optimal.

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🔬 H130 revalidating AdamW vs Lion | 35.67 (Lion at K=1) | First retest since H73 |
| Weight decay | 🔬 H136 retesting wd=5e-3 on new stack | 34.55 (wd=5e-3 at H125 stack) | H128 used wd=1e-3 — compound test needed |
| FFN dropout | 🔬 H126 sweep {0.1, 0.2} | none (no dropout currently) | First test |
| Spectral norm (in_project_slice) | 🔬 H133 active | none | Lipschitz-1 constraint on slice assignment |
| Condition-only Mixup | 🔬 H135 active | none | Geometry-safe Mixup on NACA+flow params; H129 literal-Mixup closed |
| LE+TE dual coords | 🔬 H131 active | none | OOD-targeted input repr; Texas A&M arXiv 2412.09399 |
| DSDF Fourier features | 🔬 H132 active | none | Applies H120 K=1 mechanism to signed-distance channels |
| GALE cross-attention | 🔬 H134 active (K=32, K=16) | none | Architecture-level OOD fix; arXiv 2512.20399 |
| Schedule T_max | 🏆 T_max=24 MERGED (H128) | 33.47 | Compile-enabled extension; 21 → 24 polish epochs |
| torch.compile | 🏆 reduce-overhead MERGED (H128) | 1.86× speedup | Enables T_max=24 |
| EMA weight averaging | ❌ Schedule-incompat (H124) | none | Same family as SWA/Lookahead |
| Lookahead wrapper | ❌ Schedule-incompat (H122) | none | Cosine→0 prevents late-epoch polish |
| SWA | ❌ Schedule-incompat (H121) | none | — |
| Mixup (literal, sample pairs) | ❌ Mesh identity violation (H129, H55) | none | Variable mesh; NaN in mixed batch |
| n_hidden | ❌ Closed: 96,112 negative (H127) + >128 negative (earlier). 128 is optimum. | 128 | — |
| Fourier PE K sweep | ✅ K=1 LOCKED, sweep COMPLETE (H120, H123) | 35.67 (H120B at K=1) | K=0 and scale=0.5 both fail |
| LR (Lion) | ✅ 3e-4 LOCKED | 3e-4 | — |
| β₂, β₁ | ✅ Locked 0.997 / 0.9 | 0.997 / 0.9 | — |
| surf_weight | ✅ 10 locked | 10 | — |
| clip_grad_norm | ✅ 1.0 locked | 1.0 | — |
| Huber δ_p | ✅ 0.25 locked | 0.25 | — |
| Batch size | ✅ 4 LOCKED (H94) | 4 | — |
| Fourier PE | 🏆 K=1 MERGED (H120) | 35.67 (H120B) | Sweep complete; K=1 scale=1.0 optimal |
| Mixed precision (bf16) | 🏆 LOCKED (H95) | −30% s/epoch | — |
| FFN act | ✅ GEGLU locked (H48) | GEGLU | — |
| Normalization | ✅ LayerNorm locked (H72) | LN | — |

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
| 35.67 | 33.40 | H120 Arm B: Fourier PE K=1 |
| 34.55 | 33.08 | H125 Arm A: wd=5e-3 |
| **33.47** | **32.64** | **H128 Arm A: compile + T_max=24 (CURRENT BEST)** |

Total merged gain: **−81.17 pts val (70.8% reduction from 114.63).**

## Strategic State

**OOD bottleneck cracked.** H128 is the first lever to significantly move val_geom_camber_rc (−2.02). The fact that compile+T_max=24 (a schedule/efficiency lever, not architecture) moved the OOD split is informative: it suggests underfitting was a contributor to OOD weakness, not just distribution shift. The 3 extra polish epochs at low LR may be doing more work than expected in the camber-OOD direction.

**Two open questions for cycle 49+:**
1. Does wd=5e-3 compound with T_max=24 + compile? (H136)
2. Do architecture/representation experiments (H131-H135) further reduce the now-12.3-pt camber gap on top of the new baseline?

**Cycle 49 batch focus:**
- **Tier 1 (OOD-targeted, architecture/representation):** H131 LE+TE coords, H132 DSDF Fourier, H134 GALE cross-attention.
- **Tier 2 (compound/regularization):** H133 spectral norm, H135 condition-Mixup, H136 wd-compound.
- **Tier 3 (revalidation):** H126 dropout, H130 AdamW.

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise non-finite GT. Read-only.
- H128 bundle: thorfinn also patched `train.py:evaluate_split` to skip NaN slices safely. Backwards-compatible (no change for clean splits). Merged in PR #4463.
- `train.py`: T_max is now a CLI arg (--T_max, default 15). All bf16 experiments: use --T_max 24 for new baseline (or --T_max 21 for old H99 stack).
- `train.py`: Fourier PE in place (--fourier_pe, --fourier_pe_freqs, --fourier_pe_scale). Current baseline K=1.
- `train.py`: torch.compile available via `--compile_mode reduce-overhead`. Use for all new experiments.
- H126/H130 were started on wd=1e-3 / pre-H128 baseline. Compare results to new baseline (33.4710) — may need rerun on updated stack.
