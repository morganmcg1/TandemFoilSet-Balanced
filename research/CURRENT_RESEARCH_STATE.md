# SENPAI Research State

- **Date**: 2026-05-17 (cycle 45 — PLATEAU BROKEN by H125 wd=5e-3)
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — **CURRENT BEST: H125 Arm A wd=5e-3 (val=34.5532 / test=33.0792, PR #4459).** Cycle 45: H125 merged (new best), H123/H127/H124/H129/H118 closed. Plateau broken — wd=5e-3 gained −1.11 pts val via in-dist regularization (val_single_in_dist −4.70). OOD bottleneck val_geom_camber_rc~47.8 still dominant. 8 WIP incl. new OOD-targeted architecture+representation experiments.
- **Most recent human research directive**: None received

## Current Best

**PR #4459 (H125 Arm A: wd=5e-3, edward) — val_avg=34.5532 / test 3-split=33.0792** (MERGED 2026-05-17)

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H125 Arm A (wd=5e-3)** | **34.5532** | **33.0792** | **CURRENT BEST (PR #4459)** |
| H120 Arm B (Fourier K=1) | 35.6651 | 33.3976 | Overridden (PR #4394) |
| H106 Arm B (Fourier K=4) | 35.9159 | 35.1221 | Overridden (PR #4335) |
| H99 Arm A (bf16 + T_max=21) | 37.2626 | 35.8568 | Overridden (PR #4272) |

**Cumulative R5 gain: −31.55 pts val_avg vs H37b** (66.11 → 34.55). Total: **−80.08 pts from R1 start** (114.63).

## OOD Bottleneck

**val_geom_camber_rc = 47.78 vs val_avg = 34.55** — a **13.2-pt gap**. wd=5e-3 did NOT close this (flat, ±0.3). H120 K=1 only moved it from 47.56 → 47.78. No lever has disproportionately improved this split. Active experiments targeting it: H131 (LE+TE coords), H132 (DSDF Fourier), H134 (GALE cross-attention), H135 (condition-only Mixup).

## Noise Floor

**2σ ≈ 1.67 pts** on val_avg/mae_surf_p. Test 3-split 2σ ≈ 1.02 pts.

## Round 5 Insights (cumulative)

**Confirmed improvement axes (merged):**
1. **T_max=21 (H99)**: +3.24 pts — schedule-length alignment with bf16
2. **Fourier PE K=4 (H106)**: +1.35 pts — sub-chord spatial basis
3. **Fourier K=1 (H120)**: +0.25 pts val (Δ-1.72 test) — chord-scale only; **key anti-overfitting signal**
4. **bf16 (H95)**: +0.71 pts — speed enables 21 epochs vs 15
5. **wd=5e-3 (H125)**: +1.11 pts val — in-dist memorisation regularization (val_single_in_dist −4.70)

**Closed axes (post-H125, cycle 44 complete):**
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
| **#4526** | askeladd | **H134: GALE geometry cross-attention K=32/K=16 tokens** | TOP (architecture, OOD-targeted) | ~32-34 |
| **#4509** | alphonse | **H132: DSDF Fourier K=1 (combined+ablation)** | HIGH (input repr, OOD-targeted) | ~33-35 |
| **#4480** | fern | **H131: LE+TE dual coord features (4/8 extra dims)** | HIGH (input repr, OOD-targeted) | ~33-35 |
| **#4527** | edward | **H133: spectral norm on in_project_slice** | HIGH (Lipschitz regularization) | ~33.5-35 |
| **#4529** | nezuko | **H135: condition-only Mixup α={0.2,0.5}** | HIGH (camber-axis augmentation) | ~33-35 |
| **#4460** | frieren | **H126: FFN dropout {0.1, 0.2} at K=1** | MED (baseline was wd=1e-3; compare to new baseline) | ~33-35 |
| **#4463** | thorfinn | **H128: compile + K=1 + T_max=24** | MED (extended cosine polish) | ~33.5-35 |
| **#4466** | tanjiro | **H130: AdamW vs Lion revalidation at K=1** | LOW (sanity check) | likely confirms Lion |

**Note on H126/H128/H130:** These were started on wd=1e-3 baseline (35.6651). Their results will be compared to the new wd=5e-3 baseline (34.5532). If they beat their old baseline (35.6651) but not the new one (34.5532), they should be sent back for rerun on the updated stack.

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🔬 H130 revalidating AdamW vs Lion | 35.67 (Lion at K=1) | First retest since H73 |
| Weight decay | ✅ wd=5e-3 LOCKED (H125) | 34.5532 | Δ-1.11 vs wd=1e-3; optimum near 5e-3 |
| FFN dropout | 🔬 H126 sweep {0.1, 0.2} | none (no dropout currently) | First test |
| Spectral norm (in_project_slice) | 🔬 H133 active | none | Lipschitz-1 constraint on slice assignment |
| Condition-only Mixup | 🔬 H135 active | none | Geometry-safe Mixup on NACA+flow params; H129 literal-Mixup closed |
| LE+TE dual coords | 🔬 H131 active | none | OOD-targeted input repr; Texas A&M arXiv 2412.09399 |
| DSDF Fourier features | 🔬 H132 active | none | Applies H120 K=1 mechanism to signed-distance channels |
| GALE cross-attention | 🔬 H134 active (K=32, K=16) | none | Architecture-level OOD fix; arXiv 2512.20399 |
| EMA weight averaging | ❌ Schedule-incompat (H124) | none | Same family as SWA/Lookahead |
| Lookahead wrapper | ❌ Schedule-incompat (H122) | none | Cosine→0 prevents late-epoch polish |
| SWA | ❌ Schedule-incompat (H121) | none | — |
| Mixup (literal, sample pairs) | ❌ Mesh identity violation (H129, H55) | none | Variable mesh; NaN in mixed batch |
| n_hidden | ❌ Closed: 96,112 negative (H127) + >128 negative (earlier). 128 is optimum. | 128 | — |
| Fourier PE K sweep | ✅ K=1 LOCKED, sweep COMPLETE (H120, H123) | 35.67 (H120B at K=1) | K=0 and scale=0.5 both fail |
| Schedule T_max | 🔬 H128 testing T_max=24 + compile | 37.26 (H99 at T_max=21) | — |
| LR (Lion) | ✅ 3e-4 LOCKED | 3e-4 | — |
| β₂, β₁ | ✅ Locked 0.997 / 0.9 | 0.997 / 0.9 | — |
| surf_weight | ✅ 10 locked | 10 | — |
| clip_grad_norm | ✅ 1.0 locked | 1.0 | — |
| Huber δ_p | ✅ 0.25 locked | 0.25 | — |
| Batch size | ✅ 4 LOCKED (H94) | 4 | — |
| torch.compile | 🔬 H128 active (compile + K=1 + T_max=24) | -27% s/ep alone | — |
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
| **34.55** | **33.08** | **H125 Arm A: wd=5e-3 (CURRENT BEST)** |

Total merged gain: **−80.08 pts val (69.8% reduction from 114.63).**

## Strategic State

**Plateau broken.** H125 wd=5e-3 gained −1.11 pts via in-dist memorisation reduction (val_single_in_dist dominant). The OOD bottleneck (val_geom_camber_rc ~47.8) remains the dominant gap — no lever has moved it significantly. The current 8-experiment batch splits into two tiers:

**Tier 1 (OOD-targeted, architecture/representation):** H134 GALE cross-attention, H131 LE+TE dual coords, H132 DSDF Fourier. These directly attack the 13-pt camber gap.

**Tier 2 (complementary regularization/ablations):** H133 spectral norm, H135 condition-Mixup, H126 dropout, H128 schedule, H130 AdamW. These are independent levers that may compound with Tier 1 wins.

**Open questions for cycle 45:**
1. Does GALE cross-attention (H134) reduce val_geom_camber_rc? (top OOD mechanism)
2. Do LE+TE (H131) or DSDF Fourier (H132) input features help camber OOD?
3. Does dropout (H126) compound with wd=5e-3?
4. Does condition-only Mixup (H135) synthesize camber-interpolated training samples?
5. Is Lion still optimal vs AdamW at val<35? (H130)
6. Does compile + T_max=24 give more polish than T_max=21? (H128)

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise non-finite GT (confirmed mechanism by askeladd H123: NaN * 0 = NaN in accumulate_batch). Read-only. Use 3-split excl. cruise.
- `train.py`: T_max is now a CLI arg (--T_max, default 15). All bf16 experiments: use --T_max 21.
- `train.py`: Fourier PE in place (--fourier_pe, --fourier_pe_freqs, --fourier_pe_scale). Current baseline K=1.
- H126/H128/H130 were started on wd=1e-3 baseline (35.6651). Compare results to new wd=5e-3 baseline (34.5532) — may need rerun on updated stack if they beat old but not new baseline.
