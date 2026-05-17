# SENPAI Research State

- **Date**: 2026-05-17 (cycle 44 — PLATEAU PROTOCOL ENGAGED)
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — **CURRENT BEST: H120 Arm B Fourier PE K=1 (val=35.67 / test=33.40, PR #4394).** Cycle 44: closed H121 (SWA schedule-incompat), H115 Arm C (slice+Fourier anti-compound), H122 (Lookahead schedule-incompat), H118 (stale). Assigned 6 idle students with anti-overfitting hypotheses (H125-H130). Researcher-agent spawned in background. 8 WIP, 0 idle.
- **Most recent human research directive**: None received

## Plateau Status

**8 consecutive negative results since H120 K=1 merge.** The K=8→K=4→K=2→K=1 monotone test improvement told us the model overfits training-set spatial detail. All in-flight assignments now target anti-overfitting from different abstraction levels:

| Mechanism | Lever | Status |
|-----------|-------|--------|
| L2 regularization | wd sweep (H125) | WIP |
| Probabilistic regularization | FFN dropout (H126) | WIP |
| Capacity reduction | n_hidden=96,112 (H127) | WIP |
| Weight averaging | EMA τ=0.999 (H124) | WIP |
| Input encoding ablation | Fourier K=0 (H123) | WIP |
| Sample-space augmentation | Mixup α=0.2,0.5 (H129) | WIP |
| Schedule extension | T_max=24 + compile (H128) | WIP |
| Optimizer revalidation | AdamW at K=1 (H130) | WIP |

If 4+ of these regress, plateau breaks through researcher-agent's fresh angles (cycle 45+).

## Current Best

**PR #4394 (H120 Arm B: Fourier PE K=1, askeladd) — val_avg=35.6651 / test 3-split=33.3976** (MERGED 2026-05-17)

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H120 Arm B (Fourier K=1)** | **35.6651** | **33.3976** | **CURRENT BEST (PR #4394)** |
| H106 Arm B (Fourier K=4) | 35.9159 | 35.1221 | Overridden (PR #4335) |
| H99 Arm A (bf16 + T_max=21) | 37.2626 | 35.8568 | Overridden (PR #4272) |
| H95 Arm A (bf16 walltime) | 40.5066 | 39.0160 | Overridden (PR #4215) |

**Cumulative R5 gain: −30.44 pts val_avg vs H37b** (66.11 → 35.67). Total: **−78.96 pts from R1 start** (114.63).

## Noise Floor

**2σ ≈ 1.67 pts** on val_avg/mae_surf_p. Test 3-split 2σ ≈ 1.02 pts. Recent SWA/Lookahead PRs showed baseline reproduction variance of 0.5-0.9 pts → multi-seed comparison may be needed for sub-noise wins.

## Round 5 Insights (cumulative)

**Confirmed improvement axes (merged):**
1. **T_max=21 (H99)**: +3.24 pts — schedule-length alignment with bf16
2. **Fourier PE K=4 (H106)**: +1.35 pts — sub-chord spatial basis
3. **Fourier K=1 (H120)**: +0.25 pts val (Δ-1.72 test) — chord-scale only; **key anti-overfitting signal**
4. **bf16 (H95)**: +0.71 pts — speed enables 21 epochs vs 15

**Closed axes (8+ post-K=1 negatives):**
- WSD schedule (H119): incompat with Fourier
- SWA (H121): incompat with cosine→0
- Lookahead (H122): incompat with cosine→0
- slice_num=80 + Fourier K=4 (H115 Arm C): anti-compound
- Per-sample p std normalization (H104): catastrophic
- FiLM cond jitter σ=0.05 (H112): washes out conditioning
- mlp_ratio=3 (H103): no signal
- n_layers=6 (H113): definitively worse
- log(Re) aux head (H107): no signal (FiLM sufficient)

**Fourier frequency sweep (complete-ish):**
| K | val_avg | test 3-split |
|---|---------|-------------|
| 8 | 36.91 | — |
| 4 | 35.92 | 35.12 |
| 2 | 36.20 | 34.85 |
| **1** | **35.67** | **33.40** |
| 0 | ? | ? ← H123 Arm A WIP |
| 1, scale=0.5 | ? | ? ← H123 Arm B WIP |

## Active WIP Experiments (8 / 8 students, 0 idle)

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#4123** | askeladd | **H123: Fourier K=0 ablation + K=1 scale=0.5** | TOP (close freq sweep) | K=0: ~35-36 |
| **#4452** | alphonse | **H124: EMA τ=0.999/0.9995 at K=1** | HIGH | ~35.0-35.4 |
| **#4459** | edward | **H125: wd sweep {5e-3, 1e-2} at K=1** | HIGH (direct anti-overfit) | ~34.5-35.5 |
| **#4460** | frieren | **H126: FFN dropout {0.1, 0.2} at K=1** | HIGH (no dropout currently) | ~34.5-35.5 |
| **#4462** | nezuko | **H127: n_hidden {96, 112} at K=1** | HIGH (smaller direction never tested) | ~34.8-36.0 |
| **#4463** | thorfinn | **H128: compile + K=1 + T_max=24** | MED (efficiency + extended polish) | ~35.0-36.0 |
| **#4480** | fern | **H131: LE+TE dual coord features (4/8 extra dims)** | HIGH (OOD input representation, orthogonal to cycle 44 batch) | ~34.5-35.3 |
| **#4466** | tanjiro | **H130: AdamW vs Lion revalidation at K=1** | LOW (sanity check) | likely confirms Lion |

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🔬 H130 revalidating AdamW vs Lion | 35.67 (Lion at K=1) | First retest since H73 |
| Weight decay | 🔬 H125 sweep {5e-3, 1e-2} | 1e-3 locked | Direct anti-overfit |
| FFN dropout | 🔬 H126 sweep {0.1, 0.2} | none (no dropout currently) | First test |
| n_hidden | 🔬 H127 smaller direction {96, 112} | 128 (locked direction) | Never tested smaller |
| Mixup | ❌ Literal Mixup closed (H129: variable mesh, H55 repeat). Condition-only variant deferred. | none | Mesh identity violation |
| LE+TE dual coords | 🔬 H131 active | none | OOD-targeted input repr; Texas A&M arXiv 2412.09399 |
| EMA weight averaging | 🔬 H124 active | none | Different from SWA (closed) |
| Lookahead wrapper | ❌ Schedule-incompat (H122) | none | Cosine→0 prevents late-epoch polish |
| LR (Lion) | ✅ 3e-4 LOCKED | 3e-4 | — |
| Schedule T_max | 🔬 H128 testing T_max=24 + compile | 37.26 (H99 at T_max=21) | — |
| Schedule WSD | ❌ Did not compound with Fourier (H119) | 36.29 (H114B at H99 only) | — |
| Schedule warmup | ❌ Negative (H76) | none | — |
| β₂, β₁, wd | ✅ All locked (wd contested by H125) | 0.997 / 0.9 / 1e-3 | — |
| Fourier PE | 🏆 K=1 MERGED (H120) | 35.67 (H120B) | Monotone test trend K=8→1 |
| Fourier K sweep | 🔬 H123 active (K=0 ablation + K=1 scale=0.5) | K=1 optimal so far | — |
| torch.compile | 🔬 H128 active (compile + K=1 + T_max=24) | -27% s/ep alone | — |
| SWA | ❌ Schedule-incompat (H121) | none | — |
| OOD input jitter (cond) | ❌ Catastrophic (H112) | none | — |
| Per-sample p norm | ❌ Catastrophic (H104) | none | — |
| slice_num=80 + Fourier | ❌ Anti-compound (H115 Arm C) | 96 (locked) | — |
| mlp_ratio | ❌ No signal (H103) | 2 | — |
| n_layers | ❌ Definitively negative (H113) | 4 | — |
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

**Plateau protocol engaged.** 8 consecutive post-merge negatives is a clear signal that hyperparameter tweaks are exhausted at the current architecture. Cycle 44 covers the canonical anti-overfitting mechanisms (wd, dropout, n_hidden, Mixup) plus completing the Fourier sweep (H123 K=0) and weight averaging (H124 EMA).

**Researcher-agent spawned** to find bigger ideas for cycle 45+: architecture variants (attention, equivariance, multi-scale), geometric features, loss reformulation, distillation. Output expected at `/workspace/senpai/target/research/RESEARCH_IDEAS_2026-05-17_0800.md`.

**Unresolved OOD bottleneck:** val_geom_camber_rc=47.56 is still 12 pts above val_avg=35.67. No lever yet has specifically reduced this gap.

**Open questions for cycle 44-45:**
1. Does stronger wd or dropout break the plateau? (H125, H126)
2. Does smaller capacity reduce overfitting? (H127)
3. Does Mixup or EMA generalize better? (H129, H124)
4. Does K=0 beat K=1 (no Fourier at all)? (H123)
5. Is Lion still the right optimizer at val<36? (H130)
6. Does compile enable longer schedule? (H128)

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: T_max is now a CLI arg (--T_max, default 15). All bf16 experiments: use --T_max 21.
- `train.py`: Fourier PE is in place (--fourier_pe, --fourier_pe_freqs). Current baseline uses K=1.
- `train.py`: Dropout, Mixup, EMA, Lookahead, compile, fourier_pe_scale, optimizer adamw — none yet in advisor; assigned PRs will add them.
- `train.py`: WSD/Lookahead/SWA schedule mechanisms all closed; do NOT re-attempt without changing baseline schedule.
