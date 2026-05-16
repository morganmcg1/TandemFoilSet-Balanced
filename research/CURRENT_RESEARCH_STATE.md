# SENPAI Research State

- **Date**: 2026-05-16 16:00
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active — GEGLU + n_layers=4 baseline (val=57.5750); slice/regularization phase
- **Most recent human research directive**: None received

## Current Best

**PR #3968 (H60: GEGLU + n_layers=4, thorfinn) — val_avg/mae_surf_p = 57.5750** (MERGED 2026-05-16)

Test 3-split avg (excl. cruise NaN bug): **56.4610**

| Reference | val_avg/mae_surf_p | Status |
|-----------|--------------------|--------|
| **H60 Arm B (n_layers=4 at GEGLU base)** | **57.5750** | **CURRENT BEST (PR #3968)** |
| H48 GEGLU (n_layers=5) | 58.6268 | Overridden by #3968 |
| H48 SwiGLU (same stack + swiglu) | 61.4410 | Merged historical |
| H49 Lion Arm A (H37b base) | 60.3008 | Closed PR #3859 |
| H39 Arm C (n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0) | 63.4385 | Merged PR #3683 (documentation) |
| H37b (n_head=2 + lr=1e-3 + clip=1.0, default wd) | 66.1060 | Overridden |

**Δ H60 vs H48: −1.05 pts val, −0.24 pts test.** Wins on all 3 OOD splits (rc, cruise, re_rand). Cumulative R5 gain: −8.53 pts val vs H37b.

## Key Confirmed Insights

1. **GEGLU gated FFN is a major win (H48)**: val=58.63 vs H37b 66.11. Spatial selectivity via multiplicative gating is a direct fit for CFD boundary-layer gradients. GEGLU outperforms SwiGLU by 2.8 pts.
2. **GEGLU generalizes better than it in-fits**: test gain > val gain. Gating mechanism reduces OOD sensitivity — critical for our cross-geometry/Reynolds evaluation.
3. **GEGLU has different LR sensitivity than vanilla FFN (H57 falsified)**: lr=2e-3 hurts GEGLU (+0.88 regression) even though it won +2.67 on vanilla FFN (H39 Arm C). Gate's concentrated gradient updates overshoot at higher LR. GEGLU optimum is ≤ 1e-3, possibly below it (→ H61).
4. **LR ceiling confirmed at 2e-3 (H51+H57)**: Both vanilla FFN and GEGLU architectures show no benefit from lr > 2e-3. The LR lever is fully mapped.
5. **clip_grad_norm=1.0 is the global optimum (H56)**: Lower clip (0.5, 0.7) hurts GEGLU. Gates need full gradient signal to train effectively.
6. **surf_weight=10 is locked (H54)**: surf_weight=20 starves volume signal (vol_p +27%). surf_weight=5 weakens surface constraint. surf_weight=10 is the optimal balance.
7. **T_max=15 hardcoded mismatch was the first-order fix (R1)**: 11.7-pt gain.
8. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 — merged defaults. May need re-tuning at GEGLU base (→ H64).
9. **lr monotone trend holds through 2e-3 for vanilla FFN (H39 Arm C)**: −2.67 pts vs H37b at n_head=2+wd=5e-5 stack.
10. **wd=5e-5 is LR-normalized regularization (H38)**: Both H48 and H39 used wd=5e-5.
11. **n_head=2 is the global optimum (H37b, H46)**: U-shape 8→4→2→1 confirmed.
12. **Architecture width fails (H33)**: n_hidden=192/256 regress.
13. **n_layers=3 isolated win does NOT stack with n_head=2 (H42 Arm C)**: Capacity reductions destroy each other — BUT this was pre-GEGLU. H60 revisits depth in GEGLU context.
14. **β₁=0.8 isolated win does NOT stack (H44 Arm C)**: β₁=0.9 is optimal.
15. **Schedule lever exhausted**: H43/H41C/H47/H50/H56 all failed. WSD also failed. Cosine T_max=15 is the right schedule.
16. **Lion optimizer beats AdamW at H37b base (H49)**: val=60.30 (Arm A lr=1e-4), −5.80 vs H37b. Sign-normalization removes gradient imbalance between high-Re and low-Re samples. H58 tests Lion+GEGLU compounding.
17. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.

## Active WIP Experiments

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#3965** | edward | **H58: Lion + GEGLU** (mega-stack) | **CRITICAL** | ~52-55 |
| **#3966** | fern | **H59: RMSNorm in GEGLU Transolver** | HIGH | ~57-58 |
| **#3988** | alphonse | H61: GEGLU + LR down (7e-4, 5e-4) | HIGH | ~57.5-58.5 |
| **#3990** | askeladd | H62: GEGLU + mlp_ratio (3, 4) | HIGH | ~57-58 |
| **#3991** | frieren | H63: DropPath (0.05, 0.10) | MEDIUM | ~57-58 |
| **#3992** | nezuko | H64: Huber δ_p retune (0.1, 0.5) | MEDIUM | ~57.5-58.5 |
| **#3997** | tanjiro | H65: EMA weight averaging (0.999, 0.9999) | MEDIUM | ~57.5-58.5 |
| **#4011** | thorfinn | **H66: slice_num sweep (96, 128) at n_layers=4** | HIGH | ~56.5-57.5 |

All 8 students active. Zero idle.

**H60 (n_layers=4) merged as new baseline (57.5750).** All other in-flight experiments ran on n_layers=5 — their results are still interpretable but winning levers will need re-stacking on n_layers=4 in subsequent rounds.

**H55 (Mixup) closed:** Both arms regressed +15.5 / +24.7 pts. PDE nonlinearity + mesh-identity feature corruption make raw-input Mixup the wrong inductive bias for CFD surrogates. Mixup lever exhausted for this dataset.

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| LR (vanilla FFN) | ✅ Confirmed 2e-3 ceiling | H39 Arm C: 63.44 | Monotone up to 2e-3 |
| LR (GEGLU) | 🔬 H61 testing | H48: 58.63 at 1e-3 | H57 shows 2e-3 hurts; 7e-4/5e-4 in-flight |
| clip_grad_norm | ✅ Locked at 1.0 | H20+H56 | clip=0.5/0.7 regress |
| surf_weight | ✅ Locked at 10 | H54 | surf=5/20 both regress |
| Schedule (cosine) | ✅ Exhausted | T_max=15 | H43/H41C/H47/H50 all failed |
| n_head | ✅ Locked at 2 | H37b/H46 | U-shape confirmed |
| n_hidden | ✅ Locked at 128 | H33 | Width fails |
| n_layers | ✅ Locked at 4 (H60) | 4 (new baseline) | GEGLU's per-block capacity favors shallower; H42 finding doesn't transfer |
| slice_num | 🔬 H66 testing | 64 current | Wider slice tokens for finer spatial selectivity (paired with n_layers=4 budget) |
| FFN activation | ✅ GEGLU wins | H48: 58.63 | GEGLU > SwiGLU > vanilla |
| mlp_ratio | 🔬 H62 testing | 2 current | Never swept; Llama lit suggests bigger |
| Huber δ_p | 🔬 H64 testing | 0.25 (H25) | Tuned at val=83.81; model much better now |
| Optimizer | 🔬 H58 testing | AdamW | Lion wins at H37b base by 5.8 pts |
| Normalization | 🔬 H59 testing | LayerNorm | RMSNorm may benefit GEGLU gate |
| DropPath | 🔬 H63 testing | 0.0 | Novel for this arch/size |
| Mixup augmentation | ✅ Closed negative (H55) | None | Wrong inductive bias for PDE/CFD: +15-25 pt regression both arms |
| EMA weight averaging | 🔬 H65 testing | None | Polyak/SWA — flatter minima for OOD |
| β₁ (Adam momentum) | ✅ Locked at 0.9 | H44 | 0.8 isolated win doesn't stack |
| wd | ✅ Locked at 5e-5 | H38 | LR-normalized |

## Key Open Questions

1. **Does Lion + GEGLU compound?** H58 edward (#3965). If gain is additive: ~52.8. Would be largest single gain since T_max fix.
2. **Does GEGLU's optimal LR live below 1e-3?** H61 alphonse (#3988). H57 shows 2e-3 hurts — maybe 7e-4/5e-4 is optimal for gated arch.
3. **Does mlp_ratio matter for gated FFN?** H62 askeladd (#3990). Llama uses larger expansion; GEGLU's 3-matrix structure may need more width.
4. **Does RMSNorm help GEGLU's gate?** H59 fern (#3966). LayerNorm's mean-subtraction may distort gate activation directions.
5. **Does depth become a fresh lever in GEGLU?** H60 thorfinn (#3968). n_layers=3 failed at vanilla FFN base; GEGLU's stronger per-layer capacity may change optimal depth.
6. **Does DropPath help OOD?** H63 frieren (#3991). Strong mechanistic motivation for generalization across geom/Re distributions.
7. **Does Huber δ_p need re-tuning?** H64 nezuko (#3992). H25 optimized at val=83.81; error distribution is radically different at val=58.63.
8. **Does EMA weight averaging help OOD?** H65 tanjiro (#3997). Polyak averaging finds flatter minima — should help most where the model is least confident (OOD splits).
9. **Does wider slice token representation help at the n_layers=4 budget?** H66 thorfinn (#4011). H10's R1-era result is stale; retesting at GEGLU/n_layers=4 with current capacity.
10. **What's next below 55?**:
   - GEGLU + Lion + optimal LR (if H58 + H61 both win)
   - GEGLU + RMSNorm + wider mlp (if H59 + H62 win)
   - GEGLU + deeper network + Lion (if H60 + H58 win)
   - Architecture-level: attention mechanism variants, PDE-informed residuals, FiLM conditioning improvements

## Baseline Progression

| Val avg/mae_surf_p | Test 3-split | Event |
|---|---|---|
| 114.63 | — | R1 start (FiLM only, T_max=50 mismatch) |
| 83.81 | 80.24 | H19: T_max=15 + Huber + FiLM |
| 75.50 | 73.16 | H20: clip=1.0 |
| 71.77 | 70.62 | H27b/H32: lr=1e-3 |
| 68.19 | 65.44 | H38: wd=5e-5 |
| 66.11 | 64.45 | H37b: n_head=2 + lr=1e-3 |
| 63.44 | 61.39 | H39 Arm C: + lr=2e-3 (merged #3683, documentation) |
| 58.63 | 56.70 | H48 GEGLU: + ffn_act=geglu |
| **57.58** | **56.46** | **H60: + n_layers=4** |

Total gain: **−57.0 pts val** (49.8% reduction from 114.63 to 57.58).

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `T_max=15` hardcoded in scheduler — students doing T_max sweeps must add CLI flag.
