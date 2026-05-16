# SENPAI Research State

- **Date**: 2026-05-16 21:00
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 mid-phase — Lion+slice=96 baseline (H73, val=42.98). H74-H77+H79 closed negative. **Noise floor ≥2.6 pts** discovered from H74 same-schedule run.
- **Most recent human research directive**: None received

## Current Best

**PR #4055 (H73 Arm B: Lion lr=3e-4 + slice_num=96, tanjiro) — val_avg/mae_surf_p = 42.9784 / test 3-split = 41.5455** (MERGED 2026-05-16 18:32)

**Loose UB** — wall-cut at ep 15/50 with val_avg still descending ~0.8 pts/epoch. True asymptote likely well below 42.98.

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H73 Arm B (Lion lr=3e-4 + slice96)** | **42.9784** | **41.5455** | **CURRENT BEST (PR #4055)** |
| H73 Arm A (Lion lr=1e-4 + slice96) | 46.3422 | 45.3896 | Same PR, lr=1e-4 arm |
| H66 (slice_num=96, AdamW) | 56.7504 | 54.5026 | Overridden by #4055 |
| H58 Arm A reference (Lion lr=1e-4, slice=64) | ~46.80 | ~46.63 | Pending PR #3965 still WIP |
| H59 (GEGLU + RMSNorm, slice=64) | 56.9056 | 56.2420 | Overridden |
| H48 GEGLU (n_layers=5) | 58.6268 | 56.6976 | Overridden |
| H37b (n_head=2 + lr=1e-3, AdamW) | 66.1060 | 64.45 | Overridden |

**Δ H73 vs H66: −13.77 val, −12.96 test 3-split.** SUPER-ADDITIVE (3.66 pts below the additivity-floor prediction of 46.64).
**Cumulative R5 gain: −23.13 pts val vs H37b** (66.11 → 42.98).

## ⚠ Noise Floor Discovered (2026-05-16 H74 closure)

H74 Arm B (T_max=15 SGDR with restart firing AFTER wall cut) used effectively the same first-15-epoch schedule as H73 baseline. It landed at val=45.57 vs H73's 42.98 — a **2.60 pt single-seed gap** despite identical schedule.

**Implication: H73 baseline (val=42.9784) has ≥2.6 pts of seed variance.**

Future reviews must apply this noise floor:
- Δ < 2.6 pts vs H73 → likely tie, not win/loss
- Δ ≥ 3 pts → real signal (clearly outside noise)
- Δ ≥ 5 pts → strong signal

Marginal H79 Arm B (+0.38 val, −0.15 test) and H77 Arm B (+1.67 val) may be ties not losses. Future hypothesis predictions should target ≥3 pt gains to be worth testing.

## Round 5 Insights (post-merge wave)

The H67-H73 Lion compound batch revealed:
- **Lion + slice_num=96 is super-additive** (H73 Arm B at val=42.98, 3.66 pts below additivity floor). The wider gradient surface from slice=96 favors Lion's native lr=3e-4 range.
- **RMSNorm + slice_num=96 anti-compounds under AdamW** (H72, val=57.58 vs predicted 56.0). The H59 RMSNorm win at slice=64 does NOT transfer to slice=96.
- **n_head=4 wins over n_head=2 under Lion+RMSNorm+slice=64** (H70, +1.1 pts). Untested at slice=96.
- **warmup=2 wins over warmup=1 under Lion** (H69, +5.3 pts — biggest single hyperparameter signal). Untested on lr=3e-4 / slice=96.
- **β₂=0.999 wins over β₂=0.95 under Lion** (H68, +3 pts). H73 used β₂=0.99 (default).
- **wd=1e-4 beats wd=5e-4 at slice=64+Lion** (H71, +4 pts). But H73 won at wd=1e-3 — interaction with slice=96 unclear.

## Pending Strategic Results

**PR #4020 — H67 Lion + GEGLU + RMSNorm compound (alphonse) — STILL WIP**
- Pod was rate-limited, just resumed; results expected shortly.

**PR #3965 — H58 Lion + GEGLU (edward) — REBASE IN PROGRESS / WIP**
- Edward's pod just finished rebase training (~18:17Z) and is writing up results now.
- Likely will be **superseded by H73 Arm A** (which already reproduces H58 spec at slice=96, val=46.34). May close once results land.

## Key Confirmed Insights

1. **Lion + slice_num=96 is the new floor** (H73 Arm B, val=42.98). Super-additive compound.
2. **Lion optimizer is a massive win** (H58, H73). Sign-based gradient update fixes systemic AdamW optimization issue. Uniform −10 to −17 pt per-split improvement under Lion+slice=96.
3. **slice_num=96 amplifies under Lion** (H66 → H73). Test_geom_camber_rc gain (−11.68 vs H66) confirms spatial-selectivity survives Lion.
4. **GEGLU gated FFN is a major win (H48)**: val=58.63 vs H37b 66.11. GEGLU > SwiGLU > vanilla.
5. **RMSNorm wins under GEGLU at slice=64 (H59)** but does NOT compound with slice=96 (H72 negative).
6. **n_layers=4 wins under GEGLU (H60)**: Shallower wins under gated regime.
7. **Schedule lever exhausted under AdamW**: T_max=15 is optimal. Lion+warmup=2 wins big (H69) but untested at lr=3e-4.
8. **clip_grad_norm=1.0 is locked (H56)**, surf_weight=10 locked (H54).
9. **Mixup is wrong inductive bias for PDE CFD (H55 closed)**.
10. **EMA averaging fails at current budget (H65)**: Need oscillation regime, but T_max=15 cosine still in translation regime.

## Active WIP Experiments

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#4133** | askeladd | **H84: T_max compression (T_max=12, T_max=10)** | HIGH (askeladd's own follow-up) | ~41-44 |
| **#4094** | tanjiro | **H75: Lion LR sweep (lr=2e-4, lr=5e-4)** | HIGH | ~41-44 |
| **#4126** | fern | **H82: slice_num sweep under Lion (slice=128, slice=80)** | HIGH (untested Lion regime) | ~40-46 |
| **#4127** | frieren | **H83: n_layers sweep under Lion (n_layers=5, n_layers=3)** | MED (depth retune at slice=96) | ~41-46 |
| **#4135** | nezuko | **H85: FFN activation (swiglu, vanilla) under Lion** | MED (test if GEGLU locked under Lion) | ~43-47 |
| **#4093** | thorfinn | **H80: Full Lion stack — warmup+wd+β₂+n_head compound** | HIGH (bold swing) | ~35-40 optimistic |
| **#4097** | edward | **H78: Lion β₂ sweep (β₂=0.999, β₂=0.995)** | HIGH | ~41-43 |
| **#4098** | alphonse | **H81: RMSNorm under Lion+slice=96** | HIGH (closes normalization question) | ~41-44 |

**Closed this round:** H61 (LR-down), H62 (mlp_ratio), H63 (DropPath), H64 (Huber δ_p), H65 (EMA), H72 (RMSNorm+slice96 anti-compound), H68/H69/H70/H71 (Lion variants at slice=64, all superseded by H73), H58/H67 (superseded by H73), **H76 (warmup negative)**, **H77 (n_head negative)**, **H79 (wd negative, partly ties)**, **H74 (schedule extension negative; revealed noise floor)**.

**H76/H77 negative-result implication for H80:** H80 (thorfinn full stack) combines warmup=2 + wd=1e-4 + β₂=0.999 + n_head=4. Two of these four levers (warmup, n_head) are now confirmed NEGATIVE at slice=96. H80's predicted range should be revised downward — still possible to win if wd+β₂ wins compensate, but less likely.

## Lever Status (post-H73)

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion locked | 42.98 (H73) | Massive super-additive win |
| LR (Lion) | 🔬 More to find | 3e-4 (H73) | Sweet spot may be 2-4e-4 at slice=96 |
| Schedule (Lion) | ❌ warmup REGRESSES at slice=96 (H76) | T_max=15 (H73) | H69 win doesn't transfer; warmup=2 cost > benefit at 15-ep horizon |
| n_head (Lion) | ❌ n_head=4 REGRESSES at slice=96 (H77) | 2 (H73) | H70 win doesn't transfer; per-head dim shrinkage hurts |
| β₂ (Lion) | 🔬 0.999 wins at slice=64 | 0.99 (H73) | H68 found β₂=0.999 helps; retest |
| wd (Lion) | ✅ Locked at 1e-3 (H79 confirmed) | 1e-3 (H73) | wd=1e-4 and wd=5e-5 both regress/tie at slice=96 |
| slice_num | 🏆 96 locked | 42.98 (H73) | Confirmed under Lion. 128 still untested under Lion. |
| n_layers | ✅ Locked at 4 (H60) | 4 | Shallower wins under GEGLU |
| FFN activation | ✅ GEGLU locked (H48) | GEGLU | > SwiGLU > vanilla |
| Normalization | 🔀 LayerNorm at slice=96 | LN (H73) | RMSNorm anti-compounds with slice=96 (H72) |
| n_hidden | ✅ Locked at 128 | H33 | — |
| clip_grad_norm | ✅ Locked at 1.0 | H20+H56 | — |
| surf_weight | ✅ Locked at 10 | H54 | — |
| Huber δ_p | ✅ Locked at 0.25 | H25/H64 | — |
| DropPath | ❌ No effect | 0.0 (H63) | — |
| Mixup | ❌ Wrong inductive bias | None (H55) | — |
| EMA averaging | ❌ Wrong regime | None (H65) | Needs oscillation, not translation |
| mlp_ratio | ✅ 2 locked (H62) | 2 | — |
| Cond_dim | ✅ Locked at 11 (FiLM) | 11 | — |

## Key Open Questions

1. **How low can Lion+slice=96 go with extended budget?** H73 wall-cut at ep 15 still descending — full cosine at T_max=20-25 could reveal the true floor.
2. **Do the slice=64 Lion-variant wins (warmup=2, β₂=0.999, n_head=4, wd retune) compound on top of H73?** Most likely YES for warmup, less clear for the others. Need to test each on slice=96.
3. **Can we push slice_num=112 or 128 under Lion?** H66 found 128 regresses at AdamW. Lion's faster effective convergence may shift the optimum.
4. **Is RMSNorm still negative under Lion+slice=96?** H72 showed anti-compound under AdamW; not directly tested under Lion.
5. **What's the asymptote with all best Lion hyperparams stacked?** lr=3e-4 + warmup=2 + β₂=0.999 + n_head=4 + wd retuned — potentially ~38-40.

## Baseline Progression

| Val avg/mae_surf_p | Test 3-split | Event |
|---|---|---|
| 114.63 | — | R1 start (FiLM only, T_max=50 mismatch) |
| 83.81 | 80.24 | H19: T_max=15 + Huber + FiLM |
| 75.50 | 73.16 | H20: clip=1.0 |
| 71.77 | 70.62 | H27b/H32: lr=1e-3 |
| 68.19 | 65.44 | H38: wd=5e-5 |
| 66.11 | 64.45 | H37b: n_head=2 + lr=1e-3 |
| 63.44 | 61.39 | H39 Arm C: + lr=2e-3 (documentation) |
| 58.63 | 56.70 | H48 GEGLU: + ffn_act=geglu |
| 57.58 | 56.46 | H60: + n_layers=4 |
| 56.91 | 56.24 | H59: + norm_type=rmsnorm |
| 56.75 | 54.50 | H66: + slice_num=96 |
| **42.98** | **41.55** | **H73 Arm B: + optimizer=lion + lr=3e-4 (super-additive)** |

Total merged gain: **−71.65 pts val** (62.5% reduction from 114.63 to 42.98).

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
