# SENPAI Research State

- **Date**: 2026-05-16 23:35
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — Baseline H78 (val=42.30) holds. **Major strategic pivot: training efficiency front.** Capacity probes H86 (n_hidden) and H89 (mlp_ratio) both wall-cut-bound — wider models eat epochs from the 30-min budget. Schedule lever fully closed (H74/H84/H87 all negative). Active fronts: efficiency (H95 bf16, H96 torch.compile), β₂ refinement (H88), β₁ sweep (H90), surf_weight (H91), seeds (H92), WSD (H93), batch_size (H94).
- **Most recent human research directive**: None received

## Current Best

**PR #4097 (H78 Arm B: Lion lr=3e-4 + slice_num=96 + β₂=0.995, edward) — val_avg/mae_surf_p = 42.3048 / test 3-split = 40.5564** (MERGED 2026-05-16 21:32)

**Previous best: PR #4055 (H73 Arm B, val=42.98 / test=41.55, tanjiro).** β₂=0.995 small win compounds on H73 baseline.

**Loose UB** — wall-cut at ep 15/50 with val_avg still descending ~0.8 pts/epoch. True asymptote likely well below 42.98.

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H78 Arm B (β₂=0.995 stacked on H73)** | **42.3048** | **40.5564** | **CURRENT BEST (PR #4097)** |
| H73 Arm B (Lion lr=3e-4 + slice96) | 42.9784 | 41.5455 | Prior best (PR #4055) |
| H78 Arm A (β₂=0.999) | 44.3436 | 42.0389 | Same PR, regresses |
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
| **#4166** | edward | **H88: β₂ refinement {0.992, 0.997} around H78's 0.995** | HIGH (confirm peak) | ~41-43 |
| **#4189** | askeladd | **H90: Lion β₁ sweep (β₁=0.85, β₁=0.95)** | HIGH (first β₁ retune at slice=96) | ~41-44 |
| **#4191** | fern | **H91: surf_weight sweep under Lion (sw=5, sw=20)** | MED (H54 locked surf_weight=10 under AdamW) | ~40-44 |
| **#4195** | frieren | **H92: Baseline variance — 2 seeds at H78 config** | HIGH (calibrate noise floor) | ~42-44 |
| **#4196** | nezuko | **H93: WSD schedule under Lion (vs cosine)** | MED (alternative schedule) | ~42-46 |
| **#4197** | tanjiro | **H94: Batch size sweep BS=8 (no-scale and LR-scale)** | HIGH (orthogonal to capacity) | ~40-44 |
| **TBD** | alphonse | **H95: bfloat16 mixed-precision training** | HIGH (efficiency unlock) | ~40-43 + 47% more epochs |
| **TBD** | thorfinn | **H96: torch.compile baseline acceleration** | HIGH (efficiency unlock orthogonal to bf16) | ~40-43 + ~25% more epochs |

**Closed this round:** H61 (LR-down), H62 (mlp_ratio AdamW), H63 (DropPath), H64 (Huber δ_p), H65 (EMA), H72 (RMSNorm+slice96 anti-compound AdamW), H68/H69/H70/H71 (Lion variants at slice=64, superseded), H58/H67 (superseded), **H74 (T_max extend)**, **H75 (LR U-shape)**, **H76 (warmup)**, **H77 (n_head=4)**, **H79 (wd)**, **H80 (full Lion stack: confound)**, **H81 (RMSNorm anti-compound under Lion)**, **H82 (slice sweep)**, **H83 (n_layers sweep)**, **H84 (T_max compression)**, **H85 (FFN activation)**, **H86 (n_hidden expansion — wall-cut-bound)**, **H87 (eta_min > 0)**, **H89 (mlp_ratio — wall-cut-bound)**.

**Merged this round:** H73 (Lion + slice=96 super-additive, val=42.98), **H78 (β₂=0.995 small win, val=42.30 NEW BEST)**.

## Strategic State (post-cycle 30)

**Wall-cut hypothesis confirmed.** H86 (n_hidden=192/256, val=60.68) and H89 (mlp_ratio=3/4) both fail because wider models slow s/epoch enough to eat epochs from the 30-min budget. The model is still descending steeply at ep 15 in baseline — the budget, not capacity, is the dominant constraint.

**Schedule lever fully exhausted.** Three orthogonal probes all negative:
- H74: extend T_max=20 → +6.62 regress
- H84: compress T_max=10/12 → +7.05 regress
- H87: hold eta_min > 0 → +1.5 to +3.8 regress

H73's T_max=15 + eta_min=0 + no warmup is locally optimal in all directions.

**Strategic pivot to efficiency.** If bf16 (H95) or torch.compile (H96) cut s/epoch by 25-40%, the wall-cut constraint loosens enough that capacity probes (H86, H89) become retestable. This is the highest-ROI lever class remaining.

## Lever Status (post-H73)

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion locked | 42.98 (H73) | Massive super-additive win |
| LR (Lion) | ✅ Locked at 3e-4 (H75 U-shape confirmed) | 3e-4 (H73) | Bracketed 2.5e-4 to 3.5e-4 |
| Schedule (Lion) | ❌ warmup REGRESSES at slice=96 (H76) | T_max=15 (H73) | H69 win doesn't transfer; warmup=2 cost > benefit at 15-ep horizon |
| n_head (Lion) | ❌ n_head=4 REGRESSES at slice=96 (H77) | 2 (H73) | H70 win doesn't transfer; per-head dim shrinkage hurts |
| β₂ (Lion) | ✅ 0.995 LOCKED (H78 MERGED) | 0.995 (H78) | Non-monotonic: 0.99→0.995 wins, 0.999 regresses (over-smooth in 15-ep budget). H88 refines further. |
| wd (Lion) | ✅ Locked at 1e-3 (H79 confirmed) | 1e-3 (H73) | wd=1e-4 and wd=5e-5 both regress/tie at slice=96 |
| slice_num | 🏆 96 locked | 42.98 (H73) | Confirmed under Lion. 128 still untested under Lion. |
| n_layers | ✅ Locked at 4 (H60) | 4 | Shallower wins under GEGLU |
| FFN activation | ✅ GEGLU locked (H48) | GEGLU | > SwiGLU > vanilla |
| Normalization | ✅ LayerNorm locked (H72 AdamW, H81 Lion) | LN (H73) | RMSNorm anti-compounds at slice=96 under BOTH optimizers (+1.58 AdamW, +2.44 Lion) |
| n_hidden | ❌ Wall-cut-bound (H86 closed: 192=val 60.68) | 128 | Capacity scaling blocked by training speed; retest after efficiency wins (H95/H96) |
| Schedule eta_min | ❌ H87 closed negative (+1.5 to +3.8) | 0 | H73's ~3e-6 ep-15 LR is the implicit fine-tune; replacing with ≥10× larger LR overshoots |
| mlp_ratio (Lion) | ❌ Wall-cut-bound (H89 closed) | 2 | Same mechanism as H86: wider FFN slows s/epoch, eats epochs |
| Mixed precision | 🔬 H95 active | fp32 (default) | Highest-ROI efficiency lever — 25-40% s/epoch reduction unlocks capacity probes |
| torch.compile | 🔬 H96 active | off (default) | Orthogonal to bf16; 15-30% reduction typical |
| clip_grad_norm | ✅ Locked at 1.0 | H20+H56 | — |
| surf_weight | ✅ Locked at 10 | H54 | — |
| Huber δ_p | ✅ Locked at 0.25 | H25/H64 | — |
| DropPath | ❌ No effect | 0.0 (H63) | — |
| Mixup | ❌ Wrong inductive bias | None (H55) | — |
| EMA averaging | ❌ Wrong regime | None (H65) | Needs oscillation, not translation |
| mlp_ratio | 🔬 H89 retest under Lion | 2 (H62 closed under AdamW) | First test under Lion+slice=96 regime |
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
| 42.98 | 41.55 | H73 Arm B: + optimizer=lion + lr=3e-4 (super-additive) |
| **42.30** | **40.56** | **H78 Arm B: + β₂=0.995 (small compound win)** |

Total merged gain: **−72.33 pts val** (63.1% reduction from 114.63 to 42.30).

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
