# SENPAI Research State

- **Date**: 2026-05-16 16:35
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active — Lion optimizer compounding phase (H58 rebase pending; H67-H72 in-flight)
- **Most recent human research directive**: None received

## Current Best

**PR #4011 (H66: slice_num=96 at GEGLU n_layers=4, thorfinn) — val_avg/mae_surf_p = 56.7504** (MERGED 2026-05-16 16:31)

Test 3-split avg (excl. cruise NaN bug): **54.5026**

| Reference | val_avg/mae_surf_p | test 3-split | Status |
|-----------|--------------------|--------------|--------|
| **H66 Arm A (slice_num=96)** | **56.7504** | **54.5026** | **CURRENT BEST (PR #4011)** |
| H59 (GEGLU + RMSNorm, slice=64) | 56.9056 | 56.2420 | Overridden by #4011 |
| H60 Arm B (n_layers=4 at GEGLU) | 57.5750 | 56.4610 | Overridden by #3966 |
| H48 GEGLU (n_layers=5) | 58.6268 | 56.6976 | Overridden |
| H37b (n_head=2 + lr=1e-3) | 66.1060 | 64.45 | Overridden |
| (H58 Arm A Lion+GEGLU — pending rebase) | (46.7957) | (46.6320) | **PENDING — loose UB, wall-cut** |

**Δ H66 vs H59: −0.16 val, −1.74 test 3-split.** Test-side gain concentrates on geometry-OOD (test_geom_camber_rc −3.33 pts).
**Cumulative R5 gain: −9.36 pts val vs H37b** (66.11 → 56.75).

## Pending Strategic Result

**PR #3965 — H58 Lion + GEGLU (edward) — TERMINAL but SENT BACK FOR REBASE**

- **Arm A (lr=1e-4, wd=1e-3, β=(0.9,0.99)) val_avg = 46.7957** (loose upper bound — wall-clock cut at epoch 13/50 with val still dropping ~2.4 pt/ep)
- Δ −10.11 vs current baseline 56.75 — **strongest single-PR signal of the round**
- Sent back at 15:37Z for train.py rebase onto post-H59 codebase; edward's pod recovered from rate-limit at 16:23Z and is working on rebase
- 5 Lion-compound hypotheses (H67-H71) seeded in parallel

## Key Confirmed Insights

1. **Lion optimizer is a massive win on GEGLU (H58)**: val=46.80 vs 58.63 H48 base, −11.83 pts uniform across all 4 val splits. Numbers are loose upper bounds — wall-cut mid-cosine.
2. **GEGLU gated FFN is a major win (H48)**: val=58.63 vs H37b 66.11. GEGLU outperforms SwiGLU by 2.8 pts.
3. **RMSNorm (fused kernel) wins under GEGLU (H59)**: −0.67 val_avg vs LayerNorm at H60 baseline. Win is dominated by per-epoch speedup (fused F.rms_norm kernel) yielding one extra training step within wall budget.
4. **slice_num=96 wins on test-OOD (H66)**: −0.16 val, −1.74 test 3-split vs H59 baseline. Test gain concentrates on geometry-OOD (camber_rc −3.33 pts), confirming the spatial-selectivity mechanism. slice_num=128 regresses (overfit at 1499 training samples).
5. **n_layers=4 wins under GEGLU (H60)**: Pre-GEGLU H42 finding (deeper+n_head=2 hurts) does NOT transfer to gated regime. Frees ~13% s/epoch.
6. **GEGLU has different LR sensitivity than vanilla FFN (H57 falsified)**: lr=2e-3 hurts GEGLU. Optimum ≤1e-3 for AdamW, ~1e-4 for Lion.
7. **clip_grad_norm=1.0 is locked (H56)**, surf_weight=10 locked (H54), n_head=2 locked under AdamW (H37b/H46).
8. **Schedule lever exhausted under AdamW**: Cosine T_max=15 is optimal. Lion + linear warmup still open → H69.
9. **Mixup is wrong inductive bias for PDE CFD (H55 closed)**: PDE nonlinearity + mesh-identity corruption.
10. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.

## Active WIP Experiments

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#3965** | edward | **H58 REBASE: Lion + GEGLU + RMSNorm verification** | **CRITICAL** | ~45-47 (rebase ongoing) |
| **#3988** | alphonse | H61: GEGLU + LR down (7e-4, 5e-4) at AdamW | MEDIUM | ~56.5-57 |
| **#3990** | askeladd | H62: GEGLU + mlp_ratio (3, 4) | MEDIUM | ~56-57 |
| **#3991** | frieren | H63: DropPath (0.05, 0.10) | MEDIUM | ~56-57 |
| **#3992** | nezuko | H64: Huber δ_p retune (0.1, 0.5) | MEDIUM | ~56.5-57 |
| **#3997** | tanjiro | H65: EMA weight averaging (0.999, 0.9999) | MEDIUM | ~56.5-57 |
| **#4020** | alphonse | **H67: Lion + GEGLU + RMSNorm compound** | **CRITICAL** | ~45-48 |
| **#4022** | askeladd | **H68: Lion β₂ sweep (0.95, 0.999)** | **HIGH** | ~46-48 |
| **#4023** | fern | **H69: Lion + linear LR warmup** | **HIGH** | ~45-47 |
| **#4024** | frieren | **H70: n_head sweep under Lion** | **HIGH** | ~46-48 |
| **#4025** | nezuko | **H71: Lion wd sweep (1e-4, 5e-4)** | **HIGH** | ~46-48 |
| **#4048** | thorfinn | **H72: slice_num=96 + RMSNorm compound** | **HIGH** | ~56.0-56.5 |

All 8 students WIP (some with 2 active assignments from dual-batch Lion seeding). Zero idle.

**Rate-limit note:** All 8 pods were blocked 15:30-16:22Z on shared GitHub API rate limit. Pods are recovering — edward's confirmed back at 16:23Z. H67-H71 students may still be picking up their assignments.

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion wins big (H58) | 46.80 (loose UB) | Δ −10.11. H67-H71 exploring Lion compounds |
| LR (Lion) | 🔬 H67/H69 testing | 1e-4 (H58) | H67 tests 1e-4/3e-4; H69 tests warmup |
| LR (AdamW+GEGLU) | 🔬 H61 testing | 1e-3 (H48) | H57 shows 2e-3 hurts; 7e-4/5e-4 in-flight |
| clip_grad_norm | ✅ Locked at 1.0 | H20+H56 | — |
| surf_weight | ✅ Locked at 10 | H54 | — |
| Schedule | ✅ AdamW exhausted | T_max=15 | LR warmup under Lion still open → H69 |
| n_head (AdamW) | ✅ Locked at 2 | H37b/H46 | May differ under Lion → H70 |
| n_hidden | ✅ Locked at 128 | H33 | Width fails |
| n_layers | ✅ Locked at 4 (H60) | 4 | Shallower wins under GEGLU |
| slice_num | 🏆 96 wins (H66) | 56.7504 / 54.5026 | Confirmed vs H59 baseline. 128 regresses. |
| FFN activation | ✅ GEGLU wins | H48: 58.63 | GEGLU > SwiGLU > vanilla |
| mlp_ratio | 🔬 H62 testing | 2 current | — |
| Huber δ_p | 🔬 H64 testing | 0.25 (H25) | May need retune at tighter baseline |
| Normalization | 🏆 RMSNorm wins (fused) | H59: 56.9056 | Per-epoch speedup gives extra step |
| slice_num+RMSNorm | 🔬 H72 testing | TBD | Compound of H66+H59; predicted near-additive |
| DropPath | 🔬 H63 testing | 0.0 | Novel for this arch/size |
| EMA averaging | 🔬 H65 testing | None | Polyak/SWA — flatter minima for OOD |
| β₂ (Lion) | 🔬 H68 testing | 0.99 (H58) | Lion-specific |
| wd (Lion) | 🔬 H71 testing | 1e-3 (H58) | — |

## Key Open Questions

1. **What's the true Lion+GEGLU asymptote?** H58 at val=46.80 is loose UB — cut at epoch 13/50 with val still dropping. Edward rebasing; H67-H71 explore Lion compounds.
2. **Does slice_num=96 + RMSNorm compound additively?** H72 thorfinn (#4048). Predicted: ~56.0-56.5 if additive (H66 −0.16 + H59 −0.67 = −0.83 from H60).
3. **Does Lion+RMSNorm compound?** H67 alphonse (#4020). H58 used LayerNorm; adding RMSNorm should give +0.67 pts from the fused-kernel speedup.
4. **Are β₂, wd, n_head, warmup levers under Lion well-characterized?** H68/H69/H70/H71 mapping this space.
5. **Do slice_num=96 and Lion compound?** Not yet explicitly tested — would need H67+ Lion baseline merged before testing.

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
| 57.58 | 56.46 | H60: + n_layers=4 |
| 56.91 | 56.24 | H59: + norm_type=rmsnorm (fused) |
| **56.75** | **54.50** | **H66: + slice_num=96** |
| (46.80) | (46.63) | (H58 Arm A — pending rebase; loose UB) |

Total merged gain: **−57.9 pts val** (50.5% reduction from 114.63 to 56.75). Pending Lion merge would push to ~−68 pts.

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- **H66 + H59 config nuance:** H66 was measured with LayerNorm (pre-H59 merge). The merged codebase has both flags available; run with `--norm_type rmsnorm --slice_num 96` to stack both (→ H72).
