# SENPAI Research State

- **Date**: 2026-05-17 (cycle 35)
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 late-phase — **CURRENT BEST: H95 Arm A bf16 (val=40.51 / test=39.02, PR #4215).** bf16 locked. Cycle 35 closed H97 (LR locked at 3e-4) and H94 (BS=4 locked); sent back H96 for compound compile+bf16+T_max=21 test. Capacity sweep across 4 axes under bf16: H100 (width), H101 (depth), H102 (slice_num), H103 (mlp_ratio).
- **Most recent human research directive**: None received

## Current Best

**PR #4215 (H95 Arm A: bf16 walltime, alphonse) — val_avg/mae_surf_p = 40.5066 / test 3-split = 39.0160** (MERGED 2026-05-17)

**Previous best: PR #4166 (H88 Arm B: β₂=0.997, edward, val=41.22 / test=39.53).** bf16 autocast cuts s/epoch 122→85.6 (-30%), enabling 21 epochs vs 15 in the 30-min wall budget. Quality verified by Arm B (matched ep15: val=41.54, within noise of H88). Best epoch 17 reached during rising-LR phase (T_max=15 hardcode causes cosine bounce after ep15 — H99 addresses this).

| Reference | val_avg | test 3-split | Status |
|-----------|--------:|-------------:|--------|
| **H95 Arm A (bf16 walltime)** | **40.5066** | **39.0160** | **CURRENT BEST (PR #4215)** |
| H88 Arm B (β₂=0.997) | 41.2153 | 39.5337 | Prior best (PR #4166) |
| H78 Arm B (β₂=0.995) | 42.3048 | 40.5564 | Overridden (PR #4097) |
| H73 Arm B (Lion lr=3e-4 + slice96) | 42.9784 | 41.5455 | Overridden (PR #4055) |
| H88 Arm A (β₂=0.992) | 42.2565 | 41.3459 | Plateau — ties 0.995 |
| H78 Arm A (β₂=0.999) | 44.3436 | 42.0389 | Over-smoothed |

**Cumulative R5 gain: −25.60 pts val_avg vs H37b** (66.11 → 40.51). Total: −74.12 pts from R1 start (114.63).

## Noise Floor (revised H92)

**Revised: 2σ = 1.67 pts** on val_avg/mae_surf_p (H92, n=3 seeds at H78 config). Prior 2.6-pt estimate was schedule-confounded (H74 vs H73 compared different schedules, not seeds).

Decision thresholds:
- Δ < 1.7 pts → noise (tie)
- Δ ≥ 2.5 pts → real signal
- Δ ≥ 4.0 pts → strong signal

Test 3-split 2σ ≈ 1.02 pts (tighter than val).

## Round 5 Insights (cumulative)

**bf16 unlock:** The key insight of this cycle. bf16 reduces s/epoch by ~30% without changing memory (30.46 GB), unlocking capacity probes that were previously wall-cut-bound (H86, H89). The schedule confound (T_max=15 hardcoded vs 21 epochs run) means there's likely more headroom with a fixed schedule (H99).

**Lion optimizer axis near-saturated:**
- β₂=0.997: LOCKED (H88 — plateau [0.992,0.995], sharp jump at 0.997, cliff at 0.999)
- β₁=0.9: LOCKED (H90 confirmed at β₂=0.995; H98 active at β₂=0.997)
- surf_weight=10: LOCKED under Lion (H91 — sign-update insensitive to loss magnitude weighting)
- LR=3e-4: Tentatively held (H97 active — retest at β₂=0.997)

**Capacity re-opened by bf16:** H86 (n_hidden=192) and H89 (mlp_ratio=3) were wall-cut-bound at fp32. H100 retests n_hidden=192 with bf16 (~14 epochs vs 10). H101 probes n_layers=5 with bf16 (~17 epochs). These may unlock the next improvement tier.

## Active WIP Experiments

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#4272** | alphonse | **H99: bf16 + T_max=21 schedule fix** | HIGH (T_max=15 bounce confound; fix may yield Δ-1 to -2 pts) | ~38.5-40.5 |
| **#4276** | askeladd | **H100: n_hidden=192 under bf16 (H86 retest)** | HIGH (wall-cut-bound at fp32; bf16 gives ~14 epochs) | ~38-42 |
| **#4277** | frieren | **H101: n_layers=5 depth probe under bf16** | HIGH (depth unexplored; bf16 gives ~17 epochs at n_layers=5) | ~39-42 |
| **#4291** | edward | **H102: slice_num=128 attention capacity under bf16** | HIGH (slice negative under AdamW; new regime may unlock) | ~38-42 |
| **#4292** | tanjiro | **H103: mlp_ratio=3 FFN capacity retest under bf16** | HIGH (H89 wall-cut at fp32; bf16 should reach ~17 ep) | ~38-42 |
| **#4239** | fern | **H98: β₁ retune at β₂=0.997 (β₁=0.85, β₁=0.95)** | MED (H90 confirmed β₁=0.9 near-optimal at β₂=0.995; confirm at 0.997) | ~40-42 |
| **#4196** | nezuko | **H93: WSD schedule Arms B/C (0/10/5, 0/5/10) at β₂=0.997** | MED (needs rebase; budget-aware WSD; Arm A confounded by warmup+budget mismatch) | ~39-44 |
| **#4217** | thorfinn | **H96 (sent back): compile + bf16 + T_max=21 compound** | HIGH (efficiency stack test; compile -27%, bf16 -30%, may compound) | ~38-41 |

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion locked | 42.98 (H73) | Massive super-additive win vs AdamW |
| LR (Lion) | ✅ 3e-4 LOCKED at β₂=0.997 (H97 closed) | 3e-4 (H73) | H97 both arms within noise vs H88, worse vs H95. LR optimum stable across β₂ shift. |
| Schedule T_max | 🔬 H99 active (fix T_max=21 for bf16) | 15 (H73) | T_max=15 hardcoded; confound at 21 bf16 epochs |
| Schedule warmup | ❌ Regresses at slice=96 (H76) | none | — |
| Schedule WSD | 🔬 H93 (budget-aware Arms B/C) | — | Arm A confounded; real test pending |
| n_head | ❌ n_head=4 regresses at slice=96 (H77) | 2 | — |
| β₂ (Lion) | ✅ 0.997 locked (H88) | 0.997 | Peak confirmed; cliff at 0.999 |
| β₁ (Lion) | ✅ ~0.9 locked (H90, H98 active at β₂=0.997) | 0.9 | Landscape asymmetric; 0.85 neutral, 0.95 badly regresses |
| wd (Lion) | ✅ 1e-3 locked (H79) | 1e-3 | — |
| slice_num | 🔬 H102 active (128 under bf16+β₂=0.997) | 96 (H73) | H66 had slice=128 negative at AdamW; Lion+bf16+β₂=0.997 may shift |
| n_layers | 🔬 H101 active (n_layers=5 under bf16) | 4 (locked by H60/H83) | Depth unexplored at bf16 speed |
| n_hidden | 🔬 H100 active (192 under bf16) | 128 | H86 wall-cut-bound at fp32; retesting with bf16 |
| mlp_ratio | 🔬 H103 active (3 under bf16) | 2 (default) | H89 wall-cut-bound at fp32; retesting with bf16 |
| FFN act | ✅ GEGLU locked (H48) | GEGLU | — |
| Normalization | ✅ LayerNorm locked (H72, H81) | LN | RMSNorm anti-compounds at slice=96 |
| Mixed precision (bf16) | 🏆 MERGED WINNER (H95) | −0.71 val / −30% s/epoch | T_max fix pending (H99) |
| torch.compile | 🔬 H96 sent back (compile+bf16 compound + T_max=21) | -27% s/epoch alone | First pass beat H78 -1.25; compound test pending |
| surf_weight | ✅ 10 locked (H54 AdamW, H91 Lion) | 10 | Insensitive under Lion sign-update |
| clip_grad_norm | ✅ 1.0 locked (H20, H56) | 1.0 | — |
| Huber δ_p | ✅ 0.25 locked (H25/H64) | 0.25 | — |
| Batch size | ✅ 4 LOCKED (H94 closed) | 4 | BS=6/8 worse; fewer-steps-per-epoch dominates at short budget |
| cond_dim | ✅ 11 locked (FiLM) | 11 | — |
| EMA averaging | ❌ Wrong regime (H65) | none | Needs oscillation, not translation |
| Mixup | ❌ Wrong inductive bias (H55) | none | — |
| DropPath | ❌ No effect (H63) | 0.0 | — |
| eta_min | ❌ H87 negative | 0 | — |

## Baseline Progression

| Val avg/mae_surf_p | Test 3-split | Event |
|---|---|---|
| 114.63 | — | R1 start (FiLM only, T_max=50 mismatch) |
| 83.81 | 80.24 | H19: T_max=15 + Huber + FiLM |
| 75.50 | 73.16 | H20: clip=1.0 |
| 71.77 | 70.62 | H27b/H32: lr=1e-3 |
| 68.19 | 65.44 | H38: wd=5e-5 |
| 66.11 | 64.45 | H37b: n_head=2 + lr=1e-3 |
| 58.63 | 56.70 | H48 GEGLU: + ffn_act=geglu |
| 57.58 | 56.46 | H60: + n_layers=4 |
| 56.75 | 54.50 | H66: + slice_num=96 |
| 42.98 | 41.55 | H73 Arm B: + optimizer=lion + lr=3e-4 (super-additive) |
| 42.30 | 40.56 | H78 Arm B: + β₂=0.995 |
| 41.22 | 39.53 | H88 Arm B: + β₂=0.997 (confirmed β₂ peak) |
| **40.51** | **39.02** | **H95 Arm A: + bf16 autocast (−30% s/epoch, 21 epochs)** |

Total merged gain: **−74.12 pts val** (64.7% reduction from 114.63 to 40.51).

## Strategic State

**Efficiency unlock is the current engine.** bf16 merged (+30% epochs). torch.compile (H96) pending — orthogonal. Together they could give ~40-45% more effective compute per wall-minute.

**Next tier: capacity under bf16.** H100 (n_hidden=192) and H101 (n_layers=5) are the direct beneficiaries. Both were wall-cut-bound at fp32. With 14-17 epochs achievable, they may reveal genuine model capacity headroom.

**Schedule fix is high-priority** (H99). T_max=15 hardcode with bf16's 21-epoch budget creates a bounce artifact. H95 Arm A's val=40.51 was achieved at ep17 with rising LR — fixing T_max=21 should give a cleaner, possibly better result.

**Optimizer axis saturated.** β₁, β₂, wd, surf_weight all confirmed locked. LR retune (H97) and β₁ at new baseline (H98) are confirmatory housekeeping, not expected to yield large wins.

**Open strategic questions:**
1. How much does the T_max fix actually help? (H99)
2. Does n_hidden=192 or n_layers=5 break through under bf16? (H100, H101)
3. Does torch.compile compound with bf16 or interfere? (H96 active)
4. Does WSD stable-plateau help vs cosine at the 15-epoch fp32 / 21-epoch bf16 budget? (H93 Arms B/C)

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `train.py`: `T_max=15` hardcoded at line 649 (not a CLI arg). H99 student will add `--T_max` arg.
- `train.py`: `n_hidden=128` hardcoded at line 621 (not a CLI arg). H100 student will add `--n_hidden` arg.
