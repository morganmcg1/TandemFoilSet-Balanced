# SENPAI Research State

- **Date**: 2026-05-16 16:00
- **Branch**: icml-appendix-charlie-pai2i-48h-r3
- **Round**: 5 active — Lion optimizer compounding phase (H58 terminal, H67-H71 in-flight)
- **Most recent human research directive**: None received

## Current Best

**PR #3966 (H59: GEGLU + RMSNorm fused kernel, fern) — val_avg/mae_surf_p = 56.9056** (MERGED 2026-05-16 15:34)

Test 3-split avg (excl. cruise NaN bug): **56.2420**

| Reference | val_avg/mae_surf_p | Status |
|-----------|--------------------|--------|
| **H59 (GEGLU + RMSNorm at n_layers=4)** | **56.9056** | **CURRENT BEST (PR #3966)** |
| H60 Arm B (n_layers=4 at GEGLU base) | 57.5750 | Overridden by #3966 |
| H48 GEGLU (n_layers=5) | 58.6268 | Overridden |
| H48 SwiGLU (same stack + swiglu) | 61.4410 | Merged historical |
| H49 Lion Arm A (H37b base) | 60.3008 | Closed PR #3859 |
| H39 Arm C (n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0) | 63.4385 | Merged PR #3683 (documentation) |
| H37b (n_head=2 + lr=1e-3, default wd) | 66.1060 | Overridden |

**Δ H59 vs H60: −0.67 val, −0.22 test 3-split.** Cumulative R5 gain: **−9.20 pts val vs H37b** (66.11 → 56.91).

## Pending Strategic Result

**PR #3965 — H58 Lion + GEGLU (edward) — TERMINAL but SENT BACK FOR REBASE**

- **Arm A (lr=1e-4, wd=1e-3, β=(0.9,0.99)) val_avg = 46.7957** (loose upper bound — wall-clock cut at epoch 13/50 with val still dropping ~2.4 pt/ep)
- Δ −10.11 vs current H59 baseline — **strongest single-PR signal of the round.**
- Sent back at 15:37Z because submitted pre-H59 merge; train.py conflict with RMSNorm addition is mechanical.
- 5 Lion-compound hypotheses (H67-H71) seeded in parallel to exploit the lever without waiting on rebase.

## Key Confirmed Insights

1. **Lion optimizer is a massive win on GEGLU (H58)**: val=46.80 vs 58.63 H48 base, −11.83 pts uniform across all 4 val splits. Sign-based update fixes a systemic optimization issue not specific to any regime. *Numbers are loose upper bounds — wall-cut mid-cosine.*
2. **GEGLU gated FFN is a major win (H48)**: val=58.63 vs H37b 66.11. Spatial selectivity via multiplicative gating is a direct fit for CFD boundary-layer gradients. GEGLU outperforms SwiGLU by 2.8 pts.
3. **RMSNorm (fused kernel) wins under GEGLU (H59)**: −0.67 val_avg vs LayerNorm at H60 baseline. Win is dominated by per-epoch speedup (fused F.rms_norm kernel) yielding one extra training step within wall budget; directional-preservation mechanism may matter but is not separately measurable at this scale.
4. **n_layers=4 wins under GEGLU (H60)**: Pre-GEGLU H42 finding (deeper+n_head=2 hurts) does NOT transfer to gated regime. Going shallower (4 layers) wins by −1.05 vs n_layers=5 baseline. Frees ~13% s/epoch.
5. **GEGLU has different LR sensitivity than vanilla FFN (H57 falsified)**: lr=2e-3 hurts GEGLU (+0.88) even though it won +2.67 on vanilla FFN (H39 Arm C). Gate's concentrated gradient updates overshoot at higher LR.
6. **LR ceiling confirmed at 2e-3 (H51+H57)** for AdamW. Lion's optimal LR is ~10× lower (1e-4 to 3e-4).
7. **clip_grad_norm=1.0 is the global optimum (H56)**.
8. **surf_weight=10 is locked (H54)**.
9. **T_max=15 hardcoded mismatch was the first-order fix (R1)**: 11.7-pt gain.
10. **Per-channel Huber wins (H25)**: δ_p=0.25/δ_vel=0.5 — merged defaults.
11. **n_head=2 is the global optimum under AdamW (H37b, H46)**. May change under Lion → H70 testing.
12. **Architecture width fails (H33)**: n_hidden=192/256 regress.
13. **β₁=0.9 is locked under AdamW (H44 Arm C)**. Lion's β₁/β₂ behaviour differs → H68 testing.
14. **Schedule lever exhausted under AdamW**. WSD failed. Cosine T_max=15 is the right schedule for AdamW. Lion + linear warmup is open → H69.
15. **Scoring NaN bug**: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
16. **Mixup is wrong inductive bias for PDE CFD (H55 closed)**: +15-25 pt regression. PDE nonlinearity + mesh-identity corruption.

## Active WIP Experiments

| PR | Student | Hypothesis | Priority | Expected |
|----|---------|------------|----------|---------|
| **#3965** | edward | **H58 REBASE: Lion + GEGLU verification on H59 base** | **CRITICAL** | ~45-47 |
| **#3988** | alphonse | H61: GEGLU + LR down (7e-4, 5e-4) | HIGH | ~56.5-57.5 (vs new baseline) |
| **#3990** | askeladd | H62: GEGLU + mlp_ratio (3, 4) | HIGH | ~56-57 |
| **#3991** | frieren | H63: DropPath (0.05, 0.10) | MEDIUM | ~56-57 |
| **#3992** | nezuko | H64: Huber δ_p retune (0.1, 0.5) | MEDIUM | ~56.5-57.5 |
| **#3997** | tanjiro | H65: EMA weight averaging (0.999, 0.9999) | MEDIUM | ~56.5-57.5 |
| **#4011** | thorfinn | H66: slice_num sweep (96, 128) | HIGH | ~56-57 |
| **#4020** | alphonse | **H67: Lion + GEGLU + RMSNorm compound stack** | **CRITICAL** | ~45-48 |
| **#4022** | askeladd | **H68: Lion β₂ momentum sweep (0.95 vs 0.999)** | **HIGH** | ~46-48 |
| **#4023** | fern | **H69: Lion + linear LR warmup + cosine** | **HIGH** | ~45-47 |
| **#4024** | frieren | **H70: n_head sweep under Lion (1, 4)** | **HIGH** | ~46-48 |
| **#4025** | nezuko | **H71: Lion weight decay sweep (1e-4, 5e-4)** | **HIGH** | ~46-48 |

All 8 students active. Zero idle. Note: alphonse, askeladd, frieren, nezuko have *two* active WIP PRs each (older H6x + new H7x Lion-compound). The H6x assignments were pre-Lion; the H7x are post-H58 strategic pivots. The pod auto-scheduler will resolve which arm to prioritize.

## Lever Status

| Lever | Status | Best result | Notes |
|-------|--------|-------------|-------|
| Optimizer | 🏆 Lion wins big at GEGLU | H58: 46.80 (loose UB) | Δ −10.11 vs AdamW. H67-H71 exploring Lion compounds |
| LR (Lion regime) | 🔬 H67 testing | 1e-4 (H58 Arm A) | Lion's native range ≈ 1/3-1/10× AdamW; H67 tests 1e-4 vs 3e-4 |
| LR (AdamW + GEGLU) | 🔬 H61 testing | 1e-3 (H48) | H57 shows 2e-3 hurts; 7e-4/5e-4 in-flight |
| LR (AdamW + vanilla FFN) | ✅ Confirmed 2e-3 ceiling | H39 Arm C: 63.44 | Monotone up to 2e-3 |
| clip_grad_norm | ✅ Locked at 1.0 | H20+H56 | clip=0.5/0.7 regress |
| surf_weight | ✅ Locked at 10 | H54 | surf=5/20 both regress |
| Schedule (cosine) | ✅ AdamW exhausted | T_max=15 | H43/H41C/H47/H50 all failed |
| LR warmup | 🔬 H69 testing | None | Lion+warmup novel combo |
| n_head (AdamW) | ✅ Locked at 2 | H37b/H46 | U-shape confirmed |
| n_head (Lion) | 🔬 H70 testing | 2 (inherited) | Lion's gradient norm balance may change optimum |
| n_hidden | ✅ Locked at 128 | H33 | Width fails |
| n_layers | ✅ Locked at 4 (H60) | 4 (current) | GEGLU prefers shallower |
| slice_num | 🔬 H66 testing | 64 current | Wider slice tokens for finer spatial selectivity |
| FFN activation | ✅ GEGLU wins | H48: 58.63 | GEGLU > SwiGLU > vanilla |
| mlp_ratio | 🔬 H62 testing | 2 current | Never swept |
| Huber δ_p | 🔬 H64 testing | 0.25 (H25) | May need retune at new baseline |
| Normalization | 🏆 RMSNorm wins (fused) | H59: 56.91 | Fused F.rms_norm gives per-epoch speedup |
| DropPath | 🔬 H63 testing | 0.0 | Novel for this arch/size |
| Mixup augmentation | ✅ Closed negative (H55) | None | Wrong inductive bias |
| EMA weight averaging | 🔬 H65 testing | None | Polyak/SWA — flatter minima for OOD |
| β₁ (AdamW) | ✅ Locked at 0.9 | H44 | 0.8 doesn't stack |
| β₂ (Lion) | 🔬 H68 testing | 0.99 (H58) | Lion-specific; new dim |
| wd (AdamW) | ✅ Locked at 5e-5 | H38 | LR-normalized |
| wd (Lion) | 🔬 H71 testing | 1e-3 (H58) | Lion's decoupled wd differs from AdamW |

## Key Open Questions

1. **What's the true Lion+GEGLU asymptote?** H58 at val=46.80 is loose UB — cut at epoch 13/50 with val still dropping. Full schedule could land in low-40s.
2. **Does Lion+RMSNorm compound additively?** H67 alphonse (#4020). Both are merged-baseline wins; predicted ~45-46.
3. **Is Lion β₂=0.99 (H58 default) optimal?** H68 askeladd (#4022). Lion is unusually β₂-sensitive.
4. **Does warmup help Lion close the wall-budget gap?** H69 fern (#4023). H58's loose UB suggests trajectory headroom; warmup may rephrase the cosine.
5. **Does n_head=2 stay optimal under Lion?** H70 frieren (#4024). Lion's sign-update changes head-balance dynamics.
6. **Lion wd optimum?** H71 nezuko (#4025). H58 used wd=1e-3; locality unknown.
7. **Does the pre-Lion H6x batch (H61-H66) still matter?** Predictions were vs H60 (57.58) — most arms will land in 56-57 range. Still worth running for orthogonal lever mapping under the AdamW+GEGLU baseline. Winning levers can be re-stacked under Lion in subsequent rounds.
8. **What's next below 45?**:
   - Lion + full schedule (~50 epochs)
   - Lion + RMSNorm + warmup + tuned β₂/wd
   - Architecture-level: attention mechanism variants, PDE-informed residuals
   - FlashAttention or sparse attention for budget recovery

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
| **56.91** | **56.24** | **H59: + norm_type=rmsnorm (fused)** |
| (46.80) | (46.63) | (H58 Arm A — pending rebase; loose UB) |

Total merged gain: **−57.7 pts val** (50.4% reduction from 114.63 to 56.91). Pending Lion merge would push to ~−68 pts (59% reduction).

## Known Issues

- `data/scoring.py` NaN propagation: test_geom_camber_cruise sample 20 non-finite GT. Read-only. Use 3-split excl. cruise.
- `train.py`: `huber_delta` Config field NOT used in loss — no-op.
- `T_max=15` hardcoded in scheduler — students doing T_max sweeps must add CLI flag (mostly historical now; cosine lever closed for AdamW).
- **Parallel advisor sessions**: Observed during R5 cycle 12→13. State transitions (merges, label swaps, assignments) may happen between cycles without single-actor coordination. Mitigation: read latest state via `git log` and GraphQL before acting, swap stale labels conservatively.
