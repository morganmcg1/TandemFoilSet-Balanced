# SENPAI Research State

- **Date**: 2026-05-16 (Loop 17)
- **Track**: `charlie-pai2i-24h-r1` on advisor branch `icml-appendix-charlie-pai2i-24h-r1`
- **Latest direction from human researcher team**: none received
- **Per-student GPU budget**: 1 × 96GB, 30-min wall-clock per training run

## Current research focus

**Pure L1 (#3798 frieren) is the BIG result of Loop 17**: val_avg=**86.66** (−8.75% vs SmoothL1 94.97), test_avg=**77.21** (−9.20%). Uniform improvement across all 8 val+test splits. Sent back for (a) seed-pin confirmation seed and (b) flipping `use_l1` default to True (so post-merge baseline matches the active recipe per #3127 convention).

**Active advisor baseline (still SmoothL1 until #3798 merges)**: PR #3127 — val_avg=94.97, test_avg=85.04. Expected next baseline after #3798 confirmation: val_avg≈86.66, test_avg≈77.21.

Active advisor config: `n_hidden=128, n_head=4, n_layers=5, slice_num=64, mlp_ratio=2` + EMA decay=0.999 + surf_weight=25.0 + **bf16 autocast** + **T_max=18, epochs=18** + **SmoothL1 loss (beta=1.0)** (Pure L1 in flight via #3798).

**Underfit signature persists**: loss still descending at last cosine-annealed epoch — even on the Pure L1 winning arm. Highest-EV remaining levers attack this directly via more effective epochs (#3802) or schedule shape (#3864 lr_min=1e-5).

## Loop 17 systemic findings (NEW)

1. **L1's gradient discontinuity at r=0 is likely the load-bearing mechanism, not smaller β**: SmoothL1 progression {β=1.0: 94.97, β=0.5: 91.57, β=0.25: 91.81, β=0.0: 86.66} is non-monotonic with a 4.91-unit gap from β=0.25 to Pure L1 — much larger than the inter-step gap inside the β sweep. **#3861 Charbonnier directly tests this**: Charbonnier `sqrt(eps² + r²) - eps` has L1's gradient saturation for large r but smooth gradient at r=0. If Charbonnier matches Pure L1 → smoothness doesn't matter; if Charbonnier underperforms → discontinuity is essential.

2. **Per-channel optima diverge** (#3763 finding): β=0.25 uniformly better on Ux/Uy; β=0.5 better on pressure on 2 of 4 splits. Suggests per-channel loss formulation is a future axis. Defer until Pure L1 stabilizes as baseline.

3. **Schedule completion still binds on width** (#3804 closure): n_hidden=160 regressed via cosine under-completion (17/18 epochs at 111s/epoch). Reinforces #3478 "schedule completion > raw capacity" under the loss regime that succeeds.

4. **Global gradient-budget reallocation hurts under L1-tail losses** (#3800 closure): surf_p 4× weighting regressed uniformly. Per-channel reweighting needs to coexist with L1 mechanics, not against them.

## PRs in-flight (all 8 students active, zero idle GPUs)

| PR | Student | Axis | One-line summary | Status |
|---|---|---|---|---|
| #3676 | edward | Architecture (slice) | slice_num=48 + 3-seed + 21-ep on SmoothL1 | wip (sent back L16) |
| #3798 | frieren | Loss formulation | **Pure L1 −8.75% win** — seed-pin + use_l1 default flip pending | wip (sent back L17) |
| #3802 | alphonse | Throughput | torch.compile determinism + 23-ep budget-soak on SmoothL1 | wip |
| #3861 | askeladd | Loss formulation | Charbonnier `sqrt(eps² + r²) − eps`; eps∈{1e-3, 1e-2} | dispatched Loop 17 |
| #3863 | fern | Optimizer stability | Gradient norm clipping max_norm=1.0 + per-epoch telemetry | dispatched Loop 17 |
| #3864 | nezuko | Schedule shape | CosineAnnealingLR eta_min=1e-5 | dispatched Loop 17 |
| #3865 | tanjiro | Per-block lr | LLRD γ∈{0.9, 0.75} — follow-up to #3719 global-lr closure | dispatched Loop 17 |
| #3866 | thorfinn | Weight averaging | SWA uniform tail-3 averaging (complementary to EMA) | dispatched Loop 17 |

## Recent decisions

- **Loop 17: #3798 SENT BACK** Pure L1 −8.75% win, single-seed risk + default flag needed. Asked frieren for 1 confirmation seed (seed=42) and to flip `use_l1: bool = False` → `True`.
- **Loop 17: 3 closures of dominated PRs**:
  - #3763 askeladd β sweep: β=0.5 and β=0.25 beat SmoothL1 but decisively dominated by Pure L1.
  - #3800 fern surf_p 4×: uniform regression across all 8 splits.
  - #3804 nezuko n_hidden=160: regression + cosine under-completion.
- **Loop 17: 2 stale closures** with fresh restarts:
  - #3588 tanjiro Lookahead (5+ days stale across 2 baseline shifts) → reassigned to LLRD #3865.
  - #3589 thorfinn SWA (5+ days stale) → reassigned to SWA tail fresh restart #3866.
- **Loop 17: 5 new dispatches** — askeladd Charbonnier, fern grad-clip, nezuko lr_min, tanjiro LLRD, thorfinn SWA.
- **Loop 16: #3127 MERGED** SmoothL1 −15.0% win. New best 94.97/85.04 (now the about-to-be-superseded baseline).

## Systemic findings (load-bearing context)

1. **L1 gradient discontinuity at r=0 is likely the load-bearing mechanism** for Pure L1's −8.75% win over SmoothL1. Charbonnier (#3861) directly tests this.
2. **Per-channel optima diverge by output channel** (#3763): pressure prefers β=0.5; Ux/Uy prefer β=0.25 or Pure L1.
3. **camber_rc-as-discriminator REDUCED by SmoothL1** (and further reduced by Pure L1): val_avg gap from +18.5% (MSE) → +9.3% (SmoothL1) → predicted ~+9% (Pure L1) after #3798 lands.
4. **Underfit baseline persists across loss formulations**: loss still descending at final cosine-annealed epoch even under Pure L1. Suggests more epochs (#3802) or schedule floor (#3864) remain high-EV.
5. **Schedule completion > raw capacity** at the cost of widening (#3804 closure reinforces #3478).
6. **A single global lr cannot simultaneously optimize all 4 val splits** (#3719 closure): per-split convergence profiles diverge. LLRD (#3865) is the natural per-block follow-up.
7. **LayerNorm bias-drop alone is INSUFFICIENT** for RMSNorm matched-epoch gain (closes ~14% of gap; 86% comes from mean-subtraction). Future norm work should target mean-subtraction removal, not bias removal.
8. **Per-domain weighting at narrow+bf16 ineffective** (#3585): mechanism mostly absorbed by SmoothL1's L1 tail at camber_rc; cruise budget squeeze dominates.
9. **Coord-jitter at σ=0.01 ineffective** (#3555): cost asymmetry (cruise damage > rc help); slice projection already provides soft spatial pooling.
10. **Per-channel surf_p 4× reweighting actively HURTS** under SmoothL1/L1 tail (#3800 uniform regression).
11. **torch.compile throughput win is real** (22% speedup, no graph breaks) from prior work — being re-tested for determinism on SmoothL1 in #3802.
12. **Slice mechanism is capacity-bottlenecked** (#3500): being re-tested by #3676 (slice_num=48 → 3-seed).
13. **Gated-FFN (SwiGLU) axis exhausted at narrow trunk** (prior work): MLP is too small a FLOPs fraction.
14. **Domain curriculum has +50% wall-time overhead** (DataLoader rebuild structural).
15. **Low-dimensional per-sample conditioners ruled out** (#3287 FiLM).
16. **Seed variance at narrow+bf16 is ~6 units** (#3676 finding): train.py does not seed torch/numpy/random; #3798 confirmation seed pending.

## Priority candidates if students free up next

1. **Charbonnier eps sweep continuation** (only if #3861 confirms smoothness-at-zero matters) — try eps=5e-3, eps=2e-3 to find the optimal blend.
2. **Per-channel loss formulation** (e.g., L1 for Ux/Uy + SmoothL1 β=0.5 for pressure) — the natural follow-up to #3763's per-channel divergence finding. Defer until Pure L1 stabilizes.
3. **lr_min=5e-6 / 5e-5 sweep** if #3864 wins to find the optimum floor.
4. **slice_num=96** (only if #3676 slice_num=48 + multi-seed shows slice_num=64 is over-parameterized).
5. **RMSNorm proper retest** (target mean-subtraction-removal, NOT bias-drop) — possibly with custom triton kernel.
6. **More epochs at lr=5e-4 (epochs=24 + slice_num=48)** — direct underfit test combining schedule extension with slice savings.
7. **LLRD γ sweep continuation** if #3865 wins — try γ=0.5, γ=0.85 to find optimum.
8. **Volume loss per-channel weighting** — vol_p analog of #3800 (vol residuals are unweighted) — but LOW EV given #3800's uniform regression suggests reweighting mechanisms broadly conflict with L1 tail.
9. **Seed-pinning bug fix as a systemic improvement** — add `--seed` flag + `torch.manual_seed` at startup to reduce the ~6-unit seed variance ceiling.

**Pending baseline update after #3798 merges**: new best val_avg<86.66, test_avg<77.21. All in-flight PRs (#3861, #3863, #3864, #3865, #3866, #3676, #3802) will need to be re-evaluated against the Pure L1 baseline.

Full idea list: `research/RESEARCH_IDEAS_2026-05-15_round1.md`
