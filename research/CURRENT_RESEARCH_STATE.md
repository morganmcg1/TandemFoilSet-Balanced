# SENPAI Research State

- **Date:** 2026-05-13 08:05
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2/3)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 85.9338** (PR #1751, merged 2026-05-13 07:10)
Config: 3-ep warmup + lr=7.5e-4 + cosine(**T_max=12**) + gradclip(max_norm=1.0) + Huber β=0.3 + chan_w=[1,1,5], wd=1e-4, ~0.66M model, 14 epochs (30-min cap)
bs=1 clean test_avg = **77.6488** (floor progression: 122.70 → 111.15 → 105.68 → **85.93**)

**AMP bf16 pending (highest priority):** Fern's #1477 r2 val_avg=101.22 on L2+AMP — needs rebase onto T_max=12 floor (r3). With Huber β=0.3 + T_max=12 now stacked, projected val_avg ~65-75.

**Known test NaN:** bs=4 test_geom_camber_cruise — inference-time attention. AMP bf16 + fern's non-finite-y prefilter fixes both issues. bs=1 eval always clean.

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever | Status |
|---|---|---|---|---|
| #1536 | askeladd | NaN guard + rerun on floor | Bug fix | Needs rebase onto T_max=12 |
| #1947 | alphonse | chan_w sweep under β=0.3: [1,1,3] vs [1,1,7] | Loss tuning | Needs rebase onto T_max=12 |
| #1927 | edward | Huber β lower: β=0.1, per-channel β (Ux=0.1, p=0.5) | Loss tuning | Needs rebase onto T_max=12 |
| #1489 | thorfinn | AoA flip p=0.25 on Huber floor | Augmentation | Needs rebase onto T_max=12 |
| #1477 | fern | AMP bf16 + Huber β=0.3 + T_max=12 floor stack (r3) | Training efficiency | Needs rebase onto T_max=12 |
| #2061 | tanjiro | mlp_ratio=4 (conventional transformer capacity) | Architecture | Just assigned |
| #1681 | nezuko | Weight decay 1e-4 → 5e-4 | Regularization | Needs rebase onto T_max=12 |
| #2019 | frieren | Cosine completion: T_max=11 (complete) vs eta_min=1e-7 | Schedule | Just assigned |

## Recent decisions

- **#1751 (frieren T_max=12) MERGED — NEW FLOOR**: val_avg 105.68 → 85.93 (−18.7%). The T_max=47 schedule was starved: at epoch 14, LR was still near peak (~6.9e-4). T_max=12 gets 92% cosine decay by epoch 14 (LR 5.1e-5). Gain entirely from unlocking the low-LR refinement phase. All 7 active WIPs need rebase.
- **#1559 (alphonse decoupled chan_w) CLOSED**: +9.8% mean regression. Volume term acts as joint regularizer.
- **#1947 (alphonse chan_w sweep) ASSIGNED**: [1,1,3] vs [1,1,7] under Huber β=0.3 — needs rebase onto T_max=12.
- **#1849 (edward Huber β sweep) MERGED**: val_avg 111.15 → 105.68 (−4.92%). β=0.3 best overall; cruise prefers β=0.5. Per-channel β assigned as follow-up (#1927).
- **#1524 (tanjiro grad-accum r3) CLOSED**: +6.2% regression.
- **#1891 (tanjiro OneCycleLR) CLOSED**: +3.32% regression. OneCycle structurally mismatched to 14-epoch budget — too much high-LR time, over-anneals at end. Cosine T_max=12 is strictly better for this wall-clock regime.
- **#2061 (tanjiro mlp_ratio=4) ASSIGNED**: conventional transformer choice (mlp_ratio=2 is conservative). Larger MLP per block, ~0.85M params. T_max adjusted for slower per-epoch compute.
- **#1927 (edward) ASSIGNED**: β=0.1 sweep + per-channel β; needs rebase onto T_max=12.

## Key findings so far

1. **Channel weight [1,1,5] confirmed win** (+6.4%, PR #1464, floor 133.94).
2. **Warmup + lr=1e-3 confirmed win** (+4.4%, PR #1482, floor 128.09).
3. **lr=7.5e-4 + gradient clipping confirmed win** (+4.2%, PR #1573, floor 122.70).
4. **Huber β=1.0 confirmed win** (−9.4%, PR #1801, floor 111.15).
5. **Huber β=0.3 beats β=0.5 beats β=1.0** (−4.92%, PR #1849, floor 105.68). Exception: cruise split (low residuals) prefers β=0.5.
6. **T_max=12 cosine alignment confirmed win** (−18.7%, PR #1751, floor 85.93). Schedule calibration to epoch budget is the dominant lever — prior wins were measured on a schedule-starved baseline.
7. **Seven wins stacked**: chan_w + warmup + gradclip + Huber + β=0.3 + T_max=12. New experiments start with all seven.
8. **AMP bf16 unlocks +58% epochs**: fern's non-stacked run at 18-19 epochs vs 12-14 floor. Non-finite-y prefilter fixes bs=4 test NaN.
9. **EMA + Lookahead + grad-accum fail under timeout-cut**: step throughput dominates.
10. **β-curve split interaction**: lower β helps high-residual splits (rc, single_in_dist) but hurts low-residual cruise. Per-channel β is the natural fix.
11. **Schedule calibration > schedule choice**: T_max aligned to actual epoch budget delivers qualitatively different (refinement-phase) training that T_max=47 locked out entirely.

## Round 6 hypothesis pipeline

### Critical (highest expected gain)
- **fern AMP bf16 + full stack (r3)** (#1477): Rebase onto T_max=12 floor. 6+ more epochs per run + Huber β=0.3 + T_max=12 interaction. Projected val_avg ~60-70.
- **edward per-channel β** (#1927): β=0.1 for velocity, β=0.5 for pressure. May fix cruise regression while improving high-residual splits. Needs rebase.

### In flight (all need rebase onto T_max=12 floor)
- **alphonse chan_w sweep** (#1947): [1,1,3] vs [1,1,7] under β=0.3 + T_max=12.
- **tanjiro OneCycleLR** (#1891): Per-batch schedule at T_max=12 floor.
- **askeladd NaN guard** (#1536): Clean test_avg measurement at full floor.
- **thorfinn AoA flip** (#1489): p=0.25 augmentation at full floor.
- **nezuko WD=5e-4** (#1681): Regularization lever at full floor.
- **frieren** (#2019): T_max=11 (epochs=14, cosine completes) vs T_max=12 + eta_min=1e-7 (extends tail). Follow-up to the biggest win on this branch.

### Next round queue
- **AMP + wider model n_hidden=160**: once fern's AMP merges, unlock VRAM headroom.
- **AMP + slice_num=128**: more attention heads per epoch budget.
- **torch.compile reduce-overhead** on top of AMP.
- **β=0.05 or β=0.0 (pure L1)**: if per-channel β doesn't resolve cruise regression.
- **Checkpoint averaging (SWA-lite)**: average last N checkpoint weights at eval.
- **T_max fine-tuning**: T_max=11 vs T_max=13 vs T_max=15 to find optimal half-cycle length.
- **eta_min sweep**: smaller eta_min (1e-7) to extend very late decay phase.
