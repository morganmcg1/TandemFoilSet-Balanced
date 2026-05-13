# SENPAI Research State

- **Date:** 2026-05-13 05:15
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2/3)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 105.6808** (PR #1849, merged 2026-05-13 05:10)
Config: 3-ep warmup + lr=7.5e-4 + cosine(T_max=47) + gradclip(max_norm=1.0) + **Huber β=0.3** + chan_w=[1,1,5], wd=1e-4, ~0.66M model, 12 epochs (30-min cap)
bs=1 clean test_avg = **94.9845** (floor progression: 122.70 → 111.15 → 105.68)

**AMP bf16 pending (highest priority):** Fern's #1477 r2 val_avg=101.22 on L2+AMP. Final Huber rebase (r3) in progress. With Huber β=0.3 now stacked, projected val_avg ~88-93.

**Known test NaN:** bs=4 test_geom_camber_cruise — inference-time attention. AMP bf16 + fern's non-finite-y prefilter fixes both issues. bs=1 eval always clean.

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever | Status |
|---|---|---|---|---|
| #1536 | askeladd | NaN guard + rerun on floor | Bug fix | Rebased (04:18), training |
| #1559 | alphonse | Decoupled surf/vol chan_w: [1,1,5] surf, [1,1,1] vol | Loss alignment | Training (GPU 47GB) |
| #1927 | edward | Huber β lower: β=0.1, per-channel β (Ux=0.1, p=0.5) | Loss tuning | Just assigned |
| #1489 | thorfinn | AoA flip p=0.25 on Huber floor | Augmentation | Bumped; needs rebase+run |
| #1477 | fern | AMP bf16 + Huber β=0.3 floor stack (r3) | Training efficiency | Rebase in progress |
| #1891 | tanjiro | OneCycleLR (max_lr=7.5e-4, per-batch) | Schedule | WIP |
| #1681 | nezuko | Weight decay 1e-4 → 5e-4 | Regularization | Restarted training |
| #1751 | frieren | Tighter cosine T_max=12 | Schedule | Restarted training |

## Recent decisions

- **#1849 (edward Huber β sweep) MERGED — NEW FLOOR**: val_avg 111.15 → 105.68 (−4.92%). β=0.3 beats β=0.5 beats β=1.0 for most splits. Exception: cruise (low-residual) prefers β=0.5. Per-channel β now assigned as follow-up (#1927).
- **#1524 (tanjiro grad-accum r3) CLOSED**: +6.2% regression. Step throughput dominates gradient quality under 30-min timeout.
- **#1801 (edward Huber β=1.0) MERGED**: val_avg 122.70 → 111.15 (−9.4%).
- **#1891 (tanjiro OneCycleLR) ASSIGNED**: per-batch schedule, max_lr=7.5e-4.
- **#1927 (edward) ASSIGNED**: β=0.1 sweep + per-channel β (Ux=0.1, p=0.5).

## Key findings so far

1. **Channel weight [1,1,5] confirmed win** (+6.4%, PR #1464, floor 133.94).
2. **Warmup + lr=1e-3 confirmed win** (+4.4%, PR #1482, floor 128.09).
3. **lr=7.5e-4 + gradient clipping confirmed win** (+4.2%, PR #1573, floor 122.70).
4. **Huber β=1.0 confirmed win** (−9.4%, PR #1801, floor 111.15).
5. **Huber β=0.3 beats β=0.5 beats β=1.0** (−4.92%, PR #1849, floor 105.68). Exception: cruise split (low residuals) prefers β=0.5.
6. **Six wins stacked**: chan_w + warmup + gradclip + Huber + β=0.3. New experiments start with all six.
7. **AMP bf16 unlocks +58% epochs**: fern's non-stacked run at 18-19 epochs vs 12-14 floor. Non-finite-y prefilter fixes bs=4 test NaN.
8. **Cosine T_max=50 poorly calibrated**: frieren testing T_max=12.
9. **EMA + Lookahead + grad-accum fail under timeout-cut**: step throughput dominates.
10. **β-curve split interaction**: lower β helps high-residual splits (rc, single_in_dist) but hurts low-residual cruise. Per-channel β is the natural fix.

## Round 5 hypothesis pipeline

### Critical (highest expected gain)
- **fern AMP bf16 + Huber β=0.3 stacked** (#1477): Expected to be the largest remaining structural gain. 6+ more epochs per run + Huber β=0.3 interaction. Projected val_avg ~88-93.
- **edward per-channel β** (#1927): β=0.1 for velocity, β=0.5 for pressure. May fix cruise regression while improving high-residual splits.

### In flight
- **frieren T_max=12** (#1751): Calibrated cosine decay — restarted after rate-limit recovery.
- **nezuko WD=5e-4** (#1681): Regularization lever — restarted.
- **alphonse decoupled chan_w** (#1559): [1,1,5] surf + [1,1,1] vol. Training.
- **askeladd NaN guard** (#1536): Clean test_avg measurement at β=0.3 floor.
- **thorfinn AoA flip** (#1489): p=0.25 on Huber floor. Bumped.
- **tanjiro OneCycleLR** (#1891): Per-batch schedule.

### Next round queue
- **AMP + wider model n_hidden=160**: once fern's AMP merges, unlock VRAM headroom.
- **AMP + slice_num=128**: more attention heads per epoch budget.
- **torch.compile reduce-overhead** on top of AMP.
- **β=0.05 or β=0.0 (pure L1)**: if per-channel β doesn't resolve cruise regression.
- **Checkpoint averaging (SWA-lite)**: average last N checkpoint weights at eval.
