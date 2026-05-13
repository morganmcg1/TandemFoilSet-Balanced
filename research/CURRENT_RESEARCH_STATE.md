# SENPAI Research State

- **Date:** 2026-05-13 03:10
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2/3)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 111.1516** (PR #1801, merged 2026-05-13 03:05)
Config: 3-ep warmup + lr=7.5e-4 + cosine(T_max=47) + gradclip(max_norm=1.0) + Huber β=1.0 + chan_w=[1,1,5], wd=1e-4, ~0.66M model, 13-14 epochs (30-min cap)
bs=1 clean test_avg = **99.0565** (first sub-100 result; all 4 splits finite)

**AMP bf16 pending (highest priority):** Fern's #1477 had val_avg=94.55 WITHOUT chan_w/warmup stack. Expected stacked result well below 100 when she completes rebase+rerun.

**Known test NaN:** bs=4 test_geom_camber_cruise — inference-time attention numerics. AMP bf16 may fix. bs=1 eval always clean.

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever | Status |
|---|---|---|---|---|
| #1536 | askeladd | NaN guard + rerun on floor | Bug fix | Rebased/CLEAN, awaiting rerun |
| #1559 | alphonse | Decoupled surf/vol chan_w: [1,1,5] surf, [1,1,1] vol | Loss alignment | Training (GPU 70GB seen) |
| #1524 | tanjiro | grad-accum=4 at lr=7.5e-4 (rebased) | Batch size | Rebased/CLEAN, awaiting rerun |
| #1489 | thorfinn | chan_w + per-sample AoA flip p=0.25 | Augmentation | WIP |
| #1477 | fern | AMP bf16 + floor stack rebase | Training efficiency | Rebased, training (GPU 66GB) |
| #1849 | edward | Huber β sweep: β=0.5 and β=0.3 | Loss tuning | Just assigned |
| #1681 | nezuko | Weight decay 1e-4 → 5e-4 | Regularization | Training (GPU 43GB seen) |
| #1751 | frieren | Tighter cosine T_max=12 | Schedule | Training (GPU 42GB seen) |

## Recent decisions

- **#1801 (edward Huber β=1.0) MERGED — NEW FLOOR**: val_avg 122.70 → 111.15 (−9.4%), bs=1 test 110.25 → 99.06 (−10.2%). First sub-100 test result. L2→Huber is significant. Largest gain: single_in_dist −15.9%.
- **#1708 (edward Lookahead) CLOSED**: val_avg 143.62 (+17%). Same regime mismatch as EMA.
- **#1573 (frieren) MERGED**: val_avg 128.09 → 122.70 (−4.2%). lr=7.5e-4 + gradclip.
- **#1477 (fern) SENT BACK**: val_avg=94.55 impressive but config missing chan_w+warmup. Rebase+rerun required.

## Key findings so far

1. **Channel weight [1,1,5] confirmed win** (+6.4%, PR #1464, floor 133.94).
2. **Warmup + lr=1e-3 confirmed win** (+4.4%, PR #1482, floor 128.09).
3. **lr=7.5e-4 + gradient clipping confirmed win** (+4.2%, PR #1573, floor 122.70).
4. **Huber β=1.0 confirmed win** (−9.4%, PR #1801, floor 111.15). L2/L1 train-metric mismatch was hurting OOD splits — Huber fixes this. Stacked in advisor train.py.
5. **Five wins stacked**: chan_w + warmup + gradclip + Huber. New experiments start with all five.
6. **AMP bf16 unlocks epoch budget**: fern's non-stacked run got 19 epochs (vs 12-14) at 33GB VRAM vs 42GB. Epoch gain is the primary driver.
7. **AMP bf16 fixes bs=4 inference NaN**: fern's bs=4 test_avg=84.64 was clean — no fallback needed.
8. **EMA + Lookahead fail in rapid-descent regime**: weight-averaging operators harmful here.
9. **Cosine T_max=50 poorly calibrated**: only ~25% of decay in 12-14 epoch budget.
10. **Grad-accum=4 at old floor**: 2.4% gain at eff_bs=16. Stacking in progress at new floor.

## Round 4/5 hypothesis pipeline

### Critical (highest expected gain)
- **fern AMP bf16 stacked** (#1477): expected to be largest remaining gain. 7+ more epochs per run.
- **edward Huber β sweep** (#1849): β=0.5 and β=0.3 vs merged β=1.0. Quick win or confirms β=1.0 optimal.

### In flight
- **askeladd NaN guard** (#1536): clean test_avg measurement at new floor.
- **alphonse decoupled chan_w** (#1559): may find better vol-channel weighting.
- **tanjiro grad-accum=4** (#1524): eff_bs=16 + Huber stacked.
- **thorfinn AoA flip** (#1489): OOD aug.
- **nezuko WD=5e-4** (#1681): regularization.
- **frieren T_max=12** (#1751): calibrated cosine decay.

### Next round queue
- **AMP + wider model n_hidden=192**: if fern's AMP merges, unlock this.
- **AMP + slice_num=128**: more attention heads per epoch budget.
- **torch.compile reduce-overhead** on top of AMP.
- **Sort-by-size sampler**: reduce pad_collate waste.
- **Huber β for vol path only** vs Huber everywhere: ablation.
