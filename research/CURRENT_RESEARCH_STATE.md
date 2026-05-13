# SENPAI Research State

- **Date:** 2026-05-13 04:20
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2/3)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 111.1516** (PR #1801, merged 2026-05-13 03:05)
Config: 3-ep warmup + lr=7.5e-4 + cosine(T_max=47) + gradclip(max_norm=1.0) + Huber β=1.0 + chan_w=[1,1,5], wd=1e-4, ~0.66M model, 13-14 epochs (30-min cap)
bs=1 clean test_avg = **99.0565** (first sub-100 result; all 4 splits finite)

**AMP bf16 pending (highest priority):** Fern's #1477 r2 seeds had val_avg 104.71/101.22 (mean 102.97) on L2+AMP+floor. Huber merged during her r2 rebase — r3 rebase onto Huber floor in progress. Expected stacked result: val_avg ~91-95 (sub-floor 111.15 guaranteed based on both seeds).

**Known test NaN:** bs=4 test_geom_camber_cruise fixed by fern's non-finite-y prefilter. Once #1477 merges, bs=4 test_avg will be clean across all splits.

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever | Status |
|---|---|---|---|---|
| #1536 | askeladd | NaN guard + rerun on floor | Bug fix | Rebased, awaiting rerun |
| #1559 | alphonse | Decoupled surf/vol chan_w: [1,1,5] surf, [1,1,1] vol | Loss alignment | Training (GPU 65GB) |
| #1891 | tanjiro | OneCycleLR (max_lr=7.5e-4, pct_start=0.15, per-batch) | Schedule | Just assigned |
| #1489 | thorfinn | chan_w + per-sample AoA flip p=0.25 on Huber floor | Augmentation | Bumped; needs rebase+run |
| #1477 | fern | AMP bf16 + Huber floor stack (r3 rebase) | Training efficiency | Final rebase in progress |
| #1849 | edward | Huber β sweep: β=0.5 and β=0.3 | Loss tuning | WIP |
| #1681 | nezuko | Weight decay 1e-4 → 5e-4 | Regularization | Training (GPU 66GB, 100%) |
| #1751 | frieren | Tighter cosine T_max=12 | Schedule | Training (GPU 67GB, 99%) |

## Recent decisions

- **#1524 (tanjiro grad-accum r3) CLOSED**: val_avg=118.09 (+6.2% regression). Under timeout-cut, accum=4 → 4× fewer optimizer steps (1313 vs 5250). Step throughput dominates over gradient quality.
- **#1477 (fern AMP r2) SENT BACK**: val_avg seed-b=101.22 (−16.1% vs old floor), but branch rebased onto pre-Huber HEAD — needs ONE more rebase onto d0b582f + one confirm seed.
- **#1801 (edward Huber β=1.0) MERGED — NEW FLOOR**: val_avg 122.70 → 111.15 (−9.4%), bs=1 test 99.06 (first sub-100).
- **#1708 (edward Lookahead) CLOSED**: regime mismatch (+17% regression).
- **#1891 (tanjiro OneCycleLR) ASSIGNED**: per-batch schedule with max_lr=7.5e-4, pct_start=0.15, final_div_factor=1e4.

## Key findings so far

1. **Channel weight [1,1,5] confirmed win** (+6.4%, PR #1464, floor 133.94).
2. **Warmup + lr=1e-3 confirmed win** (+4.4%, PR #1482, floor 128.09).
3. **lr=7.5e-4 + gradient clipping confirmed win** (+4.2%, PR #1573, floor 122.70).
4. **Huber β=1.0 confirmed win** (−9.4%, PR #1801, floor 111.15). L2/L1 train-metric mismatch was hurting OOD splits — Huber fixes this.
5. **Five wins stacked**: chan_w + warmup + gradclip + Huber. New experiments start with all five.
6. **AMP bf16 unlocks epoch budget**: fern's non-stacked r2 run got 18-19 epochs (vs 12-14) at 33GB VRAM vs 42GB. +58% epochs, −37% epoch time, −22% VRAM.
7. **AMP bf16 fixes bs=4 inference NaN**: fern's non-finite-y prefilter eliminates test_geom_camber_cruise NaN.
8. **EMA + Lookahead + grad-accum fail in rapid-descent regime**: weight-averaging and step-reducing operators all harmful under 30-min timeout.
9. **Cosine T_max=50 poorly calibrated**: only ~25% of decay at 13-14 epoch budget. Frieren testing T_max=12.
10. **Grad-accum dead end**: 4× fewer optimizer steps costs more than cleaner-gradient gain at current floor.

## Round 4/5 hypothesis pipeline

### Critical (highest expected gain)
- **fern AMP bf16 stacked** (#1477): final rebase in progress. Expected to be the dominant single-PR gain — val_avg projected ~91-95.
- **edward Huber β sweep** (#1849): β=0.5 and β=0.3. Quick win or confirms β=1.0 optimal.

### In flight
- **frieren T_max=12** (#1751): calibrated cosine decay — 25% of budget in annealing → 100% of budget.
- **nezuko WD=5e-4** (#1681): regularization lever.
- **alphonse decoupled chan_w** (#1559): [1,1,5] surf + [1,1,1] vol.
- **askeladd NaN guard** (#1536): clean test_avg measurement at new floor.
- **thorfinn AoA flip** (#1489): AoA flip p=0.25 on Huber floor — bumped to rebase+rerun.
- **tanjiro OneCycleLR** (#1891): per-batch stepping, max_lr=7.5e-4, pct_start=0.15, final_div=1e4.

### Next round queue
- **AMP + wider model n_hidden=160-192**: once fern's AMP merges, unlock with freed VRAM headroom.
- **AMP + slice_num=128**: more attention heads per epoch budget.
- **torch.compile reduce-overhead** on top of AMP.
- **Checkpoint averaging (SWA-lite)**: average last N checkpoint weights at eval time.
- **Huber β for vol path only** vs Huber everywhere: ablation.
- **Lion optimizer**: sign-only updates, potentially faster convergence at tight compute budgets.
