# SENPAI Research State

- **Date:** 2026-05-13 02:10
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2/3)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 122.7043** (PR #1573, merged 2026-05-13)
Config: 3-ep warmup + lr=7.5e-4 + cosine(T_max=47) + gradclip(max_norm=1.0), bs=4, chan_w=[1,1,5], wd=1e-4, ~0.66M model, 12 epochs (timeout-cut)
bs=1 clean test_avg = **110.2527** (all 4 splits finite)

**AMP bf16 teaser:** Fern's PR #1477 achieved val_avg=**94.55** (−23%) BUT on a config missing chan_w+warmup stack. Sent back for rebase onto floor. Expect fully stacked result ≪ 100 once rerun completes — this will be the dominant next gain.

**Known test NaN:** bs=4 test_geom_camber_cruise — deterministic inference-time attention numerics. bs=1 eval is clean. AMP bf16 may fix inference NaN (fern's run showed clean bs=4 test_avg=84.64).

**eval_bs1.py** now in advisor branch — use for clean bs=1 test evaluation.

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever | Status |
|---|---|---|---|---|
| #1536 | askeladd | NaN guard + lr=1e-3 rerun on current floor | Bug fix / measurement | Training (GPU 50% active) |
| #1559 | alphonse | Decoupled surf/vol chan_w: [1,1,5] surf, [1,1,1] vol | Loss alignment | Training (train.py modified) |
| #1524 | tanjiro | grad-accum=4 at lr=7.5e-4 (rebased onto floor) | Stacking / opt | Rebased+CLEAN, awaiting run |
| #1489 | thorfinn | chan_w + per-sample AoA flip p=0.25 | Augmentation | WIP (training?) |
| #1477 | fern | AMP bf16 + floor stack + bug fix (REBASE NEEDED) | Training efficiency | Sent-back; awaiting rebase+run |
| #1801 | edward | Huber/SmoothL1 loss for pressure channel β=1.0 | Loss function | Just assigned |
| #1681 | nezuko | Weight decay 1e-4 → 5e-4 | Regularization | WIP (training?) |
| #1751 | frieren | Tighter cosine: --epochs 15 → T_max=12 | Schedule | WIP (training?) |

## Recent decisions

- **#1477 (fern) SENT BACK 2026-05-13 02:05**: val_avg=94.55 (−23%) but config reverts chan_w+warmup. Rebase onto floor, keep AMP bf16 + bug fix, use lr=7.5e-4.
- **#1708 (edward Lookahead) CLOSED 2026-05-13 02:00**: val_avg=143.62 (+17% vs floor). Same rapid-descent regime mismatch as EMA #1603. Edward reassigned to Huber loss (#1801).
- **#1573 (frieren) MERGED — NEW FLOOR**: val_avg 128.09 → 122.70 (−4.2%), bs=1 test 110.25 (−6.1%).
- **#1524 (tanjiro) rebased**: now MERGEABLE/CLEAN, awaiting new rerun at lr=7.5e-4.
- **Rate limit recovered 2026-05-13 ~01:50Z**: all 8 pods now polling correctly.

## Key findings so far

1. **Channel weight [1,1,5] is a confirmed win** (+6.4%, PR #1464, floor 133.94).
2. **Warmup + lr=1e-3 is a confirmed win** (+4.4%, PR #1482, floor 128.09).
3. **lr=7.5e-4 + gradient clipping is a confirmed win** (+4.2%, PR #1573, floor 122.70).
4. **Three wins stacked in advisor train.py**: chan_w + warmup + gradclip. New experiments start with all three.
5. **chan_w response curve non-monotonic** — p=10 14% worse, optimal ≈5.
6. **AMP bf16 is potentially massive**: Fern's run got 19 epochs vs floor's 12 in same wall clock (VRAM 32→42GB). Val_avg 94.55 even WITHOUT chan_w/warmup. Stacked result expected ≪ 100.
7. **AMP bf16 fixes bs=4 inference NaN**: fern's bs=4 test_avg=84.64 with no NaN in test_geom_camber_cruise — bf16 numerics stabilize the attention computation.
8. **Lookahead + EMA both fail in rapid-descent regime**: weight-averaging operators are harmful when model makes fast descent (>10 MAE/epoch). #1603 and #1708 both closed for this reason.
9. **Per-sample AoA flip (p=0.25) fixes Uy** (−50%). Primary metric flat without chan_w stack.
10. **Cosine T_max=50 barely decays in 12-14 epoch budget** — T_max=12-14 is better calibration.
11. **pad_collate expensive at bs=8** (84 GB). Grad-accum is the correct lever.
12. **Test NaN Type 1** (data bug, 000020.pt): fix via evaluate_split prefilter. Askeladd #1536 + fern #1477 both implement this; fern's is cleaner.
13. **Test NaN Type 2** (numerical, bs=4 inference): AMP bf16 may fix this entirely.

## Round 4/5 hypothesis pipeline

### Highest priority (in flight)
- **fern AMP bf16 rebase** (#1477): CRITICAL — expected to be largest improvement. Rerun on floor stack.
- **tanjiro grad-accum=4** (#1524): rebased, awaiting run.
- **askeladd NaN guard rerun** (#1536): first clean test_avg at current floor.
- **alphonse decoupled chan_w** (#1559): training.
- **thorfinn AoA flip** (#1489): training.
- **nezuko WD=5e-4** (#1681): training.
- **frieren T_max=12 cosine** (#1751): training.
- **edward Huber loss** (#1801): just assigned.

### Next round (queue)
- **AMP bf16 + wider model**: if fern's AMP win merges, unlock n_hidden=192 (VRAM now fits).
- **AMP + slice_num=128**: 19+ epoch budget with larger attention, no bs change.
- **torch.compile reduce-overhead**: additive on top of AMP bf16 (fern's suggestion).
- **Sort-by-size sampler**: reduce pad_collate waste, raise effective bs.
- **Cosine warm restarts (SGDR)**: now that epoch budget is expanding with AMP.
- **DropPath stochastic depth** (reg): if regularization proves beneficial.
- If askeladd/fern NaN guard merges: close #1536 or supersede.
