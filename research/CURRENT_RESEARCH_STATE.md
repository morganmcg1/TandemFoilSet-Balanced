# SENPAI Research State

- **Date:** 2026-05-13 01:05
- **Branch:** `icml-appendix-charlie-pai2g-24h-r2`
- **Track:** Charlie no-W&B 24h/48h logging-ablation arm (round 2/3)
- **Most recent human researcher direction:** none on this branch

## Current floor

**val_avg/mae_surf_p = 122.7043** (PR #1573, merged 2026-05-13)
Config: 3-ep warmup + lr=7.5e-4 + cosine(T_max=47) + gradclip(max_norm=1.0), bs=4, chan_w=[1,1,5], wd=1e-4, ~0.66M model, 12 epochs (timeout-cut)
bs=1 clean test_avg = **110.2527** (all 4 splits finite; val 122.70 beats prior floor 128.09 by 4.2%)

**Known test NaN:** bs=4 test_geom_camber_cruise still NaN — deterministic inference-time attention numerics edge case with specific batch compositions at this model+weight. Train-side gradclip doesn't fix it. bs=1 eval is fully clean. Askeladd's #1536 addresses the separate data-bug NaN (Type 1).

**eval_bs1.py** now in advisor branch — use for clean bs=1 test evaluation going forward.

## Active experiments (WIP)

| PR | Student | Hypothesis | Lever | Round |
|---|---|---|---|---|
| #1536 | askeladd | NaN guard + lr=1e-3 rerun stacked on NEW floor | Bug fix / measurement | 3 (training) |
| #1559 | alphonse | Decoupled surf/vol chan_w: [1,1,5] surf, [1,1,1] vol | Loss alignment | 3 (training) |
| #1524 | tanjiro | chan_w + grad-accum=4 + lr=7.5e-4 + T_max=14 | Stacking / opt | 3-revised (training) |
| #1489 | thorfinn | chan_w + per-sample AoA flip p=0.25 | Stacking / aug | 3-revised (training) |
| #1477 | fern | AMP bf16 + gradient clipping | Training efficiency | 1 (recovering, now training) |
| #1708 | edward | Lookahead optimizer (k=5, α=0.5) wrapping AdamW | Optimizer | 4 (training) |
| #1681 | nezuko | Weight decay 1e-4 → 5e-4 | Regularization | 3 (training) |
| #1751 | frieren | Tighter cosine: --epochs 15 → T_max=12 aligned to budget | Schedule | 4 (training) |

## Recent decisions

- **#1573 (frieren) MERGED — NEW FLOOR** val_avg 128.09 → 122.70 (−4.2%), bs=1 test 117.40 → 110.25 (−6.1%). lr=7.5e-4 + gradclip. val_geom_camber_cruise most improved (−12%). Gradient clip didn't fix bs=4 inference NaN (confirmed: it's inference-time, not training-time).
- **#1524 (tanjiro) re-sent-back 2026-05-13 01:13**: merge conflict + stale floor. Asked to rebase onto current advisor HEAD (41f2777) and re-run grad-accum=4 isolated lever at lr=7.5e-4.
- **Rate limit alert (2026-05-13 ~01:30+)**: 7 of 8 student pods reporting GraphQL rate-limit exhaustion (own token, separate from advisor's). Pods sleeping with "No assigned PRs" until reset (~1h per token). PR labels intact; assignments preserved.
- **#1485 (nezuko) CLOSED**: +25.4% regression, wall-clock budget binding.
- **#1536 (askeladd) SENT BACK**: NaN guard code now pushed + rebased. Awaiting lr=1e-3 rerun at new floor.
- **#1603 (edward EMA) CLOSED**: rapid-descent regime mismatch. Assigned Lookahead #1708 instead.

## Key findings so far

1. **Channel weight [1,1,5] is a confirmed win** (+6.4%, PR #1464, floor 133.94).
2. **Warmup + lr=1e-3 is a confirmed win** (+4.4%, PR #1482, floor 128.09).
3. **lr=7.5e-4 + gradient clipping is a confirmed win** (+4.2%, PR #1573, floor 122.70). val_geom_camber_cruise most improved (−12%).
4. **Three wins stacked in advisor train.py**: chan_w + warmup + gradclip. New experiments start with all three.
5. **chan_w response curve non-monotonic** — p=10 14% worse, optimal ≈5.
6. **Grad-accum=4 beats pre-chan_w floor by 2.4%** at half VRAM. Stacking in progress.
7. **Per-sample AoA flip (p=0.25) fixes Uy** (−50%). Primary metric flat without chan_w stack.
8. **Cosine T_max=50 barely decays in 12-14 epoch budget** — set T_max≈12-14.
9. **pad_collate expensive at bs=8** (84 GB). Grad-accum is the correct lever.
10. **slice_num=128 doesn't compound at 30-min cap** (close pending AMP).
11. **Test NaN Type 1** (data bug, 000020.pt): fix in train.py evaluate_split (askeladd #1536).
12. **Test NaN Type 2** (numerical, bs=4 inference): lr=1e-3 → lr=7.5e-4 REDUCES it. But bs=4 cruise NaN persists at 7.5e-4 — it's an inference-time attention computation issue. bs=1 eval is fully clean.
13. **EMA doesn't fit rapid-descent regime**: averages older-worse with newer-better. #1603 closed.
14. **eval_bs1.py** now in advisor branch for clean test-avg evaluation.
15. **Frieren's gradclip key insight**: the bs=4 NaN is deterministic and identical across lr=1e-3 and lr=7.5e-4 runs — it's a property of model weights × specific batch composition in PhysicsAttention, not optimizer-related.

## Round-4 hypothesis pipeline

### High priority (active)
- **askeladd NaN guard rerun** (#1536): first clean test_avg at current floor. Critical unlock.
- **alphonse decoupled surf/vol chan_w** (#1559): [1,1,5] surf only — vs new floor 122.70.
- **tanjiro chan_w + grad-accum** (#1524): now needs rebase on new floor (lr changed to 7.5e-4).
- **thorfinn AoA flip** (#1489): now needs rebase on new floor.
- **fern AMP bf16** (#1477): recovering from rate-limit, now training. VRAM unlock → 224-7-8 + slice_num=128 retry.
- **edward Lookahead** (#1708): k=5, α=0.5. Compatible with rapid-descent regime.
- **nezuko WD=5e-4** (#1681): regularization.

### Next round (after current WIPs complete)
- **Frieren (idle)**: assign next lever — try even lower lr (5e-4 or 6e-4) to push further into the stable regime, OR try LR=7.5e-4 with T_max=12 cosine (tighter decay for 12-epoch budget).
- **Stack Lookahead + WD** (if both win independently).
- If AMP wins: retry 224-7-8 + slice_num=128 with bf16.
- Stack gradclip + askeladd NaN guard once #1536 merges.
- Sort-by-size sampler (pad_collate waste reduction, higher effective bs).
- SmoothL1/Huber loss for pressure channel.
- Note: tanjiro #1524 and thorfinn #1489 need rebase onto new floor (lr changed from 1e-3 → 7.5e-4 in advisor train.py after #1573 merge). Their current PRs reference old floor. May need to send-back again.
