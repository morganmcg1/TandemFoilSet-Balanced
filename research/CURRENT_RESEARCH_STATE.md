# SENPAI Research State

- 2026-05-12 21:15
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **114.40** | #1519 (warmup3 + cosine T_max=13) |
| **test_avg/mae_surf_p** | **107.57** | #1564 (GT-NaN fix in evaluate_split) |

- Model still improving at epoch 13 → more compute headroom
- All 4 val splits and all 4 test splits now produce finite MAEs

## Key round-5 findings to date

| Finding | Impact |
|---|---|
| Baseline does ~13 epochs in 30 min (not 50) | All schedule hyperparams must be matched to budget |
| CosineAnnealingLR(T_max=50) barely decays in 13 epochs | Matching T_max=13 alone gave 8.6% improvement |
| 3-epoch warmup stabilises early training | Composes cleanly with schedule fix |
| SWA mechanism works (27.5% within-run gain on val) | Not yet beating merged baseline; need warmup+SWA compose |
| surf_weight=20 regresses at 14 epochs | Budget too short for the changed loss landscape |
| p-channel 3x weight regresses | Surface Ux/Uy is NOT a free task; equal-weight is already right |
| GT NaN at cruise test sample 20 (idx=20, y contains Inf) | Fixed in #1564: gt_finite_mask filter before accumulate_batch |
| Surface skip -6.2% within-run on pre-warmup baseline (#1487) | Mechanism real; needs composition with merged baseline (~107 expected) |
| BF16+batch=8 regressed val +1.5% (T_max mismatch + no LR scale) | --epochs must ALWAYS match actual budget; batch scale needs LR scale |

## Active PRs (all 8 students assigned)

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #1463 | askeladd | Warmup+cosine+SWA composed | WIP rerun (needs rebase) | Beat 114.40 |
| #1470 | edward | Instance-norm loss | WIP active | Beat 114.40 |
| #1478 | frieren | Wider model n_hidden=192 | WIP active (64.8 GB GPU) | Beat 114.40 |
| #1481 | nezuko | slice_num=128 | WIP active (87 GB GPU) | Beat 114.40 |
| #1483 | tanjiro | Gradient clipping max_norm=1.0 | WIP active (69.5 GB GPU) | Beat 114.40 |
| #1487 | thorfinn | Surface skip composed with merged baseline | WIP rerun | Beat 114.40 |
| #1565 | fern | BF16 only (batch=4) isolate precision gain | WIP rerun | Beat 114.40 |
| #1596 | alphonse | EMA of weights (decay=0.999) per gradient step | WIP new | Beat 114.40 |

## Open questions from active experiments

1. **Does EMA of weights improve generalization?** (#1596 alphonse) — Near-zero cost, no LR valley needed, proven in low-iteration regimes. Expected 2-5%.
2. **Does the surface skip compose with warmup+cosine?** (#1487 thorfinn rerun) — High prior; within-run delta of -6.2%. Expected ~107 val, could push test below 100.
3. **Does composed warmup+SWA beat warmup alone?** (#1463 askeladd) — SWA mechanism confirmed; composing with warmup recipe is most promising pending hypothesis.
4. **Does gradient clipping stabilise training?** (#1483 tanjiro) — Would improve all future experiments if it does.
5. **Does wider model help?** (#1478 frieren) — 192 hidden dims; slower per epoch so fewer epochs fit in budget.
6. **Does instance-norm loss help?** (#1470 edward) — Addresses Re dynamic range; could improve val_re_rand.
7. **Does BF16 alone (batch=4) unlock more epochs?** (#1565 fern) — Isolated from batch confound.

## Next hypotheses to queue (when students go idle)

1. **Surface skip + winning compose** — once #1487 and any of (#1483, #1463, #1596) win, compose them.
2. **Skip hidden=64** — current surf_skip is 32 hidden (675 params); scaling up may improve camber-OOD. Only after #1487 confirms composition wins.
3. **Volume-side skip** — analogous to surf_skip for volume nodes (different feature subset). Complements surface skip.
4. **LR increase (lr=1e-3)** — with warmup + clipping (if #1483 wins), pushing lr from 5e-4 to 1e-3 could improve convergence speed.
5. **Transolver++ local adaptive correction** — highest expected gain from literature, moderate engineering effort. Priority after simpler wins are exhausted.
6. **n_layers=7 with BF16** — if BF16 wins (#1565), depth increase becomes viable within budget.
