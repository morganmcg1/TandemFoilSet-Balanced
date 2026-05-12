# SENPAI Research State

- 2026-05-12 22:15
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **105.46** | #1483 (grad_clip max_norm=1.0) |
| **test_avg/mae_surf_p** | **TBD** | Source branch lacked GT-NaN fix; merged code has it — next run will produce finite test |

Merged stack: warmup3+cosine13 + GT-NaN fix (evaluate_split) + grad_clip(max_norm=1.0), AdamW(lr=5e-4, wd=1e-4), batch=4, seed=42.

## Key round-5 findings to date

| Finding | Impact |
|---|---|
| Baseline does ~13 epochs in 30 min (not 50) | All schedule hyperparams must be matched to budget |
| CosineAnnealingLR(T_max=50) barely decays in 13 epochs | Matching T_max=13 alone gave 8.6% improvement (#1519) |
| 3-epoch warmup stabilises early training | Composes cleanly with schedule fix |
| SWA mechanism works (27.5% within-run gain on val) | Not yet beating merged baseline; needs warmup+SWA compose |
| surf_weight=20 regresses at 14 epochs | Budget too short for the changed loss landscape |
| p-channel 3x weight regresses | Surface Ux/Uy is NOT a free task; equal-weight is already right |
| GT NaN at cruise test sample 20 (idx=20, y contains Inf) | Fixed in #1564: gt_finite_mask filter before accumulate_batch |
| Surface skip -6.2% within-run on pre-warmup baseline (#1487) | Mechanism real; needs composition with merged baseline (~107 expected) |
| BF16+batch=8 regressed val +1.5% (T_max mismatch + no LR scale) | --epochs must ALWAYS match actual budget; batch scale needs LR scale |
| Gradient clipping max_norm=1.0 gives -7.8% (#1483, MERGED) | Pre-clip norms 45-112 >> 1.0: clipping fires EVERY step = gradient renorm |
| EMA decay=0.999 regresses +16.1% (#1596, CLOSED) | 13-epoch monotonic regime: EMA averages "early bad model" into "late good model" |
| n_hidden=192 regresses +47.7% (#1478, CLOSED) | 185s/epoch → only 10/50 epochs ran; T_max=50 mismatch; needs BF16 to revisit |

## Active PRs (all 8 students assigned)

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #1463 | askeladd | Warmup+cosine+SWA composed | WIP (needs rebase onto #1483) | Beat 105.46 |
| #1470 | edward | Instance-norm loss | WIP active | Beat 105.46 |
| #1481 | nezuko | slice_num=128 | WIP active | Beat 105.46 |
| #1487 | thorfinn | Surface skip composed with merged baseline | WIP active | Beat 105.46 |
| #1565 | fern | BF16 only (batch=4) isolate precision gain | WIP rerun | Beat 105.46 |
| #1638 | tanjiro | LR=1e-3 exploit grad_clip stability | WIP new | Beat 105.46 |
| #1639 | alphonse | Huber/Smooth-L1 loss (delta=1.0) | WIP new | Beat 105.46 |
| #1641 | frieren | Lion optimizer (lr=1.5e-4) | WIP new | Beat 105.46 |

## Open questions from active experiments

1. **Does surface skip compose with warmup+cosine+grad_clip?** (#1487 thorfinn rerun) — Within-run δ was -6.2%. Composed with new baseline (105.46), expected ~98-100 val. Could push test below 100.
2. **Does lr=1e-3 exploit grad_clip stability?** (#1638 tanjiro) — Grad renorm bounds every update; 2× LR gives larger bounded steps. Expected 2–6%.
3. **Does Huber loss complement grad_clip?** (#1639 alphonse) — Per-sample outlier robustness + gradient-vector robustness are orthogonal. Expected on OOD splits.
4. **Does Lion optimizer suit our renorm regime?** (#1641 frieren) — Grad_clip is global renorm; Lion is per-param sign renorm. Natural experiment to see if full sign quantization helps. Expected 1–3%.
5. **Does instance-norm loss help?** (#1470 edward) — Addresses Re dynamic range; could improve val_re_rand.
6. **Does slice_num=128 help?** (#1481 nezuko) — More slice tokens in PhysicsAttention. Slower per epoch but richer geometry partitioning.
7. **Does BF16 alone (batch=4) unlock more epochs?** (#1565 fern) — Isolated from batch confound. If yes, enables revisiting wider model.
8. **Does composed warmup+SWA beat grad_clip baseline?** (#1463 askeladd) — Needs rebase onto #1483 merged stack.

## Next hypotheses to queue (when students go idle)

1. **Surface skip + lr=1e-3 + grad_clip** — once #1487 and #1638 results land, compose the winners.
2. **Skip hidden=64** — scale surf_skip MLP from 32→64 hidden after #1487 confirms composition wins.
3. **Volume-side skip** — analogous to surf_skip for volume nodes. Complements surface skip.
4. **Huber + Lion compose** — if both #1639 and #1641 win, compose them for additive gain.
5. **n_hidden=192 with BF16** — if #1565 wins and shows memory headroom, revisit wider model.
6. **n_layers=7 (depth) with BF16** — depth increase viable if BF16 wins.
7. **Transolver++ local adaptive correction** — highest expected gain from literature, moderate engineering effort. Priority after simpler wins are exhausted.
8. **max_norm sweep** — test max_norm=2.0 vs 0.5 to characterize the renorm-optimal scale (tanjiro's PR does lr but not norm sweep; a follow-up could isolate the scale question).
