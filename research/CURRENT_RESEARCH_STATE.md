# SENPAI Research State

- 2026-05-12 23:05
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **95.44** | #1638 (lr=1e-3 + grad_clip, merged 2026-05-12) |
| **test_avg/mae_surf_p** | **87.83** | #1638 — all 4 splits finite |

Merged stack: warmup3+cosine13 + GT-NaN fix (evaluate_split) + grad_clip(max_norm=1.0) + lr=1e-3, AdamW(wd=1e-4), batch=4, seed=42.

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
| **lr=1e-3 + grad_clip → −9.5% (#1638, MERGED)** | **Biggest single gain. Renorm every step = safe 2× LR. Largest gains on OOD splits.** |

## Active PRs (all 8 students assigned)

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #1463 | askeladd | Warmup+cosine+SWA+grad_clip triple compose | WIP (rebase needed onto current baseline) | Beat 95.44 |
| #1470 | edward | Instance-norm loss | WIP active | Beat 95.44 |
| #1481 | nezuko | slice_num=128 | WIP active | Beat 95.44 |
| #1565 | fern | BF16 only (batch=4) isolate precision gain | WIP rerun | Beat 95.44 |
| #1639 | alphonse | Huber/Smooth-L1 loss (delta=1.0) | WIP new | Beat 95.44 |
| #1641 | frieren | Lion optimizer (lr=1.5e-4) | WIP new | Beat 95.44 |
| #1656 | thorfinn | Dropout=0.1 in attention + MLP | WIP new | Beat 95.44 |
| tanjiro | tanjiro | LR push (lr=2e-3) or max_norm sweep | Assigning | Beat 95.44 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #1638 | tanjiro | MERGED | lr=1e-3 + grad_clip → **new baseline 95.44** (−9.5%). Biggest gain round 5. |
| #1487 | thorfinn | CLOSED | Surface skip compose (119.33 vs 105.46 = +13% worse). Schedule absorbed the skip's headroom. |
| #1483 | tanjiro | MERGED | Gradient clipping max_norm=1.0 → baseline 105.46. |
| #1596 | alphonse | CLOSED | EMA decay=0.999 regressed +16% in monotonic 13-epoch regime. |
| #1478 | frieren | CLOSED | n_hidden=192 regressed +47% (only 10/50 epochs ran). |

## Open questions from active experiments

1. **Does dropout=0.1 improve OOD generalization?** (#1656 thorfinn) — Stack is dropout=0.0 everywhere; forward-pass feature noise targets overfitting on OOD splits.
2. **Does Huber loss complement grad_clip?** (#1639 alphonse) — Per-sample outlier robustness + gradient-vector robustness are orthogonal. Expected gain on OOD splits.
3. **Does Lion optimizer suit our renorm regime?** (#1641 frieren) — Grad_clip is global renorm; Lion is per-param sign renorm.
4. **Does instance-norm loss help?** (#1470 edward) — Addresses Re dynamic range; could improve val_re_rand.
5. **Does slice_num=128 help?** (#1481 nezuko) — More slice tokens in PhysicsAttention. Slower per epoch but richer geometry partitioning.
6. **Does BF16 alone (batch=4) unlock more epochs?** (#1565 fern) — Isolated from batch confound. If yes, enables revisiting wider model.
7. **Does composed warmup+SWA+grad_clip beat current baseline?** (#1463 askeladd) — Needs rebase onto #1638 merged stack (now includes lr=1e-3).
8. **Does lr=2e-3 push further win?** (tanjiro follow-up) — If renorm is the active mechanism, 4× original LR under same clip should further improve. Key risk: 2e-3 may overshoot in the gradient renorm landscape.

## Next hypotheses to queue (when students go idle)

1. **max_norm=4.0 sweep at lr=1e-3** — test if tighter renorm (1.0) was the active driver or if loosening enables faster convergence.
2. **lr=2e-3 push** — tanjiro suggested; test renorm ceiling.
3. **Dropout + Huber compose** — if both #1656 and #1639 win, compose them.
4. **Huber + Lion compose** — if both #1639 and #1641 win, compose them.
5. **weight_decay sweep** — current 1e-4; with grad_clip normalizing gradient magnitudes, the relative effect of WD is amplified. Try wd=1e-3 or 5e-4.
6. **DropPath/stochastic depth** — alternative to plain dropout; targeted at deep transformer residual structure.
7. **n_hidden=192 with BF16** — if #1565 wins and shows memory headroom, revisit wider model with proper budget.
8. **n_layers=7 (depth) with BF16** — depth increase viable if BF16 wins.
9. **Transolver++ local adaptive correction** — highest expected gain from literature, moderate engineering effort.
10. **Activation sweep** — GELU → SwiGLU/SiLU. Common transformer improvement.
11. **Slice-token diversity regularization** — orthogonality penalty on slice_weights to prevent slice-token collapse.
