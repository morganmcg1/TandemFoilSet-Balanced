# SENPAI Research State

- 2026-05-13 01:10
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **94.22** | #1565 (BF16 autocast, merged 2026-05-13) |
| **test_avg/mae_surf_p** | **87.10** | #1565 — all 4 splits finite |
| Peak VRAM | 32.94 GB | #1565 — 22% reduction enables wider-model experiments |

Merged stack: warmup3+cosine13 + GT-NaN fix + grad_clip(max_norm=1.0) + lr=1e-3 + **BF16 autocast**, AdamW(wd=1e-4), batch=4, seed=42.

## Key round-5 findings to date

| Finding | Impact |
|---|---|
| Baseline does ~13 epochs in 30 min (not 50) | All schedule hyperparams must be matched to budget |
| CosineAnnealingLR(T_max=50) barely decays in 13 epochs | Matching T_max=13 alone gave 8.6% improvement (#1519) |
| 3-epoch warmup stabilises early training | Composes cleanly with schedule fix |
| GT NaN at cruise test sample 20 | Fixed in #1564: gt_finite_mask filter before accumulate_batch |
| Gradient clipping max_norm=1.0 gives -7.8% (#1483, MERGED) | Pre-clip norms 45-112 >> 1.0: clipping fires EVERY step = gradient renorm |
| lr=1e-3 + grad_clip → −9.5% (#1638, MERGED) | Biggest single gain. Renorm every step = safe 2× LR. |
| **BF16 autocast → −1.3% val, −22% VRAM (#1565, MERGED)** | **Unlocks n_hidden=192, n_layers=7, batch=8 experiments** |
| EMA decay=0.999 regresses +16.1% (#1596, CLOSED) | 13-epoch monotonic regime: EMA averages early bad model |
| n_hidden=192 regresses +47.7% (#1478, CLOSED) | Only 10/50 epochs ran; T_max=50 mismatch; now revisitable with BF16 |
| surf_weight=20 regresses (#1459, CLOSED) | Budget too short for changed loss landscape |
| Surface skip regresses (#1487, CLOSED) | Schedule absorbed the skip headroom |

## Active PRs

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #1755 | fern | n_hidden=192 + BF16 (wider model unlocked by BF16 VRAM cut) | WIP | Beat 94.22 |
| #1656 | thorfinn | Dropout=0.1 in attention + MLP | WIP | Beat 94.22 |
| #1641 | frieren | Lion optimizer (lr=1.5e-4) | WIP | Beat 94.22 |
| #1639 | alphonse | Huber/Smooth-L1 loss (delta=1.0) | WIP (crashloop) | Beat 94.22 |
| #1481 | nezuko | slice_num=128 | WIP | Beat 94.22 |
| #1470 | edward | Instance-norm loss | WIP | Beat 94.22 |
| #1463 | askeladd | Warmup+SWA+grad_clip compose | WIP (rebase needed) | Beat 94.22 |
| tanjiro | tanjiro | IDLE — assigning follow-up after #1683 close | Assigning | Beat 94.22 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #1683 | tanjiro | CLOSED | LR2e3/maxnorm=4 sweep — both arms +0.9–1.2% val + test regress (renorm-ceiling confirmed). |
| #1565 | fern | MERGED | BF16 autocast → **new baseline 94.22** (−1.3%). VRAM −22% unlocks wider models. |
| #1638 | tanjiro | MERGED | lr=1e-3 + grad_clip → baseline 95.44 (−9.5%). |
| #1487 | thorfinn | CLOSED | Surface skip compose (+13% worse). |
| #1483 | tanjiro | MERGED | grad_clip max_norm=1.0 → baseline 105.46. |
| #1596 | alphonse | CLOSED | EMA decay=0.999 regressed +16%. |
| #1478 | frieren | CLOSED | n_hidden=192 regressed +47% (now revisitable with BF16). |

## Open questions from active experiments

1. **Does n_hidden=192 + BF16 break baseline?** (#1755 fern) — wider model unlocked by VRAM cut
2. **Does dropout=0.1 improve OOD?** (#1656 thorfinn) — orthogonal regularization
3. **Does Huber loss complement grad_clip?** (#1639 alphonse, currently crashlooping)
4. **Does Lion optimizer suit renorm regime?** (#1641 frieren)
5. **Does instance-norm loss help val_re_rand?** (#1470 edward)
6. **Does slice_num=128 help?** (#1481 nezuko)
7. **Does composed warmup+SWA beat current baseline?** (#1463 askeladd, rebase needed)
8. **Does longer cosine (epochs=16) with BF16 budget exploit the speed win?** (next: tanjiro)

### Renorm-regime ceiling confirmed (#1683 closure insight)

Optimization-side knobs (LR, clip threshold) appear tapped out in the pre-BF16 stack. Both Arm A (2× LR) and Arm B (4× max_norm) test the same direction (4× effective post-clip step) and both regress on test. The renorm regime ceiling sits near 95.44 val / 87.83 test in FP32. With BF16 now landed, the path forward is architecture (width, depth), training duration (more epochs in same budget), or new regularisation (dropout, droppath, swa).

## Next hypotheses to queue (when students go idle)

1. **n_hidden=192 with BF16** — BF16 now merged; 32.94 GB peak leaves 63 GB headroom. n_hidden=192 estimated ~10% more params. High expected gain.
2. **n_layers=7 (depth) with BF16** — same rationale; depth may complement width.
3. **batch=8 + BF16** — BF16 reduced VRAM by 9 GB; try batch=8 with BF16 and LR scale (2×) to test batch-stability hypothesis.
4. **Longer cosine (epochs=16)** — BF16 saves 23% wall time; at 100s/epoch, 16 epochs = 26.7 min (within 30-min cap). Test if more training time helps.
5. **max_norm=4.0 sweep** (if #1683 arm B is informative, close before assigning separately)
6. **Dropout + Huber compose** — if both #1656 and #1639 win, compose them.
7. **weight_decay sweep** — wd=1e-4 → 1e-3 or 5e-4; grad_clip amplifies relative WD effect.
8. **DropPath/stochastic depth** — targeted at transformer residual structure.
9. **Activation sweep** — GELU → SwiGLU/SiLU.
