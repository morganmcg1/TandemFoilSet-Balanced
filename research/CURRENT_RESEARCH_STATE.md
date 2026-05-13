# SENPAI Research State

- 2026-05-13 01:30
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **73.15** | #1641 (Lion lr=3e-4, merged 2026-05-13) |
| **test_avg/mae_surf_p** | **66.76** | #1641 — all 4 splits finite |
| Peak VRAM | 42.11 GB | #1641 — FP32 run; merged stack has BF16 so ~33 GB expected |

Merged stack: warmup3+cosine13 + GT-NaN fix + grad_clip(max_norm=1.0) + **Lion(lr=3e-4, wd=6e-5)** + **BF16 autocast**, batch=4, seed=42.

## Key round-5 findings to date

| Finding | Impact |
|---|---|
| Baseline does ~13 epochs in 30 min (not 50) | All schedule hyperparams must be matched to budget |
| CosineAnnealingLR(T_max=50) barely decays in 13 epochs | Matching T_max=13 alone gave 8.6% improvement (#1519) |
| 3-epoch warmup stabilises early training | Composes cleanly with schedule fix |
| GT NaN at cruise test sample 20 | Fixed in #1564: gt_finite_mask filter before accumulate_batch |
| Gradient clipping max_norm=1.0 gives −7.8% (#1483, MERGED) | Pre-clip norms 45-112 >> 1.0: clipping fires EVERY step = gradient renorm |
| lr=1e-3 + grad_clip → −9.5% (#1638, MERGED) | Biggest gain so far. Renorm every step = safe 2× LR. |
| BF16 autocast → −1.3% val, −22% VRAM (#1565, MERGED) | Unlocks n_hidden=192, n_layers=7, batch=8 experiments |
| **Lion optimizer (lr=3e-4) → −22.4% val (#1641, MERGED)** | **Largest single gain. Per-parameter sign quantization >> global L2 renorm** |
| LR/clip ceiling confirmed (#1683, CLOSED) | Optimization-side knobs (LR, clip) tapped out; both arms regressed on test |
| EMA decay=0.999 regresses +16.1% (#1596, CLOSED) | 13-epoch monotonic regime: EMA averages early bad model |
| n_hidden=192 regresses +47.7% (#1478, CLOSED) | Only 10/50 epochs ran; T_max=50 mismatch; revisitable with BF16+Lion |
| surf_weight=20 regresses (#1459, CLOSED) | Budget too short for changed loss landscape |
| Surface skip regresses (#1487, CLOSED) | Schedule absorbed the skip headroom |

## Active PRs

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #1780 | tanjiro | Lion + longer cosine (epochs=16, BF16) — exploit non-convergence at epoch 13 | WIP | Beat 73.15 |
| #1782 | frieren | Lion LR scan (2e-4, 2.5e-4, 4e-4) — narrow optimal between 1.5e-4 and 3e-4 | WIP | Beat 73.15 |
| #1755 | fern | n_hidden=192 + BF16 (wider model unlocked by BF16 VRAM cut) | WIP — needs Lion rebase | Beat 73.15 |
| #1656 | thorfinn | Dropout=0.1 in attention + MLP on Lion stack | WIP — needs Lion rebase | Beat 73.15 |
| #1639 | alphonse | Huber/Smooth-L1 loss (delta=1.0) on Lion stack | WIP — needs Lion rebase | Beat 73.15 |
| #1481 | nezuko | slice_num=128 | WIP | Beat 73.15 |
| #1470 | edward | Instance-norm loss | WIP | Beat 73.15 |
| #1463 | askeladd | Warmup+SWA+grad_clip compose on Lion stack | WIP — needs Lion rebase | Beat 73.15 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #1641 | frieren | **MERGED** | Lion optimizer (lr=3e-4) → **new baseline 73.15** (−22.4%). Largest single-PR gain. |
| #1683 | tanjiro | CLOSED | LR2e3/maxnorm=4 sweep — both arms regress on test (renorm-ceiling confirmed). |
| #1565 | fern | MERGED | BF16 autocast → baseline 94.22 (−1.3%). VRAM −22% unlocks wider models. |
| #1638 | tanjiro | MERGED | lr=1e-3 + grad_clip → baseline 95.44 (−9.5%). |
| #1487 | thorfinn | CLOSED | Surface skip compose (+13% worse). |
| #1483 | tanjiro | MERGED | grad_clip max_norm=1.0 → baseline 105.46. |
| #1596 | alphonse | CLOSED | EMA decay=0.999 regressed +16%. |
| #1478 | frieren | CLOSED | n_hidden=192 regressed +47% (now revisitable with BF16+Lion). |

## Open questions from active experiments

1. **Does n_hidden=192 + BF16 break baseline?** (#1755 fern) — wider model with old AdamW stack; now needs re-eval against Lion baseline 73.15
2. **Does dropout=0.1 improve OOD on Lion stack?** (#1656 thorfinn) — may need rebase onto Lion baseline
3. **Does Huber loss complement Lion?** (#1639 alphonse, crashlooping) — combo test on new Lion stack
4. **Does instance-norm loss help val_re_rand with Lion?** (#1470 edward)
5. **Does slice_num=128 help?** (#1481 nezuko)
6. **Does SWA compose with Lion?** (#1463 askeladd, rebase needed)

### Lion optimizer insight (PR #1641 analysis)

Lion's per-parameter sign update is strictly stronger than global L2 renorm. The 13-epoch trajectory shows monotonic improvement still at epoch 13 — **not converged**. This means:
- More epochs (budget: 16 with BF16) will improve further
- LR scan (lr=2e-4 to 2.5e-4) may find a better minimum
- Architecture experiments (n_hidden=192, n_layers=7) should be tested on the Lion stack

## Next hypotheses to queue (when students go idle)

### Top priority — Lion follow-ups (both tanjiro and frieren idle)
1. **Lion + longer cosine (epochs=16, BF16)** — both arms non-converged at epoch 13; 16 epochs at ~100s = 26.7 min, within budget. Highest expected gain.
2. **Lion lr scan (2e-4, 2.5e-4)** — fill the gap between winning 3e-4 and arm 1 1.5e-4.
3. **Lion + BF16 baseline check** — run one epoch 13 run with the new merged stack (BF16+Lion) to establish the true combined baseline (Lion ran without BF16).

### Architecture experiments (with Lion stack)
4. **n_hidden=192 + Lion** — width experiment on the new optimizer (fern is running n_hidden=192 on old stack; may need Lion rebase).
5. **n_layers=7 + Lion** — depth instead of width.
6. **batch=8 + Lion** — Lion's uniform steps may compose well with larger effective batch.

### Regularisation on Lion stack
7. **Dropout=0.1 + Lion** — if thorfinn's #1656 was with AdamW, retesting on Lion is valuable.
8. **Lion β2=0.999** — default is 0.99; at B=4 with high per-step noise, slower momentum may help.
9. **SWA + Lion** — weight averaging after Lion convergence; askeladd's #1463 was with AdamW.

### Architecture exploration
10. **DropPath / stochastic depth** — targeted at Transolver's residual structure.
11. **Activation sweep** — GELU → SwiGLU/SiLU (on Lion stack).
