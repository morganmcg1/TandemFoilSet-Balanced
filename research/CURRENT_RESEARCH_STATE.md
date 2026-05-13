# SENPAI Research State

- 2026-05-13 03:25
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
| #1755 | fern | n_hidden=160 + n_hidden=192 (lr=4e-4) — budget-cliff follow-up after first arm tied val + test regressed | WIP (sent back) | Beat 73.15 |
| #1656 | thorfinn | Dropout=0.1 in attention + MLP on Lion stack | WIP — needs Lion rebase | Beat 73.15 |
| #1639 | alphonse | Huber/Smooth-L1 loss (delta=1.0) on Lion stack | WIP — needs Lion rebase | Beat 73.15 |
| #1481 | nezuko | slice_num=128 | WIP | Beat 73.15 |
| #1470 | edward | Instance-norm loss | WIP | Beat 73.15 |
| #1844 | askeladd | Lion β2: 0.99 → 0.999 (slower momentum for B=4 noise) | WIP | Beat 73.15 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #1463 | askeladd | CLOSED | SWA from epoch 25 on Lion stack — val +2.99 / test +3.53 regression. Avg in early bad weights + SWALR fights Lion's cosine. camber_rc improvement (−3.67) is real but minority signal. |
| #1641 | frieren | **MERGED** | Lion optimizer (lr=3e-4) → **new baseline 73.15** (−22.4%). Largest single-PR gain. |
| #1683 | tanjiro | CLOSED | LR2e3/maxnorm=4 sweep — both arms regress on test (renorm-ceiling confirmed). |
| #1565 | fern | MERGED | BF16 autocast → baseline 94.22 (−1.3%). VRAM −22% unlocks wider models. |
| #1638 | tanjiro | MERGED | lr=1e-3 + grad_clip → baseline 95.44 (−9.5%). |
| #1487 | thorfinn | CLOSED | Surface skip compose (+13% worse). |
| #1483 | tanjiro | MERGED | grad_clip max_norm=1.0 → baseline 105.46. |
| #1596 | alphonse | CLOSED | EMA decay=0.999 regressed +16%. |
| #1478 | frieren | CLOSED | n_hidden=192 regressed +47% (now revisitable with BF16+Lion). |

## Open questions from active experiments

1. **Does Lion+epochs=16 (BF16) extend the monotonic improvement past 13?** (#1780 tanjiro)
2. **Does Lion lr 2e-4 or 2.5e-4 beat lr=3e-4?** (#1782 frieren)
3. **Does intermediate width n_hidden=160 (full 13 epochs) or n_hidden=192 + lr=4e-4 (recover lost epoch) clear the budget cliff?** (#1755 fern, 2-arm follow-up)
4. **Does Lion β2=0.999 (slower momentum) help on B=4 noisy gradients?** (#1844 askeladd)
5. **Does dropout=0.1 improve OOD on Lion stack?** (#1656 thorfinn) — needs Lion rebase
6. **Does Huber loss complement Lion?** (#1639 alphonse) — needs Lion rebase
7. **Does instance-norm loss help val_re_rand with Lion?** (#1470 edward)
8. **Does slice_num=128 help?** (#1481 nezuko)

## Confirmed dead ends (Lion stack)

- **SWA from mid-training (#1463 askeladd, CLOSED)**: regresses val +2.99 / test +3.53 in 13-epoch monotonic regime. SWALR perturbs Lion's cosine; averaging early checkpoints poisons the average. *Partial signal:* val_geom_camber_rc improves −3.67 — preserve for revisit when training budget extends to 24+ epochs.

## Next hypotheses to queue (when students go idle)

### Currently active (don't duplicate)
- #1780 tanjiro: Lion + epochs=16
- #1782 frieren: Lion LR scan (2e-4, 2.5e-4)
- #1755 fern: width sweep n_hidden=160 + n_hidden=192/lr=4e-4
- #1844 askeladd: Lion β2 sweep 0.99 → 0.999
- #1656 thorfinn: Dropout=0.1 (needs Lion rebase)
- #1639 alphonse: Huber loss (needs Lion rebase)
- #1481 nezuko: slice_num=128
- #1470 edward: instance-norm loss

### Queued ideas (no current assignee)

1. **batch=8 + Lion** — Lion's uniform steps may compose well with larger effective batch; BF16 leaves headroom.
2. **n_layers=7 + Lion** — depth instead of width; same VRAM headroom story as #1755.
3. **DropPath / stochastic depth** — targeted at Transolver's residual structure.
4. **Activation sweep** — GELU → SwiGLU/SiLU (on Lion stack).
5. **Mixup / CutMix on point clouds** — pair perturbation as input regularization; complementary to weight regularization.
6. **EMA (lighter than SWA) starting after final cosine descent** — averages only the last 2-3 stable epochs; avoids #1463 failure modes (no SWALR, no early averaging).
7. **Layer-wise LR decay** — different LR per Transolver layer (lower for early layers, higher for later) on Lion stack.
8. **Surface vs volume loss reweighting under Lion** — sweep surf_weight ∈ {5, 15, 20} now that the optimizer changed.

### Lion optimizer insight (PR #1641 analysis)

Lion's per-parameter sign update is strictly stronger than global L2 renorm. The 13-epoch trajectory shows monotonic improvement still at epoch 13 — **not converged**. This means:
- More epochs (#1780 tests this; up to 16 fits in budget) will improve further
- LR scan (#1782 fills 2e-4, 2.5e-4 gap)
- β2 sweep (#1844 tests momentum half-life on noisy gradients)
- Architecture experiments (#1755 fern's wider model 2-arm; #1656 dropout) should clear budget cliff to validate on Lion stack
