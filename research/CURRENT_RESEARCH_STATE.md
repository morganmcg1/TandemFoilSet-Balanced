# SENPAI Research State

- 2026-05-13 04:40
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **66.32** | #1639 (Huber δ=0.5, merged 2026-05-13) |
| **test_avg/mae_surf_p** | **61.14** | #1639 — all 4 splits finite |
| Peak VRAM | 32.95 GB | #1639 — BF16 |
| s/epoch | ~101 s | BF16+Lion |

Merged stack: warmup3+cosine + GT-NaN fix + grad_clip(max_norm=1.0) + **Lion(lr=3e-4, wd=6e-5)** + **BF16 autocast** + **Huber δ=0.5 loss**, epochs=**16** (new standard), batch=4, seed=42.

**Reproduce current best (Huber δ=0.5, note: merged code is 13 epochs by default; use --epochs 16 to get combined stack):**
```bash
cd target/ && python train.py --epochs 16 --experiment_name huber_ep16_baseline_check --agent <student>
```

## Key round-5 findings to date

| Finding | Impact |
|---|---|
| Baseline does ~13 epochs in 30 min (not 50) | All schedule hyperparams must be matched to budget |
| CosineAnnealingLR(T_max=50) barely decays in 13 epochs | Matching T_max=13 alone gave 8.6% improvement (#1519) |
| 3-epoch warmup stabilises early training | Composes cleanly with schedule fix |
| GT NaN at cruise test sample 20 | Fixed in #1564 |
| Gradient clipping max_norm=1.0 → −7.8% (#1483, MERGED) | Renorm-every-step regime |
| lr=1e-3 + grad_clip → −9.5% (#1638, MERGED) | Renorm ceiling: safe 2× LR |
| BF16 autocast → −1.3% val, −22% VRAM (#1565, MERGED) | Enables 16 epochs in budget |
| **Lion optimizer (lr=3e-4) → −22.4% val (#1641, MERGED)** | **Per-parameter sign quantization >> global L2 renorm** |
| **Lion + epochs=16 → −9.2% further (#1780, MERGED)** | **Non-converged at 13 epochs; cosine tail provides 3 more improvement epochs** |
| **Huber δ=0.5 → −9.3% further (#1639, MERGED)** | **Per-element outlier capping stacks with grad_clip; δ=0.5 > δ=1.0 uniformly** |
| LR/clip ceiling confirmed (#1683, CLOSED) | Optimization-side knobs tapped out at AdamW stage |
| SWA mid-training regresses +4.1% (#1463, CLOSED) | Averages early bad checkpoints; SWALR fights Lion cosine |
| EMA decay=0.999 regresses +16.1% (#1596, CLOSED) | 13-epoch monotonic regime: early averaging always hurts |
| n_hidden=160 → −1.71 val / −0.51 test on OLD baseline (#1755, SENT BACK 2nd time) | Clean width gain on old Lion stack; needs re-run on new Huber+epochs=16 stack to confirm composition |
| n_hidden=192 confirmed dead (#1755 Arm B, lr=4e-4) | Higher LR doesn't recover lost epoch; grad_norm spike at ep6; 2× regression evidence (this + original PR) |
| Lion LR 2.5e-4 marginally better (#1782, SENT BACK) | 71.54 on 13-epoch stack; below new 66.32 baseline; needs re-run at epochs=16+Huber |

## Active PRs

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #1879 | tanjiro | Compound: Huber δ=0.5 + epochs=16 — test both wins compose | WIP | Beat 66.32 |
| #1880 | alphonse | Huber δ scan: δ=0.3 and δ=0.2 on epochs=16 — find optimal δ floor | WIP | Beat 66.32 |
| #1782 | frieren | Lion LR scan re-run (2.5e-4, 2e-4) on Huber+epochs=16 stack | WIP (sent back) | Beat 66.32 |
| #1755 | fern | n_hidden=160 single-arm on Huber+epochs=16 stack (Arm B dropped) | WIP (sent back 2nd time) | Beat 66.32 |
| #1844 | askeladd | Lion β2: 0.99 → 0.999 (slower momentum for B=4 noise), epochs=16 | WIP | Beat 66.32 |
| #1656 | thorfinn | Dropout=0.1 in attention + MLP, epochs=16 | WIP — needs rebase | Beat 66.32 |
| #1481 | nezuko | slice_num=128, epochs=16 | WIP — needs rebase | Beat 66.32 |
| #1470 | edward | Instance-norm loss, epochs=16 | WIP — needs rebase | Beat 66.32 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #1639 | alphonse | **MERGED** | Huber δ=0.5 → **new baseline 66.32** (−9.3%). Uniformly better across all splits. δ curve not bottomed out. |
| #1780 | tanjiro | **MERGED** | Lion+epochs=16 → baseline 66.44 (−9.2%). Structural: epochs=16 is now standard. |
| #1782 | frieren | SENT BACK | val=71.54 (LR scan 2.5e-4, 13 epochs) — below new baseline 66.32; re-run at epochs=16+Huber |
| #1463 | askeladd | CLOSED | SWA → val +2.99 regression. SWALR fights Lion cosine; early checkpoint averaging bad. |
| #1641 | frieren | **MERGED** | Lion optimizer (lr=3e-4) → baseline 73.15 (−22.4%). Largest single-PR gain. |
| #1683 | tanjiro | CLOSED | LR ceiling confirmed — both arms regress on test. |
| #1565 | fern | MERGED | BF16 autocast → baseline 94.22 (−1.3%). VRAM −22%. |
| #1638 | tanjiro | MERGED | lr=1e-3 + grad_clip → baseline 95.44 (−9.5%). |

## Open questions from active experiments

1. **Does Huber+epochs=16 compound both wins?** (#1879 tanjiro) — expected ~62-65
2. **Is δ=0.3 or δ=0.2 better than δ=0.5?** (#1880 alphonse) — monotonic trend suggests yes
3. **Does LR 2.5e-4 advantage hold on Huber+epochs=16 stack?** (#1782 frieren, re-run)
4. **Does n_hidden=160/192 clear the budget cliff on Lion stack?** (#1755 fern, 2-arm)
5. **Does Lion β2=0.999 help at B=4?** (#1844 askeladd) — now running with epochs=16
6. **Does dropout=0.1 improve OOD on full combined stack?** (#1656 thorfinn)
7. **Does slice_num=128 help?** (#1481 nezuko) — with epochs=16
8. **Does instance-norm loss help val_re_rand?** (#1470 edward) — with epochs=16

## Confirmed dead ends

- **SWA mid-training (#1463)**: regresses in 13-epoch monotonic regime. Partial camber_rc signal — revisit at 24+ epochs.
- **LR/clip ceiling at AdamW stage (#1683)**: both 2× arms regress on test (renorm-ceiling). Obsoleted by Lion switch.
- **EMA decay=0.999 (#1596)**: 13-epoch monotonic regime; early averaging always hurts.
- **n_hidden=192 (#1755 Arm B, lr=4e-4)**: Higher LR can't recover lost epoch in 30-min budget; grad_norm instability at LR=4e-4. Two PRs of regression evidence. Architecture lever has shifted to n_hidden=160 only.

## Next hypotheses to queue (when students go idle)

### Currently active (don't duplicate)
- #1879 tanjiro: Huber δ=0.5 + epochs=16
- #1880 alphonse: Huber δ=0.3, δ=0.2
- #1782 frieren: LR scan 2.5e-4, 2e-4 (re-run on new stack)
- #1755 fern: width sweep n_hidden=160 + n_hidden=192/lr=4e-4
- #1844 askeladd: Lion β2 sweep 0.99 → 0.999 + epochs=16
- #1656 thorfinn: Dropout=0.1 + epochs=16
- #1481 nezuko: slice_num=128 + epochs=16
- #1470 edward: instance-norm loss + epochs=16

### Queued ideas (when students finish above)

1. **n_layers=6 + Lion + epochs=16** — depth instead of width; one extra attention+MLP layer.
2. **batch=8 + Lion + epochs=13** — larger effective batch on full stack; ~30 min with batch=8.
3. **Huber δ=0.1 + epochs=16** — if δ scan finds 0.3/0.2 wins, push further.
4. **surf_weight=15 or 5 under Huber+Lion** — optimal weighting may shift with changed loss landscape.
5. **DropPath / stochastic depth** — targeted at Transolver's residual structure.
6. **Activation sweep** — GELU → SwiGLU/SiLU (on full combined stack).
7. **Layer-wise LR decay** — different LR per Transolver layer.
8. **EMA post-convergence (last 2 epochs only)** — avoids #1463 failure mode; averages only the final stable checkpoints.
9. **Lion lr=2.5e-4 as new default** — if frieren's re-run confirms 2.5e-4 consistently beats 3e-4, update the merged default.
