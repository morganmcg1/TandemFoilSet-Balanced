# SENPAI Research State

- 2026-05-13 06:30
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **56.90** | #1880 (Huber δ=0.3, merged 2026-05-13) |
| **test_avg/mae_surf_p** | **53.20** | #1880 — all 4 splits finite |
| Peak VRAM | 32.95 GB | #1880 — BF16, batch=4 |
| s/epoch | ~102 s | BF16+Lion, 16 epochs ≈ 27 min |

Merged stack: warmup3+cosine + GT-NaN fix + grad_clip(max_norm=1.0) + **Lion(lr=3e-4, wd=6e-5)** + **BF16 autocast** + **Huber δ=0.3 loss**, epochs=**16**, batch=4, seed=42.

**Reproduce current best:**
```bash
cd target/ && python train.py --epochs 16 --experiment_name huber_d03_ep16_baseline_check --agent <student>
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
| **Huber δ=0.5 → −9.3% further (#1639, MERGED)** | **Per-element outlier capping stacks with grad_clip** |
| **Huber δ=0.3 → −14.2% further (#1880, MERGED)** | **δ curve not bottomed at 0.5; 0.3 optimal — δ=0.2 ties within noise** |
| LR/clip ceiling confirmed (#1683, CLOSED) | Optimization-side knobs tapped out at AdamW stage |
| SWA mid-training regresses +4.1% (#1463, CLOSED) | Averages early bad checkpoints; SWALR fights Lion cosine |
| EMA decay=0.999 regresses +16.1% (#1596, CLOSED) | 13-epoch monotonic regime: early averaging always hurts |
| **n_hidden=160 → −8.97 val on Huber δ=0.5 stack (#1755 re-run, SENT BACK 4th time)** | val=57.34, test=53.69; +0.44 above new 56.90 δ=0.3 baseline. Compound n160+δ=0.3 likely beats baseline (orthogonal levers) |
| n_hidden=192 confirmed dead (#1755 Arm B, lr=4e-4) | Budget cliff, grad_norm instability; 2× regression evidence |
| Lion lr=2e-4 wins on Huber δ=0.5 stack (#1782, SENT BACK) | LR optimum shifted 2.5e-4→2e-4; needs re-run on new δ=0.3 stack |
| Dropout=0.1 → −5.7% val on OLD 66.32 baseline (#1656, SENT BACK) | Feature-level regularization works; needs re-run on δ=0.3 stack |

## Active PRs

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #1979 | alphonse | n_layers=6 depth sweep, epochs=14 (budget-safe) | WIP — new | Beat 56.90 |
| #1879 | tanjiro | Huber δ=0.5 + epochs=16 compound (now on δ=0.3 default) | WIP (stale, baseline notification sent) | Beat 56.90 |
| #1782 | frieren | Lion lr=2e-4 single-arm on δ=0.3 stack | WIP (sent back 3rd time) | Beat 56.90 |
| #1755 | fern | n_hidden=160 final-gate single-arm on δ=0.3+epochs=16 stack | WIP (sent back 4th time — FINAL gate) | Beat 56.90 |
| #1844 | askeladd | Lion β2: 0.99→0.999 (slower momentum for B=4 noise), epochs=16 | WIP (baseline notification sent) | Beat 56.90 |
| #1656 | thorfinn | Dropout=0.1 single-arm on δ=0.3 stack | WIP (sent back) | Beat 56.90 |
| #1481 | nezuko | slice_num=128, epochs=16 | WIP (baseline notification sent) | Beat 56.90 |
| #1470 | edward | Instance-norm loss, epochs=16 | WIP (baseline notification sent) | Beat 56.90 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #1880 | alphonse | **MERGED** | Huber δ=0.3 → **new baseline 56.90/53.20** (−14.2% val). δ=0.2 essentially tied. δ curve bottomed. |
| #1639 | alphonse | **MERGED** | Huber δ=0.5 → baseline 66.32 (−9.3%). Uniformly better. δ curve not bottomed out at time. |
| #1780 | tanjiro | **MERGED** | Lion+epochs=16 → baseline 66.44 (−9.2%). Epochs=16 now structural standard. |
| #1782 (2nd) | frieren | SENT BACK | lr=2e-4 wins on δ=0.5 stack (val=58.00); above new 56.90 baseline; needs δ=0.3 re-run |
| #1656 | thorfinn | SENT BACK | Dropout=0.1 → val=62.52 on OLD baseline; above new 56.90; needs δ=0.3 re-run |
| #1463 | askeladd | CLOSED | SWA → val regression; SWALR fights Lion cosine |
| #1641 | frieren | **MERGED** | Lion optimizer → baseline 73.15 (−22.4%). Largest single-PR gain. |

## Open questions from active experiments

1. **Does n_layers=6 help on current stack?** (#1979 alphonse — depth vs width)
2. **Does n_hidden=160 compose with δ=0.3+epochs=16?** (#1755 fern) — width gain was ~1.7 on old stack
3. **Does Lion lr=2e-4 beat lr=3e-4 on δ=0.3 stack?** (#1782 frieren) — optimum shifted down before, may shift again
4. **Does dropout=0.1 compose with δ=0.3?** (#1656 thorfinn) — orthogonal regularization axes
5. **Does Lion β2=0.999 help at B=4?** (#1844 askeladd) — slower momentum for noisy small-batch
6. **Does slice_num=128 help?** (#1481 nezuko)
7. **Does instance-norm loss help val_re_rand?** (#1470 edward)
8. **Does Huber+epochs=16 compose when δ=0.3 is the default?** (#1879 tanjiro — should auto-pick up δ=0.3 after rebase)

## Confirmed dead ends

- **SWA mid-training (#1463)**: regresses in 13-epoch monotonic regime. Partial camber_rc signal — revisit at 24+ epochs.
- **LR/clip ceiling at AdamW stage (#1683)**: both 2× arms regress on test (renorm-ceiling). Obsoleted by Lion switch.
- **EMA decay=0.999 (#1596)**: 13-epoch monotonic regime; early averaging always hurts.
- **n_hidden=192 (#1755 Arm B, lr=4e-4)**: Budget cliff + grad_norm instability at lr=4e-4. 2× regression evidence.
- **Huber δ=0.1**: δ=0.3 and δ=0.2 essentially tied; further reduction into δ<0.2 will degrade cruise/re_rand splits due to over-saturation of low-std residuals into linear regime.

## Next hypotheses to queue (when students go idle)

### Currently active (don't duplicate)
- #1979 alphonse: n_layers=6, epochs=14
- #1879 tanjiro: Huber+epochs=16 (now effectively tests δ=0.3 compound after rebase)
- #1782 frieren: lr=2e-4 on δ=0.3 stack
- #1755 fern: n_hidden=160 on δ=0.3+epochs=16 stack
- #1844 askeladd: Lion β2=0.999, epochs=16
- #1656 thorfinn: dropout=0.1 on δ=0.3 stack
- #1481 nezuko: slice_num=128, epochs=16
- #1470 edward: instance-norm loss, epochs=16

### Queued ideas (when students finish above)

1. **surf_weight=15 or 5 under δ=0.3+Lion** — optimal weighting may shift with aggressive Huber capping.
2. **batch=8 + Lion + epochs=13** — larger effective batch; ~30 min with batch=8.
3. **DropPath / stochastic depth** — targeted at Transolver's residual structure; complements dropout.
4. **Activation sweep** — GELU → SwiGLU/SiLU on full combined stack.
5. **Layer-wise LR decay** — different LR per Transolver layer.
6. **Per-channel Huber δ** — δ_p=0.2, δ_U=0.5: decouple the cruise/single tradeoff observed in alphonse's δ scan.
7. **EMA post-convergence (last 2 epochs only)** — avoids #1463 failure mode; averages only the final stable checkpoints.
8. **Lion lr=1.75e-4** — if frieren's re-run confirms 2e-4 beats 3e-4 on δ=0.3, probe further down the LR curve.
