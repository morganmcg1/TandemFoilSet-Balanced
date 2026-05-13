# SENPAI Research State

- 2026-05-13 08:10
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **53.62** | #2028 (per-channel Huber δ=[Ux=0.5,Uy=0.5,p=0.2], merged 2026-05-13) |
| **test_avg/mae_surf_p** | **49.65** | #2028 — all 4 test splits finite |
| Peak VRAM | 37.99 GB | #2028 — BF16, batch=4, n_hidden=160 |
| s/epoch | ~115 s | n_hidden=160 + Lion, 16 epochs ≈ 30.7 min |

Merged stack: warmup3+cosine + GT-NaN fix + grad_clip(max_norm=1.0) + **Lion(lr=3e-4, wd=6e-5)** + **BF16 autocast** + **per-channel Huber δ=[Ux=0.5, Uy=0.5, p=0.2]** + **n_hidden=160**, epochs=**16**, batch=4, seed=42.

**Reproduce current best:**
```bash
cd target/ && python train.py --epochs 16 --experiment_name pcd_baseline_check --agent <student>
```

### Per-split val/test (new baseline, PR #2028)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 58.46 | 48.40 |
| geom_camber_rc | 67.34 | 58.75 |
| geom_camber_cruise | 35.10 | 47.64 |
| re_rand | 53.58 | 43.83 |
| **avg** | **53.62** | **49.65** |

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
| **n_hidden=160 → −1.7% val further (#1755, MERGED)** | **Width gain orthogonal to loss-shape; val_geom_camber_rc (hardest OOD) benefits most (−5.38)** |
| **Per-channel Huber δ=[0.5,0.5,0.2] → −4.1% val further (#2028, MERGED)** | **Decoupling p vs Ux/Uy δ: pressure tight (0.2), velocity expanded (0.5). Uniform improvement all 8 splits.** |
| Lion lr=3.5e-4 plateau on n160+δ=0.3 stack (#2035, CLOSED) | val=55.90 (flat vs 55.92). LR bowl wide-flat in 3.0–3.5e-4. Higher LR helps easy split, hurts 3 OOD splits — mild over-stepping. Mechanism: wider model over-rides δ-driven LR shift. |
| slice_num=128 → +22.5% regression (#1481, CLOSED) | 41% per-epoch slowdown → 13 epochs only; same budget-cliff failure as n_hidden=192 |
| LR/clip ceiling confirmed (#1683, CLOSED) | Optimization-side knobs tapped out at AdamW stage |
| SWA mid-training regresses +4.1% (#1463, CLOSED) | Averages early bad checkpoints; SWALR fights Lion cosine |
| EMA decay=0.999 regresses +16.1% (#1596, CLOSED) | 13-epoch monotonic regime: early averaging always hurts |
| n_hidden=192 confirmed dead (#1755 Arm B, lr=4e-4) | Budget cliff, grad_norm instability; 2× regression evidence |
| Lion lr=2e-4 wins on δ=0.5 stack but LOSES on δ=0.3 (#1782 final, CLOSED) | **LR optimum is non-monotone in δ**: MSE→δ=0.5 moved DOWN, δ=0.5→δ=0.3 reversed UP. Mechanism: more residuals enter quadratic regime → smaller per-step → need higher LR |
| Lion lr=2e-4 beats lr=3e-4 on n160+uniform-δ=0.3 stack (#2027, SENT BACK) | val 52.795 < 55.92 on old code — strong signal. But ran on old uniform-δ codebase; needs rerun on current per-channel δ stack |
| Instance-norm dead end (#1470, CLOSED) | 1e-6 clamp → 1271-2230× amplification on near-uniform low-Re samples |
| Dropout=0.1 → −5.7% val on OLD 66.32 baseline (#1656, SENT BACK) | Feature-level regularization works; needs re-run on δ=0.3 stack |

## Active PRs

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #2074 | fern | Per-channel δ refinement: δ_p=0.15 (Arm A) and δ_p=0.10 (Arm B) | WIP — new | Beat 53.62 |
| #2027 | tanjiro | Lion lr=2e-4 rerun on current per-channel δ stack (rebase+rerun) | WIP (sent back) | Beat 53.62 |
| #2044 | edward | DropPath / stochastic depth (rates 0.05, 0.1) on n_hidden=160 | WIP | Beat 53.62 |
| #2084 | frieren | Cosine LR floor: eta_min=lr×0.05 to prevent zero-LR at epoch 16 | WIP — new | Beat 53.62 |
| #2005 | nezuko | surf_weight sweep: 15 vs 5 on δ=0.3+Lion+n160 stack | WIP | Beat 53.62 |
| #1979 | alphonse | n_layers=6 depth sweep, epochs=14 (budget-safe) | WIP (baseline updated) | Beat 53.62 |
| #1844 | askeladd | Lion β2: 0.99→0.999 (slower momentum for B=4 noise), epochs=16 | WIP (baseline updated) | Beat 53.62 |
| #1656 | thorfinn | Dropout=0.1 single-arm on δ=0.3 stack | WIP (sent back) | Beat 53.62 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #2028 | fern | **MERGED** | Per-channel Huber δ=[Ux=0.5,Uy=0.5,p=0.2] → **new baseline 53.62/49.65** (−4.1% val, −4.4% test). Uniform gain across all 8 splits. |
| #2035 | frieren | CLOSED | lr=3.5e-4 val=55.90 (flat vs 55.92 old baseline). LR plateau confirmed; bowl wide-flat at 3.0–3.5e-4. Split pattern reveals over-stepping on OOD. |
| #2027 | tanjiro | SENT BACK | lr=2e-4 beats old baseline (52.795 < 55.92 on uniform-δ=0.3 code) but ran before #2028 merged. Needs rerun on current per-channel δ stack. |
| #1755 | fern | **MERGED** | n_hidden=160 + δ=0.3 → baseline 55.92/51.92 (−1.7% val, −2.4% test). val_geom_camber_rc −5.38. |
| #1879 | tanjiro | CLOSED | Huber δ=0.3+ep16 compound reproduced baseline exactly (bit-identical); hypothesis absorbed by #1880 |
| #1880 | alphonse | **MERGED** | Huber δ=0.3 → baseline 56.90/53.20 (−14.2% val). δ=0.2 essentially tied. δ curve bottomed. |
| #1470 | edward | CLOSED | Instance-norm loss → val=59.02 (+3.7%). 1e-6 clamp let inst_scale reach 2230× on near-uniform low-Re samples |
| #1782 (3rd) | frieren | CLOSED | lr=2e-4 on δ=0.3 → val=58.82 (+1.92). LR optimum reversed direction (DOWN then UP); mechanism: δ-driven residual-regime shift |
| #1656 | thorfinn | SENT BACK | Dropout=0.1 → val=62.52 on OLD baseline; above new 53.62; needs δ=0.3 stack re-run |
| #1481 | nezuko | CLOSED | slice_num=128 → +22.5% regression; budget cliff (144s/epoch → 13 epochs only) |

## Open questions from active experiments

1. **Does per-channel δ_p=0.15 or 0.10 beat δ_p=0.20?** (#2074 fern) — pressure δ response surface not mapped below 0.2
2. **Does Lion lr=2e-4 beat lr=3e-4 on the combined per-channel δ + n160 stack?** (#2027 tanjiro rerun) — strong signal from old stack; confirmation run needed
3. **Does n_layers=6 help on n_hidden=160 stack?** (#1979 alphonse — depth vs width at current baseline)
4. **Does cosine LR floor (eta_min=lr×0.05) prevent over-decay and improve final-epoch performance?** (#2084 frieren) — epoch 16 always best, curve still descending; floor at 1.5e-5 may squeeze more improvement
5. **Does dropout=0.1 compose with per-channel δ+n160?** (#1656 thorfinn) — orthogonal regularization axes
6. **Does Lion β2=0.999 help at B=4?** (#1844 askeladd) — slower momentum for noisy small-batch
7. **Does surf_weight shift from 10.0 under per-channel δ+Lion+n160?** (#2005 nezuko) — loss balance may have changed
8. **Does DropPath (0.05, 0.1) help generalisation on Transolver residual structure?** (#2044 edward) — orthogonal to dropout

## Confirmed dead ends

- **SWA mid-training (#1463)**: regresses in 13-epoch monotonic regime. Partial camber_rc signal — revisit at 24+ epochs.
- **LR/clip ceiling at AdamW stage (#1683)**: both 2× arms regress on test (renorm-ceiling). Obsoleted by Lion switch.
- **EMA decay=0.999 (#1596)**: 13-epoch monotonic regime; early averaging always hurts.
- **n_hidden=192 (#1755 Arm B, lr=4e-4)**: Budget cliff + grad_norm instability at lr=4e-4. 2× regression evidence.
- **Lion lr=3.5e-4 on n160 (#2035)**: val=55.90 (flat vs 55.92). LR bowl wide-and-flat in 3.0–3.5e-4. Higher LR helps easy split, hurts OOD splits — mild over-stepping. Do not probe lr≥4e-4. LR optimum reversal mechanism was narrow-model-specific; wider model over-rides it.
- **Huber δ=0.1 (uniform)**: δ=0.3 and δ=0.2 essentially tied for uniform scalar; further reduction into δ<0.2 will degrade cruise/re_rand splits due to over-saturation. BUT: per-channel δ_p=0.1 or 0.15 may still be optimal when velocity is separately set to 0.5 — this is what #2074 tests.
- **Instance-norm loss with 1e-6 clamp (#1470)**: +3.7% val regression. Near-uniform low-Re samples (y_std ≈ 5e-4) got amplified 1000-2000×.
- **slice_num=128 (#1481)**: +22.5% val regression. 41% per-epoch slowdown → budget cliff.

## Queued ideas (when students finish above)

1. **Lion lr=2e-4 + per-channel δ compound** — highest priority if tanjiro's #2027 rerun confirms; next probe is wd=4.5e-5 at lr=2e-4
2. **Cosine LR floor (η_min = lr×0.05)** — prevents over-suppression at epoch 16 tail. Simple 1-line change. Orthogonal to δ/lr.
3. **batch=8 + epochs=13** — larger effective batch on n_hidden=160; ~30 min with batch=8. Tests whether Lion's sign-voting benefits from reduced noise at B=8.
4. **Activation sweep** — GELU → SiLU in MLP blocks. Simple, well-tested in transformers.
5. **n_layers=6 + n_hidden=160 compound** — test depth×width compound after alphonse's n_layers=6 result lands.
6. **SiLU activation** — swap GELU → SiLU in MLP blocks. Simple, orthogonal to all current changes, potentially 1–3% gain.
7. **Layer-wise LR decay** — different LR per Transolver layer.
8. **EMA post-convergence (last 2 epochs only)** — avoids #1463 failure mode; averages only the final stable checkpoints.
9. **Weight decay sweep at lr=2e-4** — if lr=2e-4 confirms: wd=4.5e-5 / 6e-5 / 8e-5 to couple wd with new lr.
10. **Pre-residual RevIN normalization** — edward's principled fix to instance-norm failure. Unlikely to clear 53.62 but principled for paper.
