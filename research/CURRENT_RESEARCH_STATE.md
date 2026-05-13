# SENPAI Research State

- 2026-05-13 10:25
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **52.63** | #1656 (dropout=0.1 on Lion lr=2e-4 + per-channel δ + n_hidden=160, merged 2026-05-13) |
| **test_avg/mae_surf_p** | **49.22** | #1656 — all 4 splits finite |
| Peak VRAM | ~38 GB | #1656 — BF16, batch=4, n_hidden=160, dropout=0.1 |
| s/epoch | ~117 s | #1656 — 16 epochs ≈ 31 min total |

Merged stack: warmup3+cosine + GT-NaN fix + grad_clip(max_norm=1.0) + **Lion(lr=2e-4, wd=6e-5)** + **BF16 autocast** + **per-channel Huber δ=[Ux=0.5, Uy=0.5, p=0.2]** + **n_hidden=160** + **dropout=0.1**, epochs=**16**, batch=4, seed=42.

**Reproduce current best (explicit Lion flags required — train.py defaults are stale):**
```bash
cd target/ && python train.py --epochs 16 --lion_lr 2e-4 --lion_weight_decay 6e-5 --experiment_name dropout01_pcd_lr2e4_check --agent <student>
```

### Per-split val/test (new baseline, PR #1656)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 56.52 | 47.14 |
| geom_camber_rc | 67.35 | 59.44 |
| geom_camber_cruise | 34.17 | 46.76 |
| re_rand | 52.50 | 43.54 |
| **avg** | **52.63** | **49.22** |

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
| **Lion lr=2e-4 on per-channel δ+n160 → −1.6% val further (#2027, MERGED)** | **LR optimum continues moving down as loss landscape tightens. Compounds with #2028. 3/4 val splits improve.** |
| **Dropout=0.1 → −0.27% val further (#1656, MERGED)** | **Feature-level masking adds additive gain on top of gradient/loss/width regularization. Diminishing returns: 5.7%→1.0%→0.27% across 3 compound stacks. Test side stronger (−0.41%). Val curve still descending at epoch 16. New baseline 52.63/49.22.** |
| Lion lr=3.5e-4 plateau on n160+δ=0.3 stack (#2035, CLOSED) | val=55.90 (flat vs 55.92). LR bowl wide-flat in 3.0–3.5e-4. Higher LR helps easy split, hurts 3 OOD splits — mild over-stepping. Mechanism: wider model over-rides δ-driven LR shift. |
| slice_num=128 → +22.5% regression (#1481, CLOSED) | 41% per-epoch slowdown → 13 epochs only; same budget-cliff failure as n_hidden=192 |
| LR/clip ceiling confirmed (#1683, CLOSED) | Optimization-side knobs tapped out at AdamW stage |
| SWA mid-training regresses +4.1% (#1463, CLOSED) | Averages early bad checkpoints; SWALR fights Lion cosine |
| EMA decay=0.999 regresses +16.1% (#1596, CLOSED) | 13-epoch monotonic regime: early averaging always hurts |
| n_hidden=192 confirmed dead (#1755 Arm B, lr=4e-4) | Budget cliff, grad_norm instability; 2× regression evidence |
| Lion lr=2e-4 wins on δ=0.5 stack but LOSES on δ=0.3 (#1782 final, CLOSED) | **LR optimum is non-monotone in δ**: MSE→δ=0.5 moved DOWN, δ=0.5→δ=0.3 reversed UP. Mechanism: more residuals enter quadratic regime → smaller per-step → need higher LR |
| Lion lr=2e-4 beats lr=3e-4 on n160+uniform-δ=0.3 stack (#2027, SENT BACK) | val 52.795 < 55.92 on old code — strong signal. But ran on old uniform-δ codebase; needs rerun on current per-channel δ stack |
| Instance-norm dead end (#1470, CLOSED) | 1e-6 clamp → 1271-2230× amplification on near-uniform low-Re samples |
| **Dropout=0.1 composes with all current regularizers (MERGED #1656)** | 3 reruns across progressively stronger stacks: −5.7% on δ=0.5 stack, −1.0% on δ=0.3+n160+lr=3e-4, −0.27% on per-ch δ+n160+lr=2e-4. Strictly additive — regularization axes non-redundant. Final val=52.63/test=49.22. |

## Active PRs

| PR | Student | Hypothesis | Status | Target |
|---|---|---|---|---|
| #2176 | fern | SiLU activation swap: GELU → SiLU in MLP blocks | WIP — new | Beat 52.63 |
| #2100 | tanjiro | Lion lr=1.5e-4 bracket-from-below on per-channel δ + n_hidden=160 | WIP | Beat 52.63 |
| #2177 | edward | Lion weight_decay sweep at lr=2e-4: wd=4e-5 (Arm A) and wd=8e-5 (Arm B) | WIP — new | Beat 52.63 |
| #2084 | frieren | Cosine LR floor: eta_min=lr×0.05 to prevent zero-LR at epoch 16 | WIP | Beat 52.63 |
| #2005 | nezuko | surf_weight sweep: 15 vs 5 on δ=0.3+Lion+n160 stack | WIP (stale baseline) | Beat 52.63 |
| #1979 | alphonse | n_layers=6 depth sweep, epochs=14 (budget-safe) | WIP (stale baseline) | Beat 52.63 |
| #1844 | askeladd | Lion β2: 0.99→0.999 (slower momentum for B=4 noise), epochs=16 | WIP (stale baseline) | Beat 52.63 |
| #2161 | thorfinn | MLP dropout=0.1 (Arm A) + attention dropout=0.05 rate sweep (Arm B) on merged dropout=0.1 stack | WIP | Beat 52.63 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #1656 | thorfinn | **MERGED** | Dropout=0.1 on Lion lr=2e-4 + per-channel δ + n_hidden=160 → **new baseline 52.63/49.22** (−0.27% val, −0.41% test). Feature-level masking adds strictly additive gain. Val curve still descending at epoch 16. |
| #2027 | tanjiro | **MERGED** | Lion lr=2e-4 on per-channel δ + n_hidden=160 → baseline 52.78/49.42 (−1.6% val, −0.5% test). 3/4 val splits improve. Compounds with #2028. |
| #2028 | fern | **MERGED** | Per-channel Huber δ=[Ux=0.5,Uy=0.5,p=0.2] → baseline 53.62/49.65 (−4.1% val, −4.4% test). Uniform gain across all 8 splits. |
| #2035 | frieren | CLOSED | lr=3.5e-4 val=55.90 (flat vs 55.92 old baseline). LR plateau confirmed; bowl wide-flat at 3.0–3.5e-4. Split pattern reveals over-stepping on OOD. |
| #1755 | fern | **MERGED** | n_hidden=160 + δ=0.3 → baseline 55.92/51.92 (−1.7% val, −2.4% test). val_geom_camber_rc −5.38. |
| #1879 | tanjiro | CLOSED | Huber δ=0.3+ep16 compound reproduced baseline exactly (bit-identical); hypothesis absorbed by #1880 |
| #1880 | alphonse | **MERGED** | Huber δ=0.3 → baseline 56.90/53.20 (−14.2% val). δ=0.2 essentially tied. δ curve bottomed. |
| #1470 | edward | CLOSED | Instance-norm loss → val=59.02 (+3.7%). 1e-6 clamp let inst_scale reach 2230× on near-uniform low-Re samples |
| #1782 (3rd) | frieren | CLOSED | lr=2e-4 on δ=0.3 → val=58.82 (+1.92). LR optimum reversed direction (DOWN then UP); mechanism: δ-driven residual-regime shift |
| #1656 | thorfinn | **MERGED** | See above — new baseline. |
| #2074 | fern | CLOSED | δ_p sweep (0.15, 0.10) on lion_lr=1.5e-4 stack. Both lose: val 53.54/53.19 vs 52.63 baseline; **test regresses** +0.98%/+1.41%. val/test gap shrinks (−3.97 → −2.84) = over-regularization signal. δ_p=0.20 confirmed optimum; lower δ_p over-saturates pressure gradients into linear regime. |
| #2044 | edward | CLOSED | DropPath rates 0.05/0.10 catastrophically regress: val 67.40/72.80 vs 55.92 old baseline (+20.6%/+30.2%); test +17.5%/+27.7%. Mechanism: stochastic-depth ensemble needs more epochs to converge than 16-epoch/30-min budget allows. DropPath wrong-shape for this budget; within-layer dropout (already merged #1656) is the right regularization axis. |
| #1481 | nezuko | CLOSED | slice_num=128 → +22.5% regression; budget cliff (144s/epoch → 13 epochs only) |

## Open questions from active experiments

1. **Does SiLU activation beat GELU on the merged stack?** (#2176 fern, new) — simple, orthogonal to all regularization. Expected 1-3% if it composes; flat otherwise. δ_p sweep (PR #2074) answered: δ_p=0.20 is optimal.
2. **Does Lion lr=1.5e-4 continue beating lr=2e-4 (bracket-from-below)?** (#2100 tanjiro) — LR optimum has moved 3e-4 → 2e-4; testing if it keeps falling toward original Lion paper defaults
3. **Does n_layers=6 help on n_hidden=160 stack?** (#1979 alphonse — depth vs width at current baseline)
4. **Does cosine LR floor (eta_min=lr×0.05) prevent over-decay and improve final-epoch performance?** (#2084 frieren) — epoch 16 always best, curve still descending; floor at 1.5e-5 may squeeze more improvement
5. **Does lion_weight_decay=4e-5 or 8e-5 beat 6e-5 at lr=2e-4?** (#2177 edward, new) — wd was set when lr=3e-4; LR halving may have shifted wd optimum. (#2044 DropPath closed as wrong-shape for budget.)
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

1. **Dropout sweep (0.05 / 0.15)** — thorfinn's suggested follow-up. Diminishing returns at 0.1 suggests 0.05 might be closer to optimum on the fully-regularized stack. Clean 1-arm per PR.
2. **MLP dropout** — thorfinn flagged that the MLP class doesn't accept dropout kwarg. Adding dropout between GELU and post-linear in MLP is the "MLP half" of the original hypothesis — separate PR, small code change.
3. **SiLU activation** — swap GELU → SiLU in MLP blocks. Simple, orthogonal, potentially 1–3% gain. Well-tested in transformers.
4. **Weight decay sweep at lr=2e-4** — current wd=6e-5; probe wd=4e-5 and wd=8e-5 to find the optimum weight decay for the shrunken LR.
5. **batch=8** — larger effective batch on n_hidden=160; ~30 min with batch=8 (fewer steps, each less noisy). Tests Lion's sign-voting with reduced gradient noise at B=8.
6. **Layer-wise LR decay** — different LR per Transolver layer. Empirical finding in BERT: outer layers need smaller LR. With 5 layers, a decay factor 0.8–0.9 is a reasonable first arm.
7. **n_layers=6 + dropout compound** — after alphonse's depth result lands, test n_layers=6 + dropout=0.1 compound.
8. **Dropout=0.1 + DropPath compound** — if edward's DropPath arm wins, test stacking both regularization axes (feature + block).
9. **EMA post-convergence (last 2 epochs only)** — avoids #1463 failure mode; averages final stable checkpoints where val has converged.
10. **Pre-residual RevIN normalization** — principled fix to instance-norm failure mode. Lower priority but valuable for paper's ablation section.
