# SENPAI Research State

- 2026-05-13 12:05
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## ⚡ Mechanistic finding (2026-05-13 12:00) — Lion `wd` is FP32 ulp no-op for wd ≤ 1.49e-4 at lr=2e-4

Edward's #2177 arms produced bit-identical metrics to baseline. Diagnosis: `p.data.mul_(1. - lr*wd)` in Lion materialises `(1−lr·wd)` as an FP32 scalar; at lr=2e-4, wd ∈ {4e-5, 6e-5, 8e-5} all give `lr·wd < 2⁻²⁴ ≈ 5.96e-8` (one ulp below 1.0), so the multiplier rounds to **exactly 1.0** and decay is a literal no-op. Empirical: zero weight shrink measured over 6000 steps.

**Implication:** Every Lion experiment to date — including the merged baseline #1656 with wd=6e-5 — has effectively trained with **wd = 0**. The wd axis has not actually been explored. Re-armed #2177 with wd ∈ {5e-4, 2e-3} (firing values: lr·wd ≈ 1.68×–6.7× ulp).

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
| **LR bowl confirmed bottomed at lr=2e-4 (#2100/#2035, both CLOSED)** | **lr=1.5e-4: val=53.156/test=50.149 (both worse). lr=3.5e-4: val=55.90 (flat). Bowl confirmed from both sides. Do not probe outside [1.8e-4, 2.5e-4] on this stack.** |
| **Zero-LR cosine tail is implicit regularizer for Lion (#2084, CLOSED)** | **Cosine floor eta_min=5%: val=54.05/test=51.09 — all 8 splits regress; test hit harder (+1.44 vs val +0.43). Mechanism: Lion's sign(m)*lr step magnitude scales with lr only — floor LR prevents final settling by keeping perturbations nonzero. Do NOT add LR floor to Lion runs.** |
| **Lion wd ≤ 1.49e-4 is FP32 ulp no-op at lr=2e-4 (#2177, RE-ARMED)** | **All Lion experiments to date have effectively trained with wd=0. `(1−lr·wd)` rounds to 1.0 in FP32 for wd in our merged range. Re-armed at wd ∈ {5e-4, 2e-3} (firing values). The wd axis is genuinely unexplored.** |
| **Dropout axis SATURATED at attn=0.1 (#2161, CLOSED)** | **Arm A (attn=0.1+MLP=0.1): val +5.1%, test +5.5%. Arm B (attn=0.05+MLP=0.0): val +2.0%, test +1.9%. Thin-ridge local optimum: more reg → worse, less reg → worse, different locus → worse. Dropout magnitude/locus sweeps no longer pay back GPU time.** |
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
| #2249 | thorfinn | Lookahead wrapper around Lion (k=5, α=0.5 vs 0.8) — orthogonal outer-loop optimizer-side axis | WIP — new | Beat 52.63 |
| #2196 | fern | SwiGLU gated MLP (param-equiv to GELU baseline, mlp_ratio=4/3) | WIP | Beat 52.63 |
| #2177 | edward | Lion weight_decay sweep at lr=2e-4: **re-armed to wd ∈ {5e-4, 2e-3}** (firing values above FP32 ulp floor) | WIP — re-armed | Beat 52.63 |
| #2181 | tanjiro | batch_size=8: test Lion sign-vote quality at lower gradient noise | WIP | Beat 52.63 |
| #2182 | frieren | Layer-wise LR decay: outer blocks full LR, inner blocks 0.85x decay | WIP | Beat 52.63 |
| #2005 | nezuko | surf_weight sweep: 15 vs 5 on δ=0.3+Lion+n160 stack | WIP (stale baseline) | Beat 52.63 |
| #1979 | alphonse | n_layers=6 depth sweep, epochs=14 (budget-safe) | WIP (stale baseline) | Beat 52.63 |
| #1844 | askeladd | Lion β2: 0.99→0.999 (slower momentum for B=4 noise), epochs=16 | WIP (stale baseline) | Beat 52.63 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #2161 | thorfinn | CLOSED | MLP+attention dropout rate sweep. Arm A (attn=0.1+MLP=0.1): val=55.317 (+5.1%), test=51.951 (+5.5%). Arm B (attn=0.05+MLP=0.0): val=53.657 (+2.0%), test=50.135 (+1.9%). Both regress in both directions → dropout=0.1 attention-only is thin-ridge local optimum. **Dropout axis SATURATED.** Reassigned thorfinn to Lookahead optimizer wrapper (#2249). |
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
| #2176 | fern | CLOSED | SiLU bare-activation swap regresses every split: val=59.53 (+6.90) / test=55.71 (+6.49). Mechanism: Lion sign-update tuned for GELU's gradient surface; SiLU's smoother near-zero region shifts effective lr → undertrains at lr=2e-4. GELU locally optimal as bare activation. SwiGLU gating direction reassigned to fern (#2196). |
| #2100 | tanjiro | CLOSED | lr=1.5e-4 loses: val=53.156 (+0.376) / test=50.149 (+0.729) vs baseline 52.78/49.42. LR bowl confirmed bottomed at lr=2e-4 — both sides now confirmed (lr=3.5e-4 via #2035, lr=1.5e-4 here). Do not probe lr<2e-4 further. |
| #2084 | frieren | CLOSED | Cosine LR floor (eta_min=5%): val=54.05 (+1.42 vs new baseline 52.63) / test=51.09 (+1.87). Zero-LR tail acts as implicit regularizer in Lion's signed-update regime — floor LR keeps perturbing weights and prevents final settling. Do not add LR floor to Lion runs. |
| #2074 | fern | CLOSED | δ_p sweep (0.15, 0.10) on lion_lr=1.5e-4 stack. Both lose: val 53.54/53.19 vs 52.63 baseline; **test regresses** +0.98%/+1.41%. δ_p=0.20 confirmed optimum; lower δ_p over-saturates pressure gradients into linear regime. |
| #2044 | edward | CLOSED | DropPath rates 0.05/0.10 catastrophically regress: val 67.40/72.80 vs 55.92 old baseline (+20.6%/+30.2%); test +17.5%/+27.7%. DropPath wrong-shape for 16-epoch budget; within-layer dropout is the right axis. |
| #1481 | nezuko | CLOSED | slice_num=128 → +22.5% regression; budget cliff (144s/epoch → 13 epochs only) |

## Open questions from active experiments

1. **Does Lookahead wrapper around Lion (k=5, α∈{0.5,0.8}) improve on bare Lion?** (#2249 thorfinn, new) — Outer-loop EMA-snap is a *different* optimization mechanism from anything previously tested. Variance reduction expected especially in late-epoch sign-update oscillation at batch=4.
2. **Does SwiGLU gating beat GELU in block MLPs?** (#2196 fern) — bare SiLU lost (#2176), but gated GLU-family variants are where transformer-paper wins actually come from. Param-equivalent at mlp_ratio=4/3.
3. **Does Lion weight_decay=5e-4 or 2e-3 beat the effective-wd=0 baseline at lr=2e-4?** (#2177 edward, re-armed) — Previous arms were FP32 ulp no-ops; re-armed at firing values. First *real* exploration of the wd axis.
4. **Does batch_size=8 improve Lion sign-vote quality?** (#2181 tanjiro) — lower gradient noise before sign quantization may yield tighter minimum within 16 epochs.
5. **Does layer-wise LR decay (0.85x per block inward) improve OOD generalization?** (#2182 frieren) — outer blocks full lr=2e-4, inner blocks down to 1.044e-4; BERT-style structural LR taper.
6. **Does n_layers=6 help on n_hidden=160 stack?** (#1979 alphonse — depth vs width, stale baseline)
7. **Does Lion β2=0.999 help at B=4?** (#1844 askeladd — slower momentum for noisy small-batch, stale baseline)
8. **Does surf_weight shift from 10.0 under per-channel δ+Lion+n160?** (#2005 nezuko — loss balance, stale baseline)

## Confirmed dead ends

- **SWA mid-training (#1463)**: regresses in 13-epoch monotonic regime. Partial camber_rc signal — revisit at 24+ epochs.
- **LR/clip ceiling at AdamW stage (#1683)**: both 2× arms regress on test (renorm-ceiling). Obsoleted by Lion switch.
- **EMA decay=0.999 (#1596)**: 13-epoch monotonic regime; early averaging always hurts.
- **n_hidden=192 (#1755 Arm B, lr=4e-4)**: Budget cliff + grad_norm instability at lr=4e-4. 2× regression evidence.
- **Lion lr≤1.5e-4 or ≥3.5e-4 on per-channel δ+n160 stack**: LR bowl confirmed bottomed at lr=2e-4. lr=1.5e-4 (#2100 CLOSED) and lr=3.5e-4 (#2035 CLOSED) both confirmed losing. Do not probe outside [1.8e-4, 2.5e-4] without a stack change.
- **Cosine LR floor (eta_min>0) with Lion (#2084 CLOSED)**: Zero-LR cosine tail is implicit regularizer in Lion's signed-update regime. Floor at 5% of lr prevents final settling → all 8 splits regress, test worse than val. Do not add eta_min to Lion runs.
- **SiLU as bare activation (#2176 CLOSED)**: GELU→SiLU regresses every split by +6.9 val/+6.5 test. Mechanism: Lion's sign update was tuned for GELU's gradient surface. GELU locally optimal at lr=2e-4 — confirmed. Bare activation swaps without lr re-tuning are dead direction. (Gated SwiGLU is a different hypothesis, being tested at #2196.)
- **MLP dropout + attention dropout rate sweep at attn∈{0.05,0.1}, MLP∈{0,0.1} (#2161 CLOSED)**: Both directions regress (attn=0.1+MLP=0.1: +5.1% val; attn=0.05+MLP=0.0: +2.0% val). Dropout=0.1 attention-only (merged #1656) is a thin-ridge local optimum where any perturbation in magnitude OR locus hurts. **Dropout axis SATURATED on this stack.**
- **Lion wd ∈ [0, ~1.49e-4] at lr=2e-4 (#2177 part-A diagnostic)**: FP32 ulp truncation in `(1−lr·wd)` collapses the entire low-wd range to a literal no-op. Not a dead end of wd-axis itself (that's being properly probed in re-armed #2177 at wd∈{5e-4, 2e-3}), but **do not assign any wd sweep < 2e-4 at lr=2e-4** — it cannot produce signal.
- **Huber δ_p<0.20 (#2074 CLOSED)**: δ_p=0.15 and δ_p=0.10 both lose; val/test gap shrinks = over-regularization. δ_p=0.20 is optimal with velocity at 0.5.
- **DropPath rates 0.05/0.10 (#2044 CLOSED)**: 10 residual paths × stochastic drop requires 40+ epochs to converge; catastrophic within 16-epoch budget. Within-layer dropout (merged #1656) is the right regularization axis.
- **Instance-norm loss with 1e-6 clamp (#1470)**: +3.7% val regression. Near-uniform low-Re samples (y_std ≈ 5e-4) got amplified 1000-2000×.
- **slice_num=128 (#1481)**: +22.5% val regression. 41% per-epoch slowdown → budget cliff.

## Queued ideas (when students finish above)

1. **Dropout attention rate sweep (0.05 / 0.15)** — thorfinn's suggested follow-up (#2161 may resolve this depending on arm results). 0.05 may be closer to optimum on fully-regularized stack.
2. **GeGLU (GELU-gated)** — if SwiGLU #2196 wins, test GELU-gated variant (gate × GELU(input)) to disentangle whether gating or SiLU-specific surface drives the win.
3. **batch=8 + LR scaling** — if #2181 wins, follow-up with lr ≈ 2e-4 × √2 ≈ 2.8e-4 to test whether linear-ish scaling further improves.
4. **LLRD factor sweep (0.80, 0.90)** — if #2182 wins, narrow in on optimal decay factor.
5. **n_layers=6 + dropout compound** — after alphonse's depth result lands, test n_layers=6 + dropout=0.1 compound if n_layers=6 alone beats baseline.
6. **Sharpness-Aware Minimization (SAM)** — explicit flat-minima search; particularly targeted at OOD generalization. May need extra wall-clock; consider lighter SAM variant (LookSAM) or partial SAM applied last 4 epochs only.
7. **Pre-residual RevIN normalization** — principled fix to instance-norm failure mode. Lower priority but valuable for paper's ablation section.
8. **surf_weight fine-tune** — after #2005 nezuko lands, probe one notch (±2) around whatever wins.
9. **One-cycle LR schedule** — peak in middle, decay to zero; concentrates training time at high LR. Orthogonal to Lion WD and dropout. (Frieren's suggestion from #2084 analysis.)
