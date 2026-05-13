# SENPAI Research State

- 2026-05-13 16:55
- No human researcher directives (no open issues)
- Round 5 Charlie no-W&B arm — 30-min wall-clock cap, local JSONL only

## ⚡ Mechanistic finding (2026-05-13 12:00) — Lion `wd` is FP32 ulp no-op for wd ≤ 1.49e-4 at lr=2e-4

Edward's #2177 arms produced bit-identical metrics to baseline. Diagnosis: `p.data.mul_(1. - lr*wd)` in Lion materialises `(1−lr·wd)` as an FP32 scalar; at lr=2e-4, wd ∈ {4e-5, 6e-5, 8e-5} all give `lr·wd < 2⁻²⁴ ≈ 5.96e-8` (one ulp below 1.0), so the multiplier rounds to **exactly 1.0** and decay is a literal no-op. Empirical: zero weight shrink measured over 6000 steps.

**Implication:** Every Lion experiment to date — including the merged baseline #1656 with wd=6e-5 — has effectively trained with **wd = 0**. The wd axis has not actually been explored. Re-armed #2177 with wd ∈ {5e-4, 2e-3} (firing values: lr·wd ≈ 1.68×–6.7× ulp).

## Merged baseline

| Metric | Value | PR |
|---|---|---|
| **val_avg/mae_surf_p** | **43.73** | #2405 (Lion β1=0.85, merged 2026-05-13) |
| **test_avg/mae_surf_p** | **41.86** | #2405 — all 4 splits finite; epoch-15 best checkpoint |
| Peak VRAM | ~42.5 GB | #2405 — unchanged from #2287 |
| s/epoch | ~126 s | #2287/#2405 — 15 epochs completed within 30-min cap |

Merged stack: warmup3+cosine + GT-NaN fix + grad_clip(max_norm=1.0) + **Lion(lr=2e-4, wd=6e-5, β1=0.85, β2=0.99)** + **BF16 autocast** + **per-channel Huber δ=[Ux=0.5, Uy=0.5, p=0.2]** + **n_hidden=160** + **GeGLU block-MLPs (hidden=216, gate×GELU)** + **dropout=0.1**, epochs=**16**, batch=4, seed=42.

**Note on wd:** wd=6e-5 is a FP32 ulp no-op at lr=2e-4 (effective wd=0). Real wd being probed on GeGLU stack at #2352.

**Reproduce current best:**
```bash
cd target/ && python train.py --epochs 16 --lion_lr 2e-4 --lion_weight_decay 6e-5 --lion_beta1 0.85 --experiment_name beta1_085_baseline_check --agent <student>
```

### Per-split val/test (new baseline, PR #2405, epoch 15)

| Split | val | test |
|---|---:|---:|
| single_in_dist | 48.34 | 41.42 |
| geom_camber_rc | 56.87 | 50.62 |
| geom_camber_cruise | 26.95 | 40.33 |
| re_rand | 42.77 | 35.09 |
| **avg** | **43.73** | **41.86** |

### Previous baseline (PR #2287 GeGLU, now superseded)
val=45.92, test=44.35 — beat this to confirm any stack improvement relative to pre-β1 work.

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
| **SwiGLU gated MLP → −9.9% val (#2196, MERGED)** | **val 52.63→47.43, test 49.22→45.01. All 8 splits improve. Gate mechanism: selective per-channel feature suppression. Hardest OOD split gained most.** |
| **GeGLU (gate×GELU) beats SwiGLU (gate×SiLU) → −3.2% val (#2287, MERGED)** | **val 47.43→45.92, test 45.01→44.35. All 8 splits improve. Mechanism: GELU inside gate aligns with Lion's optimizer-calibrated gradient surface. Resolves paradox: bare-SiLU regresses, gated-SiLU wins, gated-GELU wins more. Baseline 45.92/44.35.** |
| **Lion β1=0.85 → −4.8% val (#2405, MERGED)** | **val 45.92→43.73, test 44.35→41.86. All 4 splits improve. β1 direction-smoothness axis: lower β1 weights current gradient more in sign decision; β1=0.95 (inertial) catastrophically regresses +19.7%. At B=4 noisy regime, fresh gradient signal > EMA smoothing. New baseline 43.73/41.86.** |
| **LLRD factor=0.85 regresses +7% (#2182, CLOSED)** | **All 8 splits worse by 2.5–4.4 MAE. Lion sign-step is linearly LR-sensitive (no preconditioning) → 52% lr cut on input blocks = 52% step cut, no recovery. Transolver is too shallow (5 blocks vs BERT-12) for safe LLRD factor at 0.85.** |
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
| #2288 | frieren | Lion lr sweep on GeGLU+β1=0.85 baseline: Arm A=2.5e-4, Arm B=3e-4 | WIP (stale) | Beat 43.73 |
| #2460 | fern | LayerScale γ=0.1 on attn+ffn residuals (zero-rank-loss selective suppression) | WIP — new | Beat 43.73 |
| #2403 | tanjiro | GeGLU mlp_ratio sweep — sent back, testing swiglu_hidden=256 (mlp_ratio≈1.6) | WIP (sent back) | Beat 43.73 |
| #2422 | edward | n_head sweep: 4→8 (more heads, smaller per-head dim, attention diversity test) | WIP | Beat 43.73 |
| #2424 | nezuko | n_layers=4 (cost-recovery probe vs #2349 n_layers=6 budget-cliff result) | WIP | Beat 43.73 |
| #2432 | thorfinn | slice_num=48 (15% per-epoch cost recovery, +2 cosine-tail epochs) | WIP | Beat 43.73 |
| #1979 | alphonse | n_layers=6 depth sweep (stale pre-β1 baseline; directionally informative) | WIP (stale) | Beat 43.73 |
| #2459 | askeladd | β1 lower-bound: β1∈{0.875, 0.80} to narrow optimum below 0.85 | WIP — new | Beat 43.73 |

## Recently closed/merged

| PR | Student | Outcome | Note |
|---|---|---|---|
| #2405 | askeladd | **MERGED** | Lion β1=0.85: val=43.73 (−4.8% vs 45.92), test=41.86 (−5.6%). Arm A (β1=0.85) clear winner; Arm B (β1=0.95) catastrophically regresses +19.7%. **New baseline 43.73/41.86.** Direction-smoothness axis: lower β1 → more reactive sign update → faster val convergence with B=4 noisy gradients. |
| #2401 | fern | CLOSED | GeGLU gate on `PhysicsAttention.to_out` (hidden=56): val=52.98 (+15.4% worse). Bottleneck rank loss (160→56→160) dominates gate benefit. Param-parity ≠ capability-parity. Epoch time +17% (131s vs 112s) → 2 fewer epochs at budget. Reassigned to LayerScale (#2460). |
| #2315 | thorfinn | CLOSED | RMSNorm: pod stalled. 0 commits, 0 comments, GPU dropped to 0% over 3.5h. Hypothesis untested. Replaced with simpler single-line slice_num=48 assignment (#2432). |
| #2403 | tanjiro | SENT BACK | swiglu_hidden=320 (mlp_ratio=2): val=48.13 (+4.8%), test=46.19 (+4.2%) at only 14/16 epochs (30-min cap hit). Per-epoch overhead 20%, not 5% as expected. Val −4.5/ep at termination (extrapolated 39–43 range at ep16). Inconclusive — sent back to test swiglu_hidden=256. |
| #2352 | edward | CLOSED | Lion wd sweep on GeGLU stack. Neither arm beats primary val. Arm A (wd=2e-3): val=46.49 (+1.21%); Arm B (wd=5e-3): val=45.96 (+0.08% noise) but **test=43.90 (−1.01% real)**. Param L2 grows ~58% from init regardless of wd — sign-update dominates, wd axis is shallow on this stack. |
| #2005 | nezuko | CLOSED | surf_weight=15 on GeGLU stack: val=46.90 (+2.13%), test=44.59 (+0.54%). Both axes regress. Mechanism: Lion's sign quantization makes loss-balance reweighting a weak knob — only changes which params get stepped, not step magnitude. |
| #2349 | fern | CLOSED | n_layers=6 GeGLU: val=50.80 (+10.6%), test=47.96 (+8.1%). Budget-starved: +18% per-epoch cost → only 12/13 epochs in 30-min cap; val still descending at −4.0/ep at termination. Depth axis alive but needs wall-clock headroom. |
| #2332 | tanjiro | CLOSED | SwiGLU preprocess entry projector: val=52.54 (+10.8%), test=50.07 (+11.2%). Gating at 24-dim input is information loss (no feature diversity for routing). Principle: gating works at scale (dim ≥ 160), not at low-dim entry. |
| #1844 | askeladd | CLOSED | β2=0.999 on GeGLU stack: val=48.83 (+6.3%), test=46.36 (+4.5%). ~10× longer EMA timescale → warmup cost dominates 30-min cap. β2=0.99 confirmed optimal for this regime. β1 axis reassigned. |
| #2287 | fern | **MERGED** | GeGLU (gate×GELU) vs SwiGLU (gate×SiLU) → **new baseline 45.92/44.35** (−3.2% val, −1.5% test). All 8 splits improve. Mechanism: GELU inside gate aligns with Lion optimizer surface. |
| #2196 | fern | **MERGED** | SwiGLU gated MLP (hidden=216, param-equiv to GELU) → baseline 47.43/45.01 (−9.9% val, −8.6% test). All 8 splits improve. |
| #2181 | tanjiro | CLOSED | batch=8 at lr=2e-4: val=64.91 (+36.9%). Epoch-budget cliff: halved step count with Lion's magnitude-invariant sign update → 2× under-training. |
| #2249 | thorfinn | CLOSED | Lookahead wrapper (k=5, α=0.5/0.8): Arm A val=57.61 (+21.5%), Arm B val=53.45 (+12.7%). Lookahead+Lion needs ≥30 epochs; 16-epoch cap insufficient. |
| #2182 | frieren | CLOSED | LLRD factor=0.85: all 8 splits regress +7.3% val. Lion sign-step linearly LR-sensitive — no recovery from 52% step reduction. |
| #2161 | thorfinn | CLOSED | MLP+attention dropout rate sweep. Arm A (attn=0.1+MLP=0.1): val=55.317 (+5.1%), test=51.951 (+5.5%). Arm B (attn=0.05+MLP=0.0): val=53.657 (+2.0%), test=50.135 (+1.9%). Both regress → dropout=0.1 attention-only is thin-ridge local optimum. **Dropout axis SATURATED.** |
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

1. **Does GeGLU gating in PhysicsAttention.to_out compound on block-MLP gating?** (fern, just assigned) — If the attention output path also benefits from selective gating, all parametrized sub-layers use GeGLU — full gated architecture. Param-parity bottleneck at hidden=56.
2. **Does widening the GeGLU MLP (hidden 216→320) improve?** (tanjiro, just assigned) — More gate capacity may unlock better selective routing without changing depth. +20% total params.
3. **Does Lion β1 tune benefit at β1∈{0.85, 0.95}?** (askeladd, just assigned) — Instantaneous step direction weight is separate from EMA timescale (β2). Low warmup cost, orthogonal to β2=0.999 test.
4. **Does the Lion lr optimum shift on the GeGLU baseline?** (#2288 frieren — stale, status-checked) — lr=2e-4 proven optimal for GELU; GeGLU may shift the bowl. Probing lr∈{2.5e-4, 3e-4}.
5. **Does RMSNorm improve on the GeGLU stack?** (#2315 thorfinn — stale, status-checked) — LLaMA-recipe co-change; mean-centering may erase PhysicsAttention slice-token offsets; scale-only norm may preserve structural bias.
6. **Does n_head=8 beat n_head=4 on the GeGLU stack?** (edward, just assigned) — attention diversity vs per-head capacity tradeoff. Same total params, different shape. Common in modern transformers.
7. **Does n_layers=4 free enough budget to outperform n_layers=5?** (nezuko, just assigned) — cost-recovery probe complementing fern's n_layers=6 budget-cliff result (#2349). One fewer block → ~17 epochs instead of 15-16.
8. **Does n_layers=6 beat n_layers=5 on the stale Huber stack?** (#1979 alphonse — rebased and running, directionally informative for depth axis)

## Confirmed dead ends

- **Lion wd axis on GeGLU at lr=2e-4 (#2352 CLOSED)**: wd ∈ {6e-5, 2e-3, 5e-3} all within ±0.6 val of baseline. Param L2 grows ~58% from init regardless of wd — sign-update growth dominates the multiplicative shrink. Axis is shallow and noise-dominated. Arm B wd=5e-3 had a real-looking test improvement (−1.01%) but val regressed slightly — unconfirmed without seed pair.
- **surf_weight=15 on GeGLU+Lion stack (#2005 CLOSED)**: val +2.13%, test +0.54%, both axes regress. Lion's sign quantization makes loss-balance reweighting weak (changes which params get stepped, not step magnitude). Don't assign further surf_weight sweeps on Lion stacks.
- **SwiGLU gating at the preprocess entry projector (#2332 CLOSED)**: +10.8% val regression. Gating below dim~32 discards information rather than routing it — no feature diversity for selective suppression. **Principle: GeGLU gating only works at scale (input dim ≥ 160).**
- **n_layers=6 on GeGLU stack (#2349 CLOSED)**: +10.6% val at termination. +18% per-epoch cost → only 12 of 13 epochs within 30-min cap; still in steep-descent regime. Depth axis alive, but needs budget > 30-min or cost reduction (smaller n_hidden/slice_num).
- **Lion β2=0.999 on GeGLU stack (#1844 CLOSED)**: +6.3% val. ~10× longer EMA timescale costs too much warmup within 30-min/16-epoch cap. β2=0.99 confirmed optimal.
- **Lion β1=0.95 (#2405 Arm B)**: +19.7% val vs baseline. Inertial β1 (sign dominated by stale EMA) badly underperforms at B=4. β1=0.85 is the winner; explore β1∈{0.80, 0.875} to find lower bound of optimum.
- **GeGLU bottleneck on `to_out` (#2401 CLOSED)**: +15.4% worse. Rank cut (160→56→160) dominates gate benefit. Epoch time +17%. Gating helps in 2-layer FFN context but not 1-layer attention projection at parity params.
- **SWA mid-training (#1463)**: regresses in 13-epoch monotonic regime. Partial camber_rc signal — revisit at 24+ epochs.
- **LR/clip ceiling at AdamW stage (#1683)**: both 2× arms regress on test. Obsoleted by Lion switch.
- **EMA decay=0.999 (#1596)**: 13-epoch monotonic regime; early averaging always hurts.
- **n_hidden=192 (#1755 Arm B, lr=4e-4)**: Budget cliff + grad_norm instability at lr=4e-4. 2× regression evidence.
- **Lion lr≤1.5e-4 or ≥3.5e-4 on GELU stack**: LR bowl confirmed bottomed at lr=2e-4 on GELU. Probing lr∈{2.5e-4, 3e-4} on GeGLU now (#2288 frieren).
- **Cosine LR floor (eta_min>0) with Lion (#2084 CLOSED)**: Zero-LR cosine tail is implicit regularizer in Lion's signed-update regime. Floor at 5% of lr prevents final settling. Do not add eta_min to Lion runs.
- **SiLU as bare activation (#2176 CLOSED)**: GELU→SiLU regresses every split +6.9 val/+6.5 test. Lion's sign update tuned for GELU's gradient surface; gated SiLU (#2196) won because gating is the primitive, not the activation slope.
- **MLP dropout + attention dropout rate sweep (#2161 CLOSED)**: Both directions regress. Dropout=0.1 attention-only is thin-ridge local optimum. **Dropout axis SATURATED.**
- **batch=8 at fixed lr=2e-4 (#2181 CLOSED)**: B=8 halves per-epoch step count → 2× under-training with Lion's magnitude-invariant sign update. Requires lr_B8 ≈ 4e-4 for fair comparison. Do not assign B=8 without LR compensation.
- **Lookahead wrapper around Lion (#2249 CLOSED)**: Epoch-budget cliff. Slow-weight anchor requires ≥30 epochs to amortise lag cost; incompatible with 16-epoch hard cap.
- **LLRD factor=0.85 (#2182 CLOSED)**: All 8 splits regress +7.3% val. Lion sign-step is linearly LR-sensitive — no preconditioning to recover from 50% step reduction. Transolver (5 layers) is wrong regime for BERT-style LLRD.
- **Lion wd ∈ [0, ~1.49e-4] at lr=2e-4 (#2177 diagnostic)**: FP32 ulp truncation makes `(1−lr·wd)` a literal 1.0. Do not assign wd < 2e-4 at lr=2e-4.
- **Huber δ_p<0.20 (#2074 CLOSED)**: δ_p=0.15 and 0.10 both lose. δ_p=0.20 is optimal with velocity at 0.5.
- **DropPath rates 0.05/0.10 (#2044 CLOSED)**: Requires 40+ epochs; catastrophic within 16-epoch budget.
- **Instance-norm loss with 1e-6 clamp (#1470)**: +3.7% val regression. Near-uniform low-Re samples amplified 1000-2000×.
- **slice_num=128 (#1481)**: +22.5% val regression. 41% per-epoch slowdown → budget cliff.

## Queued ideas (when students finish above)

1. **n_layers=4 cost-recovery probe** — if n_layers=6 is budget-starved, try n_layers=4 (one fewer block → more epochs). Quick to test; if it regresses, depth axis is confirmed at N=5.
2. **GeGLU mlp2 head** — last-layer mlp2 (GELU: hidden_dim → out_dim=3) benefits from gating? Low priority given tiny output dim, but completes the full-gated architecture picture.
3. **Real wd + GeGLU compound fine-tune** — after edward's wd sweep (#2352) lands, find the wd sweet spot on the current best stack. LR and wd axes are orthogonal.
4. **GeGLU mlp_ratio=4/3×2 (hidden=432)** — if hidden=320 wins, try hidden=432 (full param-equiv to standard mlp_ratio=4 FFN). Would increase total params ~50%.
5. **Sharpness-Aware Minimization (SAM)** — explicit flat-minima search; targeted at OOD generalization. Consider lighter LookSAM variant to reduce per-step cost.
6. **surf_weight fine-tune on GeGLU stack** — after nezuko lands directional signal, probe one notch around it on the current best stack.
7. **One-cycle LR schedule** — peak-then-decay; orthogonal to wd and dropout.
8. **GeGLU last-block-only gate test** — if attention-output gating (fern) shows mixed split results, try applying output gating only to the last TransolverBlock (which also has mlp2 GELU head).
