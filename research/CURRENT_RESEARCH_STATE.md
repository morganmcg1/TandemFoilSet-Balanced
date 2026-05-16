# SENPAI Research State

- **Date:** 2026-05-16 07:40 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

**Six axes confirmed and merged.** Compound stack: Huber + bf16 + cosine T_max=15 + EMA decay=0.999 + single-shot FiLM + two-shot FiLM.

**Current best: 89.784 val_avg/mae_surf_p** (frieren #3584, two-shot FiLM on full stack, 2026-05-16 04:50)

**Fourier features CLOSED (subsumed by FiLM, #3117 R4):** Paired Δ −0.10% at R4 (was −9.10% at R2, −3.16% at R3). FiLM absorbed the spatial-frequency signal Fourier provided. Test direction reversed (+0.86%). Clean closure.

Key in-flight composition tests:
- **🔥 Schedule-Free AdamW (alphonse #3594 R2):** R1 showed **−20.75% intra-PR (71.492 vs 90.207)** on pre-two-shot-FiLM stack. Sent back for verify on full current stack. If reproduces, largest single advance in track.
- **Gradient clipping clip=1.0 (tanjiro #3511 R2):** −4.03% on pre-FiLM stack; sent back for rebase + rerun on two-shot-FiLM stack, expected ~86-88 if additive.
- **Three-shot FiLM (frieren #3681):** preprocess injection as third FiLM site, assigned, expected −1-3%.
- **Slice-num sweep (edward #3684):** test slice_num=32/64/96 on full stack; just assigned.
- **n_layers=4 (fern #3758):** depth axis ablation; 4 layers faster per epoch → more fine-tune epochs in 30-min budget. Motivated by edward #3595 depth-vs-epochs finding.
- **SDF input features (askeladd #3777):** per-node signed distance to nearest surface as explicit geometric input. Motivated by Fourier R4 residual signal on multi-foil geometry splits. Orthogonal to FiLM.

**LR axis closed:** 5e-4 is optimal for bf16+T_max=15.
**n_layers axis:** n_layers=6 regresses (+2.47%, #3595). n_layers=4 now in test (#3758). n_layers=5 remains current best.
**Fourier axis CLOSED:** Subsumed by FiLM (#3117 R4, −0.10% Δ).

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 / −1.4% vs bf16 | 100.059 (fp32) |
| #3126 | EMA decay=0.999 Karras-ramp (nezuko) | −1.06% vs bf16+T_max=15 arm | 96.464 |
| #3122 | FiLM conditioning — log Re, AoA, NACA, gap, stagger (frieren) | −4.00% vs EMA baseline | 92.606 |
| #3584 | Two-shot FiLM — attn + MLP per block, shared module, +0 params (frieren) | −3.05% vs FiLM baseline | **89.784** |

## Falsified / closed hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 | +8.3% regression | bf16 noise amplifies higher LR |
| #3321 | lr=1e-3/1.5e-3+warmup on fp32+bf16 | +2.2-12% regression (6 arms) | Higher LR dead end, two-seed confirmed |
| #3443 | lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15 | +0.78-2.60% regression | LR axis closed: 5e-4 is magnitude optimum |
| #3595 | n_layers=6 vs 5 on full FiLM stack | +2.47% regression | Depth costs wall-clock epochs; 6 layers net-negative under 30-min budget |
| #3117 | Fourier scale=2 (R2-R4) | −0.10% at R4 (tie) | FiLM absorbed Fourier's spatial-frequency signal; test direction +0.86% |
| #3365 | batch_size=6/8 on bf16 | bs=4 best (monotonic regression) | GPU compute-bound; bigger batch cuts SGD steps without reducing sec/epoch |

## Active experiments

| Student | PR | Hypothesis | Stack | Status |
|---------|----|-----------|----|----|
| thorfinn | #3390 | bf16+T_max=15/20 compose verify | bf16+T_max | Stale_wip — training |
| askeladd | #3777 | SDF input features — distance-to-surface | bf16+T_max=15+EMA+2xFiLM | Just assigned |
| tanjiro | #3511 | grad_clip=1.0 on two-shot FiLM stack (rebase) | bf16+T_max+EMA+2xFiLM | Sent back to rebase |
| nezuko | #3492 | n_hidden=192 vs 128 | bf16+T_max=15+EMA | Stale_wip — training |
| alphonse | #3594 | Schedule-Free AdamW (R2 on two-shot FiLM, post −20.75% R1) | bf16+T_max=15+EMA+2xFiLM | Sent back for verify+rebase |
| edward | #3684 | slice_num=32/64/96 sweep | bf16+T_max=15+EMA+2xFiLM | Just assigned |
| frieren | #3681 | Three-shot FiLM (preprocess + attn + MLP) | bf16+T_max=15+EMA+2xFiLM | Just assigned |
| fern | #3758 | n_layers=4 depth ablation | bf16+T_max=15+EMA+2xFiLM | Just assigned |

## Key research questions

1. **🔥 SF-AdamW + two-shot-FiLM compose (alphonse #3594 R2):** R1 showed −20.75% on pre-two-shot stack (71.492). Does it reproduce on current stack? Expected ~70-75 if additive; ~75-80 if sub-additive.
2. **Three-shot FiLM (frieren #3681):** Preprocess injection adds third conditioning site; expected −1-3%.
3. **Gradient clip=1.0 + two-shot-FiLM compose (tanjiro #3511 rebase):** Pre-FiLM signal strong (−4.03%). Expected ~86-88 if composes.
4. **n_layers=4 (fern #3758):** Faster epochs → more fine-tune at lr≈0. Depth axis closing.
5. **Slice-num sweep (edward #3684):** Richer attention modes (96) vs faster fine-tune (32). Unknown direction.
6. **Model width (nezuko #3492):** n_hidden=192 on EMA stack (pre-FiLM); confirms if width is bottleneck.
7. **SDF input features (askeladd #3777):** per-node distance-to-surface; expected −1-3% on geom_camber splits specifically.

## LR axis (closed)

**5e-4 is the optimum** for bf16+T_max=15. Both lower (2.5e-4, 3.5e-4) and higher (1e-3, 1.5e-3) directions falsified across 4+ seeds.

## Depth axis (explored)

- n_layers=6: regresses under 30-min budget (+2.47%, epoch cost kills fine-tune time)
- n_layers=4: IN TEST (#3758) — hypothesis: faster epochs → more fine-tune
- Current best stays at n_layers=5 until #3758 resolves

## Fourier axis (CLOSED)

Fourier features subsumed by FiLM. The spatial-frequency signal FiLM provides via γ,β modulation is functionally equivalent to what Fourier basis expansion added on the input side. The per-round compression (−9.10% → −3.16% → −0.10%) cleanly tracks each merged feature's absorption of the Fourier signal. Not worth re-testing unless the FiLM stack is removed.

## Potential next hypotheses (not yet assigned)

1. **Per-block independent FiLM** — give each of the 5 blocks its own conditioner instead of shared; more expressive, ~5K more params/block. Three-shot FiLM result may inform priority.
2a. **Batch size + LR scaling** — askeladd #3365 falsified flat-LR bigger batch. LR-scaled retry (lr × sqrt(B/4)) could still win; defer until SDF result is in.
2. **Lion optimizer** — gradient-sign updates directly; tanjiro's finding suggests normalized GD helps, Lion is the extreme case. SF-AdamW result may make this redundant.
3. **Sobolev loss** — gradient supervision near surface (dU/dx, dp/dx). Physically motivated, untested. Complex to implement.
4. **SDF input features** — signed distance to surface. Orthogonal to FiLM. May help geometry-variant splits (geom_camber splits benefited from Fourier at half-magnitude).
5. **n_layers=4 + n_hidden=192** — if both depth-reduction and width increase show promise, compose them together for one compact model.
6. **Lower LR with two-shot FiLM** — FiLM+EMA smoothing may widen optimal LR range (but #3443 falsified pre-FiLM; revisit if SF-AdamW doesn't land).

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond --two_shot_film`
- **GH API rate limits:** recurring ~40-min windows; student pods auto-recover.
- **test_geom_camber_cruise NaN:** cruise-sample overflow in read-only `data/scoring.py`; use 3-split partial mean for test comparisons.
- **Depth-vs-epochs insight:** Under 30-min budget, per-epoch cost matters as much as model capacity. n_layers=6 loses 3 fine-tune epochs to n_layers=5. This asymmetry is load-bearing for hypothesis design.
- **FiLM-owns-velocity insight:** Fourier per-split decomposition (#3117 R4) shows mae_surf_Ux regresses +4.64% with Fourier under FiLM stack. FiLM fully owns the velocity channel; geometric multi-foil splits (geom_camber_*) still benefit from Fourier at half-magnitude but insufficient to overcome Ux regression.
