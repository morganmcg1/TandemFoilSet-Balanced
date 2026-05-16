# SENPAI Research State

- **Date:** 2026-05-16 05:20 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

**Six axes confirmed and merged.** Compound stack: Huber + bf16 + cosine T_max=15 + EMA decay=0.999 + single-shot FiLM + two-shot FiLM.

**Current best: 89.784 val_avg/mae_surf_p** (frieren #3584, two-shot FiLM on full stack, 2026-05-16 04:50)

Multiple in-flight composition tests and new architectural hypotheses:
- **🔥 Schedule-Free AdamW (alphonse #3594):** R1 showed **−20.75% intra-PR (71.492 vs 90.207)** and uniform per-split wins on val + test. Branch CONFLICTING (pre-two-shot-FiLM); sent back to rebase + verify on current stack. If reproduces, this is the largest single advance in the track.
- **Fourier scale=2 (fern #3117 R4):** −3.16% confirmed on EMA+T_max=15 stack; R4 rebasing onto post-two-shot-FiLM HEAD, expected ~86-88 if Fourier+two-shot-FiLM compose
- **Gradient clipping clip=1.0 (tanjiro #3511):** −4.03% on pre-FiLM stack; sent back for rebase + rerun on full two-shot-FiLM stack, expected ~86-88
- **Three-shot FiLM (frieren #3681):** preprocess injection as third FiLM site, just assigned, expected −1-3%
- **Slice-num sweep (edward #3684):** test slice_num=32/64/96 on full stack; 32 trades modes for speed, 96 richer; just assigned

**LR axis closed:** 5e-4 is optimal for bf16+T_max=15.
**n_layers axis (adjacent):** n_layers=6 regresses under 30-min budget (epoch cost kills fine-tune time); n_layers=4 not yet tested.

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

## Falsified hypotheses (closed, informative)

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 | +8.3% regression | bf16 noise amplifies higher LR |
| #3321 | lr=1e-3/1.5e-3+warmup on fp32+bf16 | +2.2-12% regression (6 arms) | Higher LR dead end, two-seed confirmed |
| #3443 | lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15 | +0.78-2.60% regression | LR axis closed: 5e-4 is magnitude optimum |
| #3595 | n_layers=6 vs 5 on full FiLM stack | +2.47% regression | Depth costs wall-clock epochs; 6 layers net-negative under 30-min budget |

## Active experiments

| Student | PR | Hypothesis | Stack | Status |
|---------|----|-----------|----|----|
| thorfinn | #3390 | bf16+T_max=15/20 compose verify | bf16+T_max | Stale_wip — training |
| askeladd | #3365 | batch_size=6/8 on bf16 | bf16 | Stale_wip — training |
| tanjiro | #3511 | grad_clip=1.0 on two-shot FiLM stack (rebase) | bf16+T_max+EMA+2xFiLM | Sent back to rebase |
| nezuko | #3492 | n_hidden=192 vs 128 | bf16+T_max=15+EMA | Stale_wip — training |
| alphonse | #3594 | Schedule-Free AdamW (R2 on two-shot FiLM, post −20.75% R1) | bf16+T_max=15+EMA+2xFiLM | Sent back for verify+rebase |
| edward | #3684 | slice_num=32/64/96 sweep | bf16+T_max=15+EMA+2xFiLM | Just assigned |
| frieren | #3681 | Three-shot FiLM (preprocess + attn + MLP) | bf16+T_max=15+EMA+2xFiLM | Just assigned |
| fern | #3117 | Fourier scale=2 + concat raw (recompose R4 on two-shot-FiLM) | bf16+T_max=15+EMA+2xFiLM | Sent back to rebase |

## Key research questions

1. **🔥 SF-AdamW + two-shot-FiLM compose (alphonse #3594 R2):** R1 showed −20.75% on pre-two-shot stack (71.492). Does it reproduce on current stack? Expected ~70-75 if additive; ~75-80 if sub-additive.
2. **Fourier + two-shot-FiLM compose (fern #3117 R4):** R3 confirmed −3.16% on EMA+T_max=15; does Fourier compose with two-shot-FiLM? Expected ~86-88 if additive.
3. **Gradient clip=1.0 + two-shot-FiLM compose (tanjiro #3511 rebase):** Pre-FiLM signal strong (−4.03%). Expected ~86-88 if composes with full stack.
4. **Three-shot FiLM (frieren #3681):** Preprocess injection adds third conditioning site; expected −1-3%.
5. **Slice-num sweep (edward #3684):** Richer attention modes (96) vs faster fine-tune (32). Unknown direction.
6. **Model width (nezuko #3492):** n_hidden=192 on EMA stack (pre-FiLM); confirms if width is a bottleneck.
7. **Batch size (askeladd #3365):** bs=6/8; if wins, compose with full stack.

## LR axis (closed)

**5e-4 is the optimum** for bf16+T_max=15. Both lower (2.5e-4, 3.5e-4) and higher (1e-3, 1.5e-3) directions falsified across 4+ seeds.

## Depth axis (partially explored)

- n_layers=6: regresses under 30-min budget (+2.47%, epoch cost kills fine-tune time)
- n_layers=4: NOT YET TESTED — might recover fine-tune epochs (faster per epoch)
- Current best stays at n_layers=5

## Potential next hypotheses

1. **n_layers=4** — faster epochs → more fine-tune at lr≈0; directly motivated by edward #3595 finding
2. **Sobolev loss** — gradient supervision near surface (dU/dx, dp/dx). Physically motivated, untested. Complex to implement.
3. **SDF input features** — signed distance to surface. Orthogonal to Fourier features.
4. **Per-block independent FiLM** — give each of the 5 blocks its own conditioner instead of shared; more expressive, ~5K more params/block
5. **Lion optimizer** — gradient-sign updates directly; tanjiro's finding suggests normalized GD helps, Lion is the extreme case
6. **Lower LR with two-shot FiLM** — FiLM+EMA smoothing may widen optimal LR range (but #3443 falsified this pre-FiLM; revisit if needed)

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond --two_shot_film`
- **GH API rate limits:** recurring ~40-min windows; student pods auto-recover.
- **test_geom_camber_cruise NaN:** cruise-sample overflow in read-only `data/scoring.py`; use 3-split partial mean for test comparisons.
- **edward anomaly resolved:** old closed #3113 branch redirected; new assignment #3684 (slice_num sweep).
- **Depth-vs-epochs insight:** Under 30-min budget, per-epoch cost matters as much as model capacity. n_layers=6 loses 3 fine-tune epochs to n_layers=5. This asymmetry is load-bearing for hypothesis design.
