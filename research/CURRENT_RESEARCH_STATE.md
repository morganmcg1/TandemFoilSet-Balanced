# SENPAI Research State

- **Date:** 2026-05-16 09:05 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Current research focus

**Six axes confirmed and merged.** Compound stack: Huber + bf16 + cosine T_max=15 + EMA decay=0.999 + single-shot FiLM + two-shot FiLM.

**Current best: 89.784 val_avg/mae_surf_p** (frieren #3584, two-shot FiLM on full stack, 2026-05-16 04:50)

### 🔥 Two huge unmerged signals on stale stacks — both need composition verify (in progress)

- **🔥 T_max=20 vs T_max=15 (thorfinn #3390 R2):** R1 showed Arm C (bf16+T_max=20) = **88.229** on bf16-only stack. Mechanism: T_max=15 floors at LR=5e-8 by epoch 16, wasting 3 of 19 bf16 epochs. Sent back for verify on full FiLM stack. Predicted: **80-85** if compose.
- **🔥 Schedule-Free AdamW (alphonse #3594 R2):** R1 showed −20.75% (71.492 vs 90.207) on pre-two-shot-FiLM stack. Parallel attack on same LR-floor problem. Sent back for verify. Predicted: **70-75** if compose.

### Confirmed closed axes

- **slice_num=64 locked in** (#3684 closed): 32 and 96 both regress by +3.8-4.9%. 64 is the knee point at n_hidden=128 capacity. (Note: 96 may unlock at n_hidden=192.)
- **Three-shot FiLM falsified** (#3681 closed): +4.08% regression. Shared conditioner head over-stretched; preprocess injection site is wrong (not inside residual stream).
- **Fourier axis closed** (#3117 R4): subsumed by FiLM.
- **Batch size axis closed** (#3365): GPU compute-bound; bs=4 is optimal.

### Key in-flight experiments

| Student | PR | Hypothesis | Status | Expected range |
|---------|----|-----------|--------|---------------|
| thorfinn | #3390 | **🔥 T_max=20 on full FiLM stack** | Sent back for R2 | 80-85 |
| alphonse | #3594 | **🔥 SF-AdamW on full FiLM stack** | Sent back for R2 | 70-75 |
| nezuko | #3492 | n_hidden=192 on full FiLM stack | Sent back for R2 | 86.5-88.5 |
| tanjiro | #3511 | grad_clip=1.0 on full FiLM stack | Sent back for R2 | 86-88 |
| fern | #3758 | n_layers=4 depth ablation | Assigned | 87.5-89.5 |
| askeladd | #3777 | SDF input features | Assigned | 87-89 |
| frieren | #3829 | Per-block independent FiLM heads | Just assigned | 87-89 |
| edward | #3830 | Lookahead optimizer wrapper | Just assigned | 88-89.5 |

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
| #3321 | lr=1e-3/1.5e-3+warmup on fp32+bf16 | +2.2-12% regression (6 arms) | Higher LR dead end |
| #3443 | lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15 | +0.78-2.60% regression | LR axis closed: 5e-4 is magnitude optimum |
| #3595 | n_layers=6 vs 5 on full FiLM stack | +2.47% regression | Depth costs wall-clock epochs |
| #3117 | Fourier scale=2 (R2-R4) | −0.10% at R4 (tie) | FiLM absorbed Fourier; test direction +0.86% |
| #3365 | batch_size=6/8 on bf16 | bs=4 best (monotonic regression) | GPU compute-bound; bigger batch cuts SGD steps |
| #3684 | slice_num=32/64/96 on full FiLM stack | 64 best (+3.8-4.9% for 96 and 32) | 64 is knee point at n_hidden=128 capacity |
| #3681 | Three-shot FiLM (preprocess injection) | +4.08% regression | Shared head over-stretched; preprocess is wrong site |

## Cosine schedule axis (active — two parallel attacks in flight)

Current T_max=15 is **suboptimal for bf16's 19-epoch budget** — floors at LR=5e-8 by epoch 16, wasting 3 epochs:
- **Extend: T_max=20 (#3390 R2)** — R1 showed 88.229 on bf16-only (better than 89.784!)
- **Eliminate: SF-AdamW (#3594 R2)** — R1 showed −20.75% on pre-two-shot stack

Whichever has the larger paired Δ on full FiLM stack wins. Both may compose with other hypotheses.

## Depth axis (explored)

- n_layers=6: regresses (+2.47%, epoch cost kills fine-tune time)
- n_layers=4: IN TEST (#3758) — faster epochs → more fine-tune at lr≈0
- n_layers=5: current best

## Slice-num axis (CLOSED)

slice_num=64 is the optimum at n_hidden=128 capacity. Revisit sn=96 only if n_hidden=192 (#3492) merges.

## FiLM injection-count axis (CLOSED)

- Single-shot: merged (PR #3122, −4%)
- Two-shot: merged (PR #3584, −3.05%)
- Three-shot: closed (PR #3681, +4.08%)
- Injection-count axis at saturation. Per-block-capacity axis now in test (#3829).

## Potential next hypotheses (not yet assigned)

1. **Lion optimizer** — gradient-sign updates; may be redundant if SF-AdamW lands. Test if SF-AdamW doesn't reproduce.
2. **Sobolev / gradient loss** — physically motivated; adds gradient supervision near surface. Complex implementation. High-EV but high-effort.
3. **n_layers=4 + n_hidden=192 compound** — only if both individually win on FiLM stack.
4. **LR-scaled bigger batch** — askeladd #3365 falsified flat-LR; scaled (lr × √(B/4)) might still win.
5. **Wider FiLM MLP hidden** — `film_mlp_hidden=256`; tests conditioner body capacity orthogonal to per-block heads.

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond --two_shot_film`
- **GH API rate limits:** recurring ~40-min windows; student pods auto-recover.
- **test_geom_camber_cruise NaN:** cruise-sample overflow in read-only `data/scoring.py`; use 3-split partial mean for test comparisons.
- **Depth-vs-epochs insight:** n_layers=6 loses 3 fine-tune epochs to n_layers=5. Wall-clock cost is the binding constraint.
- **Cosine T_max=15 wasted-epochs insight:** LR floors at epoch 16 on bf16's 19-epoch budget. Largest unfixed inefficiency. Two parallel attacks in flight.
- **FiLM-owns-velocity insight:** Fourier R4 per-split shows mae_surf_Ux +4.64% with Fourier — FiLM owns velocity channel.
- **slice_num=96 + n_hidden=192 interaction:** student suggested they may compose; hold for post-nezuko evaluation.
- **Stack staleness pattern:** Multiple PRs ran on pre-FiLM stack and need rebase verifies (thorfinn, nezuko, tanjiro, alphonse). Baseline moved fast through Round 4.
