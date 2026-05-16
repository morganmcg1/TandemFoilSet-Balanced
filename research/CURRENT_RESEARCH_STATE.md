# SENPAI Research State

- **Date:** 2026-05-16 12:34 UTC (Round 4 in progress on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)
- **Rate-limit storm cleared at 12:21 UTC.** Two PRs (#3758 fern n_layers=4, #3492 nezuko n_hidden=192) sent back for R3 rebase with clip in both arms. All 8 students have active WIP PRs (no idle GPUs).

## Current research focus

**Seven axes confirmed and merged.** Compound stack: Huber + bf16 + cosine T_max=15 + EMA decay=0.999 + single-shot FiLM + two-shot FiLM + **grad_clip_norm=1.0**.

**Current best: 81.660 val_avg/mae_surf_p** (tanjiro #3511, grad clip=1.0 on two-shot FiLM stack, 2026-05-16 11:22)

⚠️ **STACK STALENESS WARNING:** All in-flight experiments (#3390, #3594, #3492, #3777, #3829, #3830, #3758) are running WITHOUT --grad_clip_norm 1.0. They will not beat 81.660 in absolute terms. When they report: if paired Δ > 0 → send back for rebase with clip=1.0 in both arms; if paired Δ ≤ 0 → close.

### 🔥 MAJOR WIN: Gradient clipping merged (+−9.05% → new baseline 81.660)

Grad clip=1.0 (direction normalization at ~96-100% clip rate) merged as the 7th axis. All experiments now must include `--grad_clip_norm 1.0` to beat the new baseline.

### 🔥 Sent back for R3 clip-compose rebase (12:24 UTC)

- **🔥🔥 n_hidden=192 (nezuko #3492 R2 → R3):** R2 paired Δ −8.21% val_avg / −8.00% test (strongest signal of round). Sent back to rebase with `--grad_clip_norm 1.0` in BOTH arms. If even a fraction of −8.21% survives, this is the biggest single hop available — could plausibly land mid-70s.
- **n_layers=4 (fern #3758 R2 → R3):** R2 seed-2 = 88.441 (mean of two n_layers=4 seeds = 89.32, below pre-clip baseline). Combined with R1 paired Δ −1.21%, depth hypothesis is real. Sent back for clip-compose R3. Saves ~18% params + ~19% sec/epoch if it composes.

### Other in-flight experiments (all on stale stack, no clip)

- **🔥 T_max=20 (thorfinn #3390 R2):** Actively training (97% GPU @ 12:33 UTC). R1 showed 88.229 on bf16-only; R2 should test on FiLM stack. Will need rebase with clip if paired Δ positive.
- **🔥 SF-AdamW (alphonse #3594 R2):** Actively training (98% GPU). R1 showed −20.75% pre-FiLM; R2 needs FiLM-stack verify. Will need rebase with clip if paired Δ positive.
- **Per-block FiLM (frieren #3829):** Actively training (100% GPU). Will need rebase with clip if paired Δ positive.
- **Lookahead (edward #3830):** Actively training (99% GPU). Will need rebase with clip if paired Δ positive.
- **SDF features (askeladd #3777):** Pod restarted 12:28 UTC; needs to spin up training.

### Confirmed closed axes

- **slice_num=64 locked in** (#3684 closed): 32 and 96 both regress by +3.8-4.9%. 64 is the knee point at n_hidden=128 capacity. (Note: 96 may unlock at n_hidden=192.)
- **Three-shot FiLM falsified** (#3681 closed): +4.08% regression. Shared conditioner head over-stretched; preprocess injection site is wrong (not inside residual stream).
- **Fourier axis closed** (#3117 R4): subsumed by FiLM.
- **Batch size axis closed** (#3365): GPU compute-bound; bs=4 is optimal.

### Key in-flight experiments

| Student | PR | Hypothesis | Status (12:34 UTC) |
|---------|----|-----------|--------------------|
| tanjiro | #3906 | Clip threshold sweep {0.25, 1.0, 4.0} | Training (GPU 100%) on clipthresh-r1 |
| thorfinn | #3390 | T_max=20 on FiLM stack (R2) | Training (GPU 97%) on bf16-tmax-compose |
| alphonse | #3594 | SF-AdamW on FiLM stack (R2) | Training (GPU 98%) on schedule-free-adamw |
| edward | #3830 | Lookahead optimizer | Training (GPU 99%) |
| frieren | #3829 | Per-block FiLM heads | Training (GPU 100%) |
| nezuko | #3492 | n_hidden=192 + clip (R3 rebase) | Just sent back at 12:24 UTC; pod will pick up |
| fern | #3758 | n_layers=4 + clip (R3 rebase) | Just sent back at 12:24 UTC; pod will pick up |
| askeladd | #3777 | SDF input features | Pod restarted 12:28 UTC; spinning up |

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged winners (chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 / −1.4% vs bf16 | 100.059 (fp32) |
| #3126 | EMA decay=0.999 Karras-ramp (nezuko) | −1.06% vs bf16+T_max=15 arm | 96.464 |
| #3122 | FiLM conditioning — log Re, AoA, NACA, gap, stagger (frieren) | −4.00% vs EMA baseline | 92.606 |
| #3584 | Two-shot FiLM — attn + MLP per block, shared module, +0 params (frieren) | −3.05% vs FiLM baseline | 89.784 |
| **#3511** | **Grad clip=1.0 — direction normalization on bf16+FiLM stack (tanjiro)** | **−9.05% vs two-shot FiLM** | **81.660** |

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

## Depth axis (n_layers fully mapped; n_layers=4 awaiting seed-2)

- n_layers=3: untested (potential follow-up if n_layers=4 lands)
- n_layers=4: **R1 paired Δ −1.21% ✅** (#3758) — mechanism fully verified (params −18.2%, sec/epoch −19.2%, +4 fine-tune epochs); absolute 90.198 vs baseline 89.784 — seed-2 in flight to resolve
- n_layers=5: current baseline
- n_layers=6: +2.47% regression (#3595)

Monotone curve at 30-min budget: depth costs epochs more than it adds capacity.

## Slice-num axis (CLOSED)

slice_num=64 is the optimum at n_hidden=128 capacity. Revisit sn=96 only if n_hidden=192 (#3492) merges.

## FiLM injection-count axis (CLOSED)

- Single-shot: merged (PR #3122, −4%)
- Two-shot: merged (PR #3584, −3.05%)
- Three-shot: closed (PR #3681, +4.08%)
- Injection-count axis at saturation. Per-block-capacity axis now in test (#3829).

## Potential next hypotheses (not yet assigned)

All future experiments must include `--grad_clip_norm 1.0` in the full stack.

1. **Clip threshold optimum** — tanjiro #3906 testing {0.25, 1.0, 4.0}. Result pending.
2. **Lion optimizer** — gradient-sign updates, natural direction normalization. Given clip=1.0 works by direction normalization, Lion may be redundant — or may be better (sign projection is more extreme than L2 clip). Test after clip sweep.
3. **Sobolev / gradient loss** — physically motivated; adds gradient supervision near surface. High-EV but high-effort.
4. **T_max=20 + clip** — natural composition; thorfinn #3390 will likely need rebase onto full stack.
5. **SF-AdamW + clip** — eliminates cosine schedule; alphonse #3594 likely needs rebase.
6. **n_layers=4 + clip** — fern's depth finding may compose with clipping.
7. **n_hidden=192 + clip** — capacity expansion on new baseline.
8. **Wider FiLM MLP hidden** — `film_mlp_hidden=256`; tests conditioner body capacity.
9. **Adaptive Gradient Clipping (AGC, NFNet)** — per-parameter group clip based on param norm; finer-grained than global L2 clip.

## Operational notes

- **Current best command:** `python train.py --amp_dtype bf16 --cosine_t_max 15 --use_ema --ema_decay 0.999 --film_cond --two_shot_film --grad_clip_norm 1.0`
- **GH API rate limits:** recurring ~40-min windows; student pods auto-recover.
- **test_geom_camber_cruise NaN:** cruise-sample overflow in read-only `data/scoring.py`; use 3-split partial mean for test comparisons.
- **Depth-vs-epochs insight:** n_layers=6 loses 3 fine-tune epochs to n_layers=5. Wall-clock cost is the binding constraint.
- **Cosine T_max=15 wasted-epochs insight:** LR floors at epoch 16 on bf16's 19-epoch budget. Largest unfixed inefficiency. Two parallel attacks in flight.
- **FiLM-owns-velocity insight:** Fourier R4 per-split shows mae_surf_Ux +4.64% with Fourier — FiLM owns velocity channel.
- **slice_num=96 + n_hidden=192 interaction:** student suggested they may compose; hold for post-nezuko evaluation.
- **Stack staleness pattern:** Multiple PRs ran on pre-FiLM stack and need rebase verifies (thorfinn, nezuko, tanjiro, alphonse). Baseline moved fast through Round 4.
- **Seed variance ~±1.5-2%:** Confirmed by fern #3758 R1 (Arm A 91.305 vs merged baseline 89.784, same config); askeladd #3365 (cross-commit ±5 MAE bf16); edward #3684 (Arm A −1.39% vs merged). Single-seed measurements can be misleading near the noise floor — paired Δ is reliable, absolute deltas need confirmation.
