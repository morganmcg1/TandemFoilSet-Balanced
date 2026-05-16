# SENPAI Research State

- **Date:** 2026-05-16 20:00 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Latest Actions This Session

- **PR #3594 (alphonse SF-AdamW R2) MERGED 15:34 UTC**: baseline 80.893 → **65.618** (−18.88%). Largest single-experiment gain of the round.
- **PR #3980 (frieren Lion) STRONG WIN, SENT BACK FOR REBASE 16:05 UTC**: paired Δ −24.43% val / −23.72% test. Arm B absolute val=63.336 < current baseline 65.618 (−3.48%), test=60.549 < 62.853 (−3.67%). **Largest paired Δ on the track.** Mergeable once branch rebased and reproduction confirmed.
- **PR #3390 (thorfinn T_max=20 R2) CLOSED 16:08 UTC**: superseded by SF-AdamW. Cosine axis formally closed.
- **PR #3777 (askeladd SDF) CLOSED 15:55 UTC**: +2.52% paired regression. Geometric-input axis closed.
- **PR #3492 (nezuko n_hidden=192 R4) CLOSED 18:31 UTC**: BUDGET subsumption — B wins per-epoch but 4-epoch penalty costs more than the gain. Capacity dormant under SF, not falsified.
- **PR #3985 (edward AGC R2) CLOSED 18:40 UTC**: MECHANISM-FLIP — AGC wins under cosine but loses under SF constant-LR. AGC axis closed under SF.
- **PR #4012 (fern Sobolev R1) SENT BACK 19:20 UTC**: stale-stack (AdamW+cosine); paired Δ −1.07% val but −0.19% test (weak transfer); non-monotone ranking. Sent back for SF rebase + 2-arm retest (Arm A SF+w=0 vs Arm B SF+w=0.3).
- **PR #4051 (thorfinn SF wd sweep) CLOSED 19:55 UTC**: all higher-wd arms regress (+2.46% to +11.99%). Polyak+EMA already saturate the implicit-regularization role; external wd has no headroom.
- **PR #4003 (tanjiro clip-R2) CLOSED 19:58 UTC**: clip=0.25 is AdamW+cosine optimum; 100% clip rate at all thresholds confirms direction-normalization is fully saturated. Best paired Δ −0.07% (within 0.97% noise floor). **NOISE FLOOR CALIBRATED: 0.97% paired from cuDNN non-determinism on identical config.**

## Absolute Baseline Drift Note

Multiple within-session controls landing 61-63 absolute vs merged baseline 65.618:
- nezuko #3492 R4 Arm A: 62.95
- frieren #3980 Arm B: 63.336
- thorfinn #4051 Arm A: 61.758

True SF-AdamW+EMA+clip=1.0 absolute is approximately **61-63** at 17 epochs. The merged 65.618 was a slightly-unlucky 16-epoch single run. **Paired Δ within session is the only reliable merge signal.** Update BASELINE.md only when a PR shows terminal results beating the control arm by >1% paired.

## Current Research Focus

**Nine axes confirmed and merged.** Compound stack: Huber + bf16 + EMA decay=0.999 + single-shot FiLM + two-shot FiLM + **SF-AdamW (no cosine)** + grad_clip_norm=1.0.

**Current best logged: 65.618 val_avg/mae_surf_p** (alphonse #3594, SF-AdamW + clip=1.0, 2026-05-16 15:34)
- True within-session absolute likely **~61-63** at 17 epochs (see drift note above)

### SF-AdamW Hyperparameter Map (status)

| Hyperparameter | Value | Status | PR |
|---|---|---|---|
| lr | 5e-4 | In sweep (#4038 askeladd) | Active |
| clip_norm | 1.0 | In sweep (#4019 alphonse 2×2) | Active |
| use_ema | True | In sweep (#4019 alphonse 2×2) | Active |
| ema_decay | 0.999 | **In sweep (#4113 tanjiro, NEW)** | Active |
| weight_decay | 1e-4 | **CLOSED — optimum** (#4051) | Done |
| sf_warmup_steps | 500 | In sweep (#4087 edward) | Active |
| batch_size | 4 | **In sweep (#4114 thorfinn, NEW)** | Active |

### Mechanism Map (updated post-wd close)

| Role | Mechanism that owns it |
|---|---|
| Loss alignment with eval metric | Huber β=1.0 |
| Compute throughput / epoch count | bf16 AMP |
| Schedule | **None — SF-AdamW eliminates scheduler** |
| Trajectory smoothing (long time scale) | EMA decay=0.999 + SF Polyak avg (overlap being probed by #4113) |
| Physics-parameter conditioning | FiLM (single + two-shot) |
| Per-step gradient geometry | grad_clip_norm=1.0 (SF-specific optimum TBD via #4019) |
| Implicit regularization | **SF Polyak averaging + EMA** (subsumes wd at 100× range — #4051 finding) |
| Schedule-free iterate convergence | **SF-AdamW** |

### "Two Mechanisms Same Role" Closed Axes Under SF

| Axis | Mechanism | Finding | PR |
|---|---|---|---|
| Regularization | Higher wd | Polyak+EMA subsume it; wd has no headroom | #4051 |
| Direction-norm (per-tensor) | AGC | Mechanism-flip: needs late-low-LR (cosine), not SF | #3985 |
| Capacity | n_hidden=192 | Budget subsumption: per-epoch wins erased by 27% throughput penalty | #3492 |
| Depth | n_layers=6 | Wall-clock epochs cost more than capacity gain | #3595, #3758 |
| Schedule | Cosine T_max=20 | Superseded by SF-AdamW | #3390 |

## In-Flight Experiments (20:00 UTC)

| Student | PR | Hypothesis | Status | vs SF baseline 65.618 |
|---------|----|----|----|----|
| ⭐ **frieren** | **#3980** | **Lion vs AdamW (post-rebase reproduction)** | Awaiting terminal result | **Strongest candidate — −24% paired, 63.336 absolute** |
| **alphonse** | **#4019** | **SF-AdamW clip×EMA 2×2 factorial** | Training since 15:38 (~4.5h in) | Directly refines SF stack |
| **askeladd** | **#4038** | **SF-AdamW LR sweep** {5e-4, 1e-3, 2e-3, 5e-3} | Training (~3.5h in) | Directly refines SF stack |
| **nezuko** | **#4081** | **FiLM head width: film_mlp_hidden ∈ {128, 192, 256}** | Training (~1.5h in) | Inductive-bias-amplification; no throughput cost |
| **edward** | **#4087** | **SF warmup steps sweep** {100, 500, 1000, 2000} | Training (~1h in) | Only SF-specific param not yet swept |
| **fern** | **#4012** | **Sobolev edge-gradient loss R2 (SF rebase + 2-arm)** | Sent back 19:20 for rebase | Orthogonal loss axis; borderline R1 signal |
| **tanjiro** | **#4113** | **EMA decay sweep SF: {0.99, 0.999, 0.9995, 0.9999}** | Just assigned 19:55 UTC | Probes Polyak-EMA time-scale overlap |
| **thorfinn** | **#4114** | **Batch size sweep SF: {4, 6, 8, 12}** | Just assigned 19:58 UTC | Tests SF benefit from lower-variance gradients |

**Primary metric:** `val_avg/mae_surf_p` (lower is better)

## Merged Winners (Chronological)

| PR | Hypothesis | val_avg delta | New val_avg |
|----|-----------|------------|---------|
| #3094 | Huber loss (alphonse) | −15.7% vs MSE | 111.531 |
| #3290 | bf16 AMP (askeladd) | −8.98% vs Huber | 101.519 |
| #3289 | Cosine T_max=15 (thorfinn) | −10.3% vs Huber fp32 | 100.059 (fp32) |
| #3126 | EMA decay=0.999 Karras-ramp (nezuko) | −1.06% vs bf16+T_max=15 arm | 96.464 |
| #3122 | FiLM conditioning (frieren) | −4.00% vs EMA baseline | 92.606 |
| #3584 | Two-shot FiLM (frieren) | −3.05% vs FiLM baseline | 89.784 |
| #3511 | Grad clip=1.0 (tanjiro) | −9.05% vs two-shot FiLM | 81.660 |
| #3906 | Clip=0.25 (tanjiro) | −3.42% paired vs clip=1.0 | 80.893 |
| **#3594** | **SF-AdamW (alphonse R2)** | **−16.80% paired vs matched cosine** | **65.618** |

## Falsified / Closed Hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting | +3-21% regression | Domain variance > channel bias |
| #3364 | lr=1e-3+warmup on bf16 | +8.3% regression | bf16 noise amplifies higher LR |
| #3321 | lr=1e-3/1.5e-3+warmup on fp32+bf16 | +2.2-12% regression (6 arms) | Higher LR dead end |
| #3443 | lr ∈ {2.5e-4, 3.5e-4} on bf16+T_max=15 | +0.78-2.60% regression | LR axis closed: 5e-4 is magnitude optimum |
| #3595 | n_layers=6 vs 5 on full FiLM stack | +2.47% regression | Depth costs wall-clock epochs |
| #3758 | n_layers=4 R3 on clip stack | +3.14% val / +3.34% test | Depth subsumed by clip |
| #3117 | Fourier scale=2 (R2-R4) | −0.10% at R4 (tie) | FiLM absorbed Fourier |
| #3365 | batch_size=6/8 on bf16 | bs=4 best | GPU compute-bound (on AdamW+cosine) |
| #3684 | slice_num=32/64/96 on FiLM stack | 64 best | 64 is knee point at n_hidden=128 |
| #3681 | Three-shot FiLM | +4.08% regression | Wrong injection site |
| #3829 | Per-block FiLM heads | Paired Δ −0.53% (noise) | FiLM saturated at n_hidden=128 |
| #3830 | Lookahead wrapper | +0.42% val | Redundant with EMA |
| #3777 | SDF input features | +2.52% val / +3.99% test | Redundant with dsdf+FiLM |
| #3390 | T_max=20 R2 | superseded mid-run | Cosine axis closed by SF-AdamW |
| #3492 | n_hidden=192 R4 SF | paired Δ +1.46% val (per-epoch wins but 4 epochs lost) | Budget subsumption under SF |
| #3985 | AGC R2 SF | paired Δ +8.45% val / +9.40% test | Mechanism-flip: AGC needs late-low-LR; SF eliminates it |
| #4051 | SF wd sweep {1e-4, 3e-4, 1e-3, 1e-2} | +2.46% to +11.99% monotone | Polyak+EMA subsume regularization role; wd=1e-4 is optimum |
| #4003 | AdamW clip-R2 {0.05, 0.1, 0.15, 0.25} | −0.07% best (noise floor 0.97%) | 100% clip rate at all thresholds; direction-norm saturated at 0.25 |

## Potential Next Research Directions (Post-Current-Sweeps)

1. **Longer training / epoch extension** — thorfinn's #4051 finding: ALL arms still improving at final epoch 17 under SF's constant LR. If batch_size sweep (#4114) enables faster epochs → more epochs → significant improvement.
2. **AdamW β1 under SF** — paper recommends β1=0.95 for SF; default is 0.9. Needs 4-line Config change to expose as CLI flag.
3. **Huber β re-tuning** — loss β=1.0 was set on AdamW+MSE stack (PR #3094); never re-tuned on SF stack. β ∈ {0.5, 1.0, 2.0, 5.0}. Needs 4-line Config change.
4. **slice_num re-tuning under SF** — slice_num=64 was best on FiLM stack (#3684), but stack has changed dramatically. Re-test 32/64/96 on SF stack.
5. **torch.compile** — 1.5-3× potential throughput gain → more epochs in budget. Risk: CFD ops may not compile cleanly. Requires testing.
6. **Lion + SF composition** — if frieren #3980 Lion reproduces, test whether Lion under SF-AdamW (with its own schedule-free variant) compounds.
