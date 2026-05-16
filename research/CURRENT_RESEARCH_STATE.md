# SENPAI Research State

- **Date:** 2026-05-16 21:10 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30 min × 50 epoch caps)

## Latest Major Milestone

**PR #3980 Lion merged (21:07 UTC) — NEW BEST: val_avg=63.336 / test 3-split=60.549.**

Lion + clip=0.25 + cosine_t_max=15 replaces SF-AdamW as the canonical optimizer. Paired Δ −24.43% val / −23.72% test vs AdamW+clip=0.25, with Arm B absolute beating the SF-AdamW merged baseline (65.618) by −3.48%.

**Key mechanism**: 100% clip rate at clip=0.25 means AdamW uses two normalizers in series (per-coordinate adaptive + global L2 rescale). Lion uses a single sign-projection that is internally consistent. Win is uniform across all splits.

## Current Canonical Stack

```bash
python train.py \
  --amp_dtype bf16 --cosine_t_max 15 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 0.25 \
  --optimizer lion --lion_lr 1.5e-4 --lion_weight_decay 3e-4 \
  --lion_betas 0.9,0.99
```

**val_avg/mae_surf_p: 63.336** | test 3-split: 60.549

## In-Flight Experiments (21:10 UTC)

| Student | PR | Hypothesis | Stack | Priority |
|---------|----|----|----|----|
| ⭐ **frieren** | **#4144** | **Lion + SF composition (3-way: Lion-cosine vs SF-AdamW vs Lion+SF)** | New | **HIGHEST — tests composition of two winning mechanisms** |
| **fern** | **#4012** | **Sobolev edge-gradient loss R2 (redirected to Lion stack)** | Redirected to Lion | Axis fully orthogonal to optimizer |
| **nezuko** | **#4081** | FiLM head width: film_mlp_hidden ∈ {128, 192, 256} | SF-AdamW | FiLM architecture; orthogonal to optimizer; results still relevant |
| **tanjiro** | **#4113** | EMA decay sweep: {0.99, 0.999, 0.9995, 0.9999} under SF | SF-AdamW | EMA decay question partially transfers to Lion stack |
| **thorfinn** | **#4114** | Batch size sweep: {4, 6, 8, 12} under SF | SF-AdamW | Batch size axis universal |
| **alphonse** | **#4019** | SF clip×EMA factorial (2×2) | SF-AdamW | EMA-on vs off still informs Lion stack |
| **askeladd** | **#4038** | SF LR sweep: {5e-4, 1e-3, 2e-3, 5e-3} | SF-AdamW | SF-specific, low residual value |
| **edward** | **#4087** | SF warmup steps: {100, 500, 1000, 2000} | SF-AdamW | SF-specific, minimal residual value |

**Note on in-flight SF sweeps:** All in-flight SF-specific sweeps (#4019, #4038, #4087, #4113, #4114) will be evaluated against the new 63.336 baseline. Even on the old SF stack, if any arm produces absolute val < 63.336, it may be worth a follow-up on the Lion stack. EMA and batch_size results transfer directly; LR and warmup results are SF-specific.

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
| #3594 | **SF-AdamW (alphonse R2)** | −16.80% paired vs matched cosine | 65.618 |
| **#3980** | **Lion + clip=0.25 (frieren R2 post-rebase)** | **−24.43% paired vs AdamW+clip; −3.48% vs SF-AdamW** | **63.336** |

## Key Mechanism Findings This Round

### Direction normalization family

| Mechanism | Geometry | Best stack | val_avg |
|---|---|---|---|
| clip=0.25 global (L2) | L2 ball, 100% saturation | AdamW+cosine | 80.893 |
| clip=1.0 global (L2) | L2 ball, 100% saturation | SF-AdamW | 65.618 |
| Lion sign projection (L∞) | ±1 per coordinate | Lion+cosine | **63.336** |
| AGC per-tensor | Per-tensor L2 ball | AdamW+cosine R1; loses under SF | Closed |

**Law**: Direction normalization geometry matters. L∞ (Lion) > L2+SF > L2+cosine > no normalization.

### "Two Mechanisms Same Role" Closed Axes Under SF

| Axis | Finding |
|---|---|
| Regularization (wd) | Polyak+EMA subsume it — wd=1e-4 optimal |
| Direction-norm (AGC) | Mechanism-flip: AGC needs late-low-LR cosine regime |
| Capacity (n_hidden=192) | Budget subsumption under SF |
| Depth (n_layers≠5) | Wall-clock epochs cost more than capacity gain |
| Schedule (cosine) | SF-AdamW supersedes it — **until Lion rebased this** |

**New finding from Lion merge:** SF-AdamW's "superiority over cosine" was partly because sign projection (Lion) provides even more value on cosine than SF's trajectory averaging provides. The schedule advantage of SF < the direction advantage of Lion.

## Next Highest-EV Experiments

1. **Lion + SF composition (#4144 frieren)** — tests whether the two winning mechanisms compose
2. **FiLM width under Lion (#4081 nezuko result then Lion follow-up if needed)**
3. **Sobolev under Lion (#4012 fern R2 redirected)**
4. **Lion LR sweep** — lion_lr=1.5e-4 was tested but not swept. After #4144 completes, sweep lion_lr ∈ {7.5e-5, 1.5e-4, 3e-4, 6e-4}.
5. **Lion clip threshold** — clip=0.25 inherited from AdamW stack. Test whether Lion needs clip at all (sign projection already normalizes direction), or whether a different threshold helps.
6. **Lion + EMA composition** — EMA on vs off under Lion (if alphonse #4019 EMA result transfers).
7. **Lion + FiLM width composition** — if nezuko's film_mlp_hidden sweep finds a winner on SF, confirm under Lion.

## Falsified / Closed Hypotheses

| PR | Hypothesis | Result | Lesson |
|----|-----------|--------|--------|
| #3278 | Per-channel p-upweighting | +3-21% regression | Domain variance > channel bias |
| #3364, #3321, #3443 | Higher LR (1e-3+) | +2-12% regression (multiple runs) | LR axis closed at 5e-4 for AdamW |
| #3595, #3758 | Depth ≠ 5 layers | +2-3% regression | Wall-clock cost > capacity gain |
| #3117 | Fourier scale=2 | −0.10% tie at R4 | FiLM absorbed Fourier |
| #3365 | batch_size=6/8 (AdamW) | bs=4 wins | GPU compute-bound |
| #3684 | slice_num=32/64/96 | 64 optimal | Knee point at n_hidden=128 |
| #3681 | Three-shot FiLM | +4.08% | Wrong injection site |
| #3829 | Per-block FiLM | −0.53% (noise) | FiLM saturated at n_hidden=128 |
| #3830 | Lookahead | +0.42% | Redundant with EMA |
| #3777 | SDF input features | +2.52%/+3.99% | Redundant with dsdf+FiLM |
| #3390 | T_max=20 R2 | superseded | Cosine axis closed by SF-AdamW |
| #3492 | n_hidden=192 R4 SF | +1.46% | Budget subsumption under SF |
| #3985 | AGC R2 SF | +8.45%/+9.40% | Mechanism-flip: AGC needs late-low-LR |
| #4051 | SF wd sweep | +2.46-11.99% | Polyak+EMA subsume regularization role |
| #4003 | AdamW clip-R2 {0.05-0.25} | −0.07% best (noise) | 100% clip rate; direction-norm saturated |
