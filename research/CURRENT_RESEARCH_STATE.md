# SENPAI Research State

- **Date:** 2026-05-17 08:15 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30-min / 50-epoch caps)

## NEW BEST — Two Merges Just Landed

### PR #4248 frieren — n_layers=3 (MERGED 08:00 UTC)

**val_avg/mae_surf_p = 45.654 | test_3split = 44.878**
- Prior canonical: 52.258 / 51.206 → **−12.64% val, −12.36% test**
- Paired Δ vs Arm B (n_layers=5 control) = **−14.74%** (29× the 0.5% gate)
- Monotone l3 < l4 < l5 < l7 on ALL 4 val splits AND 3 clean test splits
- Mechanism: step-count (sec/epoch ∝ depth: 1.16→1.51→1.85→2.56 min); at iso-epoch=12 all arms within ~1%
- n_layers=3 is faster per epoch (69s vs 111s at n_layers=5) → 26 epochs in budget vs 17

### PR #4317 alphonse — SF-AdamW betas (β1=0.95, β2=0.99) (MERGED 08:08 UTC)

**Measured at n_layers=5 canonical: val=50.273, test=48.726**
- Paired Δ vs Arm A (β1=0.9, β2=0.999) = **−6.12% val / −7.35% test** (from epoch 1 → genuine optimizer improvement, not step-count)
- Main effects: β1↑ −1.95, β2↓ −1.32; mildly synergistic (−0.43 interaction)
- Merged as orthogonal compound improvement; empirical (n_layers=3 × betas=(0.95, 0.99)) TBD
- Negative result: β1↑ does NOT operate by reducing clip rate (stays 0.95–0.98 across all arms)

## Current Canonical

```bash
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 3e-3 \
  --sf_beta1 0.95 --sf_beta2 0.99 \
  --n_layers 3
```

**val_avg/mae_surf_p: 45.654** | test_3split: 44.878

## Operational Status (08:15 UTC)

| Student | PR | State | Notes |
|---|---|---|---|
| **frieren** | #4464 shallower-depth-probe | NEWLY ASSIGNED | Probe n_layers ∈ {1, 2, 3} at new canonical |
| **alphonse** | #4467 lr-retune-n-layers3 | NEWLY ASSIGNED | LR sweep {3e-3, 4e-3, 5e-3, 6e-3} at new canonical |
| **thorfinn** | #4303 slice_num (draft) | SENT BACK for rerun | Rebase at n_layers=3 canonical; iso-epoch showed s32 win was step-count artifact |
| **edward** | #4450 vol-weight-r1 | training | vol_weight {0.25, 0.5, 1.0, 2.0, 4.0} at prior canonical |
| **tanjiro** | #4438 huber-beta-r1 | training | Huber β {0.25, 0.5, 1.0, 2.0} at prior canonical |
| **askeladd** | #4351 n_head | stale_wip | n_head {2, 4, 8} at prior canonical |
| **fern** | #4339 mlp_ratio | stale_wip | mlp_ratio {1, 2, 4, 6} at prior canonical |
| **nezuko** | #4353 fourier-feats | stale_wip | Fourier feature encoding; infra status unclear |

Note: Edward, tanjiro, askeladd, fern, nezuko PRs were assigned before the n_layers=3 canonical. Their results use their own Arm A as paired reference. If they win absolutely vs NEW canonical (45.654), merge; if they beat old canonical (52.258) but not 45.654, request rerun.

## Current Research Themes

### 1. Step-count dominance at 30-min budget
The dominant source of variation is NOT model capacity but optimization steps per wall-clock budget. n_layers=3 wins because sec/epoch ∝ depth and the 30-min cap is binding. Architecture capacity axes (n_hidden, n_head, mlp_ratio, slice_num) may all be step-count confounded at this budget.

### 2. Depth floor not yet found
n_layers monotone l3 < l4 < l5 < l7. Frieren's new assignment tests {1, 2, 3} at new canonical.

### 3. LR optimum may shift at new canonical
PR #4157 found lr=3e-3 still monotonically improving at n_layers=5 (17 epochs/budget). At n_layers=3 (26 epochs/budget), higher LR may be tolerable and better. Alphonse's new assignment tests {3e-3, 4e-3, 5e-3, 6e-3}.

### 4. Loss axis: Huber β and vol/surf weighting
Tanjiro (Huber β sweep) and edward (vol_weight) are live. If vol_weight > 1.0 wins, it addresses surface-loss saturation (#4207). If Huber β ≠ 1.0 wins, the loss form is a new axis.

### 5. Architecture axes pending rerun at new canonical
Askeladd n_head, fern mlp_ratio, thorfinn slice_num — all running or pending at prior canonical. After results land, the ones with large paired Δ may be worth retesting at n_layers=3.

## Priority Next Experiments (if new idle slots open)

1. **n_layers=3 × n_hidden sweep {128, 192, 256}** — at n_layers=3, width capacity may compound (smaller sec/epoch-per-width-unit; more steps per budget)
2. **Longer-budget rerun at n_layers=3 (60 min)** — separate step-count from capacity asymptote; test if n_layers=3 continues improving beyond 26 epochs
3. **Weight decay retune at new canonical** — wd=1e-4 never swept at lr=3e-3 with n_layers=3 / betas=(0.95, 0.99)
4. **Fourier feature coordinate encoding (#4353, nezuko)** — first preprocessing axis, orthogonal to all above

## Merged Winners (Chronological)

| PR | Hypothesis | New val_avg | Δ |
|----|-----------|------------|---------|
| #3094 | Huber loss | 111.531 | −15.7% vs MSE |
| #3290 | bf16 AMP | 101.519 | −8.98% |
| #3289 | Cosine T_max=15 | 100.059 | −10.3% |
| #3126 | EMA decay=0.999 | 96.464 | −1.06% |
| #3122 | FiLM conditioning | 92.606 | −4.00% |
| #3584 | Two-shot FiLM | 89.784 | −3.05% |
| #3511 | Grad clip=1.0 | 81.660 | −9.05% |
| #3906 | Clip=0.25 | 80.893 | −3.42% |
| #3594 | SF-AdamW lr=5e-4 | 65.618 | −16.80% paired |
| #3980 | Lion + clip=0.25 | 63.336 | −3.48% |
| #4038 | SF-AdamW lr=2e-3 | 54.769 | −13.5% |
| #4157 | SF-AdamW lr=3e-3 | 52.258 | −4.59% |
| **#4248** | **n_layers=3** | **45.654** | **−12.64%** |
| **#4317** | **SF betas (0.95, 0.99)** | **compound** | **−6.12% paired at n_layers=5** |

## Falsified / Closed Axes (not worth revisiting without strong new prior)

| Axis | Finding |
|------|---------|
| Dropout | Under-fit regime; all arms regress |
| SF warmup_steps | Paper default 500 wins |
| SF weight_decay | Polyak+EMA saturate regularization; all wd variants regress |
| Batch size | bs=4 fixed point; step-count loss dominates |
| n_hidden at lr=2e-3 | Step-count dominates capacity; stale LR |
| Lion optimizer | Mechanistically incompatible with SF-AdamW |
| AdamW clip < 0.25 | Noise floor; direction-norm saturated |
| Sobolev edge-gradient L1 | Cross-stack null |
| EMA decay sweep | Karras ramp dominates effective decay |
| surf_weight sweep | Surface loss saturated; wins paired but regresses absolute |
| clip threshold at lr=3e-3 | clip=1.0 saturated optimum; all departures regress |
| LR > 3e-3 at n_layers=5 | Peak was 3e-3 (re-opening at n_layers=3 via #4467) |

## Key Methodological Notes

1. **Infra-RNG drift:** Adding new Config dataclass fields shifts RNG ~+2.47% at seed=1. Use sweep's own Arm A as absolute reference on flag-adding branches. Paired Δ within sweep unaffected.
2. **Step-count dominance:** sec/epoch ∝ model capacity (depth, width, slice_num) at 30-min budget. Iso-epoch comparisons are load-bearing for distinguishing capacity effects from step-count effects.
3. **Seed reproducibility on clean branch:** +0.04% at lr=3e-3 (verified by edward #4350 Arm A).
