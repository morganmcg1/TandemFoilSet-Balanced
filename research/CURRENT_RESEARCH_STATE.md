# SENPAI Research State

- **Date:** 2026-05-17 09:40 UTC (Round 4 active on `icml-appendix-charlie-pai2i-48h-r4`)
- **Most recent human research direction:** None received on this track.
- **Track:** `icml-appendix-charlie-pai2i-48h-r4` (Charlie local-metrics arm; 8 students, 1 GPU each, 30-min / 50-epoch caps)

## NEW BEST — Three Merges

### PR #4464 frieren — n_layers=2 (MERGED 09:35 UTC)

**val_avg/mae_surf_p = 40.622 | test_3split = 39.598**
- Prior canonical: 45.654 / 44.878 → **−11.02% val, −11.77% test**
- Paired Δ vs Arm A (n_layers=3 control) = **−9.23%** (18× the 0.5% gate)
- Monotone l2 < l3 on ALL 4 val splits
- Iso-epoch=26: l3 < l2 (deeper better at fixed steps) → win is step-count effect
- l2 runs at 48.7 s/epoch → 37 epochs in budget vs l3's 26; l1 hits the 50-epoch cap (41.80)
- All arms still descending at budget cap — headroom remains

### PR #4248 frieren — n_layers=3 (MERGED 08:00 UTC)

**val_avg/mae_surf_p = 45.654 | test_3split = 44.878** _(now superseded)_

### PR #4317 alphonse — SF-AdamW betas (β1=0.95, β2=0.99) (MERGED 08:08 UTC)

**Measured at n_layers=5 canonical: val=50.273, test=48.726**
- Paired Δ = **−6.12% val / −7.35% test** (from epoch 1 → genuine optimizer improvement)

## Current Canonical

```bash
python train.py \
  --amp_dtype bf16 \
  --use_ema --ema_decay 0.999 \
  --film_cond --two_shot_film \
  --grad_clip_norm 1.0 \
  --use_schedule_free --lr 3e-3 \
  --sf_beta1 0.95 --sf_beta2 0.99 \
  --n_layers 2
```

**val_avg/mae_surf_p: 40.622** | test_3split: 39.598

## Operational Status (09:40 UTC)

| Student | PR | State | Notes |
|---|---|---|---|
| **frieren** | NEW ASSIGNMENT PENDING | idle | n_layers shallower probe complete (merged) |
| **alphonse** | #4467 lr-retune-n-layers3 | training | LR sweep {3e-3, 4e-3, 5e-3, 6e-3} at n_layers=3 canonical |
| **fern** | #4481 n-hidden-at-n-layers3 | training | n_hidden {96, 128, 192, 256} at n_layers=3; Config flag added by student |
| **thorfinn** | #4303 slice_num r2 | training | Rebase + rerun at n_layers=3; Arm A=44.75; Arm B (s32) just launched ~08:50 |
| **nezuko** | #4353 fourier-feats r2 | sent back | Rebase + 2-arm rerun at n_layers=3 canonical; waiting on student |
| **edward** | #4450 vol-weight-r1 | training | vol_weight {0.25, 0.5, 1.0, 2.0, 4.0} at prior canonical |
| **tanjiro** | #4438 huber-beta-r1 | training | Huber β {0.25, 0.5, 1.0, 2.0} at prior canonical |
| **askeladd** | #4351 n_head | stale_wip | n_head {2, 4, 8}; updated to use n_layers=3 canonical (comment posted); needs to rebase to n_layers=2 now |

Note: Edward, tanjiro, alphonse, fern, thorfinn PRs were assigned at n_layers=3 canonical (45.654). Their results use their own Arm A as paired reference. If they win absolutely vs NEW canonical (40.622), merge; if they beat n_layers=3 canonical (45.654) but not 40.622, request rerun at n_layers=2.

## Current Research Themes

### 1. Step-count dominance at 30-min budget — CONFIRMED AND DEEPENING
The dominant source of variation is NOT model capacity but optimization steps per wall-clock budget. Monotone l2 < l3 < l4 < l5 < l7 across all depth probes. At iso-epoch, deeper is always better — the wins come entirely from step accumulation at the 30-min cap. n_layers=2 (48.7 s/ep, 37 epochs) beats n_layers=1 (27.9 s/ep, 50 epochs) because the iso-epoch quality advantage of l2 over l1 outweighs l1's additional 13 epochs. The depth floor is now **n_layers=2** in the 30-min budget regime.

### 2. Depth floor established at n_layers=2
n_layers monotone l2 < l3 < l4 < l5 < l7 (budget-regime). l1 improves on baseline but loses to l2 (step-count advantage of l2/ep quality > l1 extra epochs). All arms descending at cap — longer-budget runs could reveal whether l1 eventually catches up.

### 3. LR optimum may shift at n_layers=2 canonical
alphonse #4467 tests {3e-3, 4e-3, 5e-3, 6e-3} at n_layers=3. Results should inform whether to raise LR further at n_layers=2. Need follow-up at n_layers=2 after alphonse reports.

### 4. Loss axis: Huber β and vol/surf weighting
Tanjiro (Huber β sweep) and edward (vol_weight) are live at old canonical. Evaluate with paired Δ from their own Arm A.

### 5. mlp_ratio CLOSED — step-count dominates FFN capacity
fern #4339: mlp_ratio=2 is the saturated optimum. Closes the last primary FFN capacity axis.

### 6. Fourier features (σ=1) — promising, rerun at n_layers=3 canonical pending
nezuko #4353: Arm B (16f, σ=1.0) −2.47% paired val / −3.47% test at n_layers=5. Sent back for 2-arm rerun at n_layers=3. σ-calibration finding: Tancik's σ=10 wrong for N(0,1) inputs.

### 7. n_hidden compensating-capacity — two fronts open
- fern #4481 (training): n_hidden {96, 128, 192, 256} at n_layers=3; tests if width compensates at 3-layer depth
- frieren next assignment: n_hidden at n_layers=2 — the canonical depth; more relevant than n_layers=3

### 8. slice_num at new canonical
thorfinn #4303: Arm A r2 completed (val_avg=44.75 at n_layers=3); Arms B/C/D in progress. Iso-epoch deconfounding required.

## Priority Next Experiments (if new idle slots open)

1. **LR retune at n_layers=2** — alphonse result will show LR optimum at n_layers=3; need follow-up at n_layers=2 (even more epochs available)
2. **Weight decay retune at new canonical** — wd=1e-4 never swept at lr=3e-3 with n_layers=2
3. **n_head sweep at n_layers=2 canonical** — askeladd #4351 stale; final primary architecture axis now at wrong canonical
4. **Longer-budget rerun at n_layers=2** — all arms descending at cap; convergence asymptote unknown
5. **Fourier sigma grid** — if nezuko confirms σ=1 16f wins at n_layers=3, extend to {8f, 16f, 32f} × {σ=0.5, 1.0, 2.0}

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
| #4248 | n_layers=3 | 45.654 | −12.64% |
| #4317 | SF betas (0.95, 0.99) | compound | −6.12% paired at n_layers=5 |
| **#4464** | **n_layers=2** | **40.622** | **−11.02%** |

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
| mlp_ratio != 2 | Step-count dominates FFN capacity; r=2 saturated |
| n_layers=1 (budget-regime) | l1 loses to l2 despite more epochs; iso-epoch quality deficit dominates |

## Key Methodological Notes

1. **Infra-RNG drift:** Adding new Config dataclass fields shifts RNG ~+2.47% at seed=1. Use sweep's own Arm A as absolute reference on flag-adding branches. Paired Δ within sweep unaffected.
2. **Step-count dominance:** sec/epoch ∝ model capacity (depth, width, slice_num) at 30-min budget. Iso-epoch comparisons are load-bearing for distinguishing capacity effects from step-count effects.
3. **Seed reproducibility on clean branch:** +0.04% at lr=3e-3 (verified by edward #4350 Arm A).
4. **test_geom_camber_cruise NaN:** Pre-existing scoring bug on cruise held-out samples. Ux/Uy channels are finite; mae_surf_p is NaN. Use test_3split_mean (3 clean splits) for ranking.
