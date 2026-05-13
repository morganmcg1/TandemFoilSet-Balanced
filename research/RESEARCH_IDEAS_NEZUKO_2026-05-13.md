# Hypothesis: AdamW eps=1e-6 for bf16 Numerical Stability

**Student**: willowpai2g48h3-nezuko
**Date**: 2026-05-13
**Axis**: Optimizer numerical stability (non-schedule, orthogonal to all in-flight PRs)

---

## Hypothesis Statement

The current AdamW optimizer uses the default `eps=1e-8`, but the training stack uses bf16 mixed precision. bf16's minimum representable positive normal value is ~1.175e-7, which means `eps=1e-8` falls below bf16 precision and is effectively treated as zero (subnormal) when computing the denominator `sqrt(v_t) + eps`. Raising eps to `1e-6` (which is representable in bf16) gives a proper floor to the effective step size, stabilizing late-training dynamics for parameters where the second moment is very small.

---

## Mechanism / Why This Is Well-Grounded

**Numerical basis**:
- bf16 has 7 bits of mantissa -> 3 decimal digits of precision. Its minimum positive normal value is ~1.175e-7.
- Default AdamW `eps=1e-8` < 1.175e-7 -> eps is a bf16 subnormal, rounds to 0.
- The AdamW update `m_t / (sqrt(v_t) + eps)` effectively becomes `m_t / sqrt(v_t)` for any parameter where `sqrt(v_t)` is not already much larger than eps.
- For parameters with small second moments (e.g. output projection layers, bias terms, or lightly-activated slices), this removes the denominator floor entirely, allowing arbitrarily large effective step sizes.
- The grad-clip (max_norm=1.0, engaging 100% of steps) bounds the raw gradient but does NOT bound the AdamW effective step size per-parameter -- that is controlled by eps.

**Precedent in large-scale bf16 training**:
- LLaMA-2 (Meta, 2023): AdamW eps=1e-5
- Mistral-7B (Mistral AI, 2023): AdamW eps=1e-5
- GPT-NeoX / Pythia (EleutherAI): eps=1e-8 -> 1e-6 in bf16 settings
- The standard recommendation when training in bf16 is eps >= 1e-7 (representable), with eps=1e-6 being the most common practical choice as it provides a safety margin above the bf16 floor.

**Why this stack is specifically susceptible**:
1. bf16 AMP was merged at PR #1715 -- the eps was never adjusted.
2. The model uses sliced PhysicsAttention over irregular meshes. Slices with sparse activation (few nodes assigned to a slice) will have small gradient magnitudes and small second moments in their attention/projection weights.
3. weight_decay=2e-4 (merged PR #2017) adds a regularization signal that interacts with per-parameter step sizes -- with eps=0 (subnormal), weight decay can over-shrink lightly-activated parameters.
4. The compute-bound regime (best=last at 35 epochs) means the optimizer is running close to its training horizon -- late-training numerical drift is more likely to matter.

**Expected mechanism**:
eps=1e-6 adds a representable denominator floor. For well-activated parameters (large v_t), the change is negligible (1e-6 << sqrt(v_t)). For lightly-activated parameters (small v_t), it prevents runaway effective step sizes. Net effect: more stable optimization trajectory, particularly for surface-pressure-critical output layers where the pressure channel (y[:,2]) has the largest dynamic range and the most varied second moments.

---

## Code Change

**File**: `train.py`
**Lines**: 453-458 (AdamW constructor)

Current code (lines 453-458):
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    betas=(0.9, 0.95),
)
```

Proposed change:
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
    betas=(0.9, 0.95),
    eps=1e-6,
)
```

**Exactly 1 line added.** No other changes. All other training configuration (lr, weight_decay, betas, scheduler, architecture, loss, AMP, compile, clip) is unchanged.

---

## Predicted Effect

**Expected Δ on val_avg/mae_surf_p**: -1% to -3% (improvement of ~0.6 to 1.8 MAE units from 58.88 baseline)

**Confidence**: Moderate. The mechanism is sound and well-precedented in bf16 LLM training. The effect size in this mesh-based regression setting is less certain than in language models -- the second moments of the PhysicsAttention slices may already be large enough that the denominator floor rarely matters. If the effect is present, it should be visible in the val_geom_camber and val_re_rand splits (OOD generalization tracks) since those are where the model is most likely to have lightly-activated slices.

**Orthogonality**: Fully orthogonal to all 8 in-flight WIP PRs:
- edward #2440: lr=3e-4 + warmup -- different from eps, would compose additively
- tanjiro #2420: lr value scan -- different axis entirely
- thorfinn #2415: DropPath -- architectural regularization, not optimizer
- frieren #2399: EMA -- weight-averaging, not optimizer step
- fern #2397: grad-clip magnitude -- affects gradient norm, not denominator floor
- alphonse #2180: dropout -- architectural regularization, not optimizer
- askeladd #2163: Huber beta -- loss shape, not optimizer

---

## Reproduce Command

```bash
python train.py \
  --epochs $SENPAI_MAX_EPOCHS \
  --lr 5e-4 \
  --weight_decay 2e-4 \
  --wandb_group eps-bf16 \
  --seed 42
```

Run with seeds 42 and 123 to confirm stability. The change is deterministic-equivalent for well-conditioned parameters, so the two seeds primarily test whether the improvement is robust across initialization.

---

## Decision Tree

```
eps=1e-6 experiment result
├── val_avg/mae_surf_p < 58.88 (improvement)
│   ├── val_geom_camber splits also improve -> mechanism confirmed: bf16 subnormal eps was hurting OOD slices
│   │   -> MERGE. Follow-up: try eps=1e-5 to see if curve has a minimum, or test interaction with edward's warmup
│   └── val_geom_camber flat, val_single_in_dist improves -> weaker confirmation
│       -> MERGE (improvement is improvement). No clear follow-up on eps axis.
├── val_avg/mae_surf_p ~= 58.88 (wash, within 0.3 MAE)
│   -> Second moments are already large enough that eps doesn't matter for this model size/regime.
│   -> CLOSE. Move nezuko to a different axis (QK normalization, or output per-channel normalization).
└── val_avg/mae_surf_p > 59.5 (regression, >1% worse)
    -> Unexpected. Would imply eps=1e-6 is too aggressive (over-regularizing effective step sizes globally).
    -> CLOSE. Unlikely but would add signal: suggests optimizer is already well-calibrated.
```

---

## Taste Rubric

**Research mode**: Diagnostic + incremental frontier refinement
**Mechanistic grounding**: 4/4 — Precise, falsifiable mechanism tied to concrete stack evidence (bf16 merged PR #1715, AdamW default eps never adjusted). External precedent from LLaMA-2, Mistral, EleutherAI.
**Research-state value**: 3/4 — Would distinguish between "optimizer denominator floor is active bottleneck" vs "second moments are always large enough to be safe". Either answer updates the research map.
**Execution value**: 4/4 — 1 line of code, fits in the 30-min wall-clock budget at 2 seeds, directly tests the paper-facing surface pressure metric.

**Overall**: High-leverage, low-cost, well-grounded diagnostic. If it wins, it's likely a permanent component of the optimal stack. If it washes, it rules out an entire class of eps-based optimizer tuning and narrows the search space.

---

## Baseline to Beat

Current baseline (PR #2017, weight_decay=2e-4):
- val_avg/mae_surf_p: **58.883**
- test_avg/mae_surf_p: **51.078**
- val_single_in_dist/mae_surf_p: 26.58
- val_geom_camber_rc/mae_surf_p: 73.82
- val_geom_camber_cruise/mae_surf_p: 49.73
- val_re_rand/mae_surf_p: 85.42

All metrics must improve or stay flat. Primary decision criterion: val_avg/mae_surf_p < 58.883.
