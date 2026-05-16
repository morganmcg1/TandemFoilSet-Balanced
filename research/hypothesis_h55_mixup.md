## Hypothesis

**H55: Mixup data augmentation for regression — linear interpolation of input fields and labels at the H39 Arm C stack.**

Mixup (Zhang et al. 2018 "mixup: Beyond Empirical Risk Minimization") generates synthetic training samples by linear interpolation: `x' = λ·x_i + (1-λ)·x_j` and `y' = λ·y_i + (1-λ)·y_j`, where λ ~ Beta(α, α). For regression tasks like CFD field prediction, it acts as an implicit regularizer enforcing local linearity in the function the model learns.

**Why this is interesting for CFD surrogates:**
1. **Pre-overfit regime — but stochastic depth (H45 DropPath) failed.** H45 showed that stochastic-depth-style regularizers don't work at our 14-epoch budget because the model isn't overfitting. Mixup is a *different kind* of regularizer: it doesn't add noise to weights or remove paths; instead it *enlarges the effective dataset* by sampling along interpolation lines between training examples. This is much closer to data augmentation than capacity reduction.
2. **The dataset is small and the OOD splits are the bottleneck.** Looking at H39 Arm C numbers: val_geom_camber_rc/mae_surf_p was much worse than val_single_in_dist. Generalization to OOD geometry/Reynolds is what we need. Mixup interpolates between *training* geometries/Reynolds-numbers — sampling along the line between two airfoils, which is itself a plausible airfoil. This directly trains for OOD-style intermediate cases.
3. **Compatible with our regression Huber loss.** Mixup with regression and a Huber loss is straightforward: the interpolated targets are still valid regression targets, and Huber's robustness to outliers is preserved on the interpolated samples.

**CFD-specific caveats:**
- *Mesh / token correspondence*: Mixup requires that two samples have aligned input structures (same number of mesh tokens). If our model uses fixed `slice_num=64` slice-token aggregation, we *can* mix at the slice-token level (slice positions vary across samples, but the slice tokens themselves are length-64). Mix at the input feature dimension after embedding. **The student should verify this — if slice positions are sample-specific and not interpolatable, mix at the model input level (raw features) instead.**
- *FiLM conditioning is sample-specific.* The 11-dim FiLM conditioning vector (Re, AoA, geom params) is also interpolated linearly. This is well-defined for Re/AoA but assumes geometry parameters interpolate sensibly — which is the same assumption as Mixup itself.
- *Per-sample augmentation only.* Apply Mixup to training batches, NEVER to validation/test. Inference is single-sample.

**Two arms test α (the Beta distribution shape):**

- **Arm A — α=0.2 + H39 Arm C stack**: Conservative Mixup. λ skewed toward 0/1 (mostly near-original samples, occasional strong mix). Standard for regression in the literature.
- **Arm B — α=0.4 + H39 Arm C stack**: More aggressive. λ broader distribution, more "real" interpolation, more synthetic samples.

Predicted val_avg if mechanism holds ≈ 62-65. If both regress (>65), Mixup hurts at this config — the implicit linearity assumption breaks down for CFD fields (which have non-linear PDE structure), and we close cleanly.

**Why this stacks well with H39 Arm C:** Mixup's regularization is data-side, not weight-side. It's orthogonal to wd, dropout, and clip. The H39 Arm C stack already uses every weight-side regularizer optimized; adding a data-side regularizer is a *different lever* that should compound if it works at all.

## Instructions

One code change in `train.py`: add Mixup to the training data path.

**1. Add Config fields** (after `clip_grad_norm` line ~404):
```python
mixup_alpha: float = 0.0   # 0 disables mixup; >0 enables Beta(alpha,alpha)
```

**2. Wire Mixup into the train step** (in the training loop, after batch fetch):
```python
if cfg.mixup_alpha > 0.0 and model.training:
    # Sample mix coefficient from Beta(alpha, alpha)
    lam = torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample().item()
    # Permute batch indices to get mixing partner
    perm = torch.randperm(batch_size, device=device)
    # Mix all input tensors that are per-sample (NOT global / metadata):
    # - x (raw point features, B, N, F)
    # - cond (FiLM conditioning, B, 11)
    # - geom (mesh / position features, B, N, D)
    # Mix the per-sample labels:
    # - y (regression targets, B, N, C)
    x = lam * x + (1 - lam) * x[perm]
    cond = lam * cond + (1 - lam) * cond[perm]
    geom = lam * geom + (1 - lam) * geom[perm]
    y = lam * y + (1 - lam) * y[perm]
# ... proceed with forward+loss as normal
```

**Important — the student must verify**:
1. **Which tensors should be mixed and which should not.** For example: per-sample tensors (inputs, labels) must be mixed; loss masks may also need mixing depending on shape; sample-level metadata (split labels, file names) should not be mixed.
2. **Whether mixing is at the raw input level or post-embedding** depending on if slice positions are sample-specific. If raw input mixing breaks (e.g., for the slice-token aggregation), apply Mixup AFTER embedding (mix the hidden representations and the targets).
3. **No Mixup at validation.** Wrap the Mixup block in `if model.training` only.

CLI flag: `--mixup_alpha`.

**Arm A — Mixup α=0.2:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h55-mixup-a02-nhead2-lr2e3-wd5e5 \
  --agent charliepai2i48h3-tanjiro \
  --mixup_alpha 0.2 \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

**Arm B — Mixup α=0.4:**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h55-mixup-a04-nhead2-lr2e3-wd5e5 \
  --agent charliepai2i48h3-tanjiro \
  --mixup_alpha 0.4 \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Report per-arm:**
- val_avg/mae_surf_p, per-split breakdown
- **Per-split OOD performance especially** — Mixup is most likely to help OOD splits (val_geom_camber_rc, val_re_rand) at the possible cost of val_single_in_dist. Report this split-by-split.
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test
- Number of epochs completed before wall, best epoch
- Per-epoch val_avg trajectory (Mixup typically slows early convergence then catches up; report whether you see this pattern)
- Per-epoch train loss (with Mixup, train loss is *higher* than without — this is normal because the model is learning on harder synthetic samples)
- Peak GPU memory

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if Arm A or Arm B diverges:** if val_avg at epoch 3 exceeds 200, kill and report. Indicates wiring bug rather than Mixup failure.

## Baseline (pending merge)

**Current best — PR #3683 — H39 Arm C: n_head=2 + lr=2e-3 + wd=5e-5 + clip=1.0 (thorfinn)** (in rebase, will merge shortly)

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p (best of 2 seeds)** | **63.4385** |
| val_avg/mae_surf_p (2nd seed) | 65.5093 |
| test_avg/mae_surf_p (3-split, excl. cruise, best seed) | **61.3910** |
| Best epoch | 15/50 (cut by timeout) |

**Beat this: val_avg/mae_surf_p < 63.44 (best) or < 64.47 (mean of 2 seeds)**

Prior baseline (merged) — **PR #3629 — H37b: n_head=2 + lr=1e-3 + clip=1.0**:
| Metric | Value |
|--------|-------|
| val_avg/mae_surf_p | 66.1060 |
| test_avg/mae_surf_p (3-split) | 64.4522 |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=1e-4 (default), AdamW (β₁=0.9, β₂=0.999).

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.

**Reproduce H39 Arm C baseline (no Mixup):**
```bash
cd target/ && python train.py --epochs 50 \
  --experiment_name h39c-nhead2-lr2e3-wd5e5-clip1 \
  --agent charliepai2i48h3-tanjiro \
  --n_head 2 --lr 2e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```
