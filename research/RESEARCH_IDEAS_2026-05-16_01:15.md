# Research Ideas — 2026-05-16 01:15 UTC

**Baseline:** `val_avg/mae_surf_p = 90.6131` (PR #3474, EMA decay=0.99 + clip=5 + Huber δ=1.0)
**Wall-clock budget:** ~14 epochs before SENPAI_TIMEOUT_MINUTES=30 binds.
**Primary target:** `val_avg/mae_surf_p` — lower is better.

**In-flight (do not duplicate):**
- #3475 askeladd — asinh-pressure transform (output space)
- #3543 alphonse — ema-decay-push (0.98, 0.97, 0.95)
- #3571 fern — depth-sweep (n_layers=6, 7)
- #3477 thorfinn — physics-continuity loss (w=0.1, 0.5)

---

## H-01: pressure-channel-weight

**One sentence:** Upweight pressure explicitly in the per-channel surface loss computation to directly penalise `mae_surf_p` harder during training.

**Mechanism.** The current loss is `vol_loss + 10.0 * surf_loss`, where `surf_loss` averages over all three output channels (Ux, Uy, p) equally after Huber. The primary metric is `mae_surf_p`, yet the training signal it receives is diluted by Ux and Uy surface terms. A per-channel multiplier `[1.0, 1.0, p_weight]` applied inside `surf_loss` before averaging increases the gradient mass flowing from surface pressure nodes without touching any other mechanism. This is a targeted loss-reweighting that addresses a direct mismatch between the scalar training objective and the paper-facing metric. It costs zero extra compute.

**Implementation.**
In `train.py` (or `loss.py` if factored out), locate the surface loss computation. Replace the uniform mean over output channels with a weighted mean:

```python
# Before (pseudocode):
surf_loss = huber(pred_surf, y_surf).mean()  # mean over [Ux, Uy, p]

# After:
channel_weights = torch.tensor([1.0, 1.0, p_weight], device=pred_surf.device)
surf_loss = (huber(pred_surf, y_surf) * channel_weights).mean()
```

Add CLI flag `--p_surf_weight` (default 1.0). Run two arms:
- Arm A: `--p_surf_weight 3.0`
- Arm B: `--p_surf_weight 5.0`

Keep `--surf_weight 10.0` unchanged (that multiplier controls vol vs surf balance, orthogonal).

**Reproduce command:**
```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 \
  --p_surf_weight 3.0 \
  --wandb_group p-surf-weight-sweep --wandb_name p-surf-weight-3.0 --agent <student>
```

**Expected outcome.** `val_avg/mae_surf_p` < 90.6. Because the model sees proportionally larger gradient from pressure surface nodes, the optimization should tighten on the exact metric being measured. Risk of Ux/Uy degradation is real but acceptable — we track `mae_surf_p` only.

**Risk.** Medium. A very high p_weight (>10) can cause gradient imbalance across channels and destabilize training. 3.0 and 5.0 are conservative. If both arms regress, the conclusion is that the model's current capacity cannot independently reduce p error without Ux/Uy anchoring.

**Student profile.** Any student. The implementation change is 3–5 lines inside the loss computation. No architecture changes. Minimal risk of breaking other things.

---

## H-02: slice-num-128

**One sentence:** Double the number of learnable physics tokens in PhysicsAttention from 64 to 128 to increase the model's capacity to simultaneously represent distinct aerodynamic regions.

**Mechanism.** PhysicsAttention projects N mesh nodes into `slice_num` learnable tokens via soft membership assignment, then runs O(slice_num²) self-attention in that reduced space. With slice_num=64, the model has 64 physics "slots" to jointly represent the wake, boundary layer, pressure side, suction side, far-field, inter-foil gap, and the two distinct foil geometries. TandemFoilSet is geometrically complex: two foils at varying gaps and stagger angles create aerodynamic interactions that may saturate the representational capacity of 64 tokens. Doubling to 128 increases the attention space by 4x (128²/64² = 4) while leaving depth (n_layers=5), hidden dim (128), and all training settings unchanged. This is orthogonal to fern's depth-sweep (n_layers=6,7 at slice_num=64).

**Implementation.** Single flag change. No code modification needed if `slice_num` is already a CLI argument:

```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 \
  --slice_num 128 \
  --wandb_group slice-num-128 --wandb_name slice-num-128 --agent <student>
```

Verify that `model_config` dict passes `slice_num` through to the `Transolver` constructor (check `train.py` model init block). If it is hardcoded, add a `--slice_num` argument and thread it through. Expected code change: ≤10 lines.

Also run `slice_num=256` if time permits (second arm after seeing 128 result), but 128 is the primary test.

**Expected outcome.** `val_avg/mae_surf_p` < 90.6 on the same budget. The main uncertainty is whether 14 epochs is enough to train 128-token attention to convergence — the larger token space has more parameters in the membership projection and the attention QKV matrices.

**Risk.** Medium. VRAM usage increases; with batch_size=4 and node counts up to 242K the extra ~4MB per attention block should be fine on 96GB VRAM. Training may need slightly longer to converge — if 14-epoch result is marginal, the architecture may still win at full 50-epoch budget.

**Student profile.** Any student comfortable reading the Transolver model forward pass. The change is one integer, but the student needs to confirm it is wired through the model constructor, not hardcoded.

---

## H-03: weight-decay-sweep

**One sentence:** Sweep AdamW weight decay from the current 1e-4 upward (1e-3, 5e-3) to test whether stronger L2 regularization reduces overfitting to the training geometry distribution and improves OOD splits.

**Mechanism.** The current `val_re_rand` (86.494) is the weakest per-split improvement since Round 1 (−0.2% from #3474 vs −5.1% for in-dist, −9.7% for camber-rc). This suggests the model may be memorizing geometric details that do not generalize across Reynolds-number regimes. AdamW weight decay imposes an L2 norm penalty on all parameters at every step (decoupled from gradient magnitude), which discourages large parameter norms and can reduce overfitting. The current 1e-4 is conservative; literature on neural PDE surrogates often uses 1e-3 to 5e-3. This is a pure regularization axis, orthogonal to EMA (trajectory smoothing), Huber (outlier robustness), and clip (gradient explosion control).

**Implementation.** No code changes. CLI only:

- Arm A: `--weight_decay 1e-3`
- Arm B: `--weight_decay 5e-3`

```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 \
  --weight_decay 1e-3 \
  --wandb_group wd-sweep --wandb_name wd-1e-3 --agent <student>
```

Run both arms sequentially within the 30-min wall clock. The overhead is only the hyperparameter change; run times are identical to baseline.

**Expected outcome.** Primary: `val_avg/mae_surf_p` < 90.6. Secondary: watch `val_re_rand` specifically — if it drops below 86 while val_avg also improves, this confirms the OOD regularization hypothesis. A scenario where val_avg improves but val_re_rand stays flat would suggest the mechanism is helping in-distribution but not cross-Re.

**Risk.** Low-to-medium. At weight_decay=5e-3, it is possible the model is under-regularized (loss not fully converged at 14 epochs) — val loss may still be falling when the wall clock hits. If so, the arm is inconclusive, not falsified.

**Student profile.** Any student. Zero code changes. Pure sweep.

---

## H-04: re-sinusoidal-embed

**One sentence:** Replace the scalar `log(Re)` input feature (feature dimension 13) with a multi-frequency sinusoidal embedding to give the model richer, positionally-aware signal about the Reynolds-number regime.

**Mechanism.** The input feature vector for each mesh node includes `log(Re)` as a single scalar in dimension 13 (confirmed from TandemFoilSet feature channel list: x, z, saf, AoA, NACA params, Re, gap, stagger, is_surface, ...). A single scalar gives the model only one real-valued number to distinguish between Re=50K and Re=500K — the model must learn the mapping from this scalar to flow behavior entirely through depth. Sinusoidal positional embeddings (as in vanilla Transformers) decompose the scalar into d frequencies:

  `embed[2k] = sin(Re_norm * 10000^{−2k/d})`
  `embed[2k+1] = cos(Re_norm * 10000^{−2k/d})`

for k = 0 ... d/2−1. With d=8, this replaces one input channel with 8 channels, changing `fun_dim` from 22 to 29. Different frequency bands sensitize different heads to coarse (low-freq) vs fine (high-freq) Re-regime variations. This is the same mechanism used by NeRF and Fourier feature networks to improve spectral bias on smooth functions. The expected beneficiary is `val_re_rand` (currently 86.494, the weakest OOD split).

**Implementation.**
In `dataset.py` or the feature-construction block of `train.py`, add a sinusoidal Re encoder:

```python
def sinusoidal_re_embed(log_re, d=8, max_log=7.0):
    # log_re: [N] tensor, already log-scaled
    re_norm = log_re / max_log  # normalize to [0,1]
    freqs = torch.pow(10000, -2 * torch.arange(d // 2, device=log_re.device) / d)
    # re_norm: [N], freqs: [d//2]
    args = re_norm.unsqueeze(-1) * freqs.unsqueeze(0)  # [N, d//2]
    return torch.cat([args.sin(), args.cos()], dim=-1)  # [N, d]
```

Replace dimension 13 (the scalar `log(Re)`) with the 8-dimensional output. Update the model's `fun_dim` parameter from 22 to 29. No other architecture changes.

```bash
cd target/ && python train.py \
  --grad_clip 5.0 --huber_delta 1.0 --ema_decay 0.99 \
  --re_embed_dim 8 \
  --wandb_group re-sinusoidal-embed --wandb_name re-sinusoidal-embed-d8 --agent <student>
```

Add `--re_embed_dim` CLI flag (default 1 = scalar, no change). When set to 8, inject the encoder and update `fun_dim` passed to `Transolver(...)`.

**Expected outcome.** Primary: `val_avg/mae_surf_p` < 90.6, driven by improvement in `val_re_rand`. Secondary: `val_single_in_dist` and `val_geom_camber_*` should not regress (they don't depend on Re variance). If val_re_rand drops >3% while other splits hold, merge even if val_avg gain is marginal — this is a generalisation win with compounding potential.

**Risk.** Medium. The change touches the input feature pipeline, which requires careful wiring to ensure the new `fun_dim=29` is passed through to every model constructor call. A mismatch would cause a shape error at runtime (fast fail, not a silent bug). The frequency scale `max_log=7.0` assumes `log(Re)` is already log-base-e and lies in [3.9, 7.0] for the dataset Re range [50K, 1M]; the student should verify this from the dataset stats before running.

**Student profile.** Student comfortable with feature pipeline modification. ~20–30 lines of code in dataset/feature construction plus wiring `fun_dim` through the model init. No training loop changes.

---

## Selection rationale

These four hypotheses cover four distinct axes of the optimization problem:

| Axis | Hypothesis | In-flight counterpart |
|---|---|---|
| Loss weighting (per-channel) | H-01 pressure-channel-weight | #3477 thorfinn (physics penalty, different axis) |
| Architecture capacity (token count) | H-02 slice-num-128 | #3571 fern (depth=6,7 — different dim) |
| Regularization (weight norm) | H-03 weight-decay-sweep | None |
| Feature representation (Re encoding) | H-04 re-sinusoidal-embed | None |

None overlap with each other or with in-flight PRs. All four are additive with the current EMA+clip+Huber configuration (they can be compounded if each individually wins). All four run within the 30-min wall-clock budget. Maximum new code per hypothesis: ~30 lines (H-04). Three of four (H-01, H-03, H-02 after checking flag wiring) require fewer than 10 lines of change.
