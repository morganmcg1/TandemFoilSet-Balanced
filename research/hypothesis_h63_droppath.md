## Hypothesis

**H63: DropPath (stochastic depth) at the residual blocks improves OOD generalization on GEGLU Transolver.**

Stochastic depth (Huang et al. 2016, CaiT/ViT in Touvron et al. 2021) randomly zeros entire residual blocks during training: `x = x + drop_path(residual_branch(x))` where `drop_path` keeps the residual branch with probability `1-p` and zeros it otherwise. Effects:
1. Forces each residual block to learn a "stable" identity-friendly representation — if a block is randomly dropped, the network must still function.
2. Acts as an implicit ensemble during training (random subsets of layers).
3. Particularly strong for OOD generalization in transformers — the model can't rely on any specific layer for any specific feature.

For TandemFoilSet OOD (camber_rc, re_rand splits): the model needs representations that generalize across geometries and Reynolds numbers. DropPath should help because it prevents any single GEGLU block from over-specializing to the training distribution's most common features.

**Linear schedule:** drop_prob at layer i = `i / (n_layers - 1) * max_drop_prob`. For 5 layers with max_drop_prob=0.1, layers see drop_probs [0.0, 0.025, 0.05, 0.075, 0.1]. This puts most regularization at deeper layers (which see more abstract features).

**Two arms (max_drop_prob sweep):**
- **Arm A: max_drop_prob=0.05** — light regularization
- **Arm B: max_drop_prob=0.1** — standard Llama/ViT-Base regularization

**Predicted:** Arm A ~57.5-58.5; Arm B ~57-58 if OOD gain is real. Expected to help OOD splits (camber_rc, re_rand) more than in_dist.

**Risk:** DropPath at small models can hurt — the regularization removes information the model needed. Both arms may regress slightly. If both go up >1 pt, dropPath is too aggressive for this 891k-param model.

## Instructions

Add DropPath as a residual-branch dropout. The implementation:

```python
class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Generate random tensor [batch, 1, 1] for per-sample drop
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor
```

Then in the Transolver block: wrap each residual branch with DropPath. For a 5-layer model with `--drop_path 0.1`:

```python
# Linear schedule across layers
drop_probs = [drop_path * i / (n_layers - 1) for i in range(n_layers)]
# Layer i: x = x + drop_path_i(attn_residual(x))
#         x = x + drop_path_i(ffn_residual(x))
```

Add `--drop_path` flag (default 0.0). The two arms run with `--drop_path 0.05` and `--drop_path 0.1`.

Run both arms:

```bash
# Arm A — drop_path=0.05 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h63-droppath05-geglu \
  --agent charliepai2i48h3-frieren \
  --drop_path 0.05 \
  --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0

# Arm B — drop_path=0.10 at GEGLU base
cd target/ && python train.py --epochs 50 \
  --experiment_name h63-droppath10-geglu \
  --agent charliepai2i48h3-frieren \
  --drop_path 0.10 \
  --ffn_act geglu \
  --n_head 2 --lr 1e-3 --weight_decay 5e-5 --clip_grad_norm 1.0
```

All other flags: FiLM cond_dim=11, huber_delta_vel=0.5, huber_delta_p=0.25, surf_weight=10, n_hidden=128, slice_num=64, T_max=15 (current merged defaults).

**Critical: verify DropPath is OFF during eval.** `model.eval()` should set `self.training=False`, but verify by:
- During training: log mean drop_path activation on a forward pass (should be ~max_drop_prob)
- During eval: log mean drop_path activation (should be 0)

**Report:**
- val_avg/mae_surf_p, per-split breakdown for both arms
- test_avg/mae_surf_p (3-split, excl. cruise) and per-split test  
- **OOD vs in_dist gain breakdown** — is the gain larger on camber_rc and re_rand (OOD) than single_in_dist (in_dist)? That's the predicted signature
- Best epoch and epochs completed
- Per-epoch val_avg trajectory — DropPath training is noisier; expect higher per-epoch variance than H48 baseline
- Peak GPU memory and mean s/epoch (DropPath is cheap — negligible overhead)

Commit `metrics.jsonl` + `metrics.yaml` + `config.yaml` for both arms.

**Stop early if diverging:** val_avg at epoch 3 > 300 → kill and report. DropPath can cause noisy early training.

## Baseline

**Current best — PR #3834 — H48: GEGLU gated FFN (askeladd, no DropPath)**

| Metric | Value |
|--------|-------|
| **val_avg/mae_surf_p** | **58.6268** |
| val_single_in_dist/mae_surf_p | 61.6193 |
| val_geom_camber_rc/mae_surf_p | 73.8983 (OOD split) |
| val_geom_camber_cruise/mae_surf_p | 40.4338 |
| val_re_rand/mae_surf_p | 58.5556 (OOD split) |
| test_avg/mae_surf_p (3-split, excl. cruise) | **56.6976** |

Config: FiLM cond_dim=11 + Huber δ_vel=0.5/δ_p=0.25 + T_max=15 + clip_grad_norm=1.0 + lr=1e-3 + n_head=2 + wd=5e-5 + ffn_act=geglu + **drop_path=0.0** (current default).

**Beat this: val_avg/mae_surf_p < 58.6268**

Predicted Arm A (drop_path=0.05) ≈ 57.5-58.5; Arm B (drop_path=0.1) ≈ 57-58.

⚠ `test_avg/mae_surf_p` will appear NaN — pre-existing scoring bug. Report 3-split excl. cruise.
