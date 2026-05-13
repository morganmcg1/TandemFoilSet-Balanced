# LayerScale on n_layers=3+slice_num=16+epochs=36: residual branch scaling

## Hypothesis

Tests LayerScale (CaiT, Touvron et al. 2021) — learnable per-channel scaling of residual branches. Adds gamma_attn and gamma_ffn vectors (shape n_hidden=128) init to 1e-4 per transformer block. Improves gradient flow at init, enables selective branch suppression. ~768 extra params total.

## Instructions

Modify TransformerBlock in train.py:
- Add: `self.gamma_attn = nn.Parameter(1e-4 * torch.ones(dim))`
- Add: `self.gamma_ffn = nn.Parameter(1e-4 * torch.ones(dim))`
- Change: `x = x + self.attn(...)` → `x = x + self.gamma_attn * self.attn(...)`
- Change: `x = x + self.ff(...)` → `x = x + self.gamma_ffn * self.ff(...)`

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-nezuko \
  --experiment_name layerscale-slicenum16-nlayers3 \
  --epochs 36 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 3 \
  --slice_num 16
```

## Baseline (PR #2348)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 35.263 | 32.248 |
| geom_camber_rc | 49.105 | 44.663 |
| geom_camber_cruise | 19.392 | 16.188 |
| re_rand | 38.431 | 28.282 |
| **avg** | **35.548** | **30.345** |
