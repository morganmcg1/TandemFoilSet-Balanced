# LayerScale on n_layers=2+slice_num=16+epochs=46: Stabilization at new depth

## Hypothesis

LayerScale (Touvron et al., CaiT 2021) adds a learnable γ to each residual branch with init=1e-4, allowing deeper transformers to train more reliably. While n_layers=2 is shallow, LayerScale should provide:

1. **Improved gradient flow**: Even at n_layers=2, LayerScale's small init forces early-training residual branches to act as identity, letting the model learn shortcuts and only gradually mix in transformer features.
2. **Counter single_in_dist regression**: The +1.21 in-dist regression at n_layers=2 may stem from over-aggressive feature mixing too early in training. LayerScale's slow ramp-up of residual contributions may help preserve early identity features that matter for in-distribution geometries.
3. **Cosine annealing complements LayerScale**: With T_max=46 and LR cosine schedule, the small γ init combined with high LR early should let the model discover when to engage the transformer layers more strongly. This is a known dynamic in CaiT and DeiT III.
4. **Low complexity, established technique**: Adds two learnable γ tensors per residual branch (one for attention, one for MLP) — total ~few hundred extra params. Established in the literature with strong empirical support.

LayerScale was previously tried at n_layers=3 (PR #2390 area, nezuko-layerscale-slicenum16-nlayers3) — let's see if it pairs better with the new winning depth stack.

## Instructions

Add LayerScale to all residual branches of the Transolver blocks. The standard implementation:

```python
# Inside Transolver_block (or equivalent block)
self.gamma_1 = nn.Parameter(1e-4 * torch.ones(hidden_dim), requires_grad=True)
self.gamma_2 = nn.Parameter(1e-4 * torch.ones(hidden_dim), requires_grad=True)

# In forward()
x = x + self.gamma_1 * self.attn(self.norm1(x))
x = x + self.gamma_2 * self.mlp(self.norm2(x))
```

Set init=1e-4 (CaiT default). All other hyperparameters match PR #2468:

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-nezuko \
  --experiment_name layerscale-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs **NEW** baseline (val=35.256 / test=30.245)
2. Per-split mae_vol_p
3. **single_in_dist val mae_surf_p** — KEY METRIC: does LayerScale recover the +1.21 in-dist regression?
4. Train loss trajectory epochs 1-5: slower start expected from LayerScale (γ near zero)?
5. Best epoch, total wall-clock, peak memory
6. Confirm γ values at end of training (are they still small, or have they grown?)

## Baseline (PR #2468)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 36.476 | 33.035 |
| geom_camber_rc | 48.297 | 44.333 |
| geom_camber_cruise | 18.326 | 15.496 |
| re_rand | 37.923 | 28.116 |
| **avg** | **35.256** | **30.245** |

**Reproduce baseline:**
```bash
cd target/ && python train.py \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 --n_layers 2 --slice_num 16
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
