# Baseline — icml-appendix-charlie-pai2f-r3

## Current Best Result

**Source:** PR #1093 — Compound baseline anchor: Lion+L1+EMA+bf16+n_layers=1+sw=28+cosine+clip (charliepai2f3-alphonse)

**Primary metric:** `val_avg/mae_surf_p = 47.3987`

**Configuration:** Lion optimizer + L1 loss + EMA(0.995) + bf16 autocast + n_layers=1 + surf_weight=28 + cosine scheduler (T_max=15) + grad_clip=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2

**Per-split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 50.0824 |
| val_geom_camber_rc | 62.7615 |
| val_geom_camber_cruise | 28.5501 |
| val_re_rand | 48.2009 |
| **val_avg** | **47.3987** |

**Training:** ~22 min, 50 epochs, batch_size=4, Peak VRAM: 9.02 GB

**Metrics path:** `target/models/model-charliepai2f3-alphonse-compound-baseline-lion-l1-ema-bf16-n1-20260429-102214/metrics.jsonl`

## Run Command

```bash
cd target/ && python train.py --n_layers 1 --bf16 --surf_weight 28.0 --optimizer lion --lr 3e-4 --weight_decay 1e-2 --loss l1 --scheduler cosine --T_max 15 --clip_grad_norm 1.0 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2 --batch_size 4
```

## Merge History

### 2026-04-29 — PR #1093: Compound baseline anchor (Lion+L1+EMA+bf16+n_layers=1+sw=28+cosine+clip)
- Previous: `val_avg/mae_surf_p = 47.7385` (charlie-pai2e-r5 reference)
- New best: `val_avg/mae_surf_p = 47.3987` (improvement: −0.3398)
- Student: charliepai2f3-alphonse
