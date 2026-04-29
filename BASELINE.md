# Baseline — icml-appendix-charlie-pai2f-r3

## Current Best Result

**Source:** PR #1148 — Extended Fourier freqs on (x,z): freqs=(1,2,4,8,16,32,64) (charliepai2f3-askeladd)

**Primary metric:** `val_avg/mae_surf_p = 43.9575`

**Configuration:** Lion optimizer + L1 loss + EMA(0.995) + bf16 autocast + n_layers=1 + surf_weight=28 + cosine scheduler (T_max=15) + grad_clip=1.0 + n_hidden=128 + n_head=4 + slice_num=64 + mlp_ratio=2 + Fourier positional encoding on (x,z) with freqs=(1,2,4,8,16,32,64)

**Per-split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| val_single_in_dist | 44.6169 |
| val_geom_camber_rc | 57.7367 |
| val_geom_camber_cruise | 26.7301 |
| val_re_rand | 46.7462 |
| **val_avg** | **43.9575** |

**Test split breakdown:**
| Split | mae_surf_p |
|-------|-----------|
| test_single_in_dist | 38.1831 |
| test_geom_camber_rc | 52.2408 |
| test_geom_camber_cruise | 22.2041 |
| test_re_rand | 37.1885 |
| **test_avg** | **37.4541** |

**Training:** ~21.5 min, 50 epochs (best epoch 48), batch_size=4, Peak VRAM: 9.40 GB, n_params: 184,903

**Metrics path:** `target/models/model-charliepai2f3-askeladd-fourier-freqs-7-20260429-121807/metrics.jsonl`

## Run Command

```bash
cd target/ && python train.py --n_layers 1 --bf16 --surf_weight 28.0 --optimizer lion --lr 3e-4 --weight_decay 1e-2 --loss l1 --scheduler cosine --T_max 15 --clip_grad_norm 1.0 --n_hidden 128 --n_head 4 --slice_num 64 --mlp_ratio 2 --batch_size 4 --fourier_pos_enc --fourier_freqs 1 2 4 8 16 32 64
```

Note: The Fourier positional encoding appends `sin(f*pi*xy)` and `cos(f*pi*xy)` for freqs in (1,2,4,8,16,32,64) to the (x,z) dims, expanding the spatial input from 2-dim to 30-dim. Input feature dimension becomes 52. Adding freq=128 regresses sharply (aliasing near mesh-element spacing scale).

## Merge History

### 2026-04-29 — PR #1148: Extended Fourier freqs on (x,z): freqs=(1,2,4,8,16,32,64) (charliepai2f3-askeladd)
- Previous: `val_avg/mae_surf_p = 44.4154` (PR #1106, Fourier pos enc freqs=(1,2,4,8,16))
- New best: `val_avg/mae_surf_p = 43.9575` (improvement: −0.4579, −1.03%)
- Student: charliepai2f3-askeladd
- Key finding: freqs=(1,2,4,8,16,32,64) beats baseline; adding freq=128 regresses (Nyquist aliasing near mesh resolution)

### 2026-04-29 — PR #1106: Fourier positional encoding on (x,z) (charliepai2f3-frieren)
- Previous: `val_avg/mae_surf_p = 47.3987` (PR #1093, compound baseline)
- New best: `val_avg/mae_surf_p = 44.4154` (improvement: −2.9833, −6.29%)
- Student: charliepai2f3-frieren
- Also included: NaN fix for test_geom_camber_cruise (non-finite GT entries in sample 20 masked via y_finite guard in evaluate_split)

### 2026-04-29 — PR #1093: Compound baseline anchor (Lion+L1+EMA+bf16+n_layers=1+sw=28+cosine+clip)
- Previous: `val_avg/mae_surf_p = 47.7385` (charlie-pai2e-r5 reference)
- New best: `val_avg/mae_surf_p = 47.3987` (improvement: −0.3398)
- Student: charliepai2f3-alphonse
