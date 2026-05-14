# Round 138 — Asymmetric surface correction head (zero-init residual on top of shared head)

## Hypothesis

Keep baseline shared output head intact. ADD a small ZERO-INIT `surf_correction = nn.Linear(96, 3)` whose output is added to surface tokens' shared head predictions. Vol tokens use shared head unchanged.

## Motivation (#2946)

#2946 askeladd `separate-surf-vol-heads`: split head_surf + head_vol → val 31.8968 (+4.37% LOSS). Head divergence DID happen (1.57 > 0.5 null) but in WRONG direction:
- Vol head ||W||=1.98 > Surf head ||W||=1.43 despite surf_weight=10 pulling 10× harder on surf
- Vol head FREE-RODE (escaped surf_weight constraint)
- Divergence concentrated on Ux/Uy (1.00, 0.98), weak on metric channel p (0.70)
- KEY INSIGHT: shared head is an implicit regularizer, not a compromise penalty

#2946 student directly recommended: *"split ONLY head_surf, keep head_vol shared with main projection — isolates 'specialization in surface direction only', avoids vol-head free-rider failure mode."*

## Architecture

```python
# In __init__ (after existing mlp2)
self.surf_correction = nn.Linear(n_hidden, 3)
nn.init.zeros_(self.surf_correction.weight)
nn.init.zeros_(self.surf_correction.bias)

# In forward
predictions = self.mlp2(x)  # shared, unchanged
surf_corr = self.surf_correction(x)
predictions = predictions + surf_mask.unsqueeze(-1) * surf_corr  # surf tokens only
```

+291 params, zero-init means step-0 = baseline behavior.

## Why this might WIN

- Vol head can't free-ride (it's still shared, surf_weight=10 still pulls on it via shared head)
- Surface gets DEDICATED additional capacity
- Zero-init: gradient drives correction only when it improves loss
- Pressure channel (metric-critical) gets first shot at specialization

## Falsifiable predictions

- WIN: surf_correction weight grows materially (||W|| > 0.1 at ep60); val < 30.5605
- WASH: surf_correction weight stays near 0; val ≈ baseline (extra params unused)
- LOSS: even zero-init asymmetric correction hurts → head-axis is fundamentally wrong place to attack

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-askeladd \
    --experiment_name "charliepai2g48h5-askeladd/asymmetric-surf-correction-head" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

## Diagnostics required

1. ||W_surf_correction|| trajectory at ep1, 5, 10, 30, 60
2. Per-channel ||W_corr[Ux/Uy/p]|| at ep60
3. Surface-prediction delta diagnostic at ep60
4. Meta-signal check (does cruise WIN / in_dist LOSS repeat?)
