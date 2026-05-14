# Round 138 — Aux supervision at block-3 (75% depth)

## Hypothesis

Move auxiliary surface loss placement from block-1 (index 1, 50% depth, #2952 LOSS) to block-3 (index 2, 75% depth). Tests whether deeper-block features are more linearly mappable to surface output, reducing representational conflict.

## Motivation (#2952)

#2952 at block-1: val 31.1363 (+1.89% LOSS), test -1.26% slight WIN.
- Aux head learned FUNDAMENTALLY DIFFERENT projection (||W_aux||=3.33, cos sim 0.05 on p channel)
- Aux/surf ratio 2.5-3.7× throughout — mid features NOT linearly mappable to surface output
- Meta-signal repeated: cruise -3.16% WIN, in_dist +5.02% LOSS hardest hit

Student suggestion #2 verbatim: *"Try block 3 (index 2, 75% depth) — closer to output, features should be more linearly-mappable to surface target by then, so aux loss won't be a 3.5× residual."*

## Architecture

Same as #2952 except capture at index 2:
```python
for i, block in enumerate(self.blocks):
    x = block(x, fx=fx, T=T)
    if i == 2:  # block-3 input = 75% depth
        mid_features = x

self.aux_head_surf = nn.Linear(96, 3)  # +291 params, same as #2952
# Loss: total_loss = vol_loss + 10*surf_loss + 0.1*aux_loss
```

## Falsifiable predictions

- WIN: aux/surf ratio ~1.0-1.5× (vs #2952's 3.5×), cos sim of aux/primary heads ~0.5-0.9, val < 30.5605
- WASH: aux at deep placement is redundant; val ≈ baseline
- LOSS: same-direction loss as #2952, closes deep supervision axis comprehensively at moderate weight

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-alphonse \
    --experiment_name "charliepai2g48h5-alphonse/aux-supervision-at-block-3" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```
