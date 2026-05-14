# Round 137 — Slice-routing temperature T=0.25 (sharper-direction symmetric closure)

## Hypothesis

Hardcode `Physics_Attention` slice-routing softmax temperature to **T=0.25** (sharper than baseline init 0.5) at all 4 blocks. Opposite-direction followup to #2944 (T=2.0 LOSS); maps the temperature axis on both sides of baseline.

## Motivation

#2944 hardcoded T=2.0 (4× softer than baseline init 0.5) → val +4.43% LOSS / test +2.02% LOSS. Block-3 entropy rose from ~0.01-0.92 nats (collapsed at baseline) to 2.34-2.91 nats (near ceiling). Decisive mechanistic conclusion: sharp routing at block-3 is LOAD-BEARING. This PR tests the opposite direction — does SHARPER routing help?

If sharp specialization is the load-bearing mechanism, pushing T further down (toward 0) should reinforce it. Counter-hypothesis: removing learnability is what hurt #2944 (not softness), in which case ANY hardcoded T (including 0.25) will LOSS by similar amount.

## Two-sided axis closure logic

| T value | val_avg | Effect on routing |
|---|---|---|
| 0.25 (this PR) | ? | 2× sharper than baseline init |
| 0.5 (baseline init, learnable) | 30.5605 | baseline |
| 2.0 (#2944, hardcoded) | 31.9159 (+4.43%) | 4× softer than baseline init |

Three outcomes:
- T=0.25 WIN (val < 30.5605): monotonic axis, sharper better
- T=0.25 WASH (val ≈ 30.5605): U-shaped axis, baseline at minimum
- T=0.25 LOSS magnitude ≈ #2944: learnability matters more than value
- T=0.25 LOSS magnitude < #2944: asymmetric axis, sharper hurts less than softer

## Implementation

Single-line change from #2944: `ROUTING_TEMPERATURE = 0.25` (was 2.0). Same 2-change structure (removes learnability + hardcodes). 8 fewer params (407,932 vs baseline 407,940).

Reuse the routing-entropy forward-hook from #2944 — at ep60, compare three points (baseline vs #2944 T=2.0 vs this PR T=0.25).

## Falsifiable predictions

- WIN: sharper helps. Try T=0.1 (extreme) and restore learnability with init=0.25.
- WASH: hardcoded T at right value ≈ baseline.
- LOSS ≈ #2944: learnability is the lever.
- LOSS < #2944: asymmetric axis.

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-tanjiro \
    --experiment_name "charliepai2g48h5-tanjiro/slice-routing-temperature-0.25" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```
