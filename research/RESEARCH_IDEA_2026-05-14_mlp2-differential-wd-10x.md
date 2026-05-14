# Round 138 — Differential weight decay 10× on mlp2 (shared head)

## Hypothesis

Apply 10× weight decay (3e-3 vs baseline 3e-4) ONLY to `mlp2` parameters (the shared output head), keeping baseline wd on all other params. Tests whether forcing the shared head to stay closer to small-norm values preserves the early-training surface-tuning that #2956 broke.

## Motivation (#2956 close)

#2956 askeladd asymmetric-surf-correction-head LOSS (val +3.29%, test +3.06%). Mechanism: zero-init surface-only correction broke co-adaptation in `mlp2`. Even though correction grew correctly on the p channel, the shared `mlp2` drifted toward vol-friendly weights — the implicit regularization came from co-adaptation pressure, not parameter sharing.

Student suggestion #2 verbatim: *"Regularize the shared head's surface-tuning. Instead of adding an asymmetric correction, constrain the shared head's drift: e.g. EMA-regularize `mlp2` toward its early-training state, or apply higher weight decay only to `mlp2`. Tests the dual hypothesis: it's not specialization that's needed, it's preserving early shared-head surface-tuning."*

Direct one-line test of the co-adaptation-preservation hypothesis.

## Architecture

```python
# Standard pattern in optimizer construction:
mlp2_params = [p for n, p in model.named_parameters() if 'mlp2' in n]
other_params = [p for n, p in model.named_parameters() if 'mlp2' not in n]

optimizer = Lion(
    [
        {'params': other_params, 'weight_decay': 3e-4},   # baseline wd
        {'params': mlp2_params, 'weight_decay': 3e-3},    # 10× wd on shared head
    ],
    lr=1.5e-4,
    betas=(0.9, 0.99)
)
```

Zero new params, ~6 lines of optimizer config change.

## Falsifiable predictions

- **WIN** (val < 30.5605): Strong regularization on mlp2 preserves co-adaptation → confirms #2956 hypothesis
- **PARTIAL** (val ≈ 30.5605 ± 1%): Differential wd helps marginally; co-adaptation mechanism partly correct
- **LOSS** (val > 31.0): Pinning mlp2 hurts — co-adaptation needs FREEDOM to evolve, not constraint

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-askeladd \
    --experiment_name "charliepai2g48h5-askeladd/mlp2-differential-wd-10x" \
    --lr 1.5e-4 \
    --weight_decay 3e-4 \
    --epochs 60
```

## Reporting

1. val_avg/test_avg vs baseline
2. Per-split val + test breakdown
3. **mlp2 weight norm trajectory** at ep1, 5, 10, 30, 60. Compare to baseline trajectory.
4. **Mlp2 weight norm vs body weight norm ratio** — does the differential decay actually keep mlp2 norm smaller?
5. Param count (407,940, unchanged)
6. Meta-signal check: cruise WIN / in_dist LOSS pattern?
7. Plain-language verdict: WIN / PARTIAL / LOSS
