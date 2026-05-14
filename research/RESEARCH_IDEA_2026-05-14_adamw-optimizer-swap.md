# Round 138 — AdamW optimizer swap (lr=5e-4, wd=0.01)

## Hypothesis

Replace the Lion optimizer with AdamW (lr=5e-4, wd=0.01, betas=(0.9, 0.999)). Tests the OPTIMIZER-FAMILY axis (Lion vs Adam-family) — fundamentally different update rule, fundamentally different exploration of the loss landscape.

## Motivation

After 134 closed taxa, EVERY closure has been on a Lion-trained model. Lion's sign(grad) update is unusual — most modern transformer literature uses AdamW. The optimizer-family axis is the single LARGEST unexplored axis in this launch.

Lion (per Chen et al 2023): `update = sign(β₁·m + (1-β₁)·g)` — sign-step preserves direction per-param but discards magnitude. AdamW: `update = m_hat / (sqrt(v_hat) + ε)` — element-wise adaptive scaling.

Conversion factor: Lion's effective learning rate is ~3-10× smaller than AdamW's typical lr. Baseline Lion@1.5e-4 → AdamW@5e-4 is the conservative 3.3× factor.

If AdamW WIN: Lion was the bottleneck. Suggests revisiting hyperparameter sweeps on AdamW.
If AdamW LOSS: Lion is the right optimizer. Closes the optimizer-family axis.

## Architecture

```python
# Replace:
# optimizer = Lion(model.parameters(), lr=1.5e-4, weight_decay=3e-4, betas=(0.9, 0.99))

# With:
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=0.01,       # standard transformer AdamW wd
    betas=(0.9, 0.999),       # standard AdamW betas
)
```

Keep all other training config unchanged (SequentialLR warmup+cosine, epochs=60).

Zero new params. Single-line optimizer construction change.

## Why this might WIN

1. **AdamW is the canonical transformer optimizer.** BERT, GPT, ViT, T5 all use AdamW. Lion is younger (2023) and less universally validated.
2. **Adaptive per-param step size.** AdamW's `sqrt(v_hat)` denominator gives implicit per-param learning rates; Lion's sign-step doesn't. May better handle the heterogeneous CFD parameter dimensions.
3. **Different convergence dynamics.** AdamW typically converges faster early but plateaus at similar final loss. In 60ep underfit regime, faster early convergence helps.
4. **Standard weight decay magnitude.** AdamW's 0.01 wd is well-tuned for transformers; Lion's 3e-4 was a guess.

## Why this might LOSS

1. **Lion was clearly working** (baseline reached). Switching optimizers may take many sweeps to match.
2. **AdamW's adaptive denom may overfit to gradient variance**, hurt OOD generalization.
3. **lr conversion may be off.** 5e-4 may be too low or too high; could need 1e-3 or 2e-4.
4. **wd interaction with optimizer.** AdamW's wd is decoupled (Loshchilov), Lion's is also decoupled — should be comparable, but absolute values differ.
5. **Memory footprint:** AdamW maintains 2 momentum buffers per param (vs Lion's 1) — ~+1.5MB. Trivial at 407k params.

## Falsifiable predictions

- **WIN** (val < 30.5605): AdamW outperforms Lion. Suggests Lion was the bottleneck. Try AdamW with lr sweep (1e-4 to 1e-3).
- **PARTIAL** (val ≈ 30.5605 ± 1%): Optimizers comparable; Lion choice was neutral.
- **LOSS** (val > 31.0): Lion is the right optimizer. Closes optimizer-family axis.

## Reproduce command

```bash
cd target/ && python train.py \
    --agent charliepai2g48h5-thorfinn \
    --experiment_name "charliepai2g48h5-thorfinn/adamw-optimizer-swap" \
    --lr 5e-4 \
    --weight_decay 0.01 \
    --epochs 60
```

NOTE: The student must verify the `--lr 5e-4` flag is plumbed to AdamW construction. If the existing argparser passes --lr/--weight_decay to a Lion-only branch, the student needs to add an `--optimizer adamw` flag OR hardcode the optimizer swap in the construction code path.

## Reporting

1. val_avg/test_avg vs baseline + per-split breakdown
2. Param count (407,940 unchanged) + momentum buffer count (verify AdamW has 2 buffers vs Lion's 1)
3. Train→val gap (typically smaller for AdamW)
4. Epochs completed, sec/epoch (slightly higher for AdamW), peak GPU memory
5. **Meta-signal check:** does cruise WIN / in_dist LOSS pattern repeat under AdamW? Critical test — meta-signal MAY be Lion-specific.
6. Plain-language verdict: WIN / PARTIAL / LOSS
