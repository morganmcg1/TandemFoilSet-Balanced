# Hypothesis: dropout-sweep (askeladd)

## Hypothesis

The current Transolver model is trained with **`dropout=0.0`** (zero regularization
inside the attention and MLP modules). The model has 2.7M parameters fitting on a
mid-sized CFD dataset. With the SOTA stack reaching val=65.30 at ep19 with
**val still descending at timeout**, there are two possible interpretations:

1. **Underfitting** — the model has more to learn; more capacity / more epochs would
   help. (This is what frieren's #3801 T_max=25 and tanjiro's #3821 plateau-tail
   are testing.)

2. **Overfitting on in-dist while under-generalizing on OOD splits** — the gap
   between val_single_in_dist (~72) and val_geom_camber_cruise (~42) plus the
   test-set OOD penalty (test_geom_camber_rc=67.58 worse than test_re_rand=54.40)
   suggests OOD splits are systematically harder. Dropout is the textbook
   regularizer for this: it forces the model to learn redundant representations
   that generalize across distributions.

This experiment tests interpretation #2. **Dropout has never been tried on this
stack across 6 rounds of optimization.** Recent transformer literature (and the
ViT lineage that Transolver descends from) consistently shows that small dropout
in attention/MLP yields modest but reliable OOD wins.

**Two arms:**

1. **Arm 1 — `dropout=0.05`** (light): minimal disruption to training dynamics,
   tests whether even small regularization helps OOD generalization.

2. **Arm 2 — `dropout=0.10`** (moderate): standard ViT-scale dropout. Stronger
   regularization signal; risks slowing convergence in our 19-epoch budget.

**Predicted improvement:** −0.2 to −1.5 if OOD generalization is the bottleneck.
The OOD splits (val_geom_camber_rc, val_re_rand) should benefit most.

**Worst case:** dropout slows learning enough that the 30-min budget runs out
before convergence, and val gets *worse*.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full SOTA stack (Lion lr=2e-4,
bf16+clip=1.0+eta_min=1e-5+T_max=21). **Do NOT change anything else.**

### 2. Expose `dropout` as a CLI flag

In `target/train.py`, find the `model_config` dict around line 488–499:

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Add `dropout=cfg.dropout` to the dict, and add to the Config dataclass:

```python
dropout: float = 0.0
```

The Transolver class already accepts `dropout` (passes it to attention and to MLP
via `Dropout` layers in `model.py`) — no model code changes needed.

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — dropout=0.05 (light regularization):**

```bash
cd target/ && python train.py \
    --dropout 0.05 \
    --wandb_group dropout-sweep \
    --wandb_name dropout-0p05 \
    --agent willowpai2i24h3-askeladd
```

**Arm 2 — dropout=0.10 (moderate regularization):**

```bash
cd target/ && python train.py \
    --dropout 0.10 \
    --wandb_group dropout-sweep \
    --wandb_name dropout-0p10 \
    --agent willowpai2i24h3-askeladd
```

### 5. Key signals to report

- `val_avg/mae_surf_p` per epoch — does either arm reach below 65.30?
- **Per-split breakdown at best checkpoint** — this is the most important diagnostic:
  - In-dist (`val_single_in_dist`, `test_single_in_dist`) — likely worse with dropout
    if model was already well-fit on these
  - OOD (`val_geom_camber_rc`, `val_re_rand`, `val_geom_camber_cruise`) — likely
    better with dropout IF the underfit hypothesis is wrong
- **Convergence rate** — does dropout=0.10 slow val descent visibly? If best epoch
  ≥19 with val still descending, the budget is too short for that dropout level.
- **Train loss trajectory** — dropout should raise train loss while (hopefully) lowering
  val. A widening train↔val gap means dropout is doing its job.

### 6. Compute nansafe test metrics

```bash
cd target/ && python eval_nansafe.py <arm1_run_id>
cd target/ && python eval_nansafe.py <arm2_run_id>
```

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — frieren's lion-lr2e4 (PR #3675, merged 2026-05-16 07:30 UTC):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **65.2991** |
| **test_avg_nansafe/mae_surf_p** | **60.5400** |
| test_single_in_dist | 64.0454 |
| test_geom_camber_rc | 67.5770 |
| test_geom_camber_cruise | 56.1342 |
| test_re_rand | 54.4033 |
| W&B run | `3rvfeq4g` (group: `lion-lr-sweep`) |
| Stack | Lion lr=2e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + T_max=21 |
| Model | Transolver L=5, H=128, n_head=4, slice_num=64, **dropout=0.0** |
| Best epoch | **19** (FINAL — val still descending at timeout) |

Reproduce:
```bash
cd "target/" && python train.py --lr 2e-4 --lr_T_max 21 --wandb_group lion-lr-sweep --wandb_name lion-lr2e4 --agent willowpai2i24h3-frieren
```

## Why this matters

Dropout is the most-used, lowest-cost regularizer in transformer ML. It's never
been tested on this stack. If it helps even by −0.3, that's a free win. If it
hurts, we learn that the model is genuinely underfit (not overfit) and the right
direction is more capacity / more epochs, not regularization.
