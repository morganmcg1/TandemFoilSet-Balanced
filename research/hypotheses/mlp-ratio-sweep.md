# Hypothesis: mlp-ratio-sweep (nezuko)

## Hypothesis

The Transolver's per-layer feed-forward (MLP) block has hidden dim `n_hidden *
mlp_ratio`. The current value `mlp_ratio=2` (MLP hidden = 256 at H=128) was
inherited from the initial repo config and has **never been tuned** across 6
rounds of optimization. The standard transformer convention is `mlp_ratio=4`
(MLP hidden = 4 × embedding dim), used in BERT, GPT, ViT, etc. — but Transolver
ships with the smaller `2×` ratio.

Your prior H=160 test cleanly answered the per-token-width capacity question:
**widening n_hidden does not beat SOTA** (H=160 val=65.78 vs SOTA 65.30 — essentially
tied on val, marginally better on test). This narrows the capacity question to
the OTHER axes:

- **slice_num** (alphonse #3876 testing concurrently): how many basis vectors
  the field decomposes into
- **mlp_ratio** (THIS PR): how much non-linear processing each layer applies
  per-token

These are independent capacity dimensions. The fact that more per-token width
(H=160) didn't help suggests the bottleneck may not be representational width
but **per-layer compute** — how much non-linear refinement each layer can apply
to its representation before passing it on. `mlp_ratio` is exactly that lever.

**Two arms:**

1. **Arm 1 — `mlp_ratio=3`** (MLP hidden=384): conservative increase, tests
   whether modest MLP capacity helps. Param count grows ~25%, VRAM stays small.

2. **Arm 2 — `mlp_ratio=4`** (MLP hidden=512): standard transformer convention.
   Param count grows ~50%, but the model is still ~4M params total — comfortably
   below VRAM ceiling.

**Predicted improvement:** −0.2 to −1.0 if MLP capacity is the bottleneck.
Lion+T_max=21 is well-calibrated to the existing model size, but if the
bottleneck is per-token refinement capacity, more MLP width should help with
the persistently hard OOD splits (geom_camber_rc=67.58 in test, val_single_in_dist
still high).

**Worst case:** Wider MLP makes the model harder to optimize in the 30-min /
~19-epoch budget — best epoch lands earlier (under-converged), val regresses.
But seeing this would itself be informative (says budget, not capacity, is the
binding constraint).

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full SOTA stack (Lion lr=2e-4,
bf16+clip=1.0+eta_min=1e-5+T_max=21). **Do NOT change anything else.**

### 2. Expose `mlp_ratio` as a CLI flag

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
    mlp_ratio=2,      # ← hardcoded; change to cfg.mlp_ratio
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Make the change:

```python
mlp_ratio=cfg.mlp_ratio,
```

Add to the Config dataclass (default `2` to preserve baseline):

```python
mlp_ratio: int = 2
```

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — mlp_ratio=3 (MLP hidden=384, modest capacity bump):**

```bash
cd target/ && python train.py \
    --mlp_ratio 3 \
    --wandb_group mlp-ratio-sweep \
    --wandb_name mlpr-3 \
    --agent willowpai2i24h3-nezuko
```

**Arm 2 — mlp_ratio=4 (MLP hidden=512, standard transformer convention):**

```bash
cd target/ && python train.py \
    --mlp_ratio 4 \
    --wandb_group mlp-ratio-sweep \
    --wandb_name mlpr-4 \
    --agent willowpai2i24h3-nezuko
```

### 5. Key signals to report

- `val_avg/mae_surf_p` per epoch — does either arm reach below 65.30?
- **Parameter count** logged for each arm — mlp_ratio=3 should add ~25%,
  mlp_ratio=4 should add ~50%. Confirm.
- **Epoch time** — wider MLPs may slow per-epoch. If under 110 s/epoch, you
  still have time for 17+ epochs; if higher, may want to nudge T_max down.
- **Best epoch** — does best epoch shift earlier (slower convergence due to
  more params)? If best epoch is still ep18-19 at timeout, the curve is
  still descending and we're seeing the under-converged regime.
- **Per-split breakdown at best checkpoint** vs SOTA `3rvfeq4g` — especially
  watch `val_geom_camber_rc` (persistent hardest split). More MLP capacity
  could help if the model is bottlenecked on geometric pattern complexity.
- **Peak VRAM** — should be well within 96 GB ceiling for both arms.

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
| Model | Transolver L=5, H=128, n_head=4, slice_num=64, **mlp_ratio=2** |
| Best epoch | **19** (FINAL — val still descending at timeout) |

Reproduce:
```bash
cd "target/" && python train.py --lr 2e-4 --lr_T_max 21 --wandb_group lion-lr-sweep --wandb_name lion-lr2e4 --agent willowpai2i24h3-frieren
```

## Why this matters

Your prior H=160 work definitively closed the "wider n_hidden" question — extra
per-token width doesn't help on this dataset. But Transolver also ships with an
unusually narrow MLP ratio (`2×` vs the standard `4×` in modern transformers).
This is a clean, untested architectural lever. If either arm wins, we have a new
SOTA and a strong signal about what kind of capacity the model needs. If both
lose, we've closed another architectural dimension, complementing your H=160
finding and pointing the next round toward non-architectural levers
(loss/augmentation/data).

Together with alphonse's slice_num sweep (capacity in basis-vector dimension) and
askeladd's dropout sweep (regularization), the three concurrent architectural
tests will give us a strong map of which model-design dimensions still hold
headroom on this dataset.
