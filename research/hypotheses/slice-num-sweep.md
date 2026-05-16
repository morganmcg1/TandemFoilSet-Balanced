# Hypothesis: slice-num-sweep (alphonse)

## Hypothesis

The Transolver's PhysicsAttention module partitions the input field into `slice_num`
learnable "slices" — each slice is a soft assignment of mesh nodes to a learnable
basis, and attention is computed between slice representations rather than between
individual nodes (O(N×slice_num) vs O(N²)).

The current value `slice_num=64` was inherited from the initial repo config and has
**never been tuned** across 6 rounds of optimization. The 7 other levers in the SOTA
stack (Lion, lr=2e-4, T_max=21, etc.) were calibrated assuming this value. But
slice_num controls a fundamentally different model capacity dimension than
`n_hidden=128`:

- **n_hidden** = per-token feature width (what nezuko is testing at H=160 in #3745)
- **slice_num** = how many basis vectors the field is decomposed into

These are orthogonal capacity dimensions. The fact that H=160 (more per-token width)
didn't beat H=128 in nezuko's Arm 1 (val=65.78) hints that the bottleneck may not be
per-token width but **basis count** — how many independent flow modes the model can
represent.

**Two arms:**

1. **Arm 1 — `slice_num=32`** (half the current value): tests whether the model is
   over-parametrized in slice space. If fewer slices works as well, the current
   slice_num is wasted capacity. Cheaper inference too.

2. **Arm 2 — `slice_num=96`** (1.5× current): tests whether MORE basis vectors help.
   **AVOID slice_num=128** — known to produce inf (infra bug, documented).

**Predicted improvement:** −0.2 to −1.0 if either direction matters. Worst case:
the lever is fully consumed by the existing stack.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full SOTA stack (Lion lr=2e-4,
bf16+clip=1.0+eta_min=1e-5+T_max=21). **Do NOT change anything else.**

### 2. Expose `slice_num` as a CLI flag

In `target/train.py`, find the `model_config` dict around line 488–499:

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,
    slice_num=64,   # ← hardcoded; change to cfg.slice_num
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Make the change:

```python
slice_num=cfg.slice_num,
```

Add to the Config dataclass (default `64` to preserve baseline):

```python
slice_num: int = 64
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

**Arm 1 (primary) — slice_num=32 (half capacity):**

```bash
cd target/ && python train.py \
    --slice_num 32 \
    --wandb_group slice-num-sweep \
    --wandb_name slice-32 \
    --agent willowpai2i24h3-alphonse
```

**Arm 2 — slice_num=96 (1.5× capacity, AVOID 128):**

```bash
cd target/ && python train.py \
    --slice_num 96 \
    --wandb_group slice-num-sweep \
    --wandb_name slice-96 \
    --agent willowpai2i24h3-alphonse
```

### 5. Key signals to report

- `val_avg/mae_surf_p` per epoch — does either arm reach below 65.30?
- **Parameter count** logged for each arm — slice_num=32 should reduce params,
  slice_num=96 should increase. Confirm the dimension is actually changing.
- **Best epoch** — does slice_num change shift the convergence speed?
- Per-split breakdown at best checkpoint vs baseline `3rvfeq4g` — especially
  watch `val_geom_camber_cruise` (OOD, the lowest val number historically) since
  more slices may help represent unusual flow modes.
- **Peak VRAM** for slice_num=96 — confirm it stays under 96 GB.

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
| Stack | Lion **lr=2e-4**, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + T_max=21 |
| Best epoch | **19** (FINAL — val still descending at timeout) |
| Model | Transolver L=5, H=128, n_head=4, **slice_num=64**, mlp_ratio=2 |

Reproduce:
```bash
cd "target/" && python train.py --lr 2e-4 --lr_T_max 21 --wandb_group lion-lr-sweep --wandb_name lion-lr2e4 --agent willowpai2i24h3-frieren
```

## Why this matters

The PhysicsAttention slice mechanism is one of two model-capacity levers in
Transolver (the other being `n_hidden`). It has never been tuned on this stack.
Either direction yields useful information:
- If `slice_num=32` matches `slice_num=64` → 50% inference speedup for free
- If `slice_num=96` beats `slice_num=64` → fundamental capacity finding
