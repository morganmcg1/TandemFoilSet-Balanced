# Hypothesis: n-head-sweep (thorfinn)

## Hypothesis

The Transolver's attention modules use **`n_head=4`** with `n_hidden=128` —
giving `head_dim=32`. This value was inherited from the initial repo config and
has **never been tuned** across 6 rounds of optimization. All 7 other levers in
the SOTA stack (Lion lr=2e-4, T_max=21, etc.) were calibrated assuming this value.

`n_head` controls a fundamentally different attention property than the other
levers we've explored:

- **n_hidden** = per-token feature width (nezuko's recent H=160 test landed close
  to SOTA but did not beat it — capacity-on-width axis closed)
- **slice_num** = how many basis vectors the field decomposes into (alphonse #3876
  testing this concurrently)
- **n_head** = how attention bandwidth is partitioned across parallel subspaces

At fixed `n_hidden`, `n_head` trades head-width (representational richness per head)
against the number of parallel attention computations. The literature (transformer
scaling laws, ViT ablations) consistently shows the optimum depends on the data
distribution and the role attention plays — there is no universally "correct"
n_head.

**Two arms:**

1. **Arm 1 — `n_head=2`** (head_dim=64, fewer wider heads): tests whether the
   model benefits from richer per-head subspaces. In small-data CFD where
   attention may primarily be carrying long-range geometric information (mesh
   nodes far apart in space but causally related via flow), wider heads with
   more capacity per attention pattern could help.

2. **Arm 2 — `n_head=8`** (head_dim=16, more narrower heads): tests whether the
   model benefits from more parallel attention patterns. This is the standard
   ViT-scale choice and matches the convention in most modern transformers.

**Predicted improvement:** −0.2 to −1.0 if attention bandwidth is mis-partitioned.
Both directions are plausible a priori; the data will tell us which way.

**Worst case:** n_head is fully consumed by the existing stack (no signal in
either direction), confirming `n_head=4` is locally optimal.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full SOTA stack (Lion lr=2e-4,
bf16+clip=1.0+eta_min=1e-5+T_max=21). **Do NOT change anything else.**

### 2. Expose `n_head` as a CLI flag

In `target/train.py`, find the `model_config` dict around line 488–499:

```python
model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=5,
    n_head=4,         # ← hardcoded; change to cfg.n_head
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
```

Make the change:

```python
n_head=cfg.n_head,
```

Add to the Config dataclass (default `4` to preserve baseline):

```python
n_head: int = 4
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

**Arm 1 (primary) — n_head=2 (head_dim=64, wider heads):**

```bash
cd target/ && python train.py \
    --n_head 2 \
    --wandb_group n-head-sweep \
    --wandb_name nhead-2 \
    --agent willowpai2i24h3-thorfinn
```

**Arm 2 — n_head=8 (head_dim=16, more narrow heads):**

```bash
cd target/ && python train.py \
    --n_head 8 \
    --wandb_group n-head-sweep \
    --wandb_name nhead-8 \
    --agent willowpai2i24h3-thorfinn
```

### 5. Key signals to report

- `val_avg/mae_surf_p` per epoch — does either arm reach below 65.30?
- **Parameter count** logged for each arm — n_head only affects attention
  reshape, total params should be approximately constant (slightly different
  due to QKV linear bias terms). Confirm this.
- **Best epoch** — does n_head change shift convergence speed?
- **Per-split breakdown at best checkpoint** vs SOTA `3rvfeq4g` — geom_camber_rc
  is persistently the hardest split. If wider heads (n_head=2) help that split,
  attention is bottlenecked on geometric long-range patterns.
- **Peak VRAM** — should be near-identical across arms.
- Optional: train loss trajectory — does the model converge faster/slower with
  different head counts?

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
| Model | Transolver L=5, H=128, **n_head=4**, slice_num=64, mlp_ratio=2 |
| Best epoch | **19** (FINAL — val still descending at timeout) |

Reproduce:
```bash
cd "target/" && python train.py --lr 2e-4 --lr_T_max 21 --wandb_group lion-lr-sweep --wandb_name lion-lr2e4 --agent willowpai2i24h3-frieren
```

## Why this matters

`n_head` is one of the three primary architectural levers in the Transolver model
(the others being `n_hidden` and `slice_num`). The first has been tested at H=160
(closes most of the prior gap but doesn't beat SOTA); the second is being tested
concurrently by alphonse. `n_head` completes the architectural sweep — if any
direction wins, we have a new SOTA. If both lose, we know the attention partition
is correctly calibrated and the bottleneck lies elsewhere.

This is independent from alphonse's slice_num sweep and askeladd's dropout sweep,
so the three architectural experiments together will give us a strong picture of
which model-capacity dimensions are still under-tuned.
