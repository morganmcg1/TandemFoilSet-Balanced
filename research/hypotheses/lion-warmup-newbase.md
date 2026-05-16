# Hypothesis: lion-warmup-newbase (frieren)

## Hypothesis

Your previous lion-warmup PR (#3515, closed) tested warmup on the bare Lion baseline
(no clip, no bf16) and found it didn't help: lion-warmup1 val=100.80, lion-warmup2
val=104.91, both worse than Lion alone (94.08).

The new merged baseline (PR #3427, val=69.86) has a critical difference: `clip_grad_norm
(max_norm=1.0)` is now engaged on **99.7% of steps**. Alphonse's analysis showed the
median pre-clip grad norm is 16.6, so clip=1.0 is reshaping the gradient input on every
update — a mechanism that effectively bounds the early-epoch instability that warmup was
intended to address.

The key question: **are warmup and clip=1.0 redundant, or do they provide complementary
benefits?**

- **Redundant (clip already handles it)**: warmup will add no improvement over the baseline
  → warmup has the same ~100% clip engagement rate → val ≈ baseline 69.86
- **Complementary (warmup adds a separate benefit)**: warmup affects LR trajectory (starts
  at 0, ramps to 1e-4 over 2 epochs), while clip controls step magnitude within each LR
  value. Even with clip, the LR trajectory itself matters — starting at a very low LR
  prevents the model from jumping to a poor initial basin. → val < 69.86

This is a direct test of mechanism interaction: LR schedule vs gradient magnitude control.

**Predicted improvement:** small (0–3 points) if the mechanisms are complementary; near
zero if clip subsumes warmup's function. Best case: val 66–69.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+Huber+bf16+clip=1.0
+eta_min=1e-5) is the default. Add ONLY warmup.

### 2. Add warmup to the scheduler (if not already in merged train.py)

The merged train.py may or may not include `--warmup_epochs` from your prior PR
(it was on the old Huber baseline branch, not the Lion branch). Check first.

If NOT present, add to Config:
```python
warmup_epochs: int = 0
```

Add a linear warmup using SequentialLR:
```python
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

if cfg.warmup_epochs > 0:
    warmup_sched = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=cfg.warmup_epochs
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS - cfg.warmup_epochs,
        eta_min=cfg.eta_min
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[cfg.warmup_epochs]
    )
else:
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=cfg.eta_min)
```

### 3. Add fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — 2-epoch warmup:**
```bash
cd target/ && python train.py \
    --warmup_epochs 2 \
    --wandb_group lion-warmup-newbase \
    --wandb_name lion-warmup2-stack \
    --agent willowpai2i24h3-frieren
```

**Arm 2 — 1-epoch warmup (shorter, less training time lost):**
```bash
cd target/ && python train.py \
    --warmup_epochs 1 \
    --wandb_group lion-warmup-newbase \
    --wandb_name lion-warmup1-stack \
    --agent willowpai2i24h3-frieren
```

### 5. Report key signals

- val_avg/mae_surf_p per epoch
- **% steps clipped at clip=1.0** — does warmup change the clipping rate in early epochs?
- LR trajectory: does warmup visibly affect the LR in early epochs in W&B?
- Epoch 1 val (baseline was ~195 at ep1 on bare Lion, ~160-180 on old stack) — does warmup
  reduce the ep1 spike?
- Best epoch: still final epoch, or earlier convergence?

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — alphonse's lion-bf16-clip-floor (PR #3427, merged 2026-05-16):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **69.8562** |
| **test_avg_nansafe/mae_surf_p** | **65.8812** |
| W&B run | `f6lnbssy` (group: `bf16-stable`) |
| Stack | Lion + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 |
| Key note | clip=1.0 engaged at 99.7% of steps; val still descending at ep19 |

Your prior warmup result on OLD Lion baseline (for comparison):
- lion-warmup2 (`d1y7x4vv`): best_val=104.91 on OLD bare Lion (94.08) — warmup didn't help
- lion-warmup1 (`6ey6nh75`): best_val=100.80 on OLD bare Lion — slightly better but still regression

Reproduce baseline: `cd "target/" && python train.py --wandb_group bf16-stable --wandb_name lion-bf16-clip-floor --agent willowpai2i24h3-alphonse`
