# Hypothesis: lion-lr-sweep (frieren)

## Hypothesis

The current SOTA stack (Lion+bf16+clip=1.0+eta_min=1e-5+T_max=21, PR #3596, val=65.74)
uses **lr=1e-4** (the Lion default). The new T_max=21 cosine schedule means the LR now
sweeps from 1e-4 down to ~1.2e-5 within the 19-epoch budget — but the descent is still
heavily concentrated in the first ~16 epochs.

**Key mechanism insight from your prior PR**: Lion's update is `LR · sign(momentum)`.
The per-step displacement is FULLY determined by LR. Unlike AdamW (where grad magnitude
scales the update), in Lion a higher LR **directly and linearly** increases the step size.
This means:
- clip=1.0 bounds the gradient that feeds momentum — it doesn't bound the actual step
- There is no implicit gradient-magnitude safety valve
- LR is the **single lever** that controls update magnitude

Given this, the question is: **is lr=1e-4 the right starting point?** The val curve was
still descending at epoch 18 (best) in the merged SOTA run. That implies the optimizer
hasn't found a plateau — larger steps might reach a better basin faster within the 30-min
budget.

**Why clip=1.0 + higher LR might work**: When clip engages at 99.7% of steps and
normalizes the gradient to norm=1 before it enters momentum, Lion sees a consistent
signal direction at every step. Scaling LR then scales how fast it moves along that
consistent signal. This is different from raising LR in AdamW where the high-curvature
regions (large grad norms) would dominate — here, clip has already equalized the input.

**Risks**: If LR is already optimal, raising it will cause divergence or noisy
exploration that doesn't settle in 19 epochs. The cosine schedule will still decay LR,
but starting at 2e-4 means the effective mid-training LR (~epoch 9-10) is 2× higher.
Monitoring epoch-by-epoch val is key.

**Predicted improvement:** −1 to −5 on val_avg/mae_surf_p vs 65.74 baseline.
Conservative: −1 to −2 from faster convergence in early epochs. Aggressive: −3 to −5
if the higher LR explores better basins that lr=1e-4 can't reach in budget.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack is:
Lion+bf16+clip=1.0+eta_min=1e-5 + **T_max=21** (just merged from tanjiro's PR #3596).
Change ONLY `--lr`.

### 2. Verify lr_T_max=21 is now the default

After tanjiro's merge, check train.py to confirm:
```python
lr_T_max: int = 0  # or 21 may now be default
```
If the default is still 0 (= MAX_EPOCHS), you must pass `--lr_T_max 21` explicitly so
your run matches the merged SOTA schedule. Do not skip this.

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — lr=2e-4 (2× default), T_max=21:**
```bash
cd target/ && python train.py \
    --lr 2e-4 \
    --lr_T_max 21 \
    --wandb_group lion-lr-sweep \
    --wandb_name lion-lr2e4 \
    --agent willowpai2i24h3-frieren
```

**Arm 2 — lr=3e-4 (3× default), T_max=21:**
```bash
cd target/ && python train.py \
    --lr 3e-4 \
    --lr_T_max 21 \
    --wandb_group lion-lr-sweep \
    --wandb_name lion-lr3e4 \
    --agent willowpai2i24h3-frieren
```

If Arm 2 diverges (val > 300 after epoch 3, or val increases monotonically), stop it
early and note the failure. Arm 1 is the primary.

### 5. Key signals to report

- val_avg/mae_surf_p per epoch — does the descent curve drop faster or reach a lower floor?
- **Early-epoch val (ep1-3)**: does higher LR cause divergence or just noisier early epochs?
- LR at each epoch (log it to W&B) — confirms the cosine schedule is engaged at the right T_max=21
- % steps clipped (does clip engagement rate change with higher LR?)
- Best epoch: early or final?
- Compare val trajectory at matched epochs vs baseline (both starting from seed=42)

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — tanjiro's lion-tmax21 stack (PR #3596, merged 2026-05-16):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **65.7375** |
| **test_avg_nansafe/mae_surf_p** | **61.7003** |
| test_single_in_dist | 61.9972 |
| test_geom_camber_rc | 69.7654 |
| test_geom_camber_cruise | 57.5355 |
| test_re_rand | 57.5030 |
| W&B run | `tew7xthq` (group: `lion-tmax-newbase`) |
| Stack | Lion lr=**1e-4** (default), wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + **T_max=21** |
| Best epoch | 18 (not final — epoch 19 regresses mildly) |
| VRAM | 33 GB / 96 GB |

Reproduce baseline:
```bash
cd "target/" && python train.py --lr_T_max 21 --wandb_group lion-tmax-newbase --wandb_name lion-tmax21 --agent willowpai2i24h3-tanjiro
```

Your prior warmup experiments for reference (both closed, all 4 arms worse than baseline):
- warmup2-stack (rzmszrhy): val=76.12 on old baseline (69.86)
- warmup1-stack (ay7uz94m): val=81.42 on old baseline (69.86)
- warmup2 OLD (d1y7x4vv): val=104.91 on old Lion (94.08)
- warmup1 OLD (6ey6nh75): val=100.80 on old Lion (94.08)
