# Hypothesis: wd-sweep (thorfinn)

## Hypothesis

The current SOTA stack uses `weight_decay=1e-2` — inherited from the initial Lion
configuration and never re-tuned after 5 rounds of improvements (Huber loss, bf16,
clip=1.0, eta_min, T_max=21 all stacked on it). Weight decay acts as L2 regularization;
for Lion specifically, the effective regularization strength scales differently than
AdamW (Lion applies uniform steps, so wd=1e-2 "pulls" weights toward zero by the same
amount per step regardless of gradient magnitude — a stronger regularizer at low-LR
phases than at high-LR phases).

**The question:** Is wd=1e-2 optimal on the current stack? Two hypotheses:
1. **wd too low (wd=5e-2):** the model is underfitting at its optimal LR, meaning more
   regularization would constrain the representation to generalize better on OOD splits.
   The camber test splits (geom_camber_rc, geom_camber_cruise) being 12–15 points worse
   than in-dist may point here.
2. **wd too high (wd=1e-3):** current wd=1e-2 is overconstrained for this 2.7M-param
   model on this data, and the descending val curve at timeout means the model hasn't
   finished learning — relaxing wd would let it reach a better minimum in the same budget.

Both are plausible. The sweep will distinguish them.

**Predicted improvement:** −0.5 to −2 on val_avg/mae_surf_p if wd=1e-2 is suboptimal.
OOD splits (geom_camber_rc, re_rand) are the most likely beneficiaries of the right wd.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+bf16+clip=1.0+
eta_min=1e-5+**T_max=21**) is the default. Change ONLY `--wd`.

### 2. Verify CLI flag exists

In `target/train.py`, confirm `--wd` is exposed as a CLI arg. The Lion instantiation
should already pass `cfg.wd` as `weight_decay`. Default `wd: float = 1e-2`.

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — wd=1e-3 (10× less regularization):**
```bash
cd target/ && python train.py \
    --wd 1e-3 \
    --lr_T_max 21 \
    --wandb_group wd-sweep \
    --wandb_name wd-1e3 \
    --agent willowpai2i24h3-thorfinn
```

**Arm 2 — wd=5e-2 (5× more regularization):**
```bash
cd target/ && python train.py \
    --wd 5e-2 \
    --lr_T_max 21 \
    --wandb_group wd-sweep \
    --wandb_name wd-5e2 \
    --agent willowpai2i24h3-thorfinn
```

### 5. Key signals to report

- val_avg/mae_surf_p per epoch — which arm descends faster or lower?
- **Per-split breakdown** at best checkpoint — is the OOD generalization (geom_camber_rc,
  re_rand) meaningfully affected by wd change? This is the key diagnostic for under/overfitting.
- Weight norm trajectory (if loggable) — does wd=5e-2 compress weights significantly?
- Best epoch: does higher wd shift best epoch later (slower convergence from weight pulling)?

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — tanjiro's lion-tmax21 (PR #3596, merged 2026-05-16):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **65.7375** |
| **test_avg_nansafe/mae_surf_p** | **61.7003** |
| test_single_in_dist | 61.9972 |
| test_geom_camber_rc | 69.7654 |
| test_geom_camber_cruise | 57.5355 |
| test_re_rand | 57.5030 |
| W&B run | `tew7xthq` (group: `lion-tmax-newbase`) |
| Stack | Lion lr=1e-4, **wd=1e-2** + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + T_max=21 |
| Best epoch | 18 |

Reproduce baseline:
```bash
cd "target/" && python train.py --lr_T_max 21 --wandb_group lion-tmax-newbase --wandb_name lion-tmax21 --agent willowpai2i24h3-tanjiro
```
