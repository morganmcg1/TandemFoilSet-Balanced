# Hypothesis: h160-tmax-calibrated (nezuko)

## Hypothesis

Your prior H=160 run (`ese2fcr2`, group `deeper-model`) showed val=70.49 (test=65.14) —
close to the old baseline (69.86) despite running only 17 epochs at 110 s/epoch, with
T_max=50 meaning the LR never reached the productive low-LR refinement region. The val
curve was still steeply descending at epoch 17 (final).

The new SOTA stack uses T_max=21 to align the cosine schedule with the actual epoch
budget (19 epochs at ~98 s/epoch). H=160 runs at ~110 s/epoch → **~16 epochs in 30 min**.
The fix: set T_max=16 so the cosine schedule fully completes within the actual budget,
engaging the productive low-LR zone (epochs 13-16 at the equivalent LR that drove the
SOTA win at epochs 15-18 for H=128).

If the performance gap between H=128 and H=160 was entirely due to the T_max
misalignment (LR never decayed to productive range), re-running with T_max calibrated
should recover the gap and potentially show compound benefit from extra capacity +
better LR schedule.

**Predicted improvement:** H=160 + T_max=16 lands around val=65-68 (vs SOTA 65.74).
Conservative: closes within noise of SOTA (≤+1 point). Aggressive: beats SOTA if extra
capacity + lower-LR refinement compound.

Also run H=144 as a secondary arm — adds ~25% params vs H=128 (vs H=160's 47%), runs at
~103 s/epoch → ~17 epochs, T_max=17. Less risky tradeoff.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+bf16+clip=1.0+
eta_min=1e-5+**T_max=21**) is the default. You will override n_hidden and lr_T_max.

### 2. Estimate epoch times before launching full runs

Do a quick 2-epoch timing dry-run for each architecture:
```bash
cd target/ && python train.py --n_hidden 160 --lr_T_max 2 --wandb_group h160-tmax-calibrated --wandb_name h160-timing --agent willowpai2i24h3-nezuko
cd target/ && python train.py --n_hidden 144 --lr_T_max 2 --wandb_group h160-tmax-calibrated --wandb_name h144-timing --agent willowpai2i24h3-nezuko
```
Use the per-epoch wall time to confirm T_max estimates below. Adjust if timing differs materially (>10%).

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — H=160, T_max=16:**
```bash
cd target/ && python train.py \
    --n_hidden 160 \
    --lr_T_max 16 \
    --wandb_group h160-tmax-calibrated \
    --wandb_name h160-tmax16 \
    --agent willowpai2i24h3-nezuko
```

**Arm 2 — H=144, T_max=17:**
```bash
cd target/ && python train.py \
    --n_hidden 144 \
    --lr_T_max 17 \
    --wandb_group h160-tmax-calibrated \
    --wandb_name h144-tmax17 \
    --agent willowpai2i24h3-nezuko
```

If timing differs from estimates, adjust T_max to match the actual achievable epoch count
(target: T_max = epochs_in_30min - 2 for safety margin).

### 5. Key signals to report

- val_avg/mae_surf_p per epoch — does the curve reach a lower floor than before?
- **LR at each epoch** — confirm the cosine schedule decays to eta_min=1e-5 within budget
- Peak VRAM and epoch_time_s for each arm (confirm T_max calibration)
- Best epoch — does it shift earlier or stay final? (Is the curve still descending at timeout?)
- Per-split breakdown at best checkpoint
- Compare against prior H=160 run (ese2fcr2) at matched epochs — does T_max calibration alone explain the gap?

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
| W&B run | `tew7xthq` (group: `lion-tmax-newbase`) |
| Stack | Lion lr=1e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + **T_max=21** |
| Best epoch | 18 (ep19 mildly regresses) |
| n_hidden | **128** (default, what you are replacing) |

### Prior H=160 run for comparison (ese2fcr2, group `deeper-model`):

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (ep17/17) | 70.4932 |
| test_avg_nansafe/mae_surf_p | 65.1431 |
| Epoch time | 110 s/epoch |
| T_max | 50 (misaligned — LR never reached refinement zone) |
| Final LR at ep17 | ~7.4e-5 (still upper half of cosine arc) |

Reproduce baseline:
```bash
cd "target/" && python train.py --lr_T_max 21 --wandb_group lion-tmax-newbase --wandb_name lion-tmax21 --agent willowpai2i24h3-tanjiro
```
