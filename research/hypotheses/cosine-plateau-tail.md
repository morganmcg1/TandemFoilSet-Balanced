# Hypothesis: cosine-plateau-tail (tanjiro)

## Hypothesis

Your eta_min sweep (#3713) cleanly refuted the "raise the floor" hypothesis: lifting
eta_min lifts the **entire** second half of the cosine curve, not just ep17-19. Your
own analysis nailed the mechanism — the productive refinement window sits at LR ≈
1.4–2.0e-5, and cosine **passes through this window too quickly** rather than
spending time inside it.

Your suggested follow-up was the directly motivated next experiment:

> "Drop the schedule and try a constant LR around 1.4e-5 for the final 3 epochs
> (LR-finder style cooldown plateau) — would directly hold the model at the
> empirical sweet spot rather than passing through it."

This experiment tests exactly that. Replace the cosine **tail** with a constant-LR
plateau, holding the model at the productive refinement LR for 3 epochs instead of
having it pass through that band in a single epoch.

The change is orthogonal to frieren's in-flight `lion-lr-refine` (#3801), which
**extends** the cosine over more epochs (T_max=25). Yours **freezes** the cosine
at its productive low-LR value. Two complementary tests of "more time at low LR".

**Predicted improvement:** −0.2 to −1.0 on val_avg/mae_surf_p if extended residence
in the sweet-spot LR band extracts more refinement than a single-epoch crossing.
Worst case: model overfits during the plateau and slightly regresses.

## Background — natural cosine LR on the new SOTA stack

With Lion lr=2e-4, T_max=21, eta_min=1e-5 (frieren #3675 winner), the cosine LR
trajectory in the final epochs is:

```
ep | LR
15 | 5.6e-5
16 | 4.5e-5
17 | 2.7e-5
18 | 2.0e-5
19 | 1.45e-5  ← best epoch, still descending
```

The model spends ≤1 epoch in each of the productive sub-3e-5 LR bands. Holding the
LR constant at one of these values for 3 epochs gives the optimizer ~3× more steps
in the sweet spot.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion **lr=2e-4**,
bf16+clip=1.0+eta_min=1e-5+T_max=21) is the default. **Do NOT change `--lr`,
`--lr_T_max`, or `--eta_min`** — only the LR scheduler shape changes.

### 2. Implement the plateau scheduler

In `target/train.py`, modify the scheduler creation to support a cosine→constant
transition. The cleanest implementation uses `SequentialLR`:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR

if cfg.plateau_start_epoch is not None:
    cos_epochs = cfg.plateau_start_epoch
    plateau_epochs = cfg.epochs - cos_epochs

    # Cosine: lr_peak → plateau_lr over the first cos_epochs epochs.
    # We set eta_min=plateau_lr so the cosine *lands* exactly at the plateau value,
    # making the transition smooth (no discontinuity at the milestone).
    cos_sched = CosineAnnealingLR(
        optimizer,
        T_max=cos_epochs,
        eta_min=cfg.plateau_lr,
    )
    # ConstantLR with factor=1.0 holds whatever LR is current.
    # The SequentialLR sets the optimizer LR to plateau_lr at the milestone, then
    # ConstantLR holds it.
    const_sched = ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=plateau_epochs,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[cos_sched, const_sched],
        milestones=[cos_epochs],
    )
else:
    # unchanged: existing CosineAnnealingLR path
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.lr_T_max, eta_min=cfg.eta_min
    )
```

Add two CLI flags to the Config dataclass:
- `plateau_start_epoch: Optional[int] = None` (None = use existing CosineAnnealingLR)
- `plateau_lr: float = 2e-5`

**Critical:** verify the scheduler step cadence. If the existing code calls
`scheduler.step()` once per epoch, the milestone arithmetic above is correct. If it
steps per batch, multiply `cos_epochs` and `plateau_epochs` by `steps_per_epoch`
when constructing the sub-schedulers.

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — plateau at LR=1.4e-5 from ep17:**

Matches the natural ep19 cosine LR (the position where the model is "still
descending" at frieren's SOTA timeout). Tests whether 3 epochs at the **lowest**
productive LR extracts more refinement.

```bash
cd target/ && python train.py \
    --plateau_start_epoch 17 \
    --plateau_lr 1.4e-5 \
    --wandb_group cosine-plateau-tail \
    --wandb_name plateau-ep17-lr1p4e5 \
    --agent willowpai2i24h3-tanjiro
```

**Arm 2 — plateau at LR=2e-5 from ep17:**

Matches the natural ep18 cosine LR. Tests whether a slightly higher plateau LR
gives more useful per-step displacement during the 3-epoch hold (Lion update
magnitude = LR × sign, so 2e-5 produces ~43% larger steps than 1.4e-5).

```bash
cd target/ && python train.py \
    --plateau_start_epoch 17 \
    --plateau_lr 2e-5 \
    --wandb_group cosine-plateau-tail \
    --wandb_name plateau-ep17-lr2e5 \
    --agent willowpai2i24h3-tanjiro
```

### 5. Key signals to report

- **val_avg/mae_surf_p per epoch** — does either arm reach below 65.30 at ep17,
  18, or 19?
- **Per-epoch LR log** — confirm the cosine→constant transition is clean (no LR
  spike at the milestone, plateau actually held at the target value).
- **Best epoch** — does best epoch land at ep17, 18, or 19? Within-plateau
  comparison tells you which LR is the sweet spot for THIS stack.
- **Compare against frieren's baseline `3rvfeq4g`** (lr=2e-4, T_max=21) at
  matched epochs — does the plateau outperform cosine at ep17/18/19?
- **Per-split breakdown** at best checkpoint — single, rc, cruise, re_rand. The
  refinement window benefits most from the splits that were still improving at
  ep19 in tanjiro's prior data (re_rand and cruise).

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — frieren's lion-lr2e4 (PR #3675, merged 2026-05-16):

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
| Clip engagement | 98.5% |

Reproduce baseline:
```bash
cd "target/" && python train.py --lr 2e-4 --lr_T_max 21 --wandb_group lion-lr-sweep --wandb_name lion-lr2e4 --agent willowpai2i24h3-frieren
```

### Reference cosine LR trajectory (lr=2e-4, T_max=21, ep14-19):

```
ep14: LR=6.9e-5 | val=81.46
ep15: LR=5.6e-5 | val=72.13
ep16: LR=4.5e-5 | val=73.72
ep17: LR=2.7e-5 | val=68.97
ep18: LR=2.0e-5 | val=66.14
ep19: LR=1.45e-5 | val=65.30  ← best, still descending
```

The plateau experiment tests whether **holding** at the ep18 or ep19 LR value for
3 epochs is better than passing through.
