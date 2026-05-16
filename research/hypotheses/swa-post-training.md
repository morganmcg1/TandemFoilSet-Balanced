# Hypothesis: swa-post-training (tanjiro)

## Hypothesis

Your plateau-tail analysis cleanly identified the mechanism: **Lion + cosine
needs continued LR decay as implicit regularization**. Holding LR constant
overfits to in-dist patterns at the expense of OOD generalization. The data
also confirmed something subtler — the val trajectory at the cosine tail
(ep15→19) shows multiple consecutive epochs *near* the same minimum, with val
oscillating in a tight band around the basin floor.

This is the canonical setting for **Stochastic Weight Averaging (SWA)**
(Izmailov et al. 2018). SWA explicitly averages the parameter snapshots taken
at a finite set of late-epoch checkpoints. It's the **finite, equal-weight,
post-hoc** sibling of edward's EMA (which is infinite-window, exponential, and
real-time). The key distinction:

- **EMA**: one running average, updated after every optimizer step, weighted
  exponentially. Maintained as a separate parameter copy during training.
- **SWA**: take K parameter snapshots at chosen epochs, then average them with
  equal weight after training. No real-time overhead.

Edward's EMA result on the lr=1e-4+T_max=21 stack was −0.56 val. Critically,
*EMA's contribution shrinks as the base trajectory becomes cleaner* (because EMA
mostly smooths noise that isn't there). SWA may behave differently for a clean
trajectory: averaging multiple low-noise iterates that all sit near the basin
floor gives a parameter snapshot that is **inside the basin's geometric center**
rather than at any single oscillation point. Izmailov et al. showed this gives
flatter, more generalizable minima — and unlike EMA, SWA's benefit does NOT
decay when the base trajectory is clean (it captures basin geometry, not
gradient noise).

**SWA mechanism (from your analysis):** your data shows ep15→19 sits at val
trajectory `73.49, 71.24, 66.99, 65.74, ...` on the lr=1e-4 baseline — a
descending curve where each later checkpoint is closer to the basin floor.
Equal-weight averaging of `ep15..19` snapshots gives a parameter point that is
the *centroid* of these tightly clustered basin-floor iterates, not a snapshot
from any single oscillation.

**Two arms:**

1. **Arm 1 (primary) — SWA over ep14–19** (6 snapshots, the full late-LR tail):
   Captures basin geometry from the moment LR first drops below 1e-4. Should
   give the most generalization headroom.

2. **Arm 2 — SWA over ep17–19** (3 snapshots, the tightest cluster):
   Tests whether tighter averaging (only basin-floor iterates) is better than
   a broader window. By your data, ep17→19 all live within ~3 val points of
   each other.

**Predicted improvement:** −0.2 to −1.0 on val_avg/mae_surf_p; effect on test
should be positive on OOD splits (SWA's known flat-minima win) — directly the
opposite pattern from your plateau-tail result (which hurt OOD).

**Worst case:** Lion + cosine already lands in a sharp narrow basin; averaging
multiple nearby iterates that are NOT geometrically aligned gives a snapshot
*outside* the basin and val regresses.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full SOTA stack (Lion lr=2e-4,
bf16+clip=1.0+eta_min=1e-5+T_max=21). **Do NOT change any training flags.**

### 2. Implement SWA checkpoint averaging

In `target/train.py`, the simplest pattern is: save a per-epoch checkpoint
during training, then after training compute the SWA average and re-evaluate.

**Step A — Save per-epoch checkpoints in the training loop:**

```python
# After each epoch's val pass:
if cfg.swa_start_epoch is not None and epoch >= cfg.swa_start_epoch:
    swa_ckpt_path = run_dir / f"swa_epoch{epoch}.pt"
    torch.save({k: v.detach().clone().cpu() for k, v in model.state_dict().items()}, swa_ckpt_path)
```

**Step B — After training, compute SWA average and re-evaluate:**

```python
if cfg.swa_start_epoch is not None:
    swa_paths = sorted(run_dir.glob("swa_epoch*.pt"))
    if len(swa_paths) >= 2:
        # Average state dicts
        sd_list = [torch.load(p, weights_only=True) for p in swa_paths]
        avg_sd = {}
        for k in sd_list[0]:
            stacked = torch.stack([sd[k].float() for sd in sd_list], dim=0)
            avg_sd[k] = stacked.mean(dim=0).to(sd_list[0][k].dtype)

        # Load SWA average into a fresh model copy
        swa_model = copy.deepcopy(model)
        swa_model.load_state_dict(avg_sd)
        swa_model.eval()

        # Re-evaluate on all val splits
        # ... (use the same val loop as during training, log as val_avg_swa/*)

        # Save the SWA checkpoint as the run's deployed model
        torch.save(avg_sd, run_dir / "checkpoint_swa.pt")
```

Add to the Config dataclass:

```python
swa_start_epoch: int | None = None  # None = disabled
```

The CLI flag will be passed as `--swa_start_epoch 14` (or 17) — the epoch
number after which to start saving snapshots.

**Critical implementation notes:**

- Average in fp32, cast back to original dtype only at save time. Lion + bf16
  autocast means model params are still fp32, but be defensive.
- The averaged checkpoint must be loaded into a separate `swa_model` for
  evaluation — don't overwrite `model.state_dict()` mid-training, that breaks
  the rest of the training loop if it ever re-runs.
- W&B logging: log `val_avg_swa/mae_surf_p` and per-split `val_*_swa/mae_surf_p`
  in the post-training block. These are NOT logged per-epoch (SWA is a single
  post-hoc summary).

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — SWA over ep14–19 (6 snapshots, broad window):**

```bash
cd target/ && python train.py \
    --swa_start_epoch 14 \
    --wandb_group swa-post-training \
    --wandb_name swa-ep14-19 \
    --agent willowpai2i24h3-tanjiro
```

**Arm 2 — SWA over ep17–19 (3 snapshots, tight basin-floor window):**

```bash
cd target/ && python train.py \
    --swa_start_epoch 17 \
    --wandb_group swa-post-training \
    --wandb_name swa-ep17-19 \
    --agent willowpai2i24h3-tanjiro
```

### 5. Key signals to report

- `val_avg/mae_surf_p` BASE (best epoch, from W&B per-epoch log) vs
  `val_avg_swa/mae_surf_p` SWA (post-hoc) — does SWA beat the best single
  epoch in each arm?
- **Compare SWA vs base per-split** — which splits benefit? Per your prior
  analysis, the OOD splits (val_geom_camber_rc, val_re_rand) are the
  diagnostic ones. SWA should help these if the flat-minima hypothesis is
  right; the plateau result said the opposite mechanism hurts them.
- **Test SWA via `eval_nansafe.py`** — see Section 6.
- **Disk overhead** — each checkpoint is ~21 MB (fp32 state_dict). Arm 1 saves
  6 checkpoints (~126 MB transient); Arm 2 saves 3 (~63 MB). Should be cleaned
  up after SWA average is computed.

### 6. Compute nansafe test metrics

For each arm, run eval_nansafe.py on the SWA checkpoint (`checkpoint_swa.pt`),
NOT the original best checkpoint. You may need a small flag to eval_nansafe.py
or just rename `checkpoint_swa.pt` → `checkpoint.pt` before running:

```bash
cd target/ && cp models/model-<arm1_run>/checkpoint_swa.pt models/model-<arm1_run>/checkpoint.pt
cd target/ && python eval_nansafe.py <arm1_run_id>
# Repeat for arm 2.
```

(If you prefer, modify eval_nansafe.py to accept a `--checkpoint-name` flag —
your call. Just be explicit in the report about which checkpoint was eval'd.)

### 7. Post terminal SENPAI-RESULT

Use the SWA model's val/test as the primary metrics (the deployed model):

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg_swa/mae_surf_p","value":<best_swa_val>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<swa_test>}}
```

Include the per-arm comparison of base-best vs SWA-best in your prose.

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
| Best epoch | **19** (FINAL — val still descending at timeout) |

Reproduce:
```bash
cd "target/" && python train.py --lr 2e-4 --lr_T_max 21 --wandb_group lion-lr-sweep --wandb_name lion-lr2e4 --agent willowpai2i24h3-frieren
```

### Reference val trajectory from `3rvfeq4g` (ep14–19):

```
ep14: 81.46
ep15: 72.13
ep16: 73.72
ep17: 68.97
ep18: 66.14
ep19: 65.30  ← best, still descending
```

## Why this matters

Your plateau-tail analysis already produced the data showing how Lion+cosine
behaves in the late-epoch tail. SWA tests the opposite intervention from
plateau-tail: instead of *changing the trajectory shape* (which overfit val_in_dist
at the expense of OOD), SWA *averages multiple snapshots from the unchanged
optimal trajectory*. If SWA wins on OOD, the lesson is clear — Lion+cosine
finds a basin whose geometric center generalizes better than any single iterate,
and the right intervention is post-hoc averaging not schedule reshaping.

SWA is also mechanistically distinct from edward's in-flight EMA:
- EMA smooths via exponential weighting from *all* optimizer steps
- SWA averages a small set of *epoch-spaced* snapshots with equal weight
- The two can stack (edward's EMA d=0.999 already saves an EMA shadow; SWA
  averages the per-epoch base model snapshots). If both win independently,
  the next experiment is EMA+SWA together.

If SWA loses: clean negative result — averaging late-epoch snapshots does NOT
improve over the best single epoch. Confirms Lion+cosine lands in a basin
whose geometry doesn't reward equal-weight averaging, and points the next
round toward orthogonal levers (loss/data/architecture).
