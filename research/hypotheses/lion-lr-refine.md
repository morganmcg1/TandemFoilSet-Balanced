# Hypothesis: lion-lr-refine (frieren)

## Hypothesis

Your prior LR sweep (#3675, MERGED) established:
- lr=2e-4: val=65.30 (NEW SOTA), best epoch=19 (FINAL, val still descending)
- lr=3e-4: val=67.00 (overshoots basin, best epoch=18)

The optimal LR lives in the (2e-4, 3e-4) interval — strictly between them. With your
own observation that val was still descending at ep19 for lr=2e-4, there's headroom in
the LR direction as well: either find the LR peak more precisely, or change the
schedule to let the existing lr=2e-4 reach a lower floor.

**Two complementary directions, single PR:**

1. **Arm 1 (LR refinement): lr=2.5e-4** — midpoint between winner and overshoot. If the LR
   landscape is smooth, this should be the peak. If not (e.g., bimodal sensitivity to LR),
   we get useful negative information.

2. **Arm 2 (Schedule extension at lr=2e-4): T_max=25** — keeps LR higher for longer.
   At T_max=21 with lr=2e-4: ep19 LR ≈ 1.4e-5. At T_max=25: ep19 LR ≈ 4.1e-5 (3× higher).
   This rides higher peak LR through more of the budget, capturing the "still descending"
   signal from your prior result.

These two arms attack the same insight ("lr=2e-4 wasn't done at ep19") from opposite ends:
either find a faster-converging LR, or slow the schedule's decay.

**Predicted improvement:**
- Arm 1 (lr=2.5e-4): −0.2 to −1.0 if LR peak is closer to 2.5e-4.
- Arm 2 (lr=2e-4, T_max=25): −0.3 to −1.5 if higher late-stage LR allows more refinement.
- Worst case for Arm 2: model overshoots in the last 3 epochs and regresses.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion **lr=2e-4** post-#3675
merge, bf16+clip=1.0+eta_min=1e-5+T_max=21) is the default. Change ONLY `--lr` and/or
`--lr_T_max`.

### 2. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 3. Run two arms

**Arm 1 (primary) — lr=2.5e-4, T_max=21 (LR midpoint probe):**
```bash
cd target/ && python train.py \
    --lr 2.5e-4 \
    --lr_T_max 21 \
    --wandb_group lion-lr-refine \
    --wandb_name lion-lr2p5e4 \
    --agent willowpai2i24h3-frieren
```

**Arm 2 — lr=2e-4, T_max=25 (schedule extension):**
```bash
cd target/ && python train.py \
    --lr 2e-4 \
    --lr_T_max 25 \
    --wandb_group lion-lr-refine \
    --wandb_name lion-lr2e4-tmax25 \
    --agent willowpai2i24h3-frieren
```

If Arm 1 diverges (val > 300 at ep 3, or monotonically increasing), stop early — note
that lr=2.5e-4 is too high.

### 4. Key signals to report

- val_avg/mae_surf_p per epoch — does either arm reach below 65.30?
- **LR trajectory at each epoch** for Arm 2 — confirm T_max=25 keeps LR higher at ep19
  than T_max=21 (expect ~4.1e-5 vs 1.4e-5)
- Best epoch comparison: does Arm 1 peak earlier than Arm 2?
- Per-split breakdown at best checkpoint
- Compare against baseline run `3rvfeq4g` (lr=2e-4, T_max=21) at matched epochs

### 5. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 6. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — YOUR own lion-lr2e4 (PR #3675, merged 2026-05-16 07:30 UTC):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **65.2991** |
| **test_avg_nansafe/mae_surf_p** | **60.5400** |
| W&B run | `3rvfeq4g` (group: `lion-lr-sweep`) |
| Stack | Lion **lr=2e-4**, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + T_max=21 |
| Best epoch | **19** (FINAL — val still descending) |
| Clip engagement | 98.5% |

Reproduce baseline:
```bash
cd "target/" && python train.py --lr 2e-4 --lr_T_max 21 --wandb_group lion-lr-sweep --wandb_name lion-lr2e4 --agent willowpai2i24h3-frieren
```

### Reference trajectory (lr=2e-4 winner, ep 14-19):

```
ep14: 81.46
ep15: 72.13
ep16: 73.72
ep17: 68.97
ep18: 66.14
ep19: 65.30  ← best, still descending
```
