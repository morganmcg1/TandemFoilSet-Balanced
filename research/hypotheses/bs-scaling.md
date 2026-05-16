# Hypothesis: bs-scaling (askeladd)

## Hypothesis

The current SOTA baseline (PR #3427, val=69.86) uses **batch_size=4** and peaks at
**33 GB / 96 GB VRAM** — leaving **63 GB unused**. With bf16 autocast already engaged,
we have substantial headroom to scale batch size.

Larger batch size has two competing effects:
1. **Pro: less gradient noise per step** → more stable optimization, especially
   important for Lion's sign-based update (which is most useful when the gradient sign
   is reliable)
2. **Con: fewer optimizer steps per epoch** → less stochastic exploration of the loss
   surface, less data-diversity-driven regularization

For sign-based optimizers like Lion, the sign of the gradient is the entire update
signal — so noise in gradient magnitude is *less* damaging than for AdamW (which uses
magnitude). What matters is whether the *sign* is reliable. Larger batch → more reliable
sign → cleaner per-step momentum updates → potentially better convergence.

There's also a key compute interaction with our 30-min budget:
- bs=4 → 19 epochs (current) → 9.6% of available VRAM used
- bs=8 → ~10 epochs predicted (fewer steps/epoch is compensated by bigger steps in same
  wall-clock) → ~60 GB VRAM
- bs=12 → ~7 epochs predicted → ~75 GB VRAM
- bs=16 → ~5 epochs predicted → ~95 GB VRAM (near ceiling)

If larger bs improves per-epoch convergence enough, it can win at fewer epochs.

**Predicted improvement:** −1 to −5 on val_avg/mae_surf_p vs 69.86 baseline.
Best case if Lion benefits strongly from cleaner sign signal at bs=8 or 12. Worst case
if fewer total steps dominates and we end up time-bounded short of convergence.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+bf16+clip=1.0
+eta_min=1e-5) is the default. ONLY change batch_size.

### 2. Verify the batch_size CLI override

The Config already has `batch_size: int = 4`. Check that `--batch_size` is exposed as a
CLI flag (it should be, but verify). If not, add the standard CLI exposure.

The data loader uses `batch_size=cfg.batch_size` consistently for train/val/test — no
need to modify the DataLoader code.

### 3. Add a fixed seed

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — bs=8 (2× current, safe VRAM headroom):**
```bash
cd target/ && python train.py \
    --batch_size 8 \
    --wandb_group bs-scaling \
    --wandb_name bs8 \
    --agent willowpai2i24h3-askeladd
```

**Arm 2 — bs=12 (3× current, ~80% VRAM expected):**
```bash
cd target/ && python train.py \
    --batch_size 12 \
    --wandb_group bs-scaling \
    --wandb_name bs12 \
    --agent willowpai2i24h3-askeladd
```

If Arm 2 OOMs, fall back to bs=10. Do NOT attempt bs=16 — too close to ceiling, and
prior runs suggest VRAM scales roughly linearly with bs.

### 5. Report key signals

- val_avg/mae_surf_p per epoch — does the val curve descend faster at higher bs?
- Total epochs in 30 min (should be 10–12 at bs=8, 6–8 at bs=12)
- Peak VRAM (confirms scaling)
- Epoch_time_s — verify the throughput is actually better per-step at higher bs
- Best epoch (early or final?) — if best=final, val still descending and could benefit
  from more epochs

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
| batch_size | **4** (Config default) |
| Peak VRAM | **33 GB / 96 GB** (63 GB headroom) |
| Total epochs in 30 min | 19 |

Reproduce: `cd "target/" && python train.py --wandb_group bf16-stable --wandb_name lion-bf16-clip-floor --agent willowpai2i24h3-alphonse`

Your prior round-4 result for context (`3gffkqmt`, PR #3385):
- warmup2-clip50-fullstack: val=104.02, 9 epochs (throughput-confounded)
- The 2.15× slowdown was traced to suspected concurrent-pod contention — separate issue
  from bs-scaling effects, but worth comparing your epoch_time_s here to baseline's 98.18s
