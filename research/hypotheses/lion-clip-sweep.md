# Hypothesis: lion-clip-sweep (alphonse)

## Hypothesis

Your rebased run (PR #3427, `f6lnbssy`, val=69.86) revealed a critical insight about the
stack: **clip_grad_norm(max_norm=1.0) engaged on 99.7% of training steps**, with observed
pre-clip grad norms mean=24.4, median=16.6, max=337.8. At this engagement rate, clip=1.0
is not a "spike ceiling" — it is a **per-step normalizer for Lion's momentum input**, 
effectively scaling down the gradient EMA on every update.

This means clip=1.0 is the dominant active lever in the merged stack. The question
is whether 1.0 is the optimal value, or whether a tighter or looser threshold gives a
better gradient-to-momentum signal.

Two competing hypotheses:
- **Tighter clip (0.25–0.5)**: further constrains the momentum input signal at each step →
  smoother trajectory, possibly lower floor, but may slow convergence
- **Looser clip (2.0–5.0)**: allows larger gradient magnitudes into the momentum buffer →
  faster convergence but higher risk of late-epoch instability

The sweet spot is unknown. Your analysis suggests the gradient scale of Lion on this task
is systematically ~16–25 (median) before clipping. A "spike-only" clip at 2× median (~35)
would let most steps through freely; a tighter clip at 0.5 would constrain ~99.9% of
steps even more aggressively than the current 1.0.

**Predicted improvement:** unknown direction — this is a diagnostic sweep. If tighter clip
helps: expect val in 65–68 range. If looser clip helps: expect val 67–70 but with risk of
late divergence. If 1.0 is optimal: arms within noise of 69.86.

## Instructions

### 1. Start from the current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — this already has the full stack:
Lion + Huber δ=2.0 + bf16 + `clip_grad_norm(max_norm=1.0)` + `eta_min=1e-5`.

### 2. Add `--grad_clip_max_norm` CLI flag if not present

In `target/train.py`, verify the `grad_clip_norm` config field exists. If there's no CLI
override (the merged default is hard-coded at 1.0), add:

```python
grad_clip_norm: float = 1.0  # in Config dataclass
```

```python
# in training loop, before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
```

### 3. Add a fixed seed

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run three arms (priority order)

**Arm 1 (primary) — tighter clip, 1/4 of current:**
```bash
cd target/ && python train.py \
    --grad_clip_norm 0.25 \
    --wandb_group lion-clip-sweep \
    --wandb_name lion-clip0p25 \
    --agent willowpai2i24h3-alphonse
```

**Arm 2 — intermediate tighter:**
```bash
cd target/ && python train.py \
    --grad_clip_norm 0.5 \
    --wandb_group lion-clip-sweep \
    --wandb_name lion-clip0p5 \
    --agent willowpai2i24h3-alphonse
```

**Arm 3 — looser clip (2× current):**
```bash
cd target/ && python train.py \
    --grad_clip_norm 2.0 \
    --wandb_group lion-clip-sweep \
    --wandb_name lion-clip2p0 \
    --agent willowpai2i24h3-alphonse
```

If you only have time for two arms, run Arm 1 and Arm 3 — they bracket the current
value and give the clearest monotonic signal.

### 5. Report critical diagnostics per arm

In addition to the standard metrics, report for each arm:
- **% steps clipped** (train/grad_norm > clip_threshold): at clip=0.25, what % of steps
  clip? If it's still near 100%, the scale is even tighter.
- **median pre-clip grad_norm**: is the pre-clip magnitude stable across arms or does the
  momentum buffer's EMA shift when the clip changes?
- **val trajectory**: does tighter clip slow convergence (higher val early epochs) but
  reach a lower final value, or does it just hurt throughout?
- **Late-epoch stability**: does clip=2.0 show late-epoch val drift (risk of divergence)?

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>","<id3>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — alphonse's lion-bf16-clip-floor (PR #3427, merged 2026-05-16):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **69.8562** |
| val_single_in_dist | 78.4834 |
| val_geom_camber_rc | 86.8730 |
| val_geom_camber_cruise | 45.3256 |
| val_re_rand | 68.7430 |
| **test_avg_nansafe/mae_surf_p** | **65.8812** |
| W&B run | `f6lnbssy` (group: `bf16-stable`) |
| Stack | Lion lr=1e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 |
| Key note | clip=1.0 engaged at 99.7% of steps; val still descending at ep19 |

Reproduce: `cd "target/" && python train.py --wandb_group bf16-stable --wandb_name lion-bf16-clip-floor --agent willowpai2i24h3-alphonse`
