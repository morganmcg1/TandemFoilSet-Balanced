# Hypothesis: deeper-model (nezuko)

## Hypothesis

Alphonse's merged stack (Lion+bf16+clip+floor, PR #3427) used only 33.0 GB of the
96 GB H100 VRAM — **63 GB of headroom is unused**. The current Transolver config
(n_hidden=128, n_layers=5) was chosen to fit under memory constraints that no longer
bind. With bf16 enabled by default, a deeper or wider model can now fit comfortably.

The Transolver PhysicsAttention mechanism scales in VRAM roughly as:
- `n_hidden`: quadratic-ish in attention layers (QKV projections)
- `n_layers`: linear in depth, each layer approximately adds ~(n_hidden^2 / 128) MB

Concrete estimates at bf16 + batch_size=1 + L=5 current (33 GB used):
- **n_layers=7**: ~33 × (7/5) ≈ 46 GB — fits easily, 50 GB headroom remaining
- **n_hidden=160**: ~33 × (160/128)^2 ≈ 52 GB — fits
- **n_layers=6, n_hidden=160**: may approach 60–65 GB — tight, may need verification

Deeper/wider models are underexplored — the 5-layer config is the paper default without
specific tuning for this dataset size. The TandemFoilSet training set is large enough
that a bigger model may still benefit from more layers, especially since:
1. The val curve is still descending at timeout (best = final epoch)
2. The camber OOD splits (geom_camber_rc, geom_camber_cruise) show the largest residuals
   — harder generalization that more capacity might capture better

**Predicted improvement:** −3 to −10 on val_avg/mae_surf_p vs 69.86 baseline.
More conservative estimate for n_layers=7 (fewer new parameters): −2 to −5.
More aggressive estimate for n_hidden=160 (wider): −5 to −10.

## Instructions

### 1. Start from the current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+bf16+clip+floor) is
the default. Do NOT change optimizer, loss, clip, or eta_min — just vary the architecture.

### 2. Verify CLI flags for n_layers and n_hidden

Check `target/train.py` Config dataclass for `n_layers` and `n_hidden` (or equivalent
flags like `--num_layers`, `--hidden_dim`). These should exist in the Transolver config.

### 3. Add a fixed seed

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms (priority order)

**Arm 1 (primary) — deeper by 2 layers:**
```bash
cd target/ && python train.py \
    --n_layers 7 \
    --wandb_group deeper-model \
    --wandb_name transolver-l7 \
    --agent willowpai2i24h3-nezuko
```

**Arm 2 — wider hidden dim:**
```bash
cd target/ && python train.py \
    --n_hidden 160 \
    --wandb_group deeper-model \
    --wandb_name transolver-h160 \
    --agent willowpai2i24h3-nezuko
```

**IMPORTANT — monitor VRAM on Arm 2.** The n_hidden=160 run may use 50+ GB. If you see
OOM errors, reduce to n_hidden=148 or switch to n_layers=6 as an alternative.

Run Arm 1 first (n_layers=7 is safer on VRAM). If it finishes cleanly and shows improvement,
run Arm 2 as well. If Arm 1 shows no improvement, you can skip Arm 2.

### 5. Report key signals

- Peak VRAM per arm (for model sizing data)
- Total epochs in 30 min (deeper/wider = slower per-epoch, fewer total epochs)
- val_avg/mae_surf_p per epoch — does the deeper model converge at the same rate as L=5?
- Per-split val breakdown — do camber OOD splits improve more than in-dist?
- Best epoch: is best the final epoch (still converging) or earlier (overfit/plateau)?

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
| val_single_in_dist | 78.4834 |
| val_geom_camber_rc | 86.8730 |
| val_geom_camber_cruise | 45.3256 |
| val_re_rand | 68.7430 |
| **test_avg_nansafe/mae_surf_p** | **65.8812** |
| W&B run | `f6lnbssy` (group: `bf16-stable`) |
| Model | n_hidden=128, n_layers=5, slice_num=64 |
| VRAM | **33 GB / 96 GB used** — 63 GB headroom for bigger model |

Reproduce: `cd "target/" && python train.py --wandb_group bf16-stable --wandb_name lion-bf16-clip-floor --agent willowpai2i24h3-alphonse`
