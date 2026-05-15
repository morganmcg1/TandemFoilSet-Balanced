# Baseline — icml-appendix-willow-pai2i-48h-r3

The unmodified `train.py` at commit 845d9bb on this branch is the baseline. No fresh end-to-end W&B baseline run exists yet — round 3 starts cold.

## Current baseline configuration

| Component | Value |
|---|---|
| Architecture | Transolver |
| `n_hidden` | 128 |
| `n_layers` | 5 |
| `n_head` | 4 |
| `slice_num` | 64 |
| `mlp_ratio` | 2 |
| `dropout` | 0.0 |
| Optimizer | AdamW |
| `lr` | 5e-4 |
| `weight_decay` | 1e-4 |
| `batch_size` | 4 |
| `surf_weight` | 10.0 |
| Loss | MSE in normalized space, `vol_loss + 10*surf_loss` |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` |
| `epochs` | 50 |
| Sampling | Balanced `WeightedRandomSampler` over 3 domains |

## How students should report against baseline

Until a canonical baseline run exists on this branch, every assigned PR must include a **`baseline` arm** in the same `--wandb_group` as the variant arm:

```bash
# Arm 1 — baseline (unchanged train.py)
python train.py --wandb_group <hyp-slug> --wandb_name baseline --agent <student>

# Arm 2 — variant (after the code edit)
python train.py --wandb_group <hyp-slug> --wandb_name variant --agent <student>
```

Report both arms' `val_avg/mae_surf_p` and `test_avg/mae_surf_p` in the SENPAI-RESULT marker so the advisor can compare apples-to-apples.

Once a baseline is established by any student, future rounds can reference it directly without a re-run arm.
