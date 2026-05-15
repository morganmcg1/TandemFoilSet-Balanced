# Baseline — `icml-appendix-charlie-pai2i-48h-r4`

## Current best

**Track:** `icml-appendix-charlie-pai2i-48h-r4`
**Status:** Fresh research track — no baseline metrics committed yet. The first round of PRs establishes the reference number for `val_avg/mae_surf_p`.

**Primary ranking metric (lower is better):** `val_avg/mae_surf_p`
**Test metric (lower is better):** `test_avg/mae_surf_p`

## Reference configuration (unmodified `train.py`)

- Model: 5-layer Transolver, `n_hidden=128`, `n_head=4`, `slice_num=64`, `mlp_ratio=2`
- Optimizer: AdamW, `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`
- Scheduler: CosineAnnealingLR(T_max=epochs)
- Loss: MSE in normalized space, `vol_loss + 10 * surf_loss`
- Training budget per run: `SENPAI_TIMEOUT_MINUTES=30`, `SENPAI_MAX_EPOCHS=50`

## Reproduce

```bash
cd target/
python train.py --experiment_name baseline
```

## Round 1 protocol

Every PR in this fresh round runs a **paired comparison**:
- **Arm A** = unmodified baseline (one full training run)
- **Arm B** = hypothesis change (one full training run)

The student commits both `metrics.jsonl` outputs and reports both numbers in their PR. This makes each PR self-contained and robust to per-run variance until enough runs accumulate to give a stable absolute baseline. After Round 1 we will commit a stable baseline number here and the round-2 protocol can drop the paired arm.
