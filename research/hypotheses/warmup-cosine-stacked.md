# Hypothesis: warmup-cosine-stacked (askeladd)

## Hypothesis
Linear LR warmup (5 epochs) + cosine decay + gradient clipping, stacked on top of the
merged Huber loss baseline. Warmup prevents large early-epoch gradient updates from
destabilizing the Huber-loss regime; cosine annealing ensures the LR actually reaches
its tail before the 30-min timeout; grad-clip=1.0 provides a hard ceiling on update
magnitude. This lever was tested in round-3 against the MSE baseline (val_avg=109.99 vs
baseline 129.99) and showed the strongest improvement in the cohort after Huber loss.
Now testing on the Huber baseline to see if it compounds.

**Predicted improvement:** −5 to −10 on val_avg/mae_surf_p vs 107.46.

## Instructions

Modify `target/train.py` as follows:

### 1. Add warmup + cosine sequential scheduler (replace lines ~435-436)

Replace the existing `CosineAnnealingLR` with a `SequentialLR`:

```python
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

warmup_epochs = 5
warmup_sched = LinearLR(
    optimizer,
    start_factor=1.0 / warmup_epochs,  # starts at lr/5 = 1e-4
    end_factor=1.0,                      # ends at full lr = 5e-4
    total_iters=warmup_epochs,
)
cosine_sched = CosineAnnealingLR(
    optimizer,
    T_max=max(MAX_EPOCHS - warmup_epochs, 1),  # cosine over remaining epochs
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_sched, cosine_sched],
    milestones=[warmup_epochs],
)
```

### 2. Add gradient clipping (in the training loop, after loss.backward())

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ADD THIS
optimizer.step()
```

### 3. Run the experiment

```bash
cd target/ && python train.py \
    --wandb_group warmup-cosine-stacked \
    --wandb_name warmup5-cos-clip1 \
    --agent willowpai2i24h3-askeladd
```

Run 2–3 arms under `--wandb_group warmup-cosine-stacked` (same code, different seeds if
desired, or vary `warmup_epochs=2` as a second arm to compare).

## Baseline

- **val_avg/mae_surf_p:** 107.4641 (frieren PR #3248, Huber δ=2.0)
- **test_avg_nansafe/mae_surf_p:** 101.9848
- **W&B run:** `mp8s8okf`
- **Reproduce baseline:** `cd target/ && python train.py --wandb_group huber-robust-loss --wandb_name huber-delta2 --agent willowpai2i24h3-frieren`
