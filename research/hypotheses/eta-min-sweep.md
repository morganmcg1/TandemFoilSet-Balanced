# Hypothesis: eta-min-sweep (tanjiro)

## Hypothesis

Your own T_max=21 result (PR #3596, val=65.74, NEW SOTA — just merged) revealed a
striking pattern: **best epoch=18, but epoch 19 mildly regresses in BOTH arms**:

```
Arm 2 (T_max=21):
ep17: val=66.99, lr=1.78e-5
ep18: val=65.74, lr=1.45e-5  ← best
ep19: val=66.63, lr=1.20e-5  ← mild regression
```

Your own conclusion: "The very lowest LRs aren't where the best minimum lives — there
is no advantage to a perfectly-tuned eta_min floor for this model+data+budget."

This directly motivates a **higher eta_min sweep**: raise the floor so the cosine
schedule's lower-LR window never decays below the productive range. With T_max=21:
- Current eta_min=1e-5: LR at ep21=1e-5 (over-decayed)
- eta_min=2e-5: LR at ep21=2e-5 (closer to the ep17-18 useful range)
- eta_min=3e-5: LR at ep21=3e-5 (similar magnitude to ep15-16)

The cosine curve shape stays the same; only the floor changes. The expected mechanism:
the late epochs (17-19) stay in the productive LR regime instead of dipping into the
non-productive regime, potentially keeping the model improving through epoch 19.

**Predicted improvement:** −0.3 to −2.0 on val_avg/mae_surf_p vs 65.74 baseline.
Modest because the regression at ep19 is only ~0.9 points, so the headroom from
fixing it is bounded. Best case: the higher floor also keeps eps 17-18 in a better
LR window and we gain a bit more.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack
(Lion+bf16+clip=1.0+eta_min=1e-5+**T_max=21**, your merged PR #3596) is the default.
Change ONLY `--eta_min` (and keep `--lr_T_max 21`).

### 2. Verify the eta_min CLI flag exists

In `target/train.py`, check if `--eta_min` is exposed. The Config has
`eta_min: float = 1e-5` per the merged stack — confirm CLI flag works. If not, add
the standard CLI exposure for the dataclass.

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — eta_min=2e-5 (moderate raise):**
```bash
cd target/ && python train.py \
    --eta_min 2e-5 \
    --lr_T_max 21 \
    --wandb_group eta-min-sweep \
    --wandb_name eta-min-2e5 \
    --agent willowpai2i24h3-tanjiro
```

**Arm 2 — eta_min=3e-5 (stronger floor):**
```bash
cd target/ && python train.py \
    --eta_min 3e-5 \
    --lr_T_max 21 \
    --wandb_group eta-min-sweep \
    --wandb_name eta-min-3e5 \
    --agent willowpai2i24h3-tanjiro
```

### 5. Key signals to report

- val_avg/mae_surf_p per epoch — does the ep19 regression go away?
- **Specifically log LR at each epoch** to confirm the eta_min change is engaging
- Best epoch — does it shift from 18 to 19 (if higher floor helps late refinement)?
- val_avg trajectory at matched epochs vs the merged baseline (`tew7xthq`)
- Per-split breakdown at best checkpoint

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — YOUR own lion-tmax21 (PR #3596, merged 2026-05-16 04:30 UTC):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **65.7375** |
| **test_avg_nansafe/mae_surf_p** | **61.7003** |
| test_single_in_dist | 61.9972 |
| test_geom_camber_rc | 69.7654 |
| test_geom_camber_cruise | 57.5355 |
| test_re_rand | 57.5030 |
| W&B run | `tew7xthq` (group: `lion-tmax-newbase`) |
| Stack | Lion lr=1e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + **eta_min=1e-5** + T_max=21 |
| Best epoch | **18** (ep19 mildly regresses) |

Reproduce baseline:
```bash
cd "target/" && python train.py --lr_T_max 21 --wandb_group lion-tmax-newbase --wandb_name lion-tmax21 --agent willowpai2i24h3-tanjiro
```

### Your T_max=21 per-epoch val (reference, from your prior PR):

```
ep | val      | lr
15 | 73.4881  | 2.694e-5
16 | 70.6841  | 2.219e-5
17 | 66.9996  | 1.782e-5
18 | 65.7375  | 1.446e-5   ← best
19 | 66.6349  | 1.200e-5   ← regression
```

The hypothesis is: raise eta_min so the LR at epochs 17-19 stays in the productive
~2-3e-5 range and the model keeps improving instead of dipping into the regression
region at the lowest LRs.
