# Hypothesis: vol-loss-p-weight (fern)

## Hypothesis

Your prior surf_loss p-weight experiment (#3598) showed that upweighting pressure in
the *surface* loss term monotonically degrades all channels. Your own analysis correctly
diagnosed why: z-scoring gives p a balanced gradient share in surf_loss already, and
forcing more weight disrupts the shared encoder representation.

However, you noted a key asymmetry: **volume p MAE (60–100 in normalized space) is much
larger than volume Ux/Uy (1–4).** This means p may be structurally *under*weighted in
`vol_loss` even after z-scoring — because the vol MAE channels are not equally scaled.
If the volume loss is dominated by pressure anyway, explicitly upweighting vol-p might
not help. But if Ux/Uy gradients are swamping p in the backward pass through vol_loss,
a 2× vol_p weight could re-balance the gradient contribution and improve pressure
prediction without disrupting the shared representation.

This is the **complementary direction**: test the volume loss component, not the surface
loss. The mechanism is different (volume loss operates on all mesh nodes, not just
surface nodes) and the channel imbalance is potentially more severe.

**Predicted improvement:** Uncertain — could be 0 to −3 if vol-p was genuinely
underweighted. Could be +2 if the volume loss is already well-balanced and extra p
weight hurts like surf_loss did. The prior result is informative about direction.

## Instructions

### 1. Start from current advisor branch

Branch from `icml-appendix-willow-pai2i-24h-r3` — full stack (Lion+bf16+clip=1.0+
eta_min=1e-5+**T_max=21**) is the default. Change ONLY `vol_p_weight`.

### 2. Check if vol_p_weight CLI flag exists

In `target/train.py`, check if the `Config` dataclass has `vol_p_weight: float = 1.0`
and the training loss code uses it. If not, add it analogously to `p_weight` (which
you added for surf_loss in your prior PR) but targeting the volume loss term.

The volume loss computation should become:
```python
vol_ch_w = torch.tensor([1.0, 1.0, cfg.vol_p_weight], device=...)
vol_loss = (huber_per_channel * vol_ch_w).mean()
```

### 3. Fixed seed (mandatory)

```python
import torch, random, numpy as np
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
```

### 4. Run two arms

**Arm 1 (primary) — vol_p_weight=2.0:**
```bash
cd target/ && python train.py \
    --vol_p_weight 2.0 \
    --lr_T_max 21 \
    --wandb_group vol-loss-p-weight \
    --wandb_name vol-p-weight-2x \
    --agent willowpai2i24h3-fern
```

**Arm 2 — vol_p_weight=1.5 (lighter upweight as sanity check):**
```bash
cd target/ && python train.py \
    --vol_p_weight 1.5 \
    --lr_T_max 21 \
    --wandb_group vol-loss-p-weight \
    --wandb_name vol-p-weight-1p5x \
    --agent willowpai2i24h3-fern
```

If both arms regress monotonically (same pattern as surf_loss), stop early and report.
Do not launch a 3rd arm unless Arm 1 shows any improvement signal.

### 5. Key signals to report

- val_avg/mae_surf_p per epoch — do all channels track or does just p improve?
- Per-channel surface MAE (Ux, Uy, p) at best epoch — key to understanding whether
  vol_loss reweighting affects surf_loss metrics
- Gradient contribution: if possible, log vol_loss vs surf_loss magnitudes per epoch
- Compare against your prior surf_loss p-weight result — is the vol_loss channel imbalance
  different in character?
- Best epoch and whether val is still descending at timeout

### 6. Compute nansafe test metrics

Run `eval_nansafe.py` on each arm's best checkpoint.

### 7. Post terminal SENPAI-RESULT

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<id1>","<id2>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<best>},"test_metric":{"name":"test_avg_nansafe/mae_surf_p","value":<number>}}
```

## Baseline

Current best — tanjiro's lion-tmax21 (PR #3596, merged 2026-05-16):

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p** | **65.7375** |
| **test_avg_nansafe/mae_surf_p** | **61.7003** |
| test_single_in_dist | 61.9972 |
| test_geom_camber_rc | 69.7654 |
| test_geom_camber_cruise | 57.5355 |
| test_re_rand | 57.5030 |
| W&B run | `tew7xthq` (group: `lion-tmax-newbase`) |
| Stack | Lion lr=1e-4, wd=1e-2 + Huber δ=2.0 + bf16 + clip=1.0 + eta_min=1e-5 + **T_max=21** |

Reproduce baseline:
```bash
cd "target/" && python train.py --lr_T_max 21 --wandb_group lion-tmax-newbase --wandb_name lion-tmax21 --agent willowpai2i24h3-tanjiro
```

### Prior surf_loss result for reference (your #3598):
- p_weight=2.0: val=77.18, test=71.39 — ALL channels regressed
- p_weight=4.0: val=79.94, test=73.97 — monotonic degradation
- Lesson: surf_loss reweighting disrupts shared encoder representation
- Question: does vol_loss behave the same way?
