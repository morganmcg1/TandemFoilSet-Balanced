# 3-epoch lr warmup + standard cosine on n_layers=2+slice_num=16+epochs=46: test schedule HEAD

## Hypothesis

**Pivot from your own PR #2760 closure.** Your truncated cosine experiment refuted the schedule-TAIL hypothesis but revealed that **schedule shape matters** (your "polish phase" diagnostic). Standard cosine drops to near-zero in final ~5 epochs and this helps generalization. Truncated cosine eliminated that and caused OOD oversteering.

**This experiment tests the schedule HEAD complementarily.** Your epoch-1 val_avg=167.978 is the chaotic initial phase — model takes its first noisy gradient steps at peak LR=1e-4 immediately. Linear warmup over 3 epochs (lr ramps from 1e-5 to 1e-4) should give gentler initial gradients before cosine decay kicks in.

**Why warmup might help:**

1. **Direct evidence of chaotic start in your data**: epoch 1 val=167.978 dropping to epoch 10 val=74.4 — that's an 8× swing in 10 epochs. Most of that is the model recovering from initial bad gradients. Warmup would smooth this initial transition.

2. **Empirically validated in transformer training**: GPT, BERT, ViT all use lr warmup. Standard 3-10% of total epochs is typical. 3/46 = 6.5%, well within standard range.

3. **No compute cost change**: Same 46 epochs, same per-step cost. Should match baseline ~35s/epoch × 46 = ~27 min.

4. **Different perturbation from #2760**: Truncated cosine altered the late-epoch schedule; warmup alters the early-epoch schedule. These are independent perturbations of the cosine arc.

5. **OOD may benefit from gentler start**: chaotic early gradients may set up over-specialized in-distribution representations that don't transfer to OOD. Gentler start = more general representations early on.

6. **Conservative ratio (3 epochs)**: small perturbation, cheap to refute. If warmup helps, can sweep warmup_epochs=5 or 7 next.

## Code change required

Edit `train.py`:

1. **Add CLI arg** (in @dataclass around line 389):
   ```python
   warmup_epochs: int = 0  # If >0, linear warmup from lr/10 to lr over first N epochs, then cosine decay.
   ```

2. **Modify scheduler creation** (around line 445):
   ```python
   _t_max = cfg.t_max if cfg.t_max > 0 else MAX_EPOCHS
   if cfg.warmup_epochs > 0:
       warmup = torch.optim.lr_scheduler.LinearLR(
           optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg.warmup_epochs
       )
       cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=_t_max - cfg.warmup_epochs
       )
       scheduler = torch.optim.lr_scheduler.SequentialLR(
           optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
       )
   else:
       scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_t_max)
   ```

**Verify backwards compatibility BEFORE running the experiment**: with default `--warmup_epochs 0`, scheduler should be exactly the original `CosineAnnealingLR(T_max=MAX_EPOCHS)`. Test with a 2-epoch sanity check to confirm.

## Instructions

Two flag changes from PR #2468 winner: `--warmup_epochs 3` (plus the code modification above).

```bash
cd target/ && python train.py \
  --agent charliepai2g48h3-askeladd \
  --experiment_name warmup3-nlayers2-slicenum16-epochs46 \
  --epochs 46 \
  --warmup_epochs 3 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --surf_weight 10 \
  --n_layers 2 \
  --slice_num 16
```

## Reporting

1. Per-split val/test mae_surf_p vs baseline (val=35.256 / test=30.245)
2. **Per-epoch LR for epochs 1-5** — confirm warmup ramping correctly (1e-5 → 1e-4)
3. **Epoch 1 val_avg** — does warmup tame the chaotic start? (baseline was 167.978)
4. **Best epoch** — does best_epoch=46 still hold or shift?
5. **OOD splits** — geom_camber_rc and geom_camber_cruise: does gentler start help OOD?
6. **single_in_dist** — does it improve or regress?
7. Total wall-clock and per-epoch s

## Baseline (PR #2468)

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 36.476 | 33.035 |
| geom_camber_rc | 48.297 | 44.333 |
| geom_camber_cruise | 18.326 | 15.496 |
| re_rand | 37.923 | 28.116 |
| **avg** | **35.256** | **30.245** |

**Reproduce baseline:**
```bash
cd target/ && python train.py \
  --epochs 46 --lr 1e-4 --weight_decay 1e-4 --batch_size 4 \
  --surf_weight 10 --n_layers 2 --slice_num 16
```

## Results format

```
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"metric_artifacts":["<path>"],"primary_metric":{"name":"val_avg/mae_surf_p","value":<number>},"test_metric":{"name":"test_avg/mae_surf_p","value":<number>}}
```
