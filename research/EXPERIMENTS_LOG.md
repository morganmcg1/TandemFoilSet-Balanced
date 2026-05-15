# SENPAI Research Results

## 2026-05-15 14:50 — PR #3181: Gradient clipping + Huber loss for high-Re training stability
- Branch: `willowpai2i48h2-edward/grad-clip-huber`
- Student: willowpai2i48h2-edward
- Hypothesis: `grad_clip=1.0` + Huber loss (δ=1.0) stabilize training against high-Re gradient spikes; expect −3% to −10% on `val_avg/mae_surf_p`.

### Results

| Metric | Value | Note |
|---|---|---|
| `val_avg/mae_surf_p` (best @ epoch 11) | 110.5481 | primary, clean |
| `test_avg/mae_surf_p` (4 splits) | NaN | corrupted — see scoring.py bug below |
| `test_avg/mae_surf_p` (3 clean splits, partial) | 107.2103 | mean of single/rc/re_rand |
| W&B run | `p9iio40u` | https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/p9iio40u |
| Wall clock | 30.7 min (timeout) | epoch 14/50 — wall-clock bound |
| Peak GPU mem | 42.1 GB / 96 GB | room to spare |
| Pre-clip grad norm | median 16.15, p99 75.69, max 225.36 | 100% of 5,255 steps clipped at max_norm=1.0 |

Per-split val (best ckpt @ epoch 11):

| Split | mae_surf_p |
|---|---|
| val_single_in_dist | 135.7599 |
| val_geom_camber_rc | 122.7890 |
| val_geom_camber_cruise | 83.4849 |
| val_re_rand | 100.1585 |

### Conclusion

**Sent back for clip-norm sweep.** The hypothesis is well-motivated and the run was stable, but `max_norm=1.0` was vastly too aggressive — 100% of steps clipped, effective LR cut ~16×, and the model didn't converge (val trajectory: 235→126→111→128→123→113 over epochs 1–14, with timeout cutting training short). We can't disentangle "Huber+clip helps" from "model didn't converge" without a less aggressive clip.

Retry assignment: sweep `max_norm` ∈ {5.0, 10.0} with Huber δ=1.0. Same wandb_group.

### Side discoveries

- **`data/scoring.py` NaN propagation bug.** Sample `.test_geom_camber_cruise_gt/000020.pt` contains `inf` in the pressure channel. The current code computes `err = (pred - y).abs()` (which becomes `inf`) and THEN multiplies by `sample_mask`, but IEEE-754 `inf * 0 = NaN`, so the NaN propagates into the accumulator. Affects `test_avg/mae_surf_p` for any run on this branch.
  Fix: zero out non-finite-y samples in `err` before the mask multiply. Not addressed in this PR (data/scoring.py is read-only for students); needs a separate advisor-routed fix.
