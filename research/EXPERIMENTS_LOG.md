# SENPAI Research Results

## 2026-05-15 15:42 — PR #3202: Linear warmup (5 epochs) + cosine annealing
- Branch: `willowpai2i48h2-tanjiro/lr-warmup-cosine`
- Student: willowpai2i48h2-tanjiro
- Hypothesis: 5-epoch linear warmup (`start_factor=0.01`) followed by cosine decay stabilizes early-epoch transformer training; predicted −3% to −8% on `val_avg/mae_surf_p`.

### Results

| Metric | Value | Note |
|---|---|---|
| `val_avg/mae_surf_p` (best @ epoch 12) | 149.8448 | **+9.46% vs baseline (136.89) — regression** |
| `test_avg/mae_surf_p` | NaN | cruise GT inf bug |
| `test_avg/mae_surf_p` (3 valid splits) | 151.93 | +10.3% vs baseline 137.69 |
| W&B run | `kg5wb8av` | https://wandb.ai/wandb-applied-ai-team/senpai-v1/runs/kg5wb8av |
| Wall clock | 30.8 min (timeout) | epoch 14/50 — wall-clock bound |
| Peak GPU mem | 42.1 GB / 96 GB | |

Per-split val (best ckpt @ epoch 12):

| Split | tanjiro (warmup) | baseline (07efagec) | Δ |
|---|---|---|---|
| val_single_in_dist | 183.7691 | 151.8490 | +21.0% |
| val_geom_camber_rc | 177.8992 | 173.9127 | +2.3% |
| val_geom_camber_cruise | 109.3022 | 101.4053 | +7.8% |
| val_re_rand | 128.4087 | 120.3820 | +6.7% |

### Conclusion

**Sent back for budget-aware reformulation.** All 4 val splits regress versus baseline. The student's own analysis identifies the failure mode cleanly: under the 30-min wall-clock cap only ~14 epochs land, and 5 of those (~36%) sit in sub-peak warmup with the cosine tail barely activating. The model is under-converged, not stabilized.

Retry assignment: arm A = `warmup_epochs=2, T_max=48` (shape-preserved, ~14% of realized budget in warmup); arm B = `warmup_epochs=3, T_max_realized=9` with `start_factor=0.1` (cosine actually decays inside the wall-clock window). Same `wandb_group=lr-warmup-cosine`.

---

## 2026-05-15 15:41 — PR #3176: Per-channel pressure weighting in surface loss (w=3, w=5)
- Branch: `willowpai2i48h2-askeladd/pressure-channel-weight`
- Student: willowpai2i48h2-askeladd
- Hypothesis: Multiplying the squared error on the pressure channel of `surf_loss` by `p_surf_weight` redirects gradient signal toward the primary metric; predicted −5% to −15% on `val_avg/mae_surf_p`.

### Results

| Metric | baseline (w=1, `07efagec`) | arm A (w=3, `g0n1r7pq`) | Δ | arm B (w=5, `8pizb0t7`) | Δ |
|---|---|---|---|---|---|
| **`val_avg/mae_surf_p`** | **136.8873** | **134.6330** | **−1.65%** | 165.2153 | +20.69% |
| val_single_in_dist | 151.8490 | 166.7821 | **+9.83%** | 242.4408 | +59.66% |
| val_geom_camber_rc | 173.9127 | **140.7154** | **−19.09%** | 161.7334 | −6.99% |
| val_geom_camber_cruise | 101.4053 | 108.0969 | +6.60% | 114.4373 | +12.85% |
| val_re_rand | 120.3820 | 122.9376 | +2.12% | 142.2498 | +18.16% |
| best epoch | 14 | 13 | | 14 | |
| `test_avg` (3-split mean) | 137.6945 | 131.1982 | −4.72% | 167.2087 | +21.43% |

W&B runs: baseline `07efagec` (`baseline-w1-ref`), arm A `g0n1r7pq` (`p-surf-w3`), arm B `8pizb0t7` (`p-surf-w5`), all under wandb_group `pressure-channel-weight`. Peak mem ~6.6 GB per run.

### Conclusion

**Sent back for finer weight sweep.** Arm A's −1.65% on the headline is a real but fragile gain: 3 of 4 val splits regress, with a single huge RC-camber win (−19%) carrying the average. The branch's "common-recipe over single-split hacks" rule says do not lock this in as a default. Arm B (w=5) over-weights pressure into clear regression. The student themselves recommended not merging.

There is a real OOD-camber signal underneath the per-split noise (`p` weight monotonically helps RC camber), so the question becomes whether a gentler weight preserves that gain without trashing val_single_in_dist.

Retry assignment: arm C = `p_surf_weight=1.5`, arm D = `p_surf_weight=2.0` under same `wandb_group=pressure-channel-weight`. Acceptance criterion: `val_avg` improves AND `val_single_in_dist` regresses by ≤2% vs baseline 151.85.

### Side discoveries

- **NaN scoring bug** confirmed at sample-level granularity: `.test_geom_camber_cruise_gt/000020.pt` contains 761 NaN values in the pressure channel of GT. `inf * 0 = NaN` in the `err * sample_mask` chain then NaNs `test_geom_camber_cruise/mae_surf_p` and propagates to `test_avg/mae_surf_p` and `vol_loss` (which becomes `+inf`). Ux/Uy stay finite because their GT is clean. Still needs an advisor-routed fix (`data/scoring.py` is read-only for students).

---

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
