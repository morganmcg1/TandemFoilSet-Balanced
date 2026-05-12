# SENPAI Research Results — icml-appendix-willow-pai2g-48h-r5

This log records experiment outcomes for the willow-pai2g-48h-r5 track. New entries are appended at the bottom.

## 2026-05-12 18:56 — PR #1427: askeladd `surf_weight=30` (review 1, sent back)

- Branch: `willowpai2g48h5-askeladd/surf-weight-30`
- Hypothesis: raise `surf_weight` from 10 → 30 to align gradient norm with surface-pressure MAE primary metric. No code change other than the flag.
- W&B run: `kfsefiwx` (12 of 30 epochs; hit 30-min cap at 30.52 min)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 12) | **134.14** |
| `val/single_in_dist/mae_surf_p` | 159.38 |
| `val/geom_camber_rc/mae_surf_p` | 141.89 |
| `val/geom_camber_cruise/mae_surf_p` | 111.28 |
| `val/re_rand/mae_surf_p` | 124.02 |
| `test_avg/mae_surf_p` | **NaN** (non-finite pred on ≥1 test_geom_camber_cruise sample) |
| `test/test_single_in_dist/mae_surf_p` | 141.25 |
| `test/test_geom_camber_rc/mae_surf_p` | 128.46 |
| `test/test_geom_camber_cruise/mae_surf_p` | NaN |
| `test/test_re_rand/mae_surf_p` | 122.25 |
| Peak GPU memory | 42.1 GB |
| Best epoch | 12 (val curve still descending at cap) |

- Val numbers all confirmed against W&B summary.
- Direction is promising — val moves cleanly in the expected direction on all four splits. But `test_avg/mae_surf_p` is NaN-poisoned by one numerically-bad prediction on `test_geom_camber_cruise`, which violates the program.md "finite paper-facing test metric required" contract.
- Sent back with instructions to add a defensive NaN-clean in `evaluate_split` and rerun. Initially I advised cleaning `pred`; corrected after thorfinn's diagnosis (next entry) to clean `y` instead — see below for the correct workaround.

## 2026-05-12 19:01 — PR #1451: thorfinn `slice_num=128` (review 1, sent back) + scoring bug diagnosis

- Branch: `willowpai2g48h5-thorfinn/slice-num-128`
- Hypothesis: `slice_num=64 → 128` in PhysicsAttention for finer attention partitioning.
- W&B run: `jxu2le44` (11 of 30 epochs; hit 30-min cap at 31.5 min)
- Deviation from PR: `--batch_size 2` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` after first launch OOMed at batch_size=4 (peak ~86 GB on 95 GB GPU; slice_num doubling pushed activation memory over the edge on the largest cruise tandem meshes).

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 10) | **136.69** |
| `val/single_in_dist/mae_surf_p` | 176.53 |
| `val/geom_camber_rc/mae_surf_p` | 139.40 |
| `val/geom_camber_cruise/mae_surf_p` | 106.37 |
| `val/re_rand/mae_surf_p` | 124.47 |
| `test_avg/mae_surf_p` | **NaN** (same scoring bug as #1427) |
| `test_avg/mae_surf_p` (3-split mean, info only) | 132.59 |
| Peak GPU memory | 27.3 GB at batch_size=2 |
| Params | 1.86M (slice_num=128 adds ~25k vs slice_num=64) |

### Infrastructure bug — `data/scoring.py` NaN propagation (root cause for both #1427 and #1451)

Thorfinn diagnosed and reported a real bug in the scoring path:

- `.test_geom_camber_cruise_gt/000020.pt` contains NaN in `y[:, 2]` (p channel).
- `data/scoring.py` does compute a sample-level `y_finite` mask intended to exclude such samples.
- But the err computation is `err = abs(pred - y).double()` *before* masking — so `err` itself has NaN entries at those positions.
- Under IEEE 754, `NaN * 0 = NaN`, so `(err * mask.unsqueeze(-1)).sum()` propagates NaN into the channel sums even though the mask is 0 for the bad sample.
- Affected metrics: `test_geom_camber_cruise/mae_surf_p`, `test_geom_camber_cruise/mae_vol_p` (both NaN) and the equal-weight averages that include them.

**Decision:** Fix lives in `train.py` (per the `data/` read-only contract). The workaround in `evaluate_split`:

```python
sample_finite = torch.isfinite(y.reshape(y.shape[0], -1)).all(dim=-1)   # [B]
mask_eff = mask & sample_finite.unsqueeze(-1)
y_safe = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
ds, dv = accumulate_batch(pred_orig, y_safe, is_surface, mask_eff, mae_surf, mae_vol)
```

This preserves scoring's intended sample-skipping semantics without the `NaN * 0` IEEE footgun. Val is unaffected (val GT is clean).

- Sent #1451 back with the corrected fix instruction + rerun at the same batch_size=2 / expandable_segments config.
- Sent a correction to #1427 (askeladd) replacing my earlier (wrong) `nan_to_num(pred, ...)` advice with the same `y`-side fix.
- Broadcast a heads-up comment to the six in-flight WIP PRs (#1419, #1430, #1436, #1442, #1445, #1447) so they can apply the fix preemptively if their final `test_avg/mae_surf_p` comes out NaN.

## 2026-05-12 19:05 — PR #1419: alphonse bf16 autocast (review 1, sent back — round-1 leader)

- Branch: `willowpai2g48h5-alphonse/bf16-autocast`
- Hypothesis: wrap forward + loss in `torch.amp.autocast(dtype=torch.bfloat16)`; keep optimizer/master weights fp32; eval in fp32.
- W&B run: `8d4b22mt` (18 of 30 epochs in 30 min; ~101 s/epoch)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 18, still descending) | **110.84** |
| `val/single_in_dist/mae_surf_p` | 138.30 |
| `val/geom_camber_rc/mae_surf_p` | 115.25 |
| `val/geom_camber_cruise/mae_surf_p` | 88.93 |
| `val/re_rand/mae_surf_p` | 105.11 |
| `test_avg/mae_surf_p` (W&B summary) | **NaN** (same scoring bug) |
| `test_avg/mae_surf_p` (offline, clean) | **99.79** |
| `test/test_single_in_dist/mae_surf_p` | 110.00 |
| `test/test_geom_camber_rc/mae_surf_p` | 110.37 |
| `test/test_geom_camber_cruise/mae_surf_p` | 76.31 |
| `test/test_re_rand/mae_surf_p` | 102.46 |
| Peak VRAM | 32.9 GB |
| Params | 662k |

- Numerically stable — no NaN/inf during training, smooth descending val curve.
- Independently diagnosed the scoring bug AND traced upstream cause: the 761 non-finite values in test_geom_camber_cruise sample 20 are all exactly `-65504.0 = -fp_max(bf16)` — overflow leakage from bf16 preprocessing in the dataset pipeline. Confirms the bug is `-inf`, not NaN, but both fail under `(+/-inf) * 0 = NaN`. Workaround `torch.nan_to_num(y, nan=0, posinf=0, neginf=0)` handles all three.
- Best result in round 1 so far (val 110.84 vs 134.14/136.69; offline-test 99.79 vs partial-3-split 130.65/132.59 from the other two). bf16 wall-clock speedup gave 18 epochs vs 11-12 for fp32 siblings — the hypothesis works as predicted.
- Sent back with: add the 4-line eval workaround in train.py, rerun, resubmit. The merged baseline will then carry both bf16 AND the scoring fix, propagating the fix to the rest of the round.

