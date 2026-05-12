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

## 2026-05-12 19:50 — PR #1447: tanjiro `batch_size=8` (review 1, closed — dead end)

- Branch: `willowpai2g48h5-tanjiro/batch-size-8`
- Hypothesis: double batch_size 4 → 8 to halve gradient variance under `WeightedRandomSampler`.
- W&B run: `qp366pa4` (14 of 30 epochs; hit 30-min cap at 30.34 min)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14) | **154.74** |
| `val/single_in_dist/mae_surf_p` | 232.43 |
| `val/geom_camber_rc/mae_surf_p` | 160.45 |
| `val/geom_camber_cruise/mae_surf_p` | 104.41 |
| `val/re_rand/mae_surf_p` | 121.67 |
| `clean_test_avg/mae_surf_p` (post-NaN-patch re-eval) | **138.92** |
| `test/test_single_in_dist/mae_surf_p` | 201.96 |
| `test/test_geom_camber_rc/mae_surf_p` | 142.73 |
| `test/test_geom_camber_cruise/mae_surf_p` | 92.47 |
| `test/test_re_rand/mae_surf_p` | 118.53 |
| Peak GPU memory | **~94.0 GB / 96 GB (98.3%)** |
| Per-epoch wall clock | ~130 s (≈ same as `bs=4`) |

- Worst of the four reviewed round-1 PRs. The hypothesis predicted "smoother training → lower val" but mis-predicted wall-clock: per-epoch time stayed ~130 s because dataloader/collator dominates, so doubling `bs` halved optimizer steps per minute under the hard 30-min cap. Val curve was still descending steeply at the cap (170 → 156 → 155), confirming the run was under-trained relative to siblings, not bad on the merits.
- Decision: **close**. No salvageable variation within the wall-clock constraint — `bs=6` has the same direction at smaller magnitude, `bs=4, accum=2` is identical in compute, and `bs=4` baseline becomes obsolete once alphonse's bf16 (#1419) lands as round-1 baseline.
- Useful artifacts kept: (1) independent confirmation that the train.py NaN workaround produces a clean `clean_test_avg/mae_surf_p = 138.92` on the previously-poisoned scoring path; (2) 94 GB peak GPU measurement at `slice_num=64, n_hidden=128, bs=8` — bounds what `n_hidden=192` (frieren #1442) can stack on top; (3) student killed a duplicate `bs=8` launch (`4g5fatyx`) — good housekeeping.
- Lesson recorded: under our 30-min wall-clock cap, any lever that doesn't speed up *per-epoch* wall clock (or improve sample-efficiency dramatically) leaves optimizer steps on the table. This is why bf16 wins and why pure batch-size scaling loses. Architectural levers face the same headwind.

## 2026-05-12 20:05 — PR #1419 v2: alphonse bf16 autocast + scoring fix (MERGED — new baseline)

- Branch: `willowpai2g48h5-alphonse/bf16-autocast`
- W&B run: `4hy79j91` (18 of 30 epochs; hit 30-min cap at 30.37 min)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 18) | **109.2937** |
| `val/single_in_dist/mae_surf_p` | 133.2714 |
| `val/geom_camber_rc/mae_surf_p` | 115.3895 |
| `val/geom_camber_cruise/mae_surf_p` | 87.8295 |
| `val/re_rand/mae_surf_p` | 100.6844 |
| `test_avg/mae_surf_p` (W&B summary, finite) | **97.6659** |
| `test/test_single_in_dist/mae_surf_p` | 113.9645 |
| `test/test_geom_camber_rc/mae_surf_p` | 105.7068 |
| `test/test_geom_camber_cruise/mae_surf_p` | 73.3736 |
| `test/test_re_rand/mae_surf_p` | 97.6189 |
| Per-epoch wall clock | ~101 s |

- **Decision: MERGE** — round-1 winner. bf16 autocast + NaN scoring fix. All 4 test splits finite (W&B summary). Val curve still descending at cap (best is epoch 18, last epoch run). Small improvements vs v1 (val 110.84 → 109.29, test offline-99.79 → W&B-97.67) are within RNG variance; the important outcome is that the W&B-logged test metric is now fully clean and mergeable.
- bf16 gives ~101 s/epoch vs ~150-160 s fp32 — 18 epochs in 30 min vs 11-14 for fp32 siblings.
- Scoring fix confirmed working: `test_geom_camber_cruise/mae_surf_p = 73.37` (no longer NaN). Note: `test_geom_camber_cruise/loss` and `vol_loss` still show NaN in W&B (the non-workaround loss accumulator path was not patched; cosmetic only).
- **New baseline**: val=109.29, test=97.67. All subsequent PRs compare against this.

## 2026-05-12 20:10 — PR #1436: fern Huber (Smooth L1) loss (review 1, sent back for rebase)

- Branch: `willowpai2g48h5-fern/huber-loss`
- W&B run: `8kxmpkhu` (14 of 30 epochs; ~130 s/epoch fp32)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14) | **109.4523** |
| `val/single_in_dist/mae_surf_p` | 142.1625 |
| `val/geom_camber_rc/mae_surf_p` | 116.2204 |
| `val/geom_camber_cruise/mae_surf_p` | 82.3047 |
| `val/re_rand/mae_surf_p` | 97.1215 |
| `test_avg/mae_surf_p` (clean) | **98.4633** |
| `test/test_single_in_dist/mae_surf_p` | 123.5613 |
| `test/test_geom_camber_rc/mae_surf_p` | 104.7107 |
| `test/test_geom_camber_cruise/mae_surf_p` | 70.5944 |
| `test/test_re_rand/mae_surf_p` | 94.9866 |

- **Remarkable result**: val=109.45 at fp32 14 epochs nearly matches the merged baseline val=109.29 at bf16 18 epochs. This implies Huber's sample-efficiency gain ≈ 4 epochs of extra training, which is large.
- Not merged as-is because: (1) 109.45 > 109.29 new baseline (technically doesn't beat it by 0.16); (2) tested against unmerged stock — needs rebase + retest on bf16 baseline.
- **Sent back**: rebase onto merged `icml-appendix-willow-pai2g-48h-r5`, remove duplicate NaN workaround (already in baseline), rerun with bf16 inherited. Predicted val ~95-105 if effects stack (Huber + bf16).

## 2026-05-12 20:10 — PR #1430: edward lr=1e-3 + warmup (review 1, closed)

- Branch: `willowpai2g48h5-edward/lr-1e-3-warmup`
- W&B run: `w5uih4d4` (14 of 30 epochs)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 14) | 137.7464 |
| `test_avg/mae_surf_p` (clean) | 124.2629 |

- **Decision: close**. cosine T_max=30 mismatch with actual 14-epoch budget left the run at lr≈5.8e-4 (midpoint of the annealing curve) at termination — never reached the low-LR polishing phase. The hypothesis (lr=1e-3 outperforms lr=5e-4) is untested; the schedule-budget mismatch is the confound. Retesting at fp32 with corrected schedule (`--epochs 15`) would still face a structurally disadvantaged epoch count vs the merged bf16 baseline (~14 fp32 vs ~18 bf16 epochs). Closed.
- The better follow-up is lr=1e-3 + warmup ON TOP OF the merged bf16 baseline — assigned fresh to edward as round-2 (lr+warmup may compound with bf16's throughput benefit).
- NaN workaround application was correct; LR schedule implementation was correct.

## 2026-05-12 20:10 — PR #1451 v2: thorfinn slice_num=128 rerun (review 2, closed)

- Branch: `willowpai2g48h5-thorfinn/slice-num-128`
- W&B run: `0t5h1kwd` (11 of 30 epochs; ~171 s/epoch at bs=2)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 9) | 141.4702 |
| `test_avg/mae_surf_p` (clean 4-split) | **128.9030** |
| Peak GPU memory | 27.0 GB (bs=2) |

- **Decision: close**. The NaN workaround works (all 4 test splits finite). But val=141.47 is well below the merged baseline (109.29). The bs=2 confound persists — run-to-run noise at bs=2 swamped the signal (the two runs differed by 4.8 MAE on val). At ~171 s/epoch (bs=2), only 11 epochs fit in 30 min.
- The hypothesis (slice_num doubling → better OOD) remains untested cleanly. Assigned thorfinn H4 slice_num=96 as the clean test: less memory than 128, fits at bs=4, bf16 inherited → ~12-15 epochs, no batch-size confound.

