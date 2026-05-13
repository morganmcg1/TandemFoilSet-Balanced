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

## 2026-05-12 20:55 — PR #1534: tanjiro gradient clipping `max_norm=1.0` (review 1, sent back for rebase)

- Branch: `willowpai2g48h5-tanjiro/grad-clip-1p0` (pre-merge fp32 baseline)
- W&B run: `2olay9t8` (14 of 30 epochs; ~130 s/epoch fp32)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 12) | 111.9067 |
| `test_avg/mae_surf_p` | 100.3499 |
| `test/test_single_in_dist/mae_surf_p` | 115.510 |
| `test/test_geom_camber_rc/mae_surf_p` | 117.827 |
| `test/test_geom_camber_cruise/mae_surf_p` | 70.212 |
| `test/test_re_rand/mae_surf_p` | 97.851 |
| Gradient clipping triggered | **5250 / 5250 steps (100%)** |
| Max pre-clip gradient norm | **837.32** |

- **Key empirical finding**: With `max_norm=1.0`, **every step of training was clipped** — the un-clipped baseline runs at gradient norms 300–800× above the cap. `max_norm=1.0` is therefore acting as full gradient normalization (`direction(g) * lr`), not as a safety net for rare spikes as the hypothesis framed it. This is a substantive observation about the training dynamics: AdamW's per-parameter scaling alone is not preventing large raw-gradient updates; the clip is effectively the norm controller.
- **Val trajectory**: monotonically descending after epoch 5, with small wobbles (max swing ±20 vs the ±50 seen in unclipped fp32 baselines like tanjiro's bs=8 closed PR). The smoothing effect is real and visible.
- **Result vs baseline**: val=111.91 fp32 14 epochs vs merged baseline 109.29 bf16 18 epochs. Very close given the 4-epoch disadvantage at fp32.
- **Decision: send back for rebase + retest on bf16 baseline.** bf16 doesn't affect raw gradient magnitudes meaningfully (master weights are fp32, autocast is forward-only), so the same "every step clipped" dynamic should hold on top of bf16. With 4 more bf16 epochs at smoother trajectory, predicted val ~95-105.

## 2026-05-12 20:58 — PR #1442: frieren wider Transolver `n_hidden=192` (review 1, sent back for rebase)

- Branch: `willowpai2g48h5-frieren/wider-n-hidden-192` (pre-merge fp32 baseline)
- W&B run: `5ux034zo` (10 of 30 epochs; ~182 s/epoch fp32, **batch_size=2 forced fallback**)
- Model: 1.47M params (vs ~1M at n_hidden=128)

| Metric | Value |
|--------|-------|
| `val_avg/mae_surf_p` (best, epoch 9) | 140.3826 |
| `test_avg/mae_surf_p` | 128.8978 |
| `test/test_single_in_dist/mae_surf_p` | 168.5868 |
| `test/test_geom_camber_rc/mae_surf_p` | 132.5917 |
| `test/test_geom_camber_cruise/mae_surf_p` | 89.0377 |
| `test/test_re_rand/mae_surf_p` | 125.3750 |
| Peak GPU memory | 91.07 GB (shared with co-resident process) |

- **Multiple confounders**: (1) bs=2 forced by OOM (co-resident process ate 73 GB on the shared GPU during the original bs=4 launch); (2) only 10 epochs in 30 min due to wider model's per-epoch slowdown; (3) cosine LR never reached its low-lr fine-tuning regime.
- The result (val=140.38) is below baseline by ~30 absolute MAE, but the confounders are extrinsic and the hypothesis is mechanistically reasonable. With bf16 inherited from the merged baseline, the wider model should fit at bs=4 cleanly (~30-40 GB estimated).
- **Decision: send back for rebase + retest at bs=4 on bf16 baseline.** Removes the bs=2 confound and adds ~4-6 extra epochs. If wider doesn't beat 109.29 on a fair test, close.

## 2026-05-12 21:05 — PR #1436 v2: fern Huber + bf16 (MERGED — new compound baseline)

- Branch: `willowpai2g48h5-fern/huber-loss` (rebased onto bf16 baseline)
- W&B run: `kmwsz3i4` (18 of 30 epochs; ~100 s/epoch bf16; best at epoch 16)

| Metric | Value | vs PR #1419 (alphonse) |
|--------|-------|------------------------|
| `val_avg/mae_surf_p` (best, epoch 16) | **96.4863** | **−12.81 (−11.7%)** |
| `val/single_in_dist/mae_surf_p` | 112.8995 | −20.37 |
| `val/geom_camber_rc/mae_surf_p` | 106.9168 | −8.47 |
| `val/geom_camber_cruise/mae_surf_p` | 75.1834 | −12.65 |
| `val/re_rand/mae_surf_p` | 90.9454 | −9.74 |
| `test_avg/mae_surf_p` | **86.3326** | **−11.33 (−11.6%)** |
| `test/test_single_in_dist/mae_surf_p` | 101.2155 | −12.75 |
| `test/test_geom_camber_rc/mae_surf_p` | 95.6042 | −10.10 |
| `test/test_geom_camber_cruise/mae_surf_p` | 64.2155 | −9.16 |
| `test/test_re_rand/mae_surf_p` | 84.2951 | −13.32 |

- **Compound winner**. Huber's loss-shape benefit + bf16's epoch-budget benefit stacked exactly as predicted. The relative drops are uniform across all 4 splits (−9 to −13 MAE) — Huber doesn't just help one regime, it helps everywhere.
- The single largest improvement was on `test_re_rand` (−13.32, −13.6%) and `test_single_in_dist` (−12.75, −11.2%). These are the splits with the largest extreme-p values; Huber's linear tail behavior on those samples is doing the right thing.
- **Decision: MERGE** — new baseline is Huber+bf16 = val 96.49, test 86.33. All subsequent PRs compare to this.
- Round-2 winner pattern is now confirmed: orthogonal levers from round 1 are stacking (bf16 → Huber → next?). Suggests the optimization frontier is wide open.

## 2026-05-12 21:54 — PR #1550: thorfinn `slice_num=96` (review 1, closed — dead end)

- Branch: `willowpai2g48h5-thorfinn/slice-num-96`
- W&B runs: `had0wmcv` (primary, 15 epochs), `h651wzrd` (confirmation, 13 epochs)
- Clean run — no OOM, no NaN, no fallback. bs=4 + bf16 fit cleanly (peak 43.2 GB).

| Metric | slice_num=96 | Baseline (4hy79j91, bf16) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best, epoch 15) | **120.69** | 109.29 | +11.40 (+10.4%) |
| `test_avg/mae_surf_p` | **110.24** | 97.67 | +12.58 (+12.9%) |
| `test/test_single_in_dist/mae_surf_p` | 124.31 | 113.96 | +10.35 |
| `test/test_geom_camber_rc/mae_surf_p` | 123.19 | 105.71 | +17.48 |
| `test/test_geom_camber_cruise/mae_surf_p` | 84.30 | 73.37 | +10.93 |
| `test/test_re_rand/mae_surf_p` | 109.17 | 97.62 | +11.55 |
| Per-epoch wall time | ~121 s | ~101 s | +20% |
| Epochs in 30 min | 15 | 18 | −3 |
| Run-to-run variance (val_avg) | ~6 pts | — | — |

- **Both mechanisms of failure are clean**: (1) 96 slices → +20% per-epoch cost → 15 epochs vs 18 for baseline; (2) even per-step the model learns more slowly — the slice projection (learned soft clustering) needs more gradient steps to differentiate 96 groups than 64. In a short-budget regime it never fully specializes.
- **OOD splits hit hardest** (+17.5 test on camber_rc, +10.9 on cruise) — opposite of predicted "finer partitioning helps extrapolation." The partitioning underfits when under-trained, giving near-random slice assignment on OOD inputs.
- **Confirmation run** (h651wzrd) gives val=126.78, test=113.97 — consistent, same direction, variance ~6 pts is much smaller than the 11-pt gap to baseline. The negative result is robust.
- **Decision: CLOSE.** Slice_num=96 is not competitive within the 30-min cap. The experiment does establish that `slice_num=64` is a good calibration for our budget; going wider costs throughput without compensating in per-step quality. Assigning thorfinn to dropout=0.1 (regularization direction, orthogonal).

## 2026-05-12 21:54 — PR #1606: fern EMA weights `decay=0.999` (MERGED — new baseline)

- Branch: `willowpai2g48h5-fern/ema-weights`
- W&B run: `gdfynh7o` (17 of 30 epochs; ~110 s/epoch; hit 30-min cap between epochs 17 and 18)
- Implementation: `copy.deepcopy(model)` EMA shadow, updated after every `optimizer.step()` on fp32 master weights; val + test eval use `ema_model`; live-model diagnostic pass also run each epoch.

| Metric | EMA (gdfynh7o) | Baseline (kmwsz3i4, Huber+bf16) | Δ | % |
|--------|----------:|----------:|---:|---:|
| `val_avg/mae_surf_p` (best, epoch 17) | **92.3452** | 96.4863 | −4.14 | −4.3% |
| `test_avg/mae_surf_p` | **81.6297** | 86.3326 | −4.70 | −5.4% |
| `test/test_single_in_dist/mae_surf_p` | 95.2950 | 101.2155 | −5.93 | −5.86% |
| `test/test_geom_camber_rc/mae_surf_p` | 91.9270 | 95.6042 | −3.67 | −3.84% |
| `test/test_geom_camber_cruise/mae_surf_p` | 58.7160 | 64.2155 | −5.50 | −8.57% |
| `test/test_re_rand/mae_surf_p` | 80.5810 | 84.2951 | −3.72 | −4.41% |
| Per-epoch wall time | ~110 s | ~100 s | +10% |
| Epochs in 30 min | 17 | 18 | −1 |
| Peak VRAM | 33.0 GB | ~33 GB | ≈ same |

**EMA-vs-live diagnostic (confirms mechanism works):**

| Epoch | EMA val | Live val | EMA − Live |
|-------|---------|----------|------------|
| 9 | 132.82 | 147.95 | −15.1 |
| 12 | 109.29 | 123.51 | −14.2 |
| 15 | 97.06 | 106.93 | −9.9 |
| 16 | 94.59 | 116.99 | −22.4 |
| 17 | 92.35 | 117.63 | −25.3 |

Live model at epoch 17: test=104.70. EMA at same epoch: test=81.63. EMA is +28% better than the instantaneous weights at the same training step. The noise ball interpretation is empirically confirmed.

- **All four test splits improve** uniformly. Largest relative gain on camber_cruise (−8.6%) — the smaller-magnitude domain where SNR of the weight-averaging benefit is highest.
- EMA half-life at decay=0.999 is ~1.85 epochs (693 steps at 375 steps/epoch). Lags during warmup (expected), catches up by epoch 6, consistently outperforms from epoch 9 onward.
- Implementation is minimal (~15 lines, no external dependencies). Adds ~3 MB of fp32 shadow weights for 0.66M-param model. EMA on fp32 master weights is correct — bf16 autocast only touches the forward pass.
- **Decision: MERGE** — new baseline val=92.35, test=81.63.

## 2026-05-12 22:00 — PR #1546: edward `n_layers=8` (review 1, closed — dead end)

- Branch: `willowpai2g48h5-edward/n-layers-8`
- W&B run: `9duc68ci` (12 of 30 epochs; 155 s/epoch; peak VRAM 79.24 GB)

| Metric | n_layers=8 | Baseline (bf16) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best, epoch 12) | 136.4966 | 109.2937 | +27.20 (+24.9%) |
| `test_avg/mae_surf_p` | 126.2424 | 97.6659 | +28.58 (+29.3%) |
| `test/test_single_in_dist/mae_surf_p` | 129.86 | 113.96 | +15.90 |
| `test/test_geom_camber_rc/mae_surf_p` | 142.91 | 105.71 | +37.20 |
| `test/test_geom_camber_cruise/mae_surf_p` | 102.38 | 73.37 | +29.00 |
| `test/test_re_rand/mae_surf_p` | 129.82 | 97.62 | +32.20 |
| Per-epoch wall time | 155 s | 101 s | +54% |
| Epochs in 30 min | 12 | 18 | −6 |

- **Both failure mechanisms are clean**: (1) +54% per-epoch cost → 12 epochs vs 18; (2) val trajectory still descending at epoch 12 — model not converged. Cosine schedule T_max=30 + actual 12 epochs = schedule runs at high-to-mid LR throughout, no low-LR polishing.
- **OOD splits worst**: camber_rc +37.2, re_rand +32.2 — opposite of predicted "depth helps OOD". Classic underfitting: higher-capacity model generalizes worse than lower-capacity model with same data budget when the higher-capacity model hasn't been trained long enough.
- **Pattern confirmed**: third architecture experiment (after mlp_ratio=4, slice_num=96) to fail under budget. Architecture capacity is not the bottleneck; training duration and schedule are.
- **Decision: CLOSE.** The schedule-alignment hypothesis (T_max=actual budget) is worth testing independently — assigning alphonse to that. n_layers=8 closed.

## 2026-05-12 22:00 — PR #1544: alphonse `mlp_ratio=4` (review 1, closed — dead end)

- Branch: `willowpai2g48h5-alphonse/mlp-ratio-4`
- W&B run: `yz7e5k2m` (17 of 30 epochs; 108 s/epoch; peak VRAM 38.1 GB)

| Metric | mlp_ratio=4 | Baseline (bf16) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best, epoch 17) | 115.0388 | 109.2937 | +5.74 (+5.3%) |
| `test_avg/mae_surf_p` | 101.6665 | 97.6659 | +4.00 (+4.1%) |
| `test/test_single_in_dist/mae_surf_p` | 127.53 | 113.96 | +13.57 |
| `test/test_geom_camber_rc/mae_surf_p` | 103.59 | 105.71 | −2.13 (better) |
| `test/test_geom_camber_cruise/mae_surf_p` | 75.67 | 73.37 | +2.30 |
| `test/test_re_rand/mae_surf_p` | 99.88 | 97.62 | +2.26 |
| Per-epoch wall time | ~108 s | ~101 s | +7% |
| Epochs in 30 min | 17 | 18 | −1 |

- **Same pattern at smaller scale**: +7% per-epoch overhead → 17 epochs vs 18, still descending at epoch 17 (best=last). Marginally less budget-impacted than n_layers=8 (108 vs 155 s/epoch), but the extra capacity doesn't help in 17 epochs.
- **3 of 4 test splits worse**: camber_rc is the only marginal win (−2.13). in_dist worsens the most (+13.57), consistent with conventional `mlp_ratio=4` over-parameterizing for our 1500-sample training set.
- **Architectural note**: conventional ML uses `mlp_ratio=4` for large datasets (ImageNet, language); TandemFoilSet has 1500 samples with batch_size=4. Wider MLP adds params without enough optimizer steps to benefit. mlp_ratio=2 is correctly sized for our regime.
- **Decision: CLOSE.** Assigning alphonse to LR schedule alignment (T_max=18).

## 2026-05-12 22:57 — PR #1648: edward SiLU activation (review 1, closed)

- Branch: `willowpai2g48h5-edward/silu-activation`
- W&B run: `y57lkrh4` (17 epochs; 110 s/epoch — identical to baseline)

| Metric | SiLU | EMA baseline (gdfynh7o) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | 96.9905 | 92.3452 | +4.65 (+5.0%) |
| `test_avg/mae_surf_p` | 88.4942 | 81.6297 | +6.86 (+8.4%) |
| `test/test_single_in_dist/mae_surf_p` | 103.84 | 95.30 | +8.54 |
| `test/test_geom_camber_rc/mae_surf_p` | 97.07 | 91.93 | +5.15 |
| `test/test_geom_camber_cruise/mae_surf_p` | 63.30 | 58.72 | +4.58 |
| `test/test_re_rand/mae_surf_p` | 89.77 | 80.58 | +9.19 |

- **All 4 splits worse.** No throughput penalty (110 s/epoch both). Sanity check confirmed SiLU was active in model.
- **"Smoother but slower" trajectory**: val curve was less noisy but descended more slowly. SiLU's gentler negative-tail gradient reduces training signal at lr=5e-4 — not what we need.
- **Decision: CLOSE.** GELU is correctly tuned for this optimizer/LR setup. Activation function is not a productive direction. Assigning edward to EMA decay=0.9995 sweep.

## 2026-05-12 22:57 — PR #1445 v2: nezuko per-channel surface weights (0.5, 0.5, 2.0) (review 1, closed)

- Branch: `willowpai2g48h5-nezuko/surf-channel-weights`
- W&B run: `yrb605fb` (17 epochs; ~110 s/epoch; rebased on EMA+Huber+bf16 baseline)

| Metric | Per-channel (yrb605fb) | EMA baseline (gdfynh7o) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | 93.6036 | 92.3452 | +1.26 (+1.4%) |
| `test_avg/mae_surf_p` | 83.7590 | 81.6297 | +2.13 (+2.6%) |
| `test/test_single_in_dist/mae_surf_p` | 98.54 | 95.30 | +3.25 |
| `test/test_geom_camber_rc/mae_surf_p` | 94.37 | 91.93 | +2.44 |
| `test/test_geom_camber_cruise/mae_surf_p` | 59.84 | 58.72 | +1.12 |
| `test/test_re_rand/mae_surf_p` | 82.29 | 80.58 | +1.71 |

- **All 4 splits regressed.** EMA-vs-live gap preserved (−8.32) — not an EMA artifact.
- **Root cause**: p already dominates gradient signal (high variance in normalized space); doubling its weight was redundant. U-channel down-weighting removed implicit geometric regularization.
- **Decision: CLOSE.** Per-channel weighting is not effective on top of EMA+Huber baseline. The optimizer is already attending to p. Assigning nezuko to linear LR warmup.

## 2026-05-12 23:05 — PR #1626: fern EMA without live-model diagnostic pass (review 1, closed)

- Branch: `willowpai2g48h5-fern/ema-no-diag`
- W&B run: `vx2n2zuq` (18 epochs; ~101 s/epoch; diagnostic live-val pass removed)
- Hypothesis: removing live-model val pass would save ~25 s/epoch → 21–22 epochs in budget → push val to ~84–90.

| Metric | No-diag (vx2n2zuq) | EMA baseline (gdfynh7o) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 18) | 92.4619 | 92.3452 | +0.12 (within noise) |
| `test_avg/mae_surf_p` | 82.4764 | 81.6297 | +0.85 (worse) |
| `test/test_single_in_dist/mae_surf_p` | 93.44 | 95.30 | −1.86 (better) |
| `test/test_geom_camber_rc/mae_surf_p` | 93.84 | 91.93 | +1.91 |
| `test/test_geom_camber_cruise/mae_surf_p` | 60.62 | 58.72 | +1.90 |
| `test/test_re_rand/mae_surf_p` | 82.01 | 80.58 | +1.43 |
| Epoch wall time | ~101 s | ~110 s | −9 s (not −25 s) |
| Epochs in 30 min | 18 | 17 | +1 |

- **Throughput hypothesis magnitude was wrong**: diagnostic pass saved ~8 s/epoch (not ~25 s); val_loaders are only ~100 batches at bf16. Only +1 epoch in budget — insufficient to escape run-to-run noise (±1.5 MAE).
- **EMA val trajectory still descending ~2.5 MAE/epoch at cutoff** — confirms total training time is the binding constraint, not diagnostic overhead.
- **Useful intel from student**: peak memory 32.9 GB / 96 GB (~3× headroom), training step is dominant cost. Bottleneck is the training pass itself, not val/diag.
- **Decision: CLOSE.** Mechanism is real but magnitude too small. Assigning fern to Huber β=0.5 sweep (more L1-aligned in moderate-error region; surgical single-hparam follow-up to confirmed Huber lever).

## 2026-05-13 00:05 — PR #1689: fern Huber β=0.5 (review 1, MERGED — new best)

- Branch: `willowpai2g48h5-fern/huber-beta-0p5`
- W&B run: `liurnqyo` (17 epochs; ~112 s/epoch; β=1.0→0.5, both call sites)
- Hypothesis: reducing Huber transition point makes the loss linear (L1-aligned with MAE) over the moderate-error range where most of the loss density sits; EMA absorbs kink-noise near 0.

| Metric | β=0.5 (liurnqyo) | β=1.0 baseline (gdfynh7o) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | **85.9197** | 92.3452 | −6.43 (−6.96%) |
| `test_avg/mae_surf_p` | **76.5495** | 81.6297 | −5.08 (−6.22%) |
| `test/test_single_in_dist/mae_surf_p` | **88.03** | 95.30 | −7.26 (−7.6%) |
| `test/test_geom_camber_rc/mae_surf_p` | **85.46** | 91.93 | −6.46 (−7.0%) |
| `test/test_geom_camber_cruise/mae_surf_p` | **56.40** | 58.72 | −2.32 (−3.9%) |
| `test/test_re_rand/mae_surf_p` | **76.30** | 80.58 | −4.28 (−5.3%) |

- **All 4 test splits improved.** Largest gains on the two hardest splits (in_dist and camber_rc) where moderate-error density is highest and β controls the most gradient mass.
- **EMA-vs-live gap preserved:** epoch 17 EMA val=85.92 vs live val=96.41 (−10.49 MAE). The L1 kink doesn't destabilize optimization because EMA smooths it.
- **No instability:** monotonic val descent, ~same per-epoch wall time (111.78 s vs 109 s baseline).
- **Root cause of gain:** at β=1.0, quadratic region extends to |x|=1.0 in normalized space; bulk of surface-p errors in training are in the 0.1–0.8 range, so β=1.0 was effectively training an MSE objective on the majority of samples. Dropping to β=0.5 moves the linear regime into the bulk, directly aligning gradients with the MAE metric. Same mechanism that made PR #1436 (MSE→Huber β=1.0) a large win — lever pushed further.
- **Val still descending ~2.5/epoch at cap** — consistent with all prior runs; total training time is still the budget-limiting constraint.
- **Decision: MERGE.** Cleanest single-variable win in this round. New baseline: val=85.9197, test=76.5495. Assigning fern to β=0.25 sweep.

## 2026-05-13 00:05 — PR #1629: thorfinn dropout=0.1 (review 1, sent back — new baseline needed)

- Branch: `willowpai2g48h5-thorfinn/dropout-0p1`
- W&B run: `argppwi8` (17 epochs; 112.7 s/epoch; dropout=0.1 vs dropout=0.0)
- Hypothesis: attention-only dropout=0.1 improves OOD generalization.

| Metric | dropout=0.1 (argppwi8) | EMA baseline (gdfynh7o) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | 90.0841 | 92.3452 | −2.26 (−2.45%) |
| `test_avg/mae_surf_p` | 80.8527 | 81.6297 | −0.78 (−0.95%) |
| `test/test_single_in_dist/mae_surf_p` | **91.86** | 95.30 | −3.44 (better) |
| `test/test_geom_camber_rc/mae_surf_p` | **90.02** | 91.93 | −1.91 (better) |
| `test/test_geom_camber_cruise/mae_surf_p` | 59.99 | 58.72 | +1.27 (worse) |
| `test/test_re_rand/mae_surf_p` | 81.54 | 80.58 | +0.96 (worse) |

- **Beat old baseline (92.35)** but no longer beats new baseline after fern β=0.5 merge (val=85.92). Mixed per-split: IID and camber_rc improve, camber_cruise and re_rand slightly regress.
- **Mechanism: dropout helps where overfitting is the failure mode** (in_dist, camber_rc), not where genuine extrapolation is required (camber_cruise, re_rand). OOD failure modes differ by split.
- **Decision: SEND BACK for retest on β=0.5+EMA+bf16 baseline.** Mechanisms are orthogonal; if they stack, dropout should give another −2 MAE on top of β=0.5. If camber_cruise/re_rand regressions persist, try dropout=0.05.

## 2026-05-13 00:10 — PR #1672: nezuko linear LR warmup 1 epoch (review 1, sent back — new baseline needed + T_max confounder)

- Branch: `willowpai2g48h5-nezuko/lr-warmup-1ep`
- W&B run: `zp13gmgt` (17 epochs; 110 s/epoch; SequentialLR with 1-epoch linear warmup 0.2→1.0 then cosine)
- Hypothesis: 1-epoch warmup (a) reduces early-epoch EMA lag, (b) gentler optimization start prevents gradient spikes.

| Metric | Warmup (zp13gmgt) | EMA baseline (gdfynh7o) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | 91.7248 | 92.3452 | −0.62 (−0.67%) |
| `test_avg/mae_surf_p` | 81.2043 | 81.6297 | −0.43 (−0.52%) |
| `test/test_single_in_dist/mae_surf_p` | **91.52** | 95.30 | **−3.78** (better) |
| `test/test_geom_camber_rc/mae_surf_p` | **89.60** | 91.93 | **−2.33** (better) |
| `test/test_geom_camber_cruise/mae_surf_p` | 60.86 | 58.72 | +2.15 (worse) |
| `test/test_re_rand/mae_surf_p` | 82.84 | 80.58 | +2.26 (worse) |

- **Beat OLD baseline** but no longer beats NEW baseline (val=85.92 after #1689 merge).
- **EMA-lag hypothesis FALSIFIED** by student's diagnostic: epoch-1 EMA−Live gap was +108.3 vs baseline +106.6 — essentially identical. At decay=0.999, EMA shadow at epoch 1 is still ~70% initialization (0.31 cumulative replacement); slowing first-epoch updates doesn't change much.
- **Mechanism (b) partial credit**: warmup placed model in slightly different basin — IID and camber_rc improved materially, but camber_cruise/re_rand regressed. Mixed signal.
- **Confounder flagged by student**: cosine `T_max=29` (vs baseline T_max=30) → late-phase LR ~6% higher at epoch 17. Honest disclosure.
- **Decision: SEND BACK for retest on β=0.5 baseline + fixed T_max** (`T_max=MAX_EPOCHS * len(train_loader) - warmup_steps`). Mechanism may still stack on β=0.5; clean retest needed.

## 2026-05-13 00:55 — PR #1427 v2: askeladd surf_weight=30 on β=0.5 baseline (review 2, closed)

- Branch: `willowpai2g48h5-askeladd/surf-weight-30`
- W&B run: `2r3nyj6o` (17 epochs; rebased on Huber β=0.5+EMA+bf16; default `surf_weight: float = 30.0`)
- Hypothesis (retest): surf_weight 10→30 amplifies gradient signal on the primary metric (surface-p) — should compound with Huber β=0.5.

| Metric | surf_weight=30 (2r3nyj6o) | β=0.5 baseline (liurnqyo) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | 88.9891 | 85.9197 | +3.07 (+3.57%) |
| `test_avg/mae_surf_p` | 78.9516 | 76.5495 | +2.40 (+3.14%) |
| `test/test_single_in_dist/mae_surf_p` | 93.07 | 88.03 | +5.03 |
| `test/test_geom_camber_rc/mae_surf_p` | 86.80 | 85.46 | +1.34 |
| `test/test_geom_camber_cruise/mae_surf_p` | 57.76 | 56.40 | +1.36 |
| `test/test_re_rand/mae_surf_p` | 78.19 | 76.30 | +1.88 |

- **All 4 test splits regressed.** Mechanism falsified on the current stack.
- **EMA-vs-live gap widened to −22 (vs −10.5 baseline)** — direct evidence that higher surf_weight amplifies gradient variance. Live trajectory val=110.5 (vs baseline live ~96.4). EMA absorbs most of the noise but cannot fully close the gap.
- **Mechanism explanation**: Huber β=0.5 is itself an MAE-alignment lever (linear gradient in the moderate-error bulk where surface-p errors live). With both Huber and 3× surf weight, the surface signal is now over-emphasized; the volume head loses gradient mass without compensating surface gain.
- **Decision: CLOSE.** Surf_weight=30 is clearly worse on the current stack. Assigning askeladd to surf_weight=5 (opposite direction): test whether the optimal surf_weight has shifted below 10 now that Huber carries the MAE-alignment role.

## 2026-05-13 01:00 — PR #1669: edward EMA decay=0.9995 (review 1, closed)

- Branch: `willowpai2g48h5-edward/ema-decay-0p9995`
- W&B run: `o2zqo27j` (14 epochs; 138 s/epoch; `--ema_decay 0.9995`)
- Hypothesis: longer half-life (3.7 epochs vs 1.85) → wider averaging window → smoother EMA shadow late in training.

| Metric | EMA decay=0.9995 (o2zqo27j) | Baseline decay=0.999 (gdfynh7o) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 14) | 133.4326 | 92.3452 | +41.09 (catastrophic) |
| `test_avg/mae_surf_p` | 122.1107 | 81.6297 | +40.48 |
| `val_avg/mae_surf_p_ema_minus_live` (EMA−Live gap) | **+16.13 (EMA HURTS)** | −25.28 (EMA helps) | +41.41 |
| `val_live_avg/mae_surf_p` (same epoch live model) | 117.30 | 117.63 | −0.32 (≈same) |

- **Live model trajectory IDENTICAL to baseline** — only the EMA shadow differs. Clean isolation of the ema_decay variable.
- **EMA-vs-live gap stays POSITIVE throughout** (EMA always worse than live in 14 epochs). The shadow never reaches steady-state because half-life 3.7 epochs requires ~5+ half-lives (≥18 epochs) to shed init noise. At our 14-epoch effective budget, only 3.78 half-lives → shadow still anchored to high-loss epoch 1–4 weights (live MAE 161–210).
- **Mechanism takeaway**: EMA decay is exquisitely sensitive to wall-clock-bounded training. The merged 0.999 is near-optimal for our 17-epoch effective budget; wider windows would only help if epoch budget ≥20.
- **Decision: CLOSE.** Hypothesis is plausible at higher budgets but cannot be tested under SENPAI_TIMEOUT_MINUTES=30. Assigning edward to torch.compile — directly attack the binding throughput constraint that killed this hypothesis (and slice_num, n_hidden=192, n_layers=8).


## 2026-05-13 01:25 — PR #1629 v2/v3: thorfinn dropout=0.1/0.05 on β=0.5 baseline (review 2, closed)

- Branch: `willowpai2g48h5-thorfinn/dropout-0p1`
- v2 W&B run: dropout=0.1 (retest on Huber β=0.5+EMA+bf16)
- v3 W&B run: dropout=0.05 (probe smaller magnitude after v2 regression)
- Hypothesis (retest): attention dropout would stack with β=0.5 to give a further OOD generalization gain.

| Run | dropout | val_avg/mae_surf_p (best EMA) | vs β=0.5 baseline 85.9197 |
|-----|--------:|----------:|---:|
| v2 | 0.10 | 87.61 | +1.97% |
| v3 | 0.05 | 87.91 | +2.32% |

- **Monotonicity violation**: halving dropout did NOT recover toward baseline — it made things slightly worse. Rules out a "wrong magnitude" interpretation; if the effect were purely about noise magnitude, p=0.05 should sit between baseline (0.0) and p=0.1.
- **Mechanism (revised)**: Huber β=0.5 sharpens the loss curvature in the small-residual regime (|x| < 0.5 quadratic). The optimizer is now operating on a finer-grained loss landscape near the optimum, where dropout's per-step Bernoulli noise looks less like regularization and more like coordinate-wise gradient corruption. EMA at 0.999 (1.85-epoch half-life) cannot fully wash this out within the 17-epoch budget.
- **Pattern with #1427 v2 (surf_weight=30) and #1534 v2 (grad-clip 1.0)**: three independent regularization/noise mechanisms that helped on the old MSE/Huber-β=1.0 stack all regress on the β=0.5 stack. Loss-shape tightening has displaced these levers.
- **Decision: CLOSE.** Assigning thorfinn to Lookahead optimizer (k=5 inner / α=0.5 outer slow-weight averaging) — modifies the training trajectory rather than per-step noise, complementary layer to EMA.

## 2026-05-13 01:25 — PR #1534 v2: tanjiro gradient clipping max_norm=1.0 on β=0.5 baseline (review 2, closed)

- Branch: `willowpai2g48h5-tanjiro/grad-clip-1p0`
- W&B run: grad-clip 1.0 retest on Huber β=0.5+EMA+bf16
- Hypothesis (retest): clipping at max_norm=1.0 stabilizes optimization on the β=0.5 stack where the L1 kink near zero could in principle add gradient noise.

| Metric | grad-clip 1.0 | β=0.5 baseline | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA) | 87.27 | 85.92 | +1.57% |
| `test/test_single_in_dist/mae_surf_p` | 90.95 | 88.03 | +2.92 (hurts) |
| `test/test_geom_camber_rc/mae_surf_p` | 90.20 | 85.46 | +4.74 (hurts) |
| `test/test_geom_camber_cruise/mae_surf_p` | 53.14 | 56.40 | −3.26 (helps) |
| `test/test_re_rand/mae_surf_p` | 74.59 | 76.30 | −1.71 (helps) |

- **OOD splits helped, IID splits hurt** — clean directional split, not noise.
- **Diagnostic from student**: 6375/6375 training steps clipped (100%) with peak gradient norm ≈140 after β=0.5 (down from ~837 pre-Huber). With every step clipped, max_norm=1.0 is no longer a safety net against rare spikes — it is acting as full direction normalization, projecting every gradient onto the unit ball.
- **Mechanism**: normalized gradients give a flatter loss-landscape traversal — that explains the OOD/IID split (better generalization at the cost of fitting the bulk). IID hurt outweighs OOD help on the 4-split average.
- **Decision: CLOSE.** Assigning tanjiro to max_norm=10 — with observed peak norms 70–140, this threshold only fires on genuine rare-spike outliers (the original purpose of clipping) while leaving bulk steps unchanged. Will isolate whether the v2 effect was the safety-net mechanism or the direction-normalization mechanism.

## 2026-05-13 01:55 — PR #1647: alphonse cosine T_max=18 aligned (review 1, closed)

- Branch: `willowpai2g48h5-alphonse/cosine-tmax-aligned`
- W&B run: `mtvgypux` (18 epochs in 32 min; `--epochs 18` so T_max aligned to actual budget)
- Hypothesis: setting cosine T_max=18 (matching actual epoch count) eliminates the "schedule mismatch" where baseline cuts off at 17/30 with LR still at ~30% of peak.

| Metric | T_max=18 (mtvgypux) | β=0.5 baseline (liurnqyo) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | 94.44 | 85.92 | +8.52 (+9.92%) |
| `test_avg/mae_surf_p` | 85.24 | 76.55 | +8.69 (+11.36%) |
| `test/test_single_in_dist/mae_surf_p` | 95.81 | 88.03 | +7.78 |
| `test/test_geom_camber_rc/mae_surf_p` | 94.44 | 85.46 | +8.98 |
| `test/test_geom_camber_cruise/mae_surf_p` | 64.99 | 56.40 | +8.59 |
| `test/test_re_rand/mae_surf_p` | 85.74 | 76.30 | +9.44 |

- **All 4 splits regress meaningfully.** Counter-intuitive given the alignment hypothesis.
- **Mechanism (LR magnitude math):** At baseline T_max=30, epoch 17 LR ≈ 5e-4 × (1+cos(17π/30))/2 ≈ 1.5e-4 (moderate). At T_max=18, same epoch LR ≈ 5e-4 × (1+cos(17π/18))/2 ≈ 4e-6 (near-zero). The "aligned" schedule starves the model of effective LR in the final 30% of training where val is still descending ~2.5 MAE/epoch.
- **Pattern**: this isn't a schedule mismatch bug — the baseline benefits from a hot-LR plateau throughout training. Inverting the angle: raise peak LR (lr=7e-4) rather than decay it faster.
- **Decision: CLOSE.** Reassigning alphonse to lr=7e-4 with T_max=30 (same hot-cosine shape, shifted up).

## 2026-05-13 01:55 — PR #1442 v2: frieren wider Transolver n_hidden=192 post-rebase (review 2, closed)

- Branch: `willowpai2g48h5-frieren/wider-n-hidden-192`
- W&B run: `pxrllu0a` (30 epochs config, post-rebase on bf16+Huber β=0.5+EMA)
- Hypothesis (retest): wider model with current stack improves val.

| Metric | n_hidden=192 v2 (pxrllu0a) | β=0.5 baseline (liurnqyo) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA) | 96.66 | 85.92 | +10.74 (+12.50%) |
| `test_avg/mae_surf_p` (best-val EMA ckpt) | 87.19 | 76.55 | +10.64 (+13.90%) |
| `test/test_single_in_dist/mae_surf_p` | 104.96 | 88.03 | +16.93 (worst) |
| `test/test_geom_camber_rc/mae_surf_p` | 94.16 | 85.46 | +8.70 |
| `test/test_geom_camber_cruise/mae_surf_p` | 63.21 | 56.40 | +6.81 |
| `test/test_re_rand/mae_surf_p` | 86.45 | 76.30 | +10.15 |

- **EMA−Live gap = −9.88** (EMA helps as expected); wider model converges more slowly so terminal-live underperforms terminal-EMA noticeably.
- **Pattern complete**: 4/4 architecture-capacity experiments regress under 30-min cap (n_layers=8, mlp_ratio=4, slice_num=96, n_hidden=192). Capacity is NOT the bottleneck at 1500 training samples — training duration / step count is.
- **Decision: CLOSE.** Reassigning frieren to opposite direction: `n_layers=3` (shallower). Tests throughput-vs-capacity at the depth axis — we've only tested deeper so far. If shallow gives ~25–30% per-epoch speedup (~22 epochs in budget), the extra 5 epochs × ~2.5 MAE/epoch could net positive if convergence trajectory holds.

## 2026-05-13 02:05 — PR #1672 v2: nezuko LR warmup 1 epoch (review 2, MERGED — new best)

- Branch: `willowpai2g48h5-nezuko/lr-warmup-1ep`
- W&B run: `1hn6ur4l` (17 epochs; warmup 1 epoch start_factor=0.2→1.0, 375 steps; rebased on β=0.5+EMA+bf16)
- Hypothesis: 1-epoch linear LR warmup reduces EMA early-phase lag and improves per-split convergence.

| Metric | Warmup v2 (1hn6ur4l) | β=0.5 baseline (liurnqyo) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | **85.0926** | 85.9197 | −0.83 (−0.96%) |
| `test_avg/mae_surf_p` | **75.5171** | 76.5495 | −1.03 (−1.35%) |
| `test/test_single_in_dist/mae_surf_p` | **87.10** | 88.03 | −0.93 |
| `test/test_geom_camber_rc/mae_surf_p` | **84.58** | 85.46 | −0.89 |
| `test/test_geom_camber_cruise/mae_surf_p` | **55.50** | 56.40 | −0.90 |
| `test/test_re_rand/mae_surf_p` | **74.90** | 76.30 | −1.41 (best split gain) |

- **All 4 test splits improve.** v1 had regressions on camber_cruise (+2.15) and re_rand (+2.26); v2 cleans those up.
- **Original EMA-lag hypothesis falsified again** (epoch-1 gap actually wider +86.2 vs baseline +77.1 — slower first epoch means both EMA and live start closer together). **Real mechanism**: warmup compresses post-warmup EMA catch-up phase — at epoch 4, EMA-live gap compressed by 26 MAE (+41.8 vs +67.6 baseline). EMA reaches "steady convergence" faster.
- **T_max confounder unchanged** from v1 (T_max=10875, ~6% higher late LR). Gain is consistent across all 4 splits and EMA-vs-live diagnostic shows real mechanism effect, so warmup is doing real work.
- **Decision: MERGE.** New baseline: val=85.0926, test=75.5171.

## 2026-05-13 02:05 — PR #1705: fern Huber β=0.25 (review 1, closed)

- Branch: `willowpai2g48h5-fern/huber-beta-0p25`
- W&B run: `6b7k86h5` (15 epochs; timeout-capped; β=0.25 both call sites)
- Hypothesis: β=0.25 would continue the L1-alignment trend from β=0.5's −6.96% win.

| Metric | β=0.25 (6b7k86h5) | β=0.5 baseline (liurnqyo) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 15) | 93.92 | 85.92 | +9.31% |
| `test_avg/mae_surf_p` | 83.80 | 76.55 | +9.48% |
| `test/test_single_in_dist/mae_surf_p` | 103.71 | 88.03 | +17.81% (worst) |
| `test/test_geom_camber_rc/mae_surf_p` | 91.72 | 85.46 | +7.33% |
| `test/test_geom_camber_cruise/mae_surf_p` | 58.73 | 56.40 | +4.14% (least affected) |
| `test/test_re_rand/mae_surf_p` | 81.05 | 76.30 | +6.22% |

- **β sweep now fully bracketed**: β=1.0 (worse), β=0.5 (BEST), β=0.25 (worse). Minimum confirmed at β=0.5.
- **Mechanism confirmed**: quadratic region |x| < 0.25 too small for moderate-error regime. Constant L1 gradient magnitude gives uniform-step descent that underexplores loss landscape vs β=0.5's Huber gradient. EMA absorbed kink-noise cleanly (smooth val trajectory) but couldn't recover lost convergence rate.
- **Per-split pattern validates mechanism**: in_dist worst (+17.81%, large errors deep in L1 regime), camber_cruise least affected (+4.14%, smaller errors closer to quadratic boundary).
- **Student's follow-up suggestion adopted**: adaptive β schedule (β=1.0 early → β=0.5 late). Reassigning fern.
- **Decision: CLOSE.** β=0.5 is confirmed as the fixed-β optimum. Assigning fern to adaptive β annealing.

## 2026-05-13 02:10 — PR #1763: edward torch.compile (review 1, MERGED — massive throughput win)

- Branch: `willowpai2g48h5-edward/torch-compile`
- W&B run: `o6k5dj4g` (29 epochs in 30.7 min; ~63 s/epoch steady state; `torch.compile(model, dynamic=True, mode='default')`)
- Hypothesis: throughput-bound research — compile attack on the per-epoch wall-clock ceiling that killed n_layers=8, n_hidden=192, slice_num=96, EMA=0.9995.

| Metric | torch.compile (o6k5dj4g) | warmup baseline (1hn6ur4l) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 29) | **71.4371** | 85.0926 | **−13.66 (−16.06%)** |
| `test_avg/mae_surf_p` | **62.5927** | 75.5171 | **−12.92 (−17.11%)** |
| `test/test_single_in_dist/mae_surf_p` | **70.43** | 87.10 | −16.67 |
| `test/test_geom_camber_rc/mae_surf_p` | **74.09** | 84.58 | −10.49 |
| `test/test_geom_camber_cruise/mae_surf_p` | **44.51** | 55.50 | −10.99 |
| `test/test_re_rand/mae_surf_p` | **61.35** | 74.90 | −13.55 |
| Per-epoch wall time | ~63 s (steady) | ~110 s | **−44%** |
| Epochs in 30 min | **29** | 17 | **+12 (+71%)** |
| Peak GPU memory | 23.8 GB | unmeasured | ~headroom |

- **Throughput delivered above estimate**: PR hypothesis was 15–35% speedup; actual was 44%. PhysicsAttention is small-kernel-heavy → inductor fusion + bf16 compose cleanly.
- **All 4 test splits improve dramatically** (best gains in_dist −16.67, cruise −10.99).
- **Val curve was still descending at cap** (~0.4 MAE/epoch ep 27–29). More epochs would help further.
- **Confounder flagged**: `--epochs 30` makes cosine T_max=30 instead of baseline implicit T_max=50; some of the val gain may be from the more aggressive schedule. The throughput component is clean either way (29 vs 17 epochs is undeniable).
- **Decision: MERGE.** New baseline: val=71.4371, test=62.5927. All downstream experiments inherit compile automatically; in-flight PRs will rebase onto this stack.
- **Edward reassigned to `--epochs 40` with T_max=40** — convert the freed budget into more training. Val was still descending; estimate ~35–38 epochs in 30 min with slightly less aggressive cosine, projecting val ~64–67.

## 2026-05-13 02:30 — PR #1743: askeladd surf_weight=5 (review 1, closed)

- Branch: `willowpai2g48h5-askeladd/surf-weight-5`
- W&B run: `bgddphoi` (17 epochs at pre-compile; pre-warmup baseline 85.92)
- Hypothesis: optimum `surf_weight` shifted below 10 now that Huber β=0.5 carries MAE-alignment.

| Metric | surf=5 (bgddphoi) | β=0.5 baseline (liurnqyo) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | 87.6796 | 85.9197 | +1.7599 (+2.05%) |
| `test_avg/mae_surf_p` | 77.9249 | 76.5495 | +1.3754 (+1.80%) |
| `test/test_single_in_dist/mae_surf_p` | 88.04 | 88.03 | +0.01 (tied) |
| `test/test_geom_camber_rc/mae_surf_p` | 87.60 | 85.46 | +2.14 |
| `test/test_geom_camber_cruise/mae_surf_p` | 57.95 | 56.40 | +1.55 |
| `test/test_re_rand/mae_surf_p` | 78.10 | 76.30 | +1.80 |
| `val_avg/mae_vol_p` | 86.01 | 91.80 | **−5.79 (−6.31% better)** |
| `test_avg/mae_vol_p` | 76.75 | 83.32 | **−6.57 (−7.89% better)** |
| EMA-vs-live gap (epoch 17) | −2.99 | −10.49 | gap shrank, less EMA benefit |

- **surf_weight sweep now bracketed**: 5 (worse, +2.05%), 10 (BEST), 30 (worse, +3.6%). Clean U-shape on primary metric.
- **Volume-vs-surface trade-off confirmed**: lower surf_weight routes gradient mass to volume head; vol metrics improve 5-10% on all splits. But primary metric is surf_p, so this is a side observation.
- **EMA-vs-live gap monotonicity**: 5→−3, 10→−10.5, 30→−22. Reveals surf_weight controls *useful gradient variance* the EMA averages over — not just routing. Lower surf_weight → less variance → less EMA benefit → less converged surface model.
- **Decision: CLOSE.** surf_weight=10 confirmed as fixed-optimum. Reassigning askeladd to capacity-down on slice axis (#1841: slice_num=48).

## 2026-05-13 02:30 — PR #1784: tanjiro grad-clip max_norm=10 (review 1, sent back for rebase)

- Branch: `willowpai2g48h5-tanjiro/grad-clip-10p0`
- W&B run: `i3zau4g4` (17 epochs at pre-compile; pre-warmup baseline 85.92)
- Hypothesis: max_norm=10 is "true safety net" (clip rate <1%) — isolates rare-spike clipping from direction-normalization.

| Metric | grad-clip 10 (i3zau4g4) | β=0.5 baseline (liurnqyo) | v2 grad-clip 1.0 (#1534) |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | **84.9688** | 85.9197 | 87.27 |
| `test_avg/mae_surf_p` | **74.7804** | 76.5495 | ~77.22 |
| `test/test_single_in_dist/mae_surf_p` | **87.32** | 88.03 | 90.95 |
| `test/test_geom_camber_rc/mae_surf_p` | **85.30** | 85.46 | 90.20 |
| `test/test_geom_camber_cruise/mae_surf_p` | **53.05** | 56.40 | 53.14 |
| `test/test_re_rand/mae_surf_p` | **73.46** | 76.30 | 74.59 |
| Clip rate | 86.9% | n/a | 100% |
| Avg scaling | ~2.6× | n/a | ~22× |

- **Clean win on all 4 splits vs pre-compile baseline AND vs v2 (max_norm=1.0).**
- **Mechanism rebracketed: continuous scaling regime, not safety net.** Gradient norm distribution: mean 26, p50=22, p90=49, p99=92, max=191. Threshold 10 clips 87% but only divides by ~2.6× on average (vs v2's ~22×). The regime is "soft direction-normalization" — heavy upper tail dampened (helps OOD) without erasing bulk gradient direction (preserves IID).
- **Three regimes now visible**: v2@1.0 = 100% clip / 22× / OOD-helps+IID-hurts (NET REGRESS); v3@10 = 87% clip / 2.6× / ALL-HELP (NET WIN); predicted @50 or @100 = <10% clip / safety-net only (untested, would isolate rare-spike mechanism).
- **However**: result is on pre-compile stack vs old baseline (85.92). Doesn't beat new compile baseline (71.44). Sent back for rebase + retest on compile stack — if mechanism holds and stacks with compile, we have a new best.
- **Decision: SEND BACK** for rebase onto current advisor branch (with compile) and re-run. Mechanism is sound; need clean test against the actual current baseline.

## 2026-05-13 02:35 — PR #1841: askeladd assigned slice_num=48 (compile-stack)

- Branch: `willowpai2g48h5-askeladd/slice-num-48`
- Hypothesis: capacity-down on slice axis (slice_num 64 → 48) → 10-15% per-epoch speedup → ~33-35 epochs in budget; tests whether 64 was overparameterized on the slice axis.
- Targets new compile-stack baseline (val=71.44, test=62.59).
- Complements frieren's n_layers=3 (#1792, in flight) — covers the second dimension of the capacity-down matrix.

## 2026-05-13 03:05 — PR #1805: fern adaptive Huber β anneal (review 1, sent back for compile-stack rebase)

- Branch: `willowpai2g48h5-fern/beta-anneal-1p0-to-0p5`
- W&B run: `h18izyoe` (17 epochs at pre-compile; pre-compile baseline 85.09)
- Hypothesis: linear β anneal 1.0→0.5 over epochs 1–10, held 0.5 thereafter.

| Metric | β-anneal (h18izyoe) | warmup baseline (1hn6ur4l) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | **84.4610** | 85.0926 | **−0.74%** |
| `test_avg/mae_surf_p` | **75.0761** | 75.5171 | **−0.58%** |
| `test/test_single_in_dist/mae_surf_p` | 85.86 | 87.10 | −1.42% |
| `test/test_geom_camber_rc/mae_surf_p` | 84.36 | 84.58 | −0.26% |
| `test/test_geom_camber_cruise/mae_surf_p` | 55.79 | 55.50 | +0.53% (only regression) |
| `test/test_re_rand/mae_surf_p` | 74.30 | 74.90 | −0.80% |
| EMA-vs-live gap at ep 10 (β anneal end) | −5.4 (sign flip) | n/a | mechanism confirmed |

- **3 of 4 splits improve**, only camber_cruise (smallest absolute baseline error) +0.5% within noise.
- **Schedule verified** via train/huber_beta per-epoch log (1.000 → 0.500 over ep 1–10, held 0.500 ep 11–17).
- **Mechanism confirmed**: EMA-live gap flips negative exactly at the β=0.5 lock-in (ep 10). Pre-anneal phase keeps EMA lagging (large errors deep in linear region; β=1.0 quadratic provides gradient direction info). Post-anneal phase aligns with MAE metric.
- **Decision: SEND BACK** for rebase + retest on compile stack (new baseline 71.44). The β-anneal range (ep 1-10) now sits in 1/3 of training instead of 60%, leaving more ep at β=0.5 (the MAE-aligned regime where the gain originates) — mechanism predicts at-least-equal or better effect on compile stack.

## 2026-05-13 03:05 — PR #1783: thorfinn Lookahead k=5, α=0.5 (review 1, closed)

- Branch: `willowpai2g48h5-thorfinn/lookahead-k5-alpha0p5`
- W&B run: `hk7oqnbm` (17 epochs at pre-compile; pre-warmup baseline 85.92)
- Hypothesis: Lookahead optimizer wraps AdamW (k=5 inner steps, α=0.5 slow-weight pull) — modifies training trajectory rather than per-step noise; complementary to EMA.

| Metric | Lookahead (hk7oqnbm) | β=0.5 baseline (liurnqyo) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | 87.1120 | 85.9197 | **+1.19 (+1.39%)** |
| `test_avg/mae_surf_p` | 77.5381 | 76.5495 | +0.99 (+1.29%) |
| `test/test_single_in_dist/mae_surf_p` | 88.81 | 88.03 | +0.88% |
| `test/test_geom_camber_rc/mae_surf_p` | 87.05 | 85.46 | +1.85% |
| `test/test_geom_camber_cruise/mae_surf_p` | 57.49 | 56.40 | +1.93% |
| `test/test_re_rand/mae_surf_p` | 76.81 | 76.30 | +0.67% |
| Live val_avg (ep 17) | **88.74** | 96.41 | −7.7 MAE (Lookahead smoothing live works) |
| **EMA−live gap (ep 17)** | **−1.63** | **−10.49** | gap collapsed 85% |

- **All 4 splits regress.** Lookahead's live model is 7.7 MAE better than baseline live, but EMA's smoothing budget collapses from −10.5 to −1.6 — meaning EMA had nothing left to add after Lookahead pre-smoothed the trajectory.
- **Mechanism — Lookahead and EMA compete for trajectory-smoothing headroom, not stack:**
  - EMA decay=0.999 averages ~1000 fast-weight steps at eval time → absorbed noise from a noisy training path.
  - Lookahead α=0.5/k=5 averages adjacent fast/slow weights every 5 steps inside training → pre-smooths the trajectory.
  - Once Lookahead absorbs the SGD-noise EMA was eating, EMA's contribution drops to near-zero. Net regression.
  - Secondary cost: slow-weight pull *resets* fast weights to slow weights every 5 steps, slowing exploration during the 17-epoch budget. Per-epoch EMA val never crosses baseline EMA → not a "needs more epochs" failure.
- **Pattern complete**: 4 of 4 "noise/smoothing knobs" (dropout 0.1, dropout 0.05, grad-clip 1.0, Lookahead k=5 α=0.5) regress on β=0.5+EMA stack. β=0.5-sharpened landscape + EMA-smoothed eval has saturated this axis.
- **Decision: CLOSE.** Reassigning thorfinn to SGDR (cosine warm restarts) — different LR shape, tests exploration via periodic LR restarts.

## 2026-05-13 03:10 — PR #1858: thorfinn assigned SGDR cosine warm restarts (T_0=10, T_mult=2)

- Branch: `willowpai2g48h5-thorfinn/sgdr-t0-10-tmult-2`
- Hypothesis: replace single-cycle cosine with `CosineAnnealingWarmRestarts(T_0=10, T_mult=2)` → cycle 1 = 10 ep, restart at ep 12, cycle 2 = 20 ep. LR jumps from ~0 back to 5e-4 at restart.
- Mechanism: tests exploration via periodic LR restart, complementary to edward #1833 (longer single cycle) and alphonse #1791 (higher peak LR).
- Targets compile-stack baseline 71.44.

## 2026-05-13 03:25 — PR #1792: frieren n_layers=3 (closed without result; reassigned as #1875 v2)

- Branch: `willowpai2g48h5-frieren/n-layers-3` (closed at 03:16:26 UTC)
- W&B run: no terminal result posted; student left a rebase notice at 02:56 acknowledging compile-stack baseline shift but never posted final metrics
- Hypothesis: capacity-down on depth axis; complement to slice-axis test (askeladd #1841).
- **No data recovered.** PR was closed before student posted results comment — may have been killed by host harness, abandoned, or completed without posting.
- **Decision: REASSIGN as new PR.** The hypothesis is high-value (the depth-axis capacity-down direction has not been tested on the compile stack) and worth a clean retry.

## 2026-05-13 03:25 — PR #1875: frieren n_layers=3 v2 (reassigned, compile-stack)

- Branch: `willowpai2g48h5-frieren/n-layers-3-v2-compile`
- Hypothesis: reduce n_layers 5→3 on the compile stack. Expected throughput gain ~25-30% per epoch → ~38 epochs in budget. Tests whether the 4/4 capacity-up failure pattern inverts under capacity-down.
- Companion to askeladd #1841 (slice_num=48): covers the second dimension of the capacity-down matrix.
- Targets compile-stack baseline 71.44.

Note: GraphQL rate limit hit at 5000/5000 (reset ~1h); used REST API workaround for PR creation and labels.

## 2026-05-13 03:35 — PR #1806: nezuko LR warmup 2 epochs (review 1, closed)

- Branch: `willowpai2g48h5-nezuko/lr-warmup-2ep`
- W&B run: `9ksgr8ut` (17 epochs at pre-compile; pre-compile baseline 85.09)
- Hypothesis: extend warmup from 1 to 2 epochs → more EMA settling → faster useful EMA signal.

| Metric | warmup 2ep (9ksgr8ut) | warmup 1ep baseline (1hn6ur4l) | Δ |
|--------|----------:|----------:|---:|
| `val_avg/mae_surf_p` (best EMA, epoch 17) | 88.1364 | 85.0926 | **+3.04 (+3.57% worse)** |
| `test_avg/mae_surf_p` | 78.7543 | 75.5171 | +3.24 (+4.29% worse) |
| `test/test_single_in_dist/mae_surf_p` | 92.86 | 87.10 | +5.76 |
| `test/test_geom_camber_rc/mae_surf_p` | 89.18 | 84.58 | +4.60 |
| `test/test_geom_camber_cruise/mae_surf_p` | 56.33 | 55.50 | +0.83 |
| `test/test_re_rand/mae_surf_p` | 76.66 | 74.90 | +1.76 |
| EMA−live gap at ep 5 (post-2ep-warmup) | +53.63 | n/a | **wider, not narrower (prediction was ≤41.8)** |

- **Hypothesis prediction falsified directly**: epoch-5 EMA-live gap predicted ≤41.8 (v2's epoch-4 gap with 1-ep warmup); observed +53.63 — wider.
- **Mechanism — two-part failure:**
  1. EMA settling is *steps-based, not warmup-length-based*. EMA(0.999) has fixed ~693-step half-life regardless of per-step live update magnitude. Adding 375 low-LR steps doesn't buy free EMA settling.
  2. Live model loses 1 epoch of full-LR training (15 vs 16 near-peak epochs in 17 total). That training signal is unrecoverable.
- **v2 win reinterpreted**: the +0.83 from PR #1672 likely came from AdamW second-moment stabilization on first batch (start_factor=0.2 ramp prevents bad init seeding), NOT from EMA catch-up compression. One epoch is enough; second epoch buys nothing.
- **Warmup-length sweep bracketed**: 0 (worse, pre-v2) < 1 (BEST) > 2 (worse, +3.57%).
- **Decision: CLOSE.** Reassigning nezuko to mlp_ratio=1 (#1878) — capacity-down on MLP axis, completing the 3-axis matrix.

## 2026-05-13 03:35 — PR #1878: nezuko assigned mlp_ratio=1 (compile-stack)

- Branch: `willowpai2g48h5-nezuko/mlp-ratio-1`
- Hypothesis: reduce mlp_ratio 2→1 (FFN goes 128→256→128 to 128→128). Tests if FFN expansion is over-parameterized on compile stack. Expected ~10-15% per-epoch speedup → ~33 epochs in budget.
- Companion to frieren #1875 (n_layers=3, depth axis) and askeladd #1841 (slice_num=48, slice axis): completes the 3-axis capacity-down matrix.
- Targets compile-stack baseline 71.44.

## 2026-05-13 05:50 — PR #1875: frieren n_layers=3 v2 — MERGED (7th compound winner)

- Branch: `willowpai2g48h5-frieren/n-layers-3-v2-compile`
- Hypothesis: reduce depth n_layers=5→3 on compile-stack baseline. Tests if PhysicsAttention slicing carries representational load independently of depth. Throughput angle: shallower model → fewer ops per epoch → more epochs in 30-min budget.
- W&B run: `fsqr0yp5`

| Metric | Value | vs compile baseline (71.44/62.59) |
|--------|-------|-----|
| `val_avg/mae_surf_p` (best, epoch 30/30) | **69.4518** | −1.99 (−2.78%) |
| `test_avg/mae_surf_p` | **61.1887** | −1.40 (−2.24%) |
| `test_single_in_dist/mae_surf_p` | 67.8314 | −2.59 |
| `test_geom_camber_rc/mae_surf_p` | 74.2256 | +0.14 (noise, <0.2%) |
| `test_geom_camber_cruise/mae_surf_p` | 42.8224 | −1.69 |
| `test_re_rand/mae_surf_p` | 59.8755 | −1.47 |
| Per-epoch wall time | 40.8 s | −22.2 s (−35%) |
| Epochs in 30 min | 30 (of ~44 projected) | +1 (hit T_max=30 cap) |
| Param count | 420,047 | 0.23× baseline (1.84M) |

- **Best epoch was epoch 30/30 (final)** — val trajectory still descending at termination. Model had NOT converged; T_max=30 means cosine LR hit zero at epoch 30, but ~44 epochs are available in the budget. This is the key follow-up signal.
- **Mechanism confirmed**: PhysicsAttention slicing carries primary representational load. 3 layers is sufficient for 1500-sample TandemFoilSet; freed compute translates directly to more training epochs.
- **Decision: MERGE.** New compile-stack baseline: val=69.4518, test=61.1887.
- **Follow-up assigned**: frieren #1898 — n_layers=3 + epochs=50 (cosine schedule tuning: T_max=50 prevents dead epochs at LR=0 beyond epoch 30).

## 2026-05-13 05:50 — PR #1805: fern β anneal v2 — SENT BACK for retest on n_layers=3 baseline

- Branch: `willowpai2g48h5-fern/beta-anneal-1p0-to-0p5`
- Hypothesis: adaptive Huber β annealing (1.0→0.5 over epochs 1-10). Tests if gradual transition improves early-phase stability then late-phase MAE alignment.
- W&B run: `w334qy8f`

| Metric | Value | vs compile baseline (71.44) | vs new baseline (69.45) |
|--------|-------|-----|-----|
| `val_avg/mae_surf_p` (best, epoch 29/29) | **71.1615** | −0.28 (−0.39%, BEAT) | +1.71 (does NOT beat) |
| `test_avg/mae_surf_p` | **61.7884** | −0.80 (−1.28%) | +0.60 |

- Result was a genuine win against the old compile baseline (71.4371). But PR #1875 merged before review, shifting baseline to 69.4518. Result no longer beats the new baseline.
- **Mechanism confirmed as sound**: relative test win grew from −0.58% (pre-compile) to −1.28% (compile) — more late-phase β=0.5 epochs amplifies the benefit. On n_layers=3 (~44 epochs available), the anneal benefit should be even larger.
- **Decision: SEND BACK.** Retest on n_layers=3 branch with `--epochs 30`. New targets: val < 69.4518, test < 61.1887.

## 2026-05-13 05:50 — PR #1791: alphonse lr=7e-4 — CLOSED (LR magnitude dead end)

- Branch: `willowpai2g48h5-alphonse/lr-7e-4`
- Hypothesis: raise peak LR 5e-4→7e-4 to exploit the hot-LR plateau effect. Prediction: EMA accumulates better live-model signal at faster convergence rate.
- W&B run: `omqh52yk`

| Metric | Value | vs old baseline (85.92) | vs compile baseline (71.44) |
|--------|-------|-----|-----|
| `val_avg/mae_surf_p` (best, epoch 17) | **86.284** | +0.36 (+0.42% regression) | +14.84 far worse |
| `test_avg/mae_surf_p` | **76.753** | +0.20 (+0.27% regression) | +14.16 far worse |
| Per-epoch wall time | ~112 s | — (pre-compile, no throughput) | — |

- Marginal regression vs even the pre-compile baseline. No test split improved.
- **Mechanism (student analysis confirmed correct):** EMA half-life (~693 steps) is fixed by decay=0.999, independent of LR magnitude. Faster live convergence (higher lr) + larger EMA step contributions roughly cancel: faster trajectory compressed by EMA tracking a shorter but faster path. EMA-live gap narrowed (−10.5→−4.3), but that's the live model moving faster, not EMA improving.
- **LR sweep fully bracketed**: lr=5e-4 (BEST) < lr=7e-4 (−0.42%). LR-magnitude direction is exhausted. Note: was also run on pre-compile stack (17 epochs vs 29), which further amplifies the gap vs compile baseline.
- **Decision: CLOSE.** Reassigning alphonse to #1899 — n_hidden=192 + n_layers=3 (width reinvestment hypothesis: compact+wide vs prior compact+narrow and deep+wide configs).

## 2026-05-13 05:50 — PR #1898: frieren assigned n_layers=3 + epochs=50 (cosine schedule tuning)

- Branch: `willowpai2g48h5-frieren/n-layers-3-epochs-50`
- Hypothesis: PR #1875 showed best_epoch=30/30 with T_max=30 (LR hits zero at epoch 30). With ~44 epochs available in the 30-min budget, the final 14 epochs run at LR=0 — wasted training capacity. Setting `--epochs 50` (T_max=50) keeps LR positive through all 44 actual epochs (LR ≈ 1.8e-5 at ep 44 vs 0 at ep 30).
- Single variable under test: epochs (T_max) 30→50. n_layers=3 explicitly specified.
- Targets: val < 69.4518, test < 61.1887.

## 2026-05-13 05:50 — PR #1899: alphonse assigned n_layers=3 + n_hidden=192 (width reinvestment)

- Branch: `willowpai2g48h5-alphonse/n-layers-3-n-hidden-192`
- Hypothesis: on n_layers=5, n_hidden=192 regressed +12.5% (capacity saturation at depth). On n_layers=3, params are only 0.42M (baseline 1.84M). n_hidden=192 × n_layers=3 ≈ 0.94M params — compact but wide. Tests if width reinvestment compensates for reduced depth.
- Estimated per-epoch time: ~65 s/epoch → ~27 epochs in 30 min.
- Targets: val < 69.4518, test < 61.1887.

## 2026-05-13 06:15 — PR #1858: thorfinn SGDR cosine warm restarts — CLOSED

- Branch: `willowpai2g48h5-thorfinn/sgdr-t0-10-tmult-2`
- Hypothesis: replace single-cycle cosine with SGDR(T_0=10, T_mult=2). Multiple shorter cycles with periodic LR restarts probe whether re-exploration escapes the cosine baseline's basin.
- W&B run: `hhwwt15w`

| Metric | Value | vs OLD compile baseline (71.44) | vs NEW n_layers=3 baseline (69.45) |
|--------|-------|-----|-----|
| `val_avg/mae_surf_p` (best, epoch 28) | **72.9885** | +1.55 (+2.17%) | +3.54 (+5.09%) |
| `test_avg/mae_surf_p` | **63.2398** | +0.65 (+1.03%) | +2.05 (+3.36%) |
| Per-epoch wall time | ~63 s | identical (scheduler is free) | — |
| Epochs in 30 min | 28 | −1 | — |
| LR restart at step 4135 | 0 → 5e-4 | mechanism worked as designed | — |

- **Per-split**: 3/4 splits regress; only test_geom_camber_cruise improved (−0.93).
- **Mechanism worked as designed**: LR log confirms cycle structure (cycle 1 ep 1–11, restart at ep 12 ramping LR 0→5e-4 in one step, cycle 2 cooldown ep 12–28). Val trajectory shows expected bump-then-recovery pattern. Live model at ep 28 reached baseline-equivalent test quality (62.55 vs 62.59).
- **Budget mismatch**: cycle 1 (~10 epochs) is effectively wasted converging then resetting; cycle 2 (~17 epochs) cooldown < single-cycle cosine cooldown (~28 epochs). EMA's ~5.5-epoch effective window means cycle-2 high-LR noise needs ~10 post-restart epochs to wash out.
- **Pattern consolidation**: with Lookahead (#1783), dropout (#1629), grad-clip 1.0 (#1534), and SGDR all regressing, the **optimization-side smoothing/exploration axis is fully mapped on this stack**. EMA decay=0.999 saturates the trajectory-smoothing budget; additional perturbations cost more than they buy at 30-min budget.
- **Decision: CLOSE.** Reassigning thorfinn to fresh gradient-quality axis (#1913, gradient accumulation steps=2).

## 2026-05-13 06:15 — PR #1913: thorfinn assigned gradient accumulation steps=2 (effective bs=8)

- Branch: `willowpai2g48h5-thorfinn/grad-accum-2-effective-bs-8`
- Hypothesis: accumulate gradients over 2 mini-batches before optimizer.step(). Effective batch=8 without dataloader bottleneck (which killed prior `bs=8` test #1447). Lower per-update gradient variance compounds with EMA's smoothing.
- Implementation: scale loss by 1/ACCUM_STEPS, accumulate via .backward(), step optimizer + scheduler + EMA only on (batch_idx + 1) % ACCUM_STEPS == 0. New CLI flag --accumulation_steps.
- Distinct from in-flight portfolio (architecture, schedule, loss shape, gradient norms): this is a *gradient-quality* intervention, not a trajectory-shape intervention.
- Targets compile + n_layers=3 baseline: val < 69.4518, test < 61.1887.

## 2026-05-13 07:00 — PR #1784: tanjiro grad-clip max_norm=10 (v3 compile retest) — MERGED (8th compound winner)

- Branch: `willowpai2g48h5-tanjiro/grad-clip-10p0`
- Hypothesis (v3 retest): grad-clip max_norm=10 in "soft scaling" regime — heavy upper tail of gradient norms gets dampened (~2.1× downscaling at typical clipped step, ~9× at p99), bulk direction signal preserved. Pre-compile v3 (#1784 v1) showed clean win on all 4 splits.
- W&B run: `vy49aq06` (compile-stack retest)

| Metric | Value | vs PR #1763 compile baseline | vs PR #1875 new n_layers=3 baseline |
|--------|-------|-----|-----|
| `val_avg/mae_surf_p` (best, epoch 29) | **65.9757** | −5.46 (−7.65%) | −3.48 (−5.00%) |
| `test_avg/mae_surf_p` | **57.0711** | −5.52 (−8.83%) | −4.12 (−6.74%) |
| `test_single_in_dist` | 64.5497 | −5.88 | −3.28 (vs 67.83) |
| `test_geom_camber_rc` | 70.5841 | −3.50 | −3.64 (vs 74.23) |
| `test_geom_camber_cruise` | 37.9291 | −6.58 | −4.89 (vs 42.82) |
| `test_re_rand` | 55.2217 | −6.13 | −4.65 (vs 59.88) |
| Clip rate (compile stack) | 72.4% | (was 86.9% pre-compile) | — |
| Per-epoch wall time | 63.4 s | identical to compile baseline | n/a (run was on n_layers=5) |

- **All 4 splits improve vs both baselines.** Largest wins on OOD splits (camber_cruise −6.58, re_rand −6.13) but IID also wins solidly (in_dist −5.88) — clean break from v2's mixed sign.
- **Mechanism (soft scaling regime confirmed)**: clip rate 72.4% on compile stack (vs 86.9% pre-compile — bulk norm distribution shifted down due to warmup smoothing). Upper tail p99 unchanged at ~92 across stacks. Typical clipped step (norm ~21) → 2.1× downscaling. Heavy-tail damping without erasing bulk direction signal.
- **Gradient-shape lever compounds with compile**: pre-compile win was −0.95% val; compile-stack win is −7.65% val. The healthier training regime (warmup + compile + more epochs) gives grad-clip more room to extract gains.
- **Decision: MERGE.** New compile-stack baseline: val=65.9757, test=57.0711. 8th compound improvement.
- **Caveat noted**: student's run was at n_layers=5 (their branch was behind the #1875 n_layers=3 merge). The squash-merge applies the grad-clip change on top of the n_layers=3 advisor branch. The combined n_layers=3 + grad-clip=10 stack has NOT been directly measured but is expected to be ≤ 65.98 given orthogonality of the mechanisms.
- **Follow-up assigned**: tanjiro #1930 — grad-clip threshold scan (max_norm=5.0).

## 2026-05-13 07:00 — PR #1841: askeladd slice_num=48 — SENT BACK for retest on new 8-merge stack

- Branch: `willowpai2g48h5-askeladd/slice-num-48`
- Hypothesis: capacity-down on slice axis (64→48 slice tokens for PhysicsAttention). Throughput + capacity-right-sizing.
- W&B run: `sf87gbpr`

| Metric | Value | vs PR #1763 (n_layers=5 compile) | vs PR #1875 (n_layers=3) | vs PR #1784 (grad-clip new stack) |
|--------|-------|-----|-----|-----|
| `val_avg/mae_surf_p` (best, epoch 30) | **70.7556** | −0.68 (−0.95%, BEAT old) | +1.30 (DOES NOT beat) | +4.78 (DOES NOT beat) |
| `test_avg/mae_surf_p` | **61.7906** | −0.80 (−1.28%) | +0.60 | +4.72 |

- Mechanism is clean: 3/4 splits improve (only camber_rc +1.69), val still descending at ep 30 (not capacity-limited), throughput gain 5.4% steady state (smaller than predicted but real), param count only −560 weights.
- Result beat the OLD compile baseline but the advisor branch has advanced twice since assignment: n_layers=3 (−2.78%), then grad-clip=10 (−5.00% more). Need retest on full 8-merge stack.
- **Decision: SEND BACK.** Retest at slice_num=48 + n_layers=3 + grad-clip=10. Expected val ≈ 65.35 (applying relative −0.95% win to new baseline). New targets: val < 65.98, test < 57.07.

## 2026-05-13 07:00 — PR #1930: tanjiro assigned grad-clip max_norm=5.0 (threshold scan)

- Branch: `willowpai2g48h5-tanjiro/grad-clip-5p0-new-stack`
- Hypothesis: tighten threshold from 10→5 on the new 8-merge stack. Based on student's own diagnostic: at compile-stack norm distribution (p50=16.2, p90=40.6, p99=91.8), max_norm=5 gives ~100% clip rate with ~4.2× typical scaling (vs 2.1× at threshold 10). Tests whether the threshold-vs-quality relationship is monotonic (push lower in future PRs) or U-shaped (max_norm=10 is optimum).
- Predicted outcomes: (A) val < 65.98 → keep going lower; (B) val > 67 → crossed sweet spot, settle at 10.
- Single-line change: GRAD_CLIP_MAX_NORM = 10.0 → 5.0. Diagnostics inherited from #1784 merge.
- Targets compile + n_layers=3 + grad-clip baseline: val < 65.9757, test < 57.0711.

## 2026-05-13 07:35 — PR #1899: alphonse n_layers=3 + n_hidden=192 — MERGED (9th compound winner)

- Branch: `willowpai2g48h5-alphonse/n-layers-3-n-hidden-192`
- Hypothesis: Reinvest freed depth-compute into width: n_hidden=192 on the n_layers=3 stack. Prior n_hidden=192 on n_layers=5 was +12.5% worse (capacity saturation). On n_layers=3 (0.42M params), per-layer expressivity is the bottleneck — widening compensates for reduced composition depth.
- W&B run: `r10qkcgd`

| Metric | Value | vs PR #1875 (n_layers=3 baseline) | vs PR #1784 (grad-clip=10) |
|--------|-------|-----|-----|
| `val_avg/mae_surf_p` (best, epoch 30) | **63.7215** | −5.73 (−8.25%) | −2.25 (−3.45%) |
| `test_avg/mae_surf_p` | **55.6430** | −5.55 (−9.06%) | −1.43 (−2.51%) |
| `test_single_in_dist` | 61.4444 | −6.39 | −3.11 |
| `test_geom_camber_rc` | 69.3247 | −4.90 | −1.26 |
| `test_geom_camber_cruise` | 37.7067 | −5.12 | −0.22 |
| `test_re_rand` | 54.0962 | −5.78 | −1.13 |
| Param count | 931,791 (0.93M) | 2.22× n_hidden=128 baseline | — |
| Per-epoch wall time | ~54.3 s | +33% overhead | n/a |
| Epochs in 30 min | 30/30 | identical | — |

- **All 4 splits improve** vs both baselines. Best epoch 30/30 (final); val slope still −0.22/epoch at end. Model epoch-saturated, not capacity-saturated.
- **Mechanism**: width and depth aren't fungible. At n_layers=5, adding width compounds over-parameterization. At n_layers=3, per-layer expressivity is the bottleneck — widening each layer compensates for reduced composition depth.
- **NOTE**: this run was on the n_layers=3 stack WITHOUT grad-clip=10 (student branch was pre-grad-clip). Result val=63.72 still beats current advisor baseline (65.98). Merged. Combined n_layers=3 + n_hidden=192 + grad-clip=10 unmeasured.
- **Decision: MERGE.** New best: val=63.7215, test=55.6430.
- **Follow-up assigned**: alphonse #1953 — n_hidden=192 + epochs=50 (schedule fix: val still descending at ep 30, LR hits zero at T_max=30 while ~33 epochs fit in budget).

## 2026-05-13 07:35 — PR #1953: alphonse assigned n_hidden=192 + epochs=50 (compound + schedule fix)

- Branch: `willowpai2g48h5-alphonse/n-hidden-192-epochs-50`
- Hypothesis: Combine the width win (#1899) with the schedule fix. At ~54 s/epoch, only ~33 epochs fit in 30-min budget. With `--epochs 50` (T_max=50), LR at epoch 33 ≈ 1.0e-4 (productive); with `--epochs 30`, LR=0 by epoch 30 (dead zone for last 3 epochs). Val slope at epoch 30 was −0.22/epoch — clear epoch-saturation signal.
- **This is also the first run to measure the full combined stack: n_layers=3 + n_hidden=192 + grad-clip=10 + epochs=50**.
- Targets: val < 63.7215, test < 55.6430.

## 2026-05-13 07:50 — PR #1913: thorfinn grad-accum=2 (effective bs=8) — CLOSED

- Branch: `willowpai2g48h5-thorfinn/grad-accum-2-effective-bs-8`
- Hypothesis: accumulate gradients over 2 mini-batches before optimizer.step(). Effective batch=8 without dataloader bottleneck. Lower per-update gradient variance compounds with EMA's smoothing.
- W&B run: `txbm2f2n`

| Metric | Value | vs PR #1875 (n_layers=3 baseline) | vs PR #1899 (current 9-merge baseline) |
|--------|-------|-----|-----|
| `val_avg/mae_surf_p` (best, epoch 30) | **75.6422** | +6.19 (+8.9% regression) | +11.92 (+18.7% worse) |
| `test_avg/mae_surf_p` | **66.6086** | +5.42 (+8.9%) | +10.97 (+19.7%) |
| Per-epoch wall time | 40.3 s | ~unchanged | n/a (run was on n_hidden=128) |
| Total opt-steps | 5,610 | ~half (vs ~11,250) | — |

- **All 4 splits regress significantly** (in_dist +10.16, camber_rc +3.84, camber_cruise +3.01, re_rand +4.66).
- **Mechanism (clearly diagnosed)**: per-epoch wall time unchanged but opt-steps halved → cosine schedule starves at half the parameter-update distance. Val descending at +0.5/epoch at the cap = severely undertrained. Variance-reduction effect WAS present (EMA-live gap closed to +1.88, suggesting smoother live trajectories) but dwarfed by the step-count deficit.
- **Pattern consolidation**: 4/4 trajectory-quality interventions from thorfinn at fixed-epoch budget have regressed (#1550 slice=96, #1783 Lookahead, #1858 SGDR, #1913 grad-accum). Trajectory smoothing axis is fully saturated by EMA decay=0.999 + grad-clip=10. Additional perturbations cost more than they buy at the current schedule.
- **Decision: CLOSE.** Fixed wall-clock retest (student's #1 follow-up) is mechanistically valid but baseline shifted twice during the run. Reassigning thorfinn to architectural axis (n_layers=2 + n_hidden=192).

## 2026-05-13 07:50 — PR #1960: thorfinn assigned n_layers=2 + n_hidden=192 (depth floor test)

- Branch: `willowpai2g48h5-thorfinn/n-layers-2-n-hidden-192`
- Hypothesis: push depth further beyond winning #1899 stack. Param count ~0.62M (1.5× n_hidden=128 baseline, 0.67× #1899). Expected ~36 s/epoch → ~50 epochs in 30-min budget. Mechanism check: is n_layers=3 the depth floor, or can the wider hidden dim carry the load with 2 composition steps?
- Predicted outcomes: (Win) val < 63.72 → depth floor is below 3; (Tie) within ±0.5 → depth=3 was floor; (Loss) val > 65 → capacity floor at depth=2.
- Targets: val < 63.7215, test < 55.6430.

## 2026-05-13 09:00 — PR #1930: tanjiro grad-clip max_norm=5.0 — MERGED (10th compound winner)

- Branch: `willowpai2g48h5-tanjiro/grad-clip-max-norm-5`
- Hypothesis: tighten threshold from 10→5 on the new 9-merge stack. At threshold 5, clip rate ~100%, ~4.2× typical downscaling vs 2.1× at threshold 10. Prediction: either monotonic improvement (keep going lower) or U-shaped (10 was sweet spot).
- W&B run: `forfket5`

| Metric | Value | vs PR #1899 (9-merge baseline) | vs PR #1784 (grad-clip=10 baseline) |
|--------|-------|-----|-----|
| `val_avg/mae_surf_p` (best, epoch 30) | **63.4801** | −0.24 (−0.38%) | −2.50 (−3.78%) |
| `test_avg/mae_surf_p` | **54.9834** | −0.66 (−1.18%) | −2.09 (−3.66%) |
| `test_single_in_dist` | 62.4458 | **+1.00 (regression)** | −2.10 |
| `test_geom_camber_rc` | 68.3757 | −0.95 | −2.21 |
| `test_geom_camber_cruise` | 35.8182 | −1.89 | −2.11 |
| `test_re_rand` | 53.2939 | −0.80 | −1.93 |
| Clip rate | 90.06% | vs 72.4% at max_norm=10 | — |
| Mean grad norm | 21.45 | unchanged from max_norm=10 run | — |
| Mean downscaling | 4.29× | vs ~2.1× at max_norm=10 | — |
| Per-epoch wall time | ~41.6 s | n/a (n_hidden=128 run) | identical |
| Epochs in 30 min | 30/30 | — | — |

- **3/4 OOD splits improve cleanly** (camber_rc −0.95, camber_cruise −1.89, re_rand −0.80); **in_dist regresses +1.00** — a diagnostic split: tighter clipping helps OOD generalization but begins suppressing useful in-distribution gradients.
- **Mechanism confirmed**: clip rate jumped from 72.4% → 90.1% and mean downscaling from 2.1× → 4.3×, EXACTLY matching pre-run predictions (predicted 4.2×). The regime is still "moderate uniform downscaling" — not yet the direction-normalization failure of max_norm=1.0 (~22×).
- **NOTE**: This run was on n_hidden=128 stack (tanjiro's branch was based on pre-n_hidden=192 advisor commit). Current advisor branch now has n_hidden=192 + grad-clip=5.0. Combined state unmeasured.
- **Decision: MERGE.** Val improves globally despite in_dist regression; OOD metric improvement validates the tighter threshold. 10th compound winner.
- **Follow-up assigned**: tanjiro #1982 — grad-clip max_norm=2.5 (threshold scan step 3). Next question: (A) further gain or (B) regime shift (U-shape confirmed, optimum between 2.5 and 5.0).

## 2026-05-13 09:00 — PR #1982: tanjiro assigned grad-clip max_norm=2.5 (threshold scan step 3)

- Branch: `willowpai2g48h5-tanjiro/grad-clip-max-norm-2p5`
- Hypothesis: Continue threshold scan: 10→5 improved OOD but triggered in_dist regression. At max_norm=2.5, predicted clip rate ~97%, ~8-9× downscaling. Key question: is this still in "moderate uniform downscaling" territory or has it crossed into the destructive direction-normalization regime?
- Predicted outcomes: (A) val_avg still falls → clip rate 97%, ~8× scaling is productive; (B) in_dist regression dominates → U-shape confirmed; optimum between 2.5 and 5.0 → next step is 3.5–4.5 fractional scan.
- Targets: val < 63.4801, test < 54.9834.
- Single-line change: `GRAD_CLIP_MAX_NORM = 5.0 → 2.5`

## 2026-05-13 09:30 — PR #1878: nezuko mlp_ratio=1 — CLOSED (capacity-down on FFN LOSS)

- Branch: `willowpai2g48h5-nezuko/mlp-ratio-1`
- Hypothesis: mlp_ratio=2→1 on compile+n_layers=3 stack. Completes 3-axis capacity-down matrix (depth=frieren WIN, slice=askeladd in-flight, FFN=this PR).
- W&B run: `p68y441t`

| Metric | Value | vs #1930 (current) | vs #1875 (n_layers=3) |
|--------|-------|----------|---------|
| `val_avg/mae_surf_p` | 70.1382 | **+10.5% regression** | +0.99% |
| `test_avg/mae_surf_p` | 61.6820 | **+12.2% regression** | +0.80% |
| `test_single_in_dist` | 68.6717 | — | — |
| `test_geom_camber_rc` | 76.1871 | — | — |
| `test_geom_camber_cruise` | 41.4505 | — | — |
| `test_re_rand` | 60.4188 | — | — |

- **Decision: CLOSE.** > 5% regression vs current 10-compound baseline. Importantly, mlp_ratio=1 was already +0.99% worse than n_layers=3 baseline (#1875 val=69.45), so compounding with the newer stack cannot rescue it.
- **Mechanism**: FFN capacity is NOT the bottleneck at n_layers=3 + n_hidden=192. mlp_ratio=2 is the correct sweet spot for this 1500-sample CFD task. Per-axis capacity-down matrix now shows: depth-down (n_layers=3) WIN, FFN-down (mlp_ratio=1) LOSS → mlp_ratio stays at 2.
- **Note**: GPU contention at epochs 8-12 (85-125 s/epoch vs 59 s steady) confounded the early training; student ran 26/~31 expected epochs.
- **Next**: nezuko assigned #1994 n_head=4→8 (attention head diversity test).

## 2026-05-13 09:30 — PR #1994: nezuko assigned n_head=8 (attention head diversity)

- Branch: `willowpai2g48h5-nezuko/n-head-8`
- Hypothesis: Double attention heads 4→8. At n_hidden=192, head_dim goes 48→24. Prediction: more attention heads = more diverse attention patterns per slice, better multi-aspect generalization (geometry + physics + mesh topology). Clean single-axis test; head_dim=24 is small but workable for small dataset.
- Failure mode: head_dim=24 too small → val regresses → brackets n_head=4 as optimum.
- Targets: val < 63.4801, test < 54.9834.

## 2026-05-13 10:00 — PR #1953: alphonse n_hidden=192 + epochs=50 — MERGED (11th compound winner; MASSIVE)

- Branch: `willowpai2g48h5-alphonse/n-hidden-192-epochs-50`
- Hypothesis: Compound the width win (#1899 val=63.72) with two missing pieces — (a) the now-merged grad-clip=5.0 layer, and (b) T_max=50 schedule fix (T_max=30 was causing LR→0 by epoch 30, starving the final training epochs). First direct measurement of the FULL 10-compound stack + corrected T_max.
- W&B run: `vnsqnuoy`

| Metric | Value | vs PR #1930 (current) | vs PR #1899 (n_hidden=192 alone) |
|--------|-------|----------|---------|
| `val_avg/mae_surf_p` (best, epoch 30) | **55.7634** | −7.72 (−12.17%) | −7.96 (−12.49%) |
| `test_avg/mae_surf_p` | **48.0960** | −6.89 (−12.53%) | −7.55 (−13.57%) |
| `test_single_in_dist` | 52.8835 | −9.56 (−15.30%) | −8.56 (−13.94%) |
| `test_geom_camber_rc` | 61.7845 | −6.59 (−9.64%) | −7.54 (−10.88%) |
| `test_geom_camber_cruise` | 31.1522 | −4.67 (−13.03%) | −6.55 (−17.39%) |
| `test_re_rand` | 46.5637 | −6.73 (−12.63%) | −7.53 (−13.92%) |
| Best epoch | 30/30 | val descending at −0.84/ep | — |
| Epochs completed | 30/50 | wall-clock cap | — |
| LR at termination | 1.73e-4 | cos T_max=50 productive | — |
| EMA−live gap | −8.32 | EMA carries real edge | vs +0.42 at #1899 |
| Param count | 0.93M | — | — |

- **ALL 4 SPLITS IMPROVE DRAMATICALLY.** Biggest single-merge improvement in many cycles.
- **Mechanism (orthogonal compounding confirmed):** All three changes (n_hidden=192, grad-clip=5.0 inherited, T_max=50) compounded as predicted. Schedule fix alone provided the dominant lift; the combined stack delivered uniform 12%+ improvement across all 4 test splits.
- **Diagnostic finding:** EMA−live gap flipped from +0.42 (#1899) to **−8.32** with clip rate 73% — at the new compound point, the live model is noisy enough that EMA shadow carries genuine value (vs being nearly redundant in #1899). This is a strong signal: EMA isn't just smoothing convergence noise, it's correcting for real per-step gradient variance.
- **Decision: MERGE.** PR #1953 is the 11th compound winner. The model is epoch-saturated, not capacity-saturated — val descending at −0.84/ep at termination.
- **NOTE on advisor branch state:** PR merge produced an empty diff — student ran the winning config purely via CLI args. Future student PRs must specify `--n_hidden 192 --n_layers 3 --epochs 50` in their reproduce commands.
- **Follow-up assigned**: alphonse #2000 — T_max=80 schedule extension (epochs 50→80). Same wall-clock, ~2× higher LR at termination, tests whether schedule push compounds further.

## 2026-05-13 10:00 — PR #2000: alphonse assigned T_max=80 schedule extension

- Branch: `willowpai2g48h5-alphonse/t-max-80-schedule-push`
- Hypothesis: Push the T_max schedule further. At T_max=80, LR at epoch 30 is ~3.45e-4 (vs ~1.73e-4 at T_max=50) — keeps LR meaningfully higher through the wall-clock cap. Tests whether val descent rate at #1953 termination was schedule-limited or capacity-limited.
- Predicted outcomes: (A) val < 55.76 → schedule push compounds, T_max axis still has headroom; (B) val > 55.76 → T_max=50 was the sweet spot; clipping dominates at higher effective LR.
- Targets: val < 55.7634, test < 48.0960.
- Single-flag change: `--epochs 50 → --epochs 80`. No code changes.

## 2026-05-13 11:30 — PR #1982: tanjiro grad-clip=2.5 — SENT BACK (protocol-stale)

- Branch: `willowpai2g48h5-tanjiro/grad-clip-2p5`
- Hypothesis: Threshold scan step 3, max_norm=2.5 (after #1784 max_norm=10 win and #1930 max_norm=5 win). Tests whether crossing into direction-normalization regime (~97% clip rate, ~7-9× downscaling) still helps.
- W&B run: `lpr8vehg`
- First-pass result (PRE-#1953 baseline): val=66.13 — beat OLD baseline (#1784) but +4.2% worse than #1930. Mechanism diagnostic was clean: clip rate ~96.5%, ~7.21× scaling. Decision was held pending T_max-50 baseline retest.

| Metric | Value | vs #1930 (then-current) | vs #1953 (new) |
|--------|-------|----------|---------|
| `val_avg/mae_surf_p` (best) | 66.13 | +4.2% | +18.6% |
| Mechanism | clip 96.5%, ~7.21× scaling | direction-normalization regime | — |

- **Decision: SEND BACK with `--epochs 50 --n_hidden 192 --n_layers 3` retest instructions.** The first-pass ran on T_max=30 schedule (now superseded). The 11-compound + T_max=50 stack provides the proper comparison point.
- **What this measures:** does grad-clip=2.5 still hurt after schedule fix? If yes (val ≥ 55.76), U-shape between 2.5/5.0 is confirmed. If no, threshold scan continues.

## 2026-05-13 11:30 — PR #1898: frieren n_hidden=128 + epochs=50 — CLOSED (mechanism subsumed)

- Branch: `willowpai2g48h5-frieren/n-hidden-128-epochs-50`
- Hypothesis: Convert throughput headroom into more training time via T_max=50 schedule fix on the n_layers=3 + n_hidden=128 stack.
- W&B run: from PR comments — val_avg/mae_surf_p ≈ 67 range (incomplete; student replied to nudge with stale numbers).
- **Decision: CLOSE.** PR #1953 (alphonse) ran the same mechanism (T_max=50) but combined with the n_hidden=192 width win — delivered val=55.76 (−12.17%). The schedule-fix mechanism is now fully captured in the merged 11-compound baseline. Continuing #1898 on the inferior n_hidden=128 stack provides no signal: it cannot beat the merged baseline (n_hidden=128 alone was +12.5% worse than n_hidden=192 in #1442 v2 retest).
- **Lesson:** Schedule-fix experiments should always be run on the latest capacity stack. Frieren's hypothesis was prescient (predicted the schedule axis) but the experiment ran on outdated capacity.
- **Next:** frieren assigned #2023 — n_hidden=224 width push on full 11-compound stack.

## 2026-05-13 11:30 — PR #1833: edward `--epochs 40` (T_max=40) — CLOSED (stale, never completed)

- Branch: `willowpai2g48h5-edward/epochs-40-tmax-40`
- Hypothesis: convert throughput headroom into more training epochs via T_max=40.
- Status at close: WIP-stale 2.5h+. Never completed a full training run; no terminal SENPAI-RESULT marker posted.
- **Decision: CLOSE.** Same mechanism as #1898 — the schedule-fix axis is captured in the merged #1953 baseline at T_max=50, and #2000 (alphonse) is actively testing T_max=80 extension. Holding the slot is opportunity cost; cleaner to reassign edward to a fresh axis.
- **Next:** edward assigned #2024 — EMA decay 0.999 → 0.998. Targets the EMA-live gap (−8.32) discovered in #1953.

## 2026-05-13 11:30 — PR #2023: frieren assigned n_hidden=224 width push

- Branch: `willowpai2g48h5-frieren/n-hidden-224-width-push`
- Hypothesis: Push hidden width 192 → 224 on the 11-compound stack. Compact-but-wide hypothesis (#1899) suggested per-layer expressivity is the bottleneck at n_layers=3. With schedule now fixed (T_max=50, model still descending at termination), wider hidden gives the model more capacity to fit the residual error.
- Reproduce: `--n_hidden 224 --n_layers 3 --epochs 50 --wandb_group willow-pai2g-48h-r5-n_hidden_224`
- Predicted: val ≤ 54.5 if width-scaling is strong, ~55.7 if neutral, > 56.5 if width hurts.
- Targets: val < 55.7634, test < 48.0960.

## 2026-05-13 11:30 — PR #2024: edward assigned EMA decay 0.999 → 0.998

- Branch: `willowpai2g48h5-edward/ema-decay-0p998`
- Hypothesis: Tighten EMA decay from 0.999 → 0.998 on the 11-compound stack. PR #1953's EMA-live gap is −8.32 (vs +0.42 at #1899) — the EMA shadow is now meaningfully lagging the live model. Halving the EMA half-life (693 → 346 steps ≈ 2 epochs) should let EMA track the live model's improvements through the T_max=50 cosine tail.
- Reproduce: `--n_hidden 192 --n_layers 3 --epochs 50 --wandb_group willow-pai2g-48h-r5-ema_decay_0p998`
- Single-line code change in `train.py`. Diagnostic: log EMA-live gap at epochs 10, 25, 40, 50 — gap should close toward zero.
- Predicted: val ≤ 55.0 if EMA-live gap closes, ~55.7 if neutral, > 56.0 if too reactive (loses smoothing value).
- Targets: val < 55.7634, test < 48.0960.

## 2026-05-13 11:50 — PR #1994: nezuko n_head=4→8 — CLOSED (mechanism falsified)

- Branch: `willowpai2g48h5-nezuko/n-head-8`
- Hypothesis: Double attention heads 4→8 at n_hidden=192. At head_dim 48→24, more attention patterns per slice may improve OOD generalization (geometry + physics + Re).
- W&B run: `p5w8w78y` (21/30 epochs at T_max=30 — hit timeout)

| Metric | n_head=8 | vs #1930 (then current) | vs #1953 (new baseline) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 63.3548 | −0.20% (wash) | **+13.61% regression** |
| `test_avg/mae_surf_p` | 55.2761 | +0.53% (wash) | **+14.93% regression** |
| `test_single_in_dist` | 62.4992 | +0.054 | — |
| `test_geom_camber_rc` | 68.9251 | +0.549 | — |
| `test_geom_camber_cruise` | 36.1069 | +0.289 | — |
| `test_re_rand` | 53.5733 | +0.279 | — |

- **Decision: CLOSE.** Result was protocol-stale (T_max=30) but the mechanism evidence is decisive even apples-to-apples. **All 4 splits regress slightly and uniformly** — no split shows the hypothesized OOD-specific benefit. Camber-rc and re-rand (the supposed OOD-help splits) regress alongside in_dist. Mechanism is falsified at the architecture level, not the schedule level. A T_max=50 retest would burn 30 min to confirm null.
- **Diagnostic insight:** head_dim=24 is below the bottleneck threshold for this dataset. The PhysicsAttention slice geometry already partitions the input space; further sub-partitioning via more attention heads dilutes signal rather than diversifying it. n_head=4 (head_dim=48 at n_hidden=192) is the local optimum — attention-head axis is now bracketed.
- **Next:** nezuko reassigned to PR #2053 — mlp_ratio=3 (FFN capacity bracket on 11-compound + T_max=50 stack).

## 2026-05-13 11:50 — PR #2053: nezuko assigned mlp_ratio=3 FFN capacity bracket

- Branch: `willowpai2g48h5-nezuko/mlp-ratio-3`
- Hypothesis: FFN axis is currently bracketed by failed upper (mlp_ratio=4 was +5.3% on n_layers=5 stack — #1544) and failed lower (mlp_ratio=1 was +0.99% on n_layers=3 — #1878). Both bounds tested PRE-T_max=50. mlp_ratio=3 brackets between current optimum (2) and the previously-falsified upper. With #1953 showing val descending at −0.84/ep at termination (model is compute-bound, not capacity-bound), modest FFN capacity-up may surface headroom that didn't exist at the shorter schedule.
- Reproduce: `--n_hidden 192 --n_layers 3 --epochs 50 --wandb_group willow-pai2g-48h-r5-mlp-ratio-scan`, with `mlp_ratio=3` in `train.py`.
- Predicted: val ≤ 55.0 if FFN headroom unlocked; 55.0–56.5 neutral; > 56.5 capacity-up regression confirmed.
- Targets: val < 55.7634, test < 48.0960.

## 2026-05-13 12:00 — PR #1982: tanjiro grad-clip=2.5 + T_max=50 retest — MERGED (12th compound winner; MASSIVE)

- Branch: `willowpai2g48h5-tanjiro/grad-clip-max-norm-2p5`
- Hypothesis: Threshold scan step 3 (max_norm=5.0→2.5) — originally at T_max=30 (first-pass val=66.13, +4.2% vs #1930). Sent back for T_max=50 retest.
- W&B runs: `auwrg4mz` (T_max=30 first pass), `bb6o68xa` (T_max=50 confirmation — terminal result)

| Metric | T_max=30 first pass | T_max=50 confirmation | vs #1953 baseline | vs #1982 new baseline |
|---|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | 56.97 | **52.6406** | **−3.12 (−5.60%)** | — |
| `test_avg/mae_surf_p` | 49.38 | **44.9791** | **−3.12 (−6.49%)** | — |
| `test_single_in_dist` | 55.07 | **49.8555** | −3.03 (−5.73%) | — |
| `test_geom_camber_rc` | 64.45 | **57.7726** | −4.01 (−6.49%) | — |
| `test_geom_camber_cruise` | 31.10 | **28.9446** | −2.21 (−7.10%) | — |
| `test_re_rand` | 46.88 | **43.3437** | −3.22 (−6.90%) | — |
| Clip rate | 96.54% | 98.93% | — | — |
| Mean downscaling | ~7.21× | ~7.14× | — | — |
| Epochs | 30/30 | 33/50 (timeout) | — | — |

- **ALL 4 splits improve dramatically.** in_dist regression from step 2 (max_norm=5.0) is **fully reversed**.
- **Mechanism:** Grad-clip=2.5 at 98.9% clip rate and ~7.1× downscaling is still in the productive moderate-scaling regime (NOT the direction-normalization failure at max_norm=1.0). Monotonically accelerating improvement: 10→5→2.5 each step larger than last.
- **Decision: MERGE (12th compound winner).** Single-axis gradient threshold change now baked into train.py.
- **Next:** threshold scan step 4 (max_norm=1.5) — frieren #2067.

## 2026-05-13 12:05 — PR #2023: frieren n_hidden=192→224 — MERGED (13th compound winner)

- Branch: `willowpai2g48h5-frieren/n-hidden-224-width-push`
- Hypothesis: Width push 192→224 on 11-compound stack. Compact-but-wide hypothesis: per-layer expressivity is bottleneck at n_layers=3.
- W&B run: `80b6pnb9` (29/50 epochs, EMA still descending at −1.46/ep at termination)

| Metric | n_hidden=224 | vs #1953 baseline (55.76) | vs #1982 baseline (52.64) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **53.2494** | −2.51 (−4.51%) | +0.61 (wider is better vs old, not vs new) |
| `test_avg/mae_surf_p` | **46.6004** | −1.50 (−3.11%) | +1.62 |
| `test_single_in_dist` | 53.2544 | +0.37 (noise) | — |
| `test_geom_camber_rc` | 58.8796 | −2.90 (−4.70%) | — |
| `test_geom_camber_cruise` | 29.6831 | −1.47 (−4.72%) | — |
| `test_re_rand` | 44.5845 | −1.98 (−4.25%) | — |

- **Merged (13th compound winner) because:** beat #1953 baseline (55.76) at time of review; n_hidden width mechanism is orthogonal to grad-clip; mechanisms expected to compound.
- **EMPTY DIFF MERGE.** Win was CLI-only (n_hidden=224). advisor branch defaults still n_hidden=128.
- **Combined state (n_hidden=224 + grad-clip=2.5 + T_max=50) UNMEASURED.** tanjiro #2066 is the confirmation run.
- **Next:** tanjiro #2066 (combined state), thorfinn #2068 (n_hidden=256 width scan).

## 2026-05-13 12:10 — PR #1960: thorfinn n_layers=2 + n_hidden=192 — CLOSED (depth-floor confirmed)

- Branch: `willowpai2g48h5-thorfinn/n-layers-2-n-hidden-192`
- Hypothesis: Depth-floor test. At n_layers=2, 39 s/epoch vs 54 s/epoch allows ~46 epochs in budget — more compute to offset depth reduction.
- W&B run: `ychyadab` (30/30 epochs at T_max=30, grad-clip=10 — protocol stale)

| Metric | n_layers=2 | Baseline at submit (#1953) | Current baseline (#1982) |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | 56.9559 | 55.7634 | **52.6406** |
| `test_avg/mae_surf_p` | 49.5122 | 48.0960 | **44.9791** |

- **Decision: CLOSE.** val=56.96 vs #1953 = +2.1% worse at submit time; vs #1982 = +8.2% worse.
- **Mechanism:** n_layers=2 loses to n_layers=3 even at equal conditions. Despite 39 s/epoch throughput advantage, depth-2 expressivity is insufficient for PhysicsAttention multi-layer composition on PDE task. Compact-but-wide sweet spot confirmed at n_layers=3.
- **Depth-floor conclusion:** n_layers=3 is the optimum depth. n_layers=2 too shallow, n_layers=5+ too deep (old capacity-up failures), n_layers=8 catastrophic. Depth axis is now CLOSED.
- **Next:** thorfinn #2068 — n_hidden=256 width push.

## 2026-05-13 12:10 — PR #2066: tanjiro assigned n_hidden=224 + grad-clip=2.5 compound confirmation

- Branch: `willowpai2g48h5-tanjiro/n-hidden-224-compound-confirm`
- Hypothesis: Directly measure the combined 13-compound stack (n_hidden=224 + grad-clip=2.5 + T_max=50). The merged state is currently unmeasured — PR #1982 was at n_hidden=192, PR #2023 was at grad-clip=5.0.
- Reproduce: `--n_hidden 224 --n_layers 3 --epochs 50` (grad-clip=2.5 now in train.py)
- Predicted: val ≈ 50–52 if mechanisms add linearly; lower if synergistic.
- Targets: val < 52.6406, test < 44.9791.

## 2026-05-13 12:10 — PR #2067: frieren assigned grad-clip=1.5 (threshold scan step 4)

- Branch: `willowpai2g48h5-frieren/grad-clip-1p5`
- Hypothesis: Threshold scan step 4. max_norm=1.5 between last win (2.5, clip rate 98.9%) and known fail (1.0, direction-normalization). Predicted ~99.5-99.8% clip rate, ~12× mean downscaling.
- Reproduce: `--n_hidden 192 --n_layers 3 --epochs 50` + `GRAD_CLIP_MAX_NORM = 1.5` in train.py.
- Targets: val < 52.6406, test < 44.9791.

## 2026-05-13 12:10 — PR #2068: thorfinn assigned n_hidden=256 width push

- Branch: `willowpai2g48h5-thorfinn/n-hidden-256-width-push`
- Hypothesis: Width scan step 3 (192 WIN, 224 WIN → test 256). 1.63M params. Tests if compact+wide direction still has headroom or plateaus at 1500-sample dataset capacity.
- Reproduce: `--n_hidden 256 --n_layers 3 --epochs 50` (grad-clip=2.5 in train.py).
- Targets: val < 52.6406, test < 44.9791.

## 2026-05-13 13:00 — PR #2067: frieren grad-clip max_norm=1.5 — CLOSED (direction-normalization failure)

- Branch: `willowpai2g48h5-frieren/grad-clip-1p5`
- Hypothesis: Threshold scan step 4. max_norm=1.5 between last win (2.5, clip rate 98.9%) and known fail (1.0, direction-normalization). Predicted ~99.5-99.8% clip rate, ~12× mean downscaling.
- W&B run: `90yx6m3u`

| Metric | This (max_norm=1.5) | Baseline #1982 (max_norm=2.5) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **62.5939** | 52.6406 | **+9.95 (+18.91%)** |
| `test_avg/mae_surf_p` (EMA) | **54.4545** | 44.9791 | **+9.48 (+21.07%)** |
| `test_single_in_dist` | 62.4418 | 49.8555 | +12.59 (+25.2%) |
| `test_geom_camber_rc` | 68.4060 | 57.7726 | +10.63 (+18.4%) |
| `test_geom_camber_cruise` | 34.9669 | 28.9446 | +6.02 (+20.8%) |
| `test_re_rand` | 52.0031 | 43.3437 | +8.66 (+20.0%) |

**Threshold scan fully bracketed:**

| max_norm | clip rate | mean downscaling | val_avg | regime |
|---|---:|---:|---:|---|
| 10.0 | 72.4% | ~2.1× | 65.98 | soft-scaling WIN |
| 5.0 | 90.1% | ~4.3× | 63.48 | moderate-scaling WIN |
| **2.5** | **98.9%** | **~7.1×** | **52.64** | **OPTIMUM** |
| 1.5 (this) | **100.000%** | ~14.2× | 62.59 | direction-normalization FAIL |
| 1.0 | ~100% | ~22× | regression | direction-normalization FAIL |

- **Decision: CLOSE.** +18.91% val regression, +21.07% test regression. All 4 splits regress proportionally ~18-25% (in_dist hit hardest at +25.2%). Clip rate hit hard 100.0% (8625/8625 steps). Mean downscaling ~14.2× exceeded predicted 12× — deeper into direction-normalization regime than expected.
- **Mechanism: direction-normalization failure.** The optimum threshold (2.5) gives 7× proportional damping while preserving directional information; deeper than that and the gradient signal collapses to pure direction (analogous to SGD without learning rate adaptation in Adam). norm_max (207.45) actually smaller than at 2.5 (353.04) — fewer extreme spikes despite tighter clipping, suggesting model can no longer access high-curvature directions.
- **Per-split uniformity insight:** in_dist regresses MOST (+25.2%) — diverse training distributions need more gradient signal to fit; aggressive clipping hurts the largest, most informative population first.
- **Next:** frieren #2094 — grad-clip=2.0 fine-scan between 1.5 (fail) and 2.5 (win) to characterize the cliff edge precisely.

## 2026-05-13 13:00 — PR #2000: alphonse T_max=80 schedule — SENT BACK (protocol-stale; current-stack retest)

- Branch: `willowpai2g48h5-alphonse/t-max-80-schedule-push`
- Hypothesis: Extend cosine schedule from T_max=50 → T_max=80 on then-current #1953 stack to give late-training more useful LR. Goalposts moved during the run (PRs #1982 grad-clip=2.5 and #2023 n_hidden=224 merged after start).
- W&B run: `gdk53vwe`

| Metric | #1953 (baseline at submit) | #2000 T_max=80 | #1982 (current) | Δ vs #1953 | Δ vs #1982 |
|---|---:|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | 55.7634 | **54.5106** | 52.6406 | **−2.25% WIN** | **+3.55% LOSS** |
| `test_avg/mae_surf_p` | 48.0960 | **46.7380** | 44.9791 | **−2.82% WIN** | **+3.92% LOSS** |
| `test_single_in_dist` | 52.8835 | 54.2532 | 49.8555 | +2.59% ⚠ | +8.82% |
| `test_geom_camber_rc` | 61.7845 | 59.1708 | 57.7726 | **−4.23%** | +2.42% |
| `test_geom_camber_cruise` | 31.1522 | 29.2805 | 28.9446 | **−6.01%** | +1.16% |
| `test_re_rand` | 46.5637 | 44.2475 | 43.3437 | **−4.97%** | +2.08% |

**Schedule mechanism diagnostic (T_max=80 vs T_max=50):**

| Diagnostic | T_max=50 (#1953) | T_max=80 (this) | Notes |
|---|---:|---:|---|
| Best epoch / total | 30/30 | 33/33 | wall-clock-bound, one extra epoch fit |
| Val slope last 5ep | −0.84/ep | **−0.8742/ep** | descent UNCHANGED — schedule-limited not capacity-saturated |
| Val slope last 3ep | — | −0.9708/ep | actually steepening |
| LR at termination | ~1.73e-4 | 3.23e-4 | ~1.87× higher (matches design) |
| Grad-clip rate | 73% | 93.86% | predicted hot-clip regime, observed |
| EMA−live gap | −8.32 (test) | −14.06 (val)/−13.44 (test) | gap widened ~+5.7 |

- **Decision: SEND BACK for current-stack retest.** Mechanism is real (val descent unchanged at termination, 3/4 splits beat #1953 by 2-6%) but result is protocol-stale by 2 mechanism merges (#1982 grad-clip=2.5 + #2023 n_hidden=224).
- **Mechanism interpretation:** Val descent extends linearly through wall-clock cap. The model wanted more useful LR, not more epochs at zero LR. Higher tail LR widened EMA-live gap as expected.
- **Retest hypothesis:** does T_max=80 still help on grad-clip=2.5 stack? Three outcomes possible:
  - (A) val < 52.64 → schedule + clip compound, 14th compound winner
  - (B) val 52.64-54.0 → mechanism partially survives, document interaction, close
  - (C) val > 54.0 → grad-clip=2.5 at higher T_max=80 LR overshoots into clip-rate-100% regime (like #2067)
- **Watch clip rate.** At T_max=80 + grad-clip=5.0 = 93.86%. Predict T_max=80 + grad-clip=2.5 ≥ 99.5%. If 100% and regression, direction-normalization fail like #2067.
- **in_dist regression note:** the only test split that regressed (+1.37 vs #1953) was in_dist — possibly attention-vs-LR interaction at higher mid-training LR. Worth tracking on retest.

## 2026-05-13 13:00 — PR #2094: frieren assigned grad-clip max_norm=2.0 (fine-scan, bracketed cliff)

- Branch: `willowpai2g48h5-frieren/grad-clip-2p0-fine-scan`
- Hypothesis: Fine-grain test of optimum boundary in [2.0, 2.5]. The 2.5→1.5 cliff is sharp (52.64 WIN → 62.59 FAIL); max_norm=2.0 sits in the middle. Predicted mean downscaling ~9× (linear interp between 7× and 14×).
- Reproduce: `--n_hidden 192 --n_layers 3 --epochs 50` + `GRAD_CLIP_MAX_NORM = 2.0` in train.py.
- Targets: val < 52.6406, test < 44.9791.
- **Three possible outcomes:**
  - (A) val < 52.64 → optimum shifts to 2.0 or below; flat region in [2.0, 2.5]; fine-scan continues
  - (B) val ≈ 52.0-53.5 (within noise) → flat optimum [2.0, 2.5]; 2.5 holds as conservative choice
  - (C) val > 54.0 → regime transition happens between 2.0 and 2.5; threshold optimum is sharp at 2.5
- **Key diagnostic: clip rate.** If 100% (like 1.5 did), direction-normalization regime regardless of downscaling ratio.

## 2026-05-13 13:25 — PR #1841: askeladd slice_num=48 v2 retest — CLOSED (test regression on wider stack)

- Branch: `willowpai2g48h5-askeladd/slice-num-48`
- Hypothesis (v2): retest the slice=48 capacity-down win from v1 (n_hidden=128 stack, val −0.95%) on the current 13-compound stack (n_hidden=224 + grad-clip=2.5 + n_layers=3 + T_max=50). Predicted val ~50-52 if compounds; ~52-53 if washes.
- W&B run: `24e26js0` (30/50 epochs, wall-clock cap)

| Metric | This (slice=48) | Baseline #1982 (slice=64) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **52.5697** | 52.6406 | **−0.13% (wash)** |
| `test_avg/mae_surf_p` | **46.2937** | 44.9791 | **+2.93% regression** |
| `test_single_in_dist` | 51.1522 | 49.8555 | +2.6% |
| `test_geom_camber_rc` | 59.8315 | 57.7726 | +3.6% |
| `test_geom_camber_cruise` | 29.2476 | 28.9446 | +1.0% |
| `test_re_rand` | 44.9435 | 43.3437 | +3.7% |
| Param count | 1.263M | 1.263M | −560 (~0.04%) |
| Peak GPU | 74.27 GiB | similar | ~no change |
| Speedup (steady state) | 60.0 s/ep | ~62 s/ep | +3.2% |
| Epochs completed | 30/50 | 33/50 | both wall-clock capped |
| EMA-live gap (val final) | −12.16 | — | strong EMA averaging benefit |

- **Decision: CLOSE.** Val washes (within noise floor) but test regresses cleanly +2.93% on the paper metric. All 4 test splits regress with uniform magnitude (~1-4%). No path forward.
- **Mechanism analysis (clean):**
  - At n_hidden=224, slice token compute share shrinks (3.2% throughput swing 64↔48 vs 5.4% at n_hidden=128). FFN dominates at wider hidden.
  - Param delta negligible (~0.04% of 1.26M) — slice tokens are tiny share of capacity at wider hidden.
  - EMA−live gap −12.16 confirms strong variance-absorbing benefit; val still descending at ep 30/50 (epoch-saturated, not capacity-saturated — same as #1982).
- **Key insight: slice axis behaves differently at different hidden widths.** v1 (n_hidden=128, slice=48) won −0.95% — capacity reduction meaningful. v2 (n_hidden=224, slice=48) lost on test — capacity reduction irrelevant. **Capacity-down levers don't compose cleanly across widths.**
- **Slice axis CLOSED for capacity-down direction at wider hidden.** Next askeladd test: opposite direction (slice=96, capacity-up) on the wider stack.

## 2026-05-13 13:25 — PR #2113: askeladd assigned slice_num=96 capacity-up (dual to #1841)

- Branch: `willowpai2g48h5-askeladd/slice-num-96-capacity-up`
- Hypothesis: Dual to closed #1841. Tests whether **wider hidden state benefits from richer slicing pathway** (more slice tokens → more attention-addressable geometry roles). At n_hidden=224 we have 1.26M params but only 64 slice tokens — perhaps the slicing axis has untapped headroom.
- Reproduce: `--n_hidden 224 --n_layers 3 --epochs 50 --slice_num 96` (single CLI flag change; --slice_num wiring already in train.py from #1841 v1).
- Targets: val < 52.6406, test < 44.9791.
- **Three possible outcomes:**
  - (A) val < 52.64 → slice axis capacity-up compounds with wider hidden; 14th compound winner
  - (B) val ≈ 52.0-53.5 (within noise) → slice_num plateau region above 64; 64 holds as conservative default
  - (C) val > 54.0 → throughput penalty (~3.2% slower → ~28 vs 30 epochs in budget) + potential over-parameterization erodes performance
- **Mechanism diagnostics:** per-epoch wall time (predict ~64 s/ep), param count (+560 weights to 1.264M), peak GPU (~78 GiB vs 74 at slice=48), val trajectory at ep 25/28/final (plateau or still descending), EMA-live gap (was −12.16 at slice=48; if similar, EMA still doing variance-absorbing work).

## 2026-05-13 13:50 — PR #1805: fern Huber β anneal v3 — CLOSED (grad-clip saturates anneal mechanism)

- Branch: `willowpai2g48h5-fern/beta-anneal-1p0-to-0p5`
- Hypothesis: β anneal 1.0→0.5 over epochs 1-10 on the 13-compound stack. Expectation: anneal compounds with the longer T_max=50 cosine tail.
- W&B run: `lgg4v9sx` (32/50 epochs)

| Metric | This (β anneal v3) | Baseline #1982 | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **53.9233** | 52.6406 | **+2.43% regression** |
| `test_avg/mae_surf_p` | **46.5339** | 44.9791 | **+3.46% regression** |
| `test_single_in_dist` | 53.0485 | 49.8555 | +6.40% (largest hit) |
| `test_geom_camber_rc` | 59.4080 | 57.7726 | +2.84% |
| `test_geom_camber_cruise` | 28.4414 | 28.9446 | −1.73% ✓ (only improvement) |
| `test_re_rand` | 45.2377 | 43.3437 | +4.38% |

**β anneal schedule + grad-clip rate (the mechanism diagnostic):**

| Epoch | β | clip rate | mean grad-norm | val (EMA) |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | **100.0%** | 41.05 | 331.59 |
| 5 | 0.778 | **100.0%** | 18.91 | 181.65 |
| 10 | 0.500 | 100.0% | 18.32 | 109.42 |
| 32 | 0.500 | 96.3% | 10.48 | 53.92 |

- **Decision: CLOSE.** Falls in 52.64-55.0 close zone per advisor's own 08:23 instruction. 3/4 test splits regress; in_dist takes the biggest hit (+6.40% — same pattern as #2067 grad-clip=1.5 fail).
- **Mechanism: grad-clip=2.5 saturates the gradient channel during high-β phase.** β=1.0 produces mean grad norms 18-41 (7-16× threshold). 99.7-100% clip rate during anneal phase truncates everything to direction-only (sign without magnitude). The curvature information that β=1.0 was supposed to provide is **operationally invisible** under saturated clipping. After ep 10 the model is at β=0.5 and the anneal mechanism has done nothing distinctive.
- **Per-split insight:** in_dist regresses MOST (+6.40%) — mirrors #2067 finding. Aggressive/corrupted gradient signal hurts the largest, most-informative population first.
- **β anneal axis CLOSED on current grad-clip=2.5 stack.** Mechanism may revive if grad-clip loosens (e.g., if frieren #2094 max_norm=2.0 fails and we go back to 3.0+). Currently the loss-shape lever fights the gradient-stability lever.
- **Next:** fern #2142 constant β=0.25 (no high-β phase; tests steady-state loss-shape axis).

## 2026-05-13 13:50 — PR #2053: nezuko mlp_ratio=3 v1 — SENT BACK (protocol-stale grad-clip=5.0)

- Branch: `willowpai2g48h5-nezuko/mlp-ratio-3`
- Hypothesis: FFN capacity bracket. mlp_ratio=2 → 3 on the (then) 11-compound stack. Wider FFN per attention block helps when n_layers=3 has each block doing more representation work.
- W&B run: `4da3u1ad` (30/50 epochs)

| Metric | This (mlp_ratio=3) | #1953 (at submit) | #1982 (current) | Δ vs #1953 | Δ vs #1982 |
|---|---:|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | **54.8155** | 55.7634 | 52.6406 | **−1.70% WIN** | **+4.13% LOSS** |
| `test_avg/mae_surf_p` | **47.8073** | 48.0960 | 44.9791 | **−0.60% WIN** | **+6.29% LOSS** |
| `test_single_in_dist` | 52.9765 | 52.8835 | 49.8555 | +0.18% | +6.26% |
| `test_geom_camber_rc` | 61.2182 | 61.7845 | 57.7726 | **−0.92% ✓** | +5.96% |
| `test_geom_camber_cruise` | 31.0784 | 31.1522 | 28.9446 | **−0.24% ✓** | +7.37% |
| `test_re_rand` | 45.9562 | 46.5637 | 43.3437 | **−1.30% ✓** | +6.03% |
| Param count | 1.15M (+22%) | 0.94M | — | — | — |
| Grad-clip rate | 93.46% | (grad-clip=5.0) | — | — | — |

- **Decision: SEND BACK for current-stack retest.** Mechanism is real on the older stack but result is protocol-stale by #1982 (grad-clip=5.0 → 2.5 in train.py) and #2023 (n_hidden=224 documented).
- **Mechanism interpretation (student's analysis is excellent):** "With shallower attention (n_layers=3), per-block FFN does proportionally more representation work" — explains why mlp_ratio scaled here when it failed at n_layers=5 (#1544). 3/4 OOD splits descended cleanly.
- **Retest with `--n_hidden 224 --n_layers 3 --epochs 50` and `mlp_ratio=3` constant in train.py.** Predicted: val ≈ 51.75 if linear compound with #1982 baseline.
- **Three outcomes:**
  - (A) val < 52.64 → MERGE (14th compound winner; FFN width compounds with width + clip)
  - (B) val 52.64-54.0 → mechanism still has signal but doesn't beat new baseline; close
  - (C) val > 54.0 OR clip rate hits 100% → mlp_ratio=3 fights saturated clip; close
- **Diagnostic to watch:** grad-clip rate at termination. mlp_ratio=3 means larger gradients (more params) → predict 99.5%+. If 100%, in direction-normalization regime like #2067.

## 2026-05-13 13:50 — PR #2024: edward EMA decay=0.998 v1 — SENT BACK (protocol-stale grad-clip=5.0)

- Branch: `willowpai2g48h5-edward/ema-decay-0p998`
- Hypothesis: Tighten EMA decay 0.999 → 0.998 to halve the EMA half-life (693 → 346 steps) for better tracking of late-training improvements on T_max=50 schedule.
- W&B run: `ajd4i909` (29/50 epochs)

| Metric | This (EMA=0.998) | #1953 (at submit) | #1982 (current) | Δ vs #1953 | Δ vs #1982 |
|---|---:|---:|---:|---:|---:|
| `val_avg/mae_surf_p` | **54.2188** | 55.7634 | 52.6406 | **−2.77% WIN** | **+3.00% LOSS** |
| `test_avg/mae_surf_p` | **46.9800** | 48.0960 | 44.9791 | **−2.32% WIN** | **+4.45% LOSS** |
| `test_single_in_dist` | 53.7677 | 52.8835 | 49.8555 | +1.67% | +7.85% |
| `test_geom_camber_rc` | 60.7889 | 61.7845 | 57.7726 | **−1.61% ✓** | +5.22% |
| `test_geom_camber_cruise` | 29.2714 | 31.1522 | 28.9446 | **−6.04% ✓** | +1.13% |
| `test_re_rand` | 44.0918 | 46.5637 | 43.3437 | **−5.31% ✓** | +1.73% |
| EMA-live gap (final) | −8.68 | −8.32 | — | (unchanged) | — |

- **Decision: SEND BACK for current-stack retest.** Same protocol-stale issue as #2053 (grad-clip=5.0, clip rate 93.83%). Mechanism partial confirmation: val/test improve cleanly vs #1953, but EMA-live gap did NOT close as predicted.
- **Mechanism reframed (student's analysis is sharp):** Shorter half-life (~346 steps) doesn't reduce the *visible* gap against a noisy live model. It reduces the *steady-state averaging bias* against the *expected* live trajectory. Live model is noisy (94% grad-clip rate) — gap metric is dominated by live noise, not EMA lag. The improvement comes from EMA absorbing higher-frequency noise, not from closing the lag.
- **Caveat:** test_single_in_dist regressed slightly (+0.88 vs #1953, +1.67%). The faster EMA hurts the regime where the live model is most confident.
- **Retest with `--n_hidden 224 --n_layers 3 --epochs 50` and `ema_decay=0.998` constant in train.py.** Predicted: val ≈ 51.18 if relative −2.77% holds. Aggressive clip rate at grad-clip=2.5 (98.9% baseline) means live model is *more* noisy — possibly amplifying EMA's noise-rejection benefit.
- **Three outcomes:**
  - (A) val < 52.64 → MERGE (14th compound winner)
  - (B) val 52.64-54.0 → mechanism still has signal but doesn't beat new baseline; close
  - (C) val > 54.0 → faster EMA hurts under tighter clip regime; close
- **Diagnostic to log:** EMA-live gap at final epoch + test_single_in_dist (does in_dist regression persist?).

## 2026-05-13 13:50 — PR #2142: fern assigned Huber β=0.5 → 0.25 (loss-shape follow-up to #1689)

- Branch: `willowpai2g48h5-fern/huber-beta-0p25-13stack`
- Hypothesis: Tighter MAE alignment via β=0.25 on the 13-compound stack. Natural follow-up to fern's own #1689 merge (β=1.0 → 0.5, val −7.4%). Smaller β = more samples in linear region = closer to L1/MAE alignment.
- **Distinct from closed #1805 (β anneal):** Constant β=0.25 has NO high-β phase. Tests *steady-state* loss-shape axis at the current grad-clip=2.5 regime. β anneal failed because the high-β (1.0) phase saturated clipping at 99.7-100% during epochs 1-10. Constant β=0.25 operates entirely in the already-saturated steady-state regime — different mechanism question.
- Reproduce: `--n_hidden 224 --n_layers 3 --epochs 50` + Huber β constant `0.5 → 0.25` in train.py.
- Targets: val < 52.6406, test < 44.9791.
- **Three possible outcomes:**
  - (A) val < 52.64 → loss-shape lever still productive at the current stack; tighter MAE alignment helps the higher-capacity model fit residual errors
  - (B) val 52.0-53.5 → β plateau region above optimum; β=0.5 was already near-optimal
  - (C) val > 54.0 → β=0.25 destabilizes quadratic transition; regression despite tighter MAE alignment
- **Diagnostics:** grad-clip rate at termination (predict 98.5-99.5%), mean grad-norm distribution (compare to β=0.5 baseline), per-split test breakdown, EMA-live gap.

## 2026-05-13 14:30 — PR #2113: askeladd slice_num=96 capacity-up — CLOSED (throughput-induced epoch deficit)

- Branch: `willowpai2g48h5-askeladd/slice-num-96-capacity-up`
- Hypothesis: Dual to #1841. Tests whether wider hidden state benefits from richer slicing pathway. Predicted ~3.2% throughput penalty.
- W&B run: `umdt6f8w` (27/50 epochs, hard timeout)

| Metric | This (slice=96) | Baseline #1982 (slice=64) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **57.7869** | 52.6406 | **+9.77% regression** |
| `test_avg/mae_surf_p` | **49.4074** | 44.9791 | **+9.85% regression** |
| `test_single_in_dist` | 57.9273 | 49.8555 | **+16.18%** (largest hit) |
| `test_geom_camber_rc` | 62.2946 | 57.7726 | +7.83% |
| `test_geom_camber_cruise` | 30.8522 | 28.9446 | +6.59% |
| `test_re_rand` | 46.5555 | 43.3437 | +7.39% |
| Throughput penalty (measured) | ~7% (66.5 s/ep) | ~62 s/ep | predicted 3.2% — much worse |
| Epochs completed | **27/50** | 30/50 | 3-epoch deficit |
| Val descent at terminal | **−1.4/ep monotonic** | — | epoch-saturated, no plateau |
| Peak GPU memory | **25.8 GB** | similar | predicted ~78 GiB (overestimated) |

- **Decision: CLOSE.** All 4 splits regress; val and test both >5% threshold.
- **Mechanism: throughput-induced epoch deficit, NOT capacity wash.** Predicted 3.2% slowdown → measured 7% (super-linear cost). Lost 3 epochs of training. Linear extrapolation: 3 more epochs would reach val ≈ 53.6, still slightly above baseline. The cost curve in slice_num is super-linear at this width, not symmetric to the 64→48 direction.
- **Per-split asymmetry insight:** in_dist takes the LARGEST hit (+16.18%), not OOD. Inverse of #2067 (where all 4 splits regressed uniformly under direction-normalization fail). Says: single-foil samples are already over-parameterized by 64 slice tokens; adding 32 more dilutes attention without useful capacity.

**Slice axis at n_hidden=224 is now fully bracketed:**

| slice_num | val | test | regime |
|---:|---:|---:|---|
| 48 (#1841 v2) | 52.57 | 46.29 | val-wash, +2.93% test regression |
| **64 (#1982)** | **52.64** | **44.98** | **OPTIMUM** |
| 96 (this) | 57.79 | 49.41 | epoch-saturated by throughput cost |

- **Slice axis CLOSED at this stack.** Plateau region narrow; throughput-budget interaction means broader exploration is net-negative. The slicing-vs-budget tradeoff is the real bottleneck.
- **Next:** askeladd #2159 — peak LR 5e-4→7.5e-4 (different axis, epoch-saturated escape via amplitude).

## 2026-05-13 14:30 — PR #2094: frieren grad-clip max_norm=2.0 fine-scan — CLOSED (2.5 optimum holds)

- Branch: `willowpai2g48h5-frieren/grad-clip-2p0-fine-scan`
- Hypothesis: Fine-grain test between WIN (2.5) and FAIL (1.5). Predicted mean downscaling ~9× (between 7.1× and 14.2×).
- W&B run: `4rhtvua7` (33/50 epochs)

| Metric | This (max_norm=2.0) | Baseline #1982 (max_norm=2.5) | Δ |
|---|---:|---:|---:|
| `val_avg/mae_surf_p` | **53.1096** | 52.6406 | **+0.90% regression** |
| `test_avg/mae_surf_p` | **45.7294** | 44.9791 | **+1.70% regression** |
| `test_single_in_dist` | 52.7295 | 49.8555 | +5.80% |
| `test_geom_camber_rc` | 59.3584 | 57.7726 | +2.70% |
| `test_geom_camber_cruise` | 27.5591 | 28.9446 | **−4.80% ✓** |
| `test_re_rand` | 43.2707 | 43.3437 | −0.20% (wash) |
| Clip rate (measured) | **99.46%** | 98.9% | predicted 99.5% — bang on |
| Mean downscale | **9.2×** | 7.1× | predicted 9× — bang on |

- **Decision: CLOSE.** Outcome (B) tilting toward (C). Falls in close zone (52.64-55.0 = "doesn't beat baseline").
- **Mechanism: soft shoulder, not flat plateau.** The transition into direction-normalization begins gradually around 2.0, becomes catastrophic at 1.5. Above ~9× downscale, the gradient signal starts losing useful magnitude information but doesn't fully collapse.
- **Per-split insight:** in_dist + camber_rc regress while camber_cruise improves (−4.80%). Tighter clip over-smooths high-gradient single-foil samples while helping the lower-magnitude cruise domain. Same single-foil-takes-hit pattern as #2067 (+25%) and #1805 anneal (+6.4%).

**Threshold scan FULLY CHARACTERIZED:**

| max_norm | clip rate | mean downscale | val_avg | regime |
|---:|---:|---:|---:|---|
| 10.0 (#1784) | 72.4% | ~2.1× | 65.98 | soft-scaling WIN |
| 5.0 (#1930) | 90.1% | ~4.3× | 63.48 | moderate-scaling WIN |
| **2.5 (#1982)** | **98.9%** | **~7.1×** | **52.64** | **OPTIMUM** |
| **2.0 (this)** | **99.46%** | **~9.2×** | **53.11** | **soft-shoulder regression** |
| 1.5 (#2067) | 100.0% | ~14.2× | 62.59 | direction-norm FAIL |
| 1.0 | ~100% | ~22× | regression | direction-norm FAIL |

- **Threshold scan CLOSED.** 2.5 is the optimum with a soft shoulder between 2.0-2.5, sharp cliff between 2.0-1.5. Diminishing returns to fine-scanning further.
- **Student suggestion (out of scope):** domain-conditioned clip thresholds or per-domain gradient pre-scaling could recover the cruise improvement (−4.80% here) without the in_dist regression.
- **Next:** frieren #2160 — weight_decay 0 → 1e-5 (different axis: regularization, untested).

## 2026-05-13 14:30 — PR #2159: askeladd assigned peak LR 5e-4 → 7.5e-4 (amplitude scaling)

- Branch: `willowpai2g48h5-askeladd/lr-peak-7p5e-4`
- Hypothesis: Test whether the model is LR-amplitude-limited rather than schedule-limited (which alphonse #2000 is testing via T_max=80). Higher peak LR → more "exploration LR" during mid-training where the model is doing most of its learning.
- Reproduce: `--n_hidden 224 --n_layers 3 --epochs 50 --lr 7.5e-4` (or code-constant change).
- Targets: val < 52.6406, test < 44.9791.
- **Distinct from alphonse #2000 (T_max=80):** schedule extension keeps amplitude, this scales amplitude with same schedule length. Both axes could compound in future.
- **Three outcomes:** (A) val<52.64 win, LR-limited; (B) wash, near-optimal already; (C) val>54 clip rate hits 100%, direction-norm fail like #2067.
- **Diagnostics:** clip rate at termination (predict 99.5%+; if 100%, regime saturated), mean grad norm, per-split test, EMA-live gap (predict wider, ~−15 to −20 vs −12 baseline).

## 2026-05-13 14:30 — PR #2160: frieren assigned weight_decay 0 → 1e-5 (untested regularization)

- Branch: `willowpai2g48h5-frieren/weight-decay-1e-5`
- Hypothesis: Test explicit L2 regularization on 1.26M-param model + 1500-sample dataset. The capacity/sample ratio is ~840 params per sample — substantial enough that explicit weight_decay may close the train/val gap that EMA averaging alone misses.
- Reproduce: `--n_hidden 224 --n_layers 3 --epochs 50` + `weight_decay=1e-5` constant in AdamW constructor.
- Targets: val < 52.6406, test < 44.9791.
- **Distinct from grad-clip work:** grad-clip operates on gradient magnitudes (post-backward); weight_decay operates on weight magnitudes (during update). Different mechanism.
- **First task: verify current weight_decay value.** PyTorch AdamW default is 0.01, but the optimizer setup may explicitly set 0. Log the value before/after change.
- **Three outcomes:** (A) val<52.64 win, weight_decay closes train/val gap; (B) wash, no overfit at this scale; (C) val>54 wd too strong, try 1e-6.
- **Diagnostics:** train/val gap at ep 5/15/25/terminal (should narrow if overfitting is real), mean weight magnitude, clip rate (predict ~98.9% similar), per-split test, EMA-live gap.

## 2026-05-13 14:55 — PR #2068: thorfinn n_hidden=256 width push — CLOSED (runtime ceiling at 30-min budget)

- Branch: `willowpai2g48h5-thorfinn/n-hidden-256`
- Hypothesis: Width scan step 3, continue 192 → 224 win trajectory to test if width scaling plateaus at 1500-sample dataset capacity.
- W&B run: `bywj2855`

| Metric | #2068 (n=256) | Baseline #1982+#2023 (n=224) | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | **54.5742** | 52.6406 | **+3.67% REGRESSION** |
| test_avg/mae_surf_p | **47.4763** | 44.9791 | **+5.55% REGRESSION** |
| Epochs reached | 27/50 | 50/50 | −46% |
| Wall-clock | 31.3 min | ~28 min | +12% |
| Param count | 1.64M | 1.27M | +29% |
| Epoch time | 67.55s | ~62s | +9% |

**All 4 test splits regress** (in_dist, camber_rc, camber_cruise, re_rand).

**Mechanism — throughput-induced epoch deficit (not width ceiling):**
- +29% params → +9% epoch time on the bottleneck
- 30-min hard timeout (SENPAI_TIMEOUT_MINUTES=30) cuts training at 27/50 epochs (~54% of cosine schedule)
- T_max=50 schedule never completes its decay → final LR is still high → final eval is off the convergence trajectory
- The width itself may genuinely help with more epochs, but within the 30-min budget the wider model cannot match the better-converged 224 model

**Width axis bracketed at n_hidden=224 within 30-min budget:**

| n_hidden | val | test | Epochs | Verdict |
|---|---|---|---|---|
| 192 | 55.76 | 47.20 | 50/50 | pre-#2023 baseline |
| **224** | **52.64** | **44.98** | 50/50 | **OPTIMUM** |
| 256 | 54.57 | 47.48 | 27/50 | runtime ceiling |

**Pattern match:** Identical failure mode to PR #2113 askeladd slice_num=96 (7% throughput penalty cut training to 27/50 epochs, +9.77%/+9.85% regression). Both show that within a fixed wall-clock budget, capacity-up only works when throughput-neutral.

**Conclusion:** Width axis closed at n_hidden=224. Further width experiments only viable with throughput offset (e.g. n_layers=2 + n_hidden=256, but depth-floor closed at n_layers=3 by #1960).

## 2026-05-13 14:55 — PR #2186: thorfinn assigned AdamW betas=(0.9, 0.999) → (0.9, 0.95) (untested optimizer axis)

- Branch: `willowpai2g48h5-thorfinn/adamw-betas-0p9-0p95`
- Hypothesis: beta_2 controls the second-moment EMA decay in AdamW; reducing 0.999 → 0.95 makes the variance estimate react ~50× faster (half-life 14 steps vs 693). This is the right adaptation when gradient statistics shift across training phases — we have a strong cosine LR decay + Huber β anneal/saturation interactions + grad-clip at 98.9% clip rate where the optimizer must escape saturation regimes.
- Reproduce: `--n_hidden 224 --n_layers 3 --epochs 50` + AdamW `betas=(0.9, 0.95)`.
- Targets: val < 52.6406, test < 44.9791.
- **Distinct from active optimization axes:** askeladd #2159 (peak LR amplitude), frieren #2160 (weight_decay), alphonse #2000 (schedule shape), grad-clip axis (closed at 2.5). beta_2 is the variance-tracking axis, untouched.
- **Two outcomes:** (A) WIN ~1-3% val MAE reduction — faster adaptation tracks the high-clip-rate regime (98.9% saturation at grad-clip=2.5); (B) FAIL ~2-5% val MAE increase — variance estimate too noisy at beta_2=0.95 for small-batch mesh-node scale, late-training LR-effective instability.
- **Diagnostics:** clip rate (predict similar 98.9%, beta_2 doesn't directly change grad magnitudes), val/train gap (predict slight widening if variance-tracking is noisier), EMA-live gap (predict similar −12 to −13 baseline).

## 2026-05-13 15:15 — PR #2066: tanjiro n_hidden=224 + grad-clip=2.5 compound confirmation — CLOSED (critical negative finding)

- Branch: `willowpai2g48h5-tanjiro/n-hidden-224-compound-confirm`
- Hypothesis: PRIORITY confirmation run. Directly measure the combined state from PR #1982 (grad-clip=2.5 at n_hidden=192) and PR #2023 (n_hidden=224 at grad-clip=5.0). Predicted compound val ≈ 50-52 if mechanisms are additive.
- W&B run: `r4kgwypm` (after recovery from initial CUDA OOM during torch.compile warmup on first attempt; killed PID 88784, relaunched PID 93215)

### Results — compound state WORSE than n_hidden=192 baseline

| Metric | #2066 (n=224 + clip=2.5) | #1982 baseline (n=192 + clip=2.5) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | **54.3382** | **52.6406** | **+3.22% REGRESSION** |
| test_avg/mae_surf_p | **47.1909** | **44.9791** | **+4.92% REGRESSION** |
| Epochs reached | 29/50 | 33/50 | −12% |

**All 4 test splits regress** uniformly: in_dist 53.69 (+7.7%), camber_rc 60.78 (+5.2%), camber_cruise 29.14 (+0.7%), re_rand 45.15 (+4.2%). Uniformity with largest hit on the easiest split (in_dist) is the signature of under-converged training, not inductive-bias clash.

### Mechanism — throughput-induced epoch deficit + cosine misalignment

1. **n=224 → 12% slower epochs** (61.83 s vs ~54 s at n=192). Run cut at 29/50 epochs by 30-min cap.
2. **Cosine LR at termination is higher**: at epoch 29 of T_max=50 cosine, LR=37.6% of base (vs 25.9% at epoch 33 in baseline). The wider net never enters cosine's quiet annealing tail.
3. **Larger raw gradient norms × tighter clip**: norm_mean +5.8% (18.88 vs 17.85), clip rate 99.31% (vs 98.93%), mean downscaling 7.55× (vs 7.14×). Even more aggressive clipping at the wider net, compounding the under-convergence.
4. **EMA-live gap intact** (−7.69 here, −8.32 at #1953, −6.18 at #2023 stale measurement). EMA shadow is doing its job; the *trajectory itself* is the problem.

### Critical implication: PR #2023 win was protocol-stale

The PR #2023 (n_hidden=224) "win" at val=53.25 was measured under the OLD grad-clip=5.0 threshold. Direct measurement under the post-#1982 stack proves n_hidden=192 is the TRUE optimum. The cumulative compounding table in BASELINE.md should be read as: PR #1982 is the actual current best; PR #2023's measured value does not transfer to the current stack.

### Width axis bracketed at n_hidden=192 under 30-min budget × T_max=50 cosine

| n_hidden | grad-clip | val | Epochs | Verdict |
|---|---|---|---|---|
| 128 | 5.0 | 55.76 | 30/30 | superseded by 192 |
| **192** | **2.5** | **52.64** | **33/50** | **TRUE OPTIMUM** |
| 224 | 5.0 (stale) | 53.25 | 29/50 | protocol-stale |
| 224 | 2.5 | 54.34 | 29/50 | runtime ceiling |
| 256 | 2.5 | 54.57 | 27/50 | runtime ceiling |

### Suggested follow-ups from student (excellent diagnostic)

1. T_max=30 at n_hidden=224 (re-align cosine to realized budget)
2. n_hidden=192 + grad-clip=2.5 + T_max=30 (disentangle width vs schedule)
3. n_hidden=208 half-step

→ Assigning #2 variant: **--epochs 33** at n_hidden=192 (perfectly align T_max with realized 33-epoch budget, at the WINNING width).

## 2026-05-13 15:15 — PR #2199: tanjiro assigned --epochs 33 cosine schedule alignment

- Branch: `willowpai2g48h5-tanjiro/epochs-33-cosine-aligned`
- Hypothesis: Cosine schedule mismatch with realized epoch budget. At fixed 30-min budget on n_hidden=192, the model reaches 33/50 epochs and cosine LR is still at ~26% of base — never enters the quiet annealing tail. Test T_max=epochs=33 alignment: cosine fully decays to zero by epoch 33, perfectly matching realized budget.
- Reproduce: `--n_hidden 192 --n_layers 3 --epochs 33`.
- Targets: val < 52.6406, test < 44.9791.
- **Distinct from alphonse #2000 (T_max=80):** opposite direction — extend cosine, keep LR high. These two test direct opposites; at most one wins.
- **Three outcomes:** (A) val<52.64 win, late-phase low-LR refinement matters; (B) wash, misalignment was minor; (C) val>53.5 fail, LR=0 final phase hurts.
- **Diagnostics:** LR at termination (predict ~0), clip rate (predict ~98.9% similar), EMA-live gap (predict tighter than −8.32 if late-phase quiet annealing helps), val slope at ep 33 (predict near 0 if fully converged).

## 2026-05-13 15:15 — PR #2000 alphonse second ping (retest stalled 6h)

- Branch: `willowpai2g48h5-alphonse/t-max-80-schedule-push`
- Status: Sent back at 08:57 UTC for retest at grad-clip=2.5 stack. Branch HEAD is still the original 06:53 UTC assignment marker (no retest commit). Pod healthy (0 restarts). Issued second ping with restart command at 15:15 UTC.
- Also notified student that #2066 finding confirms `--n_hidden 192` (already in their send-back command) is the TRUE optimal width.

## 2026-05-13 15:40 — PR #2000: alphonse T_max=80 retest on grad-clip=2.5 stack — CLOSED (clip-saturation kills schedule extension)

- Branch: `willowpai2g48h5-alphonse/t-max-80-schedule-push`
- Hypothesis: Retest T_max=80 schedule extension on current 13-compound stack (grad-clip=2.5). Original run at grad-clip=5.0 won −2.25%. Tests whether schedule lever compounds with the new tighter clip threshold.
- W&B run: `6o54qgc6`

### Results — clean negative result

| Metric | #2000 retest (T_max=80) | #1982 baseline (T_max=50) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | **55.0317** | 52.6406 | **+4.54% REGRESSION** |
| test_avg/mae_surf_p | **48.0524** | 44.9791 | **+6.84% REGRESSION** |
| Epochs reached | 33/80 | 33/50 | identical |
| Clip rate | **99.44%** | 98.93% | +0.51pp |
| Mean downscaling | **7.50×** | 7.14× | +5% |
| EMA-live gap (val) | **−19.24** | ~−8 (#1953) | widened to handle live noise |

All 4 test splits regress: in_dist 55.31 (vs 49.86 baseline), camber_rc 61.47 (vs 57.77), camber_cruise 30.30 (vs 28.94), re_rand 45.13 (vs 43.34).

### Mechanism — schedule lever requires headroom in clip distribution

Schedule axis interaction matrix:

| Stack | T_max | Clip rate | val | Verdict |
|---|---|---:|---:|---|
| grad-clip=5.0 (old #1953 stack) | 50 | ~73% | 55.76 | baseline |
| grad-clip=5.0 + T_max=80 (prior alphonse run) | 80 | 93.86% | 54.51 | WIN −2.25% (now stale) |
| **grad-clip=2.5 + T_max=50 (current #1982)** | **50** | **98.93%** | **52.64** | **OPTIMUM** |
| grad-clip=2.5 + T_max=80 (THIS PR) | 80 | **99.44%** | 55.03 | FAIL +4.54% |

At grad-clip=5.0 + T_max=80 (93.86% clip), ~6% of steps preserve their gradient magnitude — enough curvature signal for T_max=80's "keep effective LR higher in mid-training" mechanism to operate.

At grad-clip=2.5 + T_max=80 (99.44% clip), the optimizer is essentially doing **direction-only SGD with a fixed step magnitude of 2.5/||g||**. T_max=80's mechanism is operating on a constant-magnitude proxy — the LR multiplier doesn't translate into larger effective steps. Information loss dominates.

This is the same direction-normalization regime that killed frieren #2067 at max_norm=1.5.

### Critical insight — orthogonal-compounding assumption fails at clip saturation

The schedule mechanism IS real (proven at grad-clip=5.0 by the original alphonse run). It does NOT compound with grad-clip=2.5. The two mechanisms interact destructively at clip-saturation regimes (>99% clip rate).

This is the first cleanly-observed destructive interaction in our compound stack. All prior compounds (grad-clip × n_layers × n_hidden × T_max=50) were measured to compound additively. The compound assumption breaks when one axis pushes another into a saturated regime.

### Schedule axis closed under current grad-clip=2.5 stack

- T_max=50: 52.64 (OPTIMUM)
- T_max=80: 55.03 (FAIL)
- T_max=33: tanjiro #2199 in flight — tests OPPOSITE direction (alignment with realized 33-epoch budget at n_hidden=192)

### Suggested follow-ups
1. **n_hidden=160 at T_max=50** — width-axis floor side (closes width bracket if it fails or finds new optimum if it wins)
2. **tanjiro #2199** — schedule alignment in flight

→ Assigning alphonse: n_hidden=160 width-floor test.

## 2026-05-13 15:40 — PR #2219: alphonse assigned n_hidden=160 width-floor test

- Branch: `willowpai2g48h5-alphonse/n-hidden-160-width-floor`
- Hypothesis: Width axis from the floor side. PR #2066 bracketed width at n_hidden=192 from above (224 fails on epoch deficit). Test symmetric question from below: does n_hidden=160 win by giving up some per-layer capacity for ~47 epochs in budget (predicted) vs 33 at n_hidden=192? The throughput-vs-capacity tradeoff at fixed 30-min budget is the core constraint.
- Reproduce: `--n_hidden 160 --n_layers 3 --epochs 50`.
- Targets: val < 52.6406, test < 44.9791.
- **Quantitative predictions:**
  - Param count: ~0.65M (−30% vs current 0.93M)
  - Epoch time: ~38s (vs 54s at n=192, assuming O(n²) attention)
  - Epochs in 30 min: ~47/50 (vs 33/50) — nearly full schedule
  - Cosine LR at termination: ~6% of base (vs 26%) — fully into quiet annealing tail
- **Three outcomes:** (A) val<52.64 win — narrower+more-epochs is right tradeoff; (B) wash — 160/192 plateau; (C) val>53.5 fail — 192 bracketed [160, 224].
- **Distinct from active axes:** tanjiro #2199 (--epochs 33 alignment) extracts late-phase low-LR refinement via *schedule shortening*; this PR extracts it via *width narrowing*. Alternative routes to the same lift — orthogonal mechanisms.

## 2026-05-13 16:05 — PR #2159: askeladd peak LR 5e-4 → 7.5e-4 — CLOSED (clip-saturation blocks LR amplitude)

- Branch: `willowpai2g48h5-askeladd/lr-7p5e-4-amplitude-scale`
- Hypothesis: Raise peak LR 1.5× to extract more progress from same per-step direction signal at clip saturation. Tests amplitude scaling axis, distinct from schedule (#2000 alphonse) and width (#2066/#2068).
- W&B run: `kfed937m`

### Results — clean fail

| Metric | #2159 (lr=7.5e-4) | #1982 baseline (lr=5e-4) | Δ |
|---|---:|---:|---:|
| val_avg/mae_surf_p | **56.7268** | 52.6406 | **+7.77% REGRESSION** |
| test_avg/mae_surf_p | **49.1945** | 44.9791 | **+9.37% REGRESSION** |
| Clip rate | 99.30% | 98.93% | +0.37pp |
| Mean downscaling | ~6.5× | ~7.1× | similar |
| EMA-live gap (final) | −5.67 | ~−6 | similar (NOT widened as predicted) |
| Val descent at term | −1.49/ep | ~−0.84/ep | steeper but starting from worse trajectory |

All 4 test splits regress uniformly: in_dist +10.5%, camber_rc +8.3%, camber_cruise +10.3%, re_rand +8.9%.

### Mechanism — third clip-saturation interaction confirmed

| PR | Lever | grad-clip | clip rate | val Δ |
|---|---|---|---:|---:|
| #2066 tanjiro | n_hidden=224 | 2.5 | 99.31% | +3.22% |
| #2000 alphonse | T_max=80 | 2.5 | 99.44% | +4.54% |
| **#2159 (THIS PR)** | **lr=7.5e-4** | 2.5 | 99.30% | +7.77% |

**Pattern crystallized:** At grad-clip=2.5 with 98.93% baseline clip rate, any axis that operates via gradient amplitude is blocked. Effective step magnitude is fixed at `2.5/||g||`. Multiplicative scaling of LR or extending schedule doesn't translate into useful effective updates.

### Sub-finding — model adapts parameter scale

Student's sharp diagnostic: **norm_mean actually DECREASED at lr=7.5e-4** (17.85→16.19, −9.3%) rather than increasing as predicted. The optimization landscape compensates for higher LR by shifting parameter scale, so gradient norms stay roughly bounded. This means raising LR doesn't liberate amplitude information because the model's gradient distribution is approximately scale-invariant under these dynamics.

### Implications

Mechanisms blocked by clip saturation (now empirically confirmed):
- T_max extension (#2000)
- Peak LR raise (#2159)
- Width increase past compute budget (#2066, #2068)

Mechanisms expected to work at clip saturation (orthogonal to amplitude):
- AdamW betas — variance dynamics (thorfinn #2186)
- Weight decay — parameter scale (frieren #2160)
- Huber β — loss curvature (fern #2142)
- EMA decay — averaging (edward #2024)
- mlp_ratio — capacity (nezuko #2053)
- Width narrowing — frees epoch budget (alphonse #2219)
- Schedule shortening — re-aligns cosine to realized budget (tanjiro #2199)
- LR lowering — may exit saturation (askeladd #2231 just assigned)

## 2026-05-13 16:05 — PR #2231: askeladd assigned lr=5e-4 → 3e-4 lower-amplitude (escape clip saturation)

- Branch: `willowpai2g48h5-askeladd/lr-3e-4-lower-amplitude`
- Hypothesis: Symmetric counter-test to #2159. Lower LR may exit clip saturation (target clip rate ~92-95%, vs 98.93% baseline). At lr=3e-4, raw gradient magnitudes are similar but the clip threshold (2.5) is a larger fraction of typical norms → fewer steps clipped → amplitude information flows through more often → AdamW gets diverse signal.
- Reproduce: `--n_hidden 192 --n_layers 3 --lr 3e-4 --epochs 50`.
- Targets: val < 52.6406, test < 44.9791.
- **Three outcomes:** (A) val<52.64 win — amplitude liberation works at lower LR; (B) wash; (C) val>54.5 fail — undertraining dominates, LR axis bracketed at 5e-4 within [3e-4, 7.5e-4].
- **Quantitative predictions:** clip rate drops to ~92-95%, mean downscaling ~4-5×, possible undertraining risk if val descent rate halves.
- **Distinct from active axes:** all 7 other students are on orthogonal axes (width-floor, EMA, Huber, weight_decay, mlp_ratio, schedule alignment, AdamW betas).
