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
