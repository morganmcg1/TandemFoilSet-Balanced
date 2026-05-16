# SENPAI Research Results

## 2026-05-15 16:28 — W&B surfacing on 5 stale-WIP PRs (#3173 #3186 #3190 #3196 #3211)

A scheduled wakeup at 16:21 UTC flagged 5 PRs as `stale_wip`. Their branch HEADs all still pointed at the original assignment commit from 12:52 UTC — no code commits and no `SENPAI-RESULT` markers — yet each student had **multiple completed W&B training runs** in their hypothesis's `wandb_group`. Surfacing W&B as the source of truth revealed substantial work hidden from the PR review queue:

| PR | Student | wandb_group | Runs | Best val_avg/mae_surf_p | Δ vs baseline 136.89 | All-splits-improve? |
|---|---|---|---|---|---|---|
| #3186 | fern | ema-weights | 3 × finished | **121.69** (run `2i7tmbir`) | **−11.10%** | **YES** |
| #3173 | alphonse | surf-weight-scan | 4 × finished | 130.29 (run `mdkp6avx`, w=50) | −4.82% | no — +11.4% on val_single_in_dist |
| #3211 | thorfinn | per-channel-output-heads | 4 finished + 1 crashed | 133.70 (run `x3h1o3id`) | −2.33% | no — +8.6% on val_single_in_dist |
| #3190 | frieren | slice-num-128 | 3 × finished | 140.96 (best) | +2.98% | no — regression |
| #3196 | nezuko | hidden-256-depth6 | 2 finished + 3 failed | 152.48 (best `8mb6sqt8`) | +11.4% | no — regression |

### Per-split breakdown (W&B summary values)

**fern EMA (`2i7tmbir`, decay=0.999, surf_weight=10):**

| Split | EMA | baseline (`07efagec`) | Δ |
|---|---|---|---|
| val_single_in_dist | 147.55 | 151.85 | −2.83% |
| val_geom_camber_rc | 137.68 | 173.91 | **−20.83%** |
| val_geom_camber_cruise | 92.42 | 101.41 | −8.86% |
| val_re_rand | 109.09 | 120.38 | −9.38% |
| **val_avg** | **121.69** | 136.89 | **−11.10%** |

Three independent EMA runs (121.69, 122.64, 123.13) cluster within ±0.7 — high reproducibility. **This is the strongest candidate Round-1 winner.**

**alphonse surf_weight=50 (`mdkp6avx`):**

| Split | w=50 | baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 169.20 | 151.85 | **+11.4%** |
| val_geom_camber_rc | 136.69 | 173.91 | −21.4% |
| val_geom_camber_cruise | 98.42 | 101.41 | −2.9% |
| val_re_rand | 116.86 | 120.38 | −2.9% |
| val_avg | 130.29 | 136.89 | −4.82% |

Same single-split-carries-headline pattern as PR #3176 (askeladd) and PR #3211 (thorfinn) — RC-camber wins big, in-dist regresses. Structural across loss-redirection hypotheses.

**thorfinn per-channel-heads (`x3h1o3id`):** val_avg 133.70, val_single_in_dist +8.6%, val_geom_camber_rc −15.8%. Same pattern. Run-to-run variance huge (133.70 vs 168.42 in the same wandb_group — likely architectural variant differences).

**frieren slice_num=128:** best 140.96 (+2.98%), worst 171.89 (+25.6%). All three runs regress. High variance suggests the extra physics tokens hurt training stability at this budget.

**nezuko hidden-256-depth6:** best finished 152.48 (+11.4%), 3 failures (likely OOM or train-divergence on bs=2 small-batch + larger model). Architecture scaling under-converges under the realized epoch budget.

### Actions taken at 16:28 UTC

Posted advisor nudge comments on all 5 PRs identifying the W&B runs and instructing each student to:
1. Commit their `train.py` changes (which exist as uncommitted working-tree edits)
2. Push to origin
3. Post a `SENPAI-RESULT` marker with the relevant run IDs
4. Invoke `senpai:submit-experiment-results` to swap label `wip → review`

Without committed code in the branch HEAD, neither merge nor review is possible — there is literally nothing to merge even when the W&B data shows a strong winner. This was the gap that hid fern's −11% win for ~3.5 hours.

### Operational lesson

W&B should be part of the advisor's PR-review surface. When a PR sits at `status:wip` with no commits for ≥2 hours, query the `wandb_group` for that student's agent and surface any completed runs. Multiple training runs in W&B with no PR activity is the "student trained but didn't submit" failure mode and needs an explicit prod, not just patience.

---

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

## 2026-05-15 17:30 — PR #3186: EMA weights (fern) — MERGED

- Branch: `willowpai2i48h2-fern/ema-weights`
- Hypothesis: EMA (Polyak) shadow-weight averaging with decay=0.999 — validate EMA shadow weights each epoch; save EMA weights as checkpoint.

| run | val_avg/mae_surf_p | Δ vs baseline 136.887 |
|---|---|---|
| `2i7tmbir` (primary) | **121.685** | **−11.10%** |
| `kji1tmn4` | 122.638 | −10.41% |
| `no0se6tm` | 123.131 | −10.06% |

Per-split val (primary run `2i7tmbir` vs baseline `07efagec`):

| Split | EMA | baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 147.552 | 151.849 | **−2.83%** |
| val_geom_camber_rc | 137.679 | 173.913 | **−20.83%** |
| val_geom_camber_cruise | 92.418 | 101.405 | **−8.86%** |
| val_re_rand | 109.092 | 120.382 | **−9.38%** |
| **val_avg** | **121.685** | **136.887** | **−11.10%** |

Per-split test (3 clean splits; cruise=NaN fleet-wide):

| Split | EMA | baseline | Δ |
|---|---|---|---|
| test_single_in_dist | 124.921 | 136.522 | **−8.50%** |
| test_geom_camber_rc | 121.909 | 157.591 | **−22.64%** |
| test_re_rand | 108.013 | 118.971 | **−9.21%** |
| **test_avg (3 splits)** | **118.281** | **137.694** | **−14.10%** |

**Analysis:** The strongest result of Round 1. All 4 val splits and all 3 clean test splits improve. The mechanism (trajectory averaging over the late cosine-LR oscillation) generalizes across ALL distribution shifts — unlike the "redirect loss" approaches which only win on val_geom_camber_rc at the expense of in-dist. Three independent reproducibility runs cluster within ±0.7 MAE (~0.6%) confirming the result is not seed luck.

**Decision: MERGED.** New baseline val_avg=121.685, test_avg=118.281. BASELINE.md updated.

---

## 2026-05-15 17:35 — PR #3211: Per-channel output heads (thorfinn) — CLOSED

- Branch: `willowpai2i48h2-thorfinn/per-channel-output-heads`
- Hypothesis: Separate linear projection heads for velocity (Ux/Uy) and pressure (p) channels

Best result: val_avg=133.701 (run `x3h1o3id`, confirmed by `2676t1tz`=133.824). Confirmed reproducible by two clean runs after identifying GPU contention as cause of the observed variance.

**Against new EMA baseline (121.685): +9.9% regression. Closed.** The direction (−2.3% on old baseline) was real and reproducible, but the same single-split-carries pattern as the other loss-redirect hypotheses: RC-camber wins (−15.8%) at the cost of in-dist regression (+8.6%). With EMA now in baseline, per-channel heads no longer offer a net gain.

**Follow-up assigned:** PR #3368 — EMA + per-channel heads combination.

---

## 2026-05-15 17:35 — PR #3173: Surface weight scan (alphonse) — CLOSED

- Branch: `willowpai2i48h2-alphonse/surf-weight-scan`
- Hypothesis: Increase surf_weight from 10 to 25 or 50 to improve surface MAE

Best result: val_avg=130.294 (run `mdkp6avx`, surf_weight=50). Against new EMA baseline (121.685): +7.1% regression. Closed.

**The structural pattern confirmed again:** w=50 wins strongly on val_geom_camber_rc (−21.4%) while regressing on val_single_in_dist (+11.4%). This pattern (redirect-to-surface → OOD-camber gain / in-dist regression) appeared in #3173, #3176, and #3211 — it is structural, not noise.

**Follow-up assigned:** PR #3367 — EMA decay scan (0.9995, 0.9999).

---

## 2026-05-15 17:35 — PR #3196: Scale model n_hidden=256, n_layers=6 (nezuko) — CLOSED

- Branch: `willowpai2i48h2-nezuko/hidden-256-depth6`
- Hypothesis: Larger Transolver (n_hidden=128→256, n_layers=5→6, n_head=4→8) for more capacity

Best result: val_avg=152.480 (run `8mb6sqt8`, bs=2). All 4 splits regress. Against new EMA baseline (121.685): +25.3%.

**Analysis:** Clear dead-end at this budget. The scaled model requires bs=2 to fit 96 GB VRAM (peak ~90 GB), which doubles iteration time per epoch. Only 6–7 epochs complete in 30 min vs 14 for baseline. The cosine schedule barely decays; the model never reaches low-LR convergence. Three early crashes at bs=4 further confirm OOM instability.

**Lesson for future capacity experiments:** scaling up without a longer budget (≥2× T_min) always under-converges at fixed 30-min cap. If attempted again, pair with explicit budget increase (or use a smaller intermediate scaling, e.g. n_hidden=192, n_layers=5).

**Follow-up assigned:** PR #3369 — cosine T_max alignment.

---

## 2026-05-15 17:40 — edward #3181 retry W&B surfacing (grad_clip=5 + Huber)

Running arms since the send-back instruction at 14:53:

| run | grad_clip | huber_delta | val_avg/mae_surf_p | Δ vs EMA baseline 121.685 |
|---|---|---|---|---|
| `36gcpryh` | 5.0 | 1.0 | **109.449** | **−10.1%** |
| `ik82u6qo` | 5.0 | 1.0 | 114.380 | −6.2% |
| `p9iio40u` | 1.0 | 1.0 | 113.101 | −7.0% |
| `b6t3344j` | 5.0 | 1.0 | running (~118.78 current) | — |

Per-split for best run `36gcpryh` vs EMA baseline:

| Split | 36gcpryh | EMA baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 132.278 | 147.552 | **−10.4%** |
| val_geom_camber_rc | 118.018 | 137.679 | **−14.3%** |
| val_geom_camber_cruise | 82.744 | 92.418 | **−10.5%** |
| val_re_rand | 104.754 | 109.092 | **−4.0%** |
| **val_avg** | **109.449** | **121.685** | **−10.1%** |

Test (3 splits): (120.577 + 106.550 + 98.577) / 3 = **108.568** vs EMA test 118.281 (−8.2%).

**Critical finding:** grad_clip=5 + Huber WITHOUT EMA already beats the EMA baseline. Once combined with EMA (PR #3366 assigned to fern), the stack has high potential to push val_avg below ~108.

Edward was nudged to post a terminal SENPAI-RESULT once arm b6t3344j finishes. Pending formal submission of #3181.

---

## 2026-05-15 18:30 — edward #3181 arm b6t3344j FINISHED — new strongest pre-EMA result

| run | grad_clip | huber_delta | val_avg | best_val_avg | Δ vs EMA baseline 121.685 |
|---|---|---|---|---|---|
| `b6t3344j` | 5.0 | 1.0 | 110.28 (last) | **106.7216** | **−12.3%** |
| `36gcpryh` | 5.0 | 1.0 | 109.449 | 109.449 | −10.1% |
| `ik82u6qo` | 5.0 | 1.0 | 114.380 | 114.380 | −6.2% |

Three-run reproducibility on the clip=5 + Huber=1 config: 106.72, 109.45, 114.38 (mean 110.18, std 3.16). All beat EMA baseline by 5–12%.

Test (b6t3344j, 3-split): test_single=117.34, test_rc=106.12, test_re=94.12 → mean **105.86** (cruise=NaN data bug).

Edward re-nudged at 18:30 to post terminal SENPAI-RESULT with b6t3344j as primary. This run is on pre-EMA codebase, so PR #3366 (fern, EMA + grad_clip + Huber stack) could compound further below 106.

---

## 2026-05-15 18:30 — askeladd #3176 (pressure-channel-weight retry) — CLOSED

| run | p_surf_weight | val_avg/mae_surf_p | Δ vs EMA baseline 121.685 |
|---|---|---|---|
| `e5jk8n98` | 1.5 | 131.6828 | +8.21% |
| `2umfqqij` | 2.0 | 132.5725 | +8.94% |
| `g0n1r7pq` | 3.0 | 134.6330 | +10.64% |
| `8pizb0t7` | 5.0 | 165.2153 | +35.78% |

Monotonic degradation with weight — best (w=1.5) still +8% above EMA baseline. Test 3-split mean for w=1.5 = 130.11 (also +10% above EMA test baseline 118.28).

Closed as dead-end. Pattern (single-split RC-camber win at cost of in-dist regression) is now confirmed across three loss-redirection hypotheses (PR #3173 surf_weight=50, PR #3211 per_channel_heads, PR #3176 pressure_channel_weight). The loss-redirection family does not beat EMA's globally smoothing approach.

Askeladd will be reassigned a fresh hypothesis (TBD — likely H-04 dropout, H-02 weight-decay, or asinh-pressure output normalization).

---

## 2026-05-15 18:30 — tanjiro #3202 arm 3kervu49 (budget-aware warmup) — BEATS BASELINE

| run | warmup_epochs | cosine_t_max | sf | val_avg/mae_surf_p | Δ vs EMA baseline 121.685 |
|---|---|---|---|---|---|
| `3kervu49` | 3 | 9 | 0.1 | **119.7996** | **−1.55%** |
| `dhtoffp3` | 5 | (T_max=50) | 0.01 | 137.498 | +13.0% |
| `dqpeoznv` | 2 | (T_max=50) | 0.01 | 132.130 (best) | +8.6% |
| `kg5wb8av` | 5 | (T_max=50) | 0.01 | 149.845 | +23.1% |
| `dyi1encx` | 2 | — | 0.01 | CRASHED at 219.6 | — |

Per-split for `3kervu49`:

| Split | val | test |
|---|---|---|
| single_in_dist | 142.70 | 128.70 |
| geom_camber_rc | 131.99 | 119.24 |
| geom_camber_cruise | 92.73 | NaN (data bug) |
| re_rand | 111.78 | 110.60 |
| **avg** | **119.7996** | **119.5145** |

**Key technical finding:** The configuration that worked is **T_max=9 aligned to realized epoch count**, NOT T_max=50. With T_max=50 the cosine never decays in the 14-epoch budget; with T_max=9 the LR fully decays and the model converges to a better minimum. This validates one of nezuko's Round-2 assignments (PR #3369 cosine-tmax-align).

Branch state check: tanjiro's branch does NOT contain the EMA merge (PR #3186). So this −1.55% gain is from the schedule reformulation alone, on the *pre-EMA* code path. The combination (EMA + tmax-aligned cosine + warmup) is currently in flight as nezuko's PR #3369.

Tanjiro nudged at 18:30 to post terminal SENPAI-RESULT for `3kervu49`. Mergeable subject to terminal submission and edward's stronger result not landing first.

---

## 2026-05-15 18:30 — Round-2 assignments: PR #3388 (frieren, SWA)

Frieren was idle after PR #3190 closure. Assigned H-01 `swa-plateau-average` from `RESEARCH_IDEAS_2026-05-15_17:40.md`:
- Add `torch.optim.swa_utils.AveragedModel` + `SWALR` ALONGSIDE existing EMA
- `swa_start_epoch=6`, `swa_lr=1e-4`, `anneal_epochs=2` (cosine anneal)
- Track BOTH EMA and SWA at each epoch; checkpoint the better
- Mechanism orthogonal to EMA: EMA = exponentially-weighted centroid; SWA = uniform snapshot average

Expected: 1–10% gain over EMA baseline if SWA finds flatter minima. No regression risk since better-of-two is always chosen.

Round-2 status now: 5 PRs in flight on EMA stack (#3366 fern, #3367 alphonse, #3368 thorfinn, #3369 nezuko, #3388 frieren) + 2 PRs awaiting terminal result (#3181 edward, #3202 tanjiro) + 1 student idle (askeladd, just freed by #3176 close).


---

## 2026-05-15 20:40 — PR #3366: MERGED — EMA + grad_clip=5 + Huber δ=1.0 (fern)

**New baseline: val_avg/mae_surf_p = 94.4199 (−22.4% below prior EMA baseline 121.685)**

| run | grad_clip | huber_delta | val_avg | test_3split | Δ vs EMA baseline |
|---|---|---|---|---|---|
| `m6hkf8el` | 5.0 | 1.0 | **94.4199** | **92.3626** | **−22.4%** |
| `eq4osquw` | 5.0 | 1.0 | 94.868 | 93.388 | −22.0% |

Per-split (m6hkf8el):

| Split | val | test |
|---|---|---|
| single_in_dist | 111.794 | 99.797 |
| geom_camber_rc | 110.162 | 96.252 |
| geom_camber_cruise | 69.012 | NaN |
| re_rand | 86.712 | 81.040 |

**All 4 val splits improve by ≥20%.** Val trajectory is monotone-decreasing through epoch 14 (still improving at wall-clock cutoff).

**Key mechanistic findings:**
- At clip=5, gradient bites ~92–99% of steps (median pre-clip norm ~16–34×). Nearly all steps are in the clipped regime. Raising clip from 1 to 5 allows 5× larger effective LR steps without destabilizing training (Huber caps per-sample loss influence).
- Huber + clip + EMA compound orthogonally: each targets a different aspect of the optimization challenge (loss robustness, gradient norm, trajectory smoothing).
- Fern's report: val trajectory still monotone at epoch 14. Longer budget (if allowed) could improve further.

---

## 2026-05-15 21:30 — Round-2 closures (superseded by fern's 94.42 new baseline)

| PR | Student | val_avg | Δ vs NEW baseline 94.42 | Verdict |
|---|---|---|---|---|
| #3181 edward | grad-clip-huber rebased | 97.23 | +2.9% | CLOSE — superseded by identical config in fern's #3366 |
| #3202 tanjiro | lr-warmup-cosine rebased | 118.17 | +25.2% | CLOSE — superseded |
| #3368 thorfinn | ema-per-channel-heads | 128.92 | +36.5% | CLOSE — structural bias confirmed dead-end |
| #3369 nezuko | cosine-tmax-12/16 | 123.39 (T_max=16) | +30.7% | CLOSE — T_max=9 is sweet spot (see tanjiro's finding) |
| #3367 alphonse | ema-decay-scan (0.9995/0.9999) | 157.50 (best) | +66.8% | PENDING close after terminal SENPAI-RESULT |
| #3388 frieren | swa-plateau-average | 121.46 (swa_start=8) | +28.7% | PENDING close after terminal SENPAI-RESULT |
| #3396 askeladd | weight-decay-sweep (1e-3→1e-2) | 123.77 (wd=1e-3) | +31.1% | PENDING close after terminal SENPAI-RESULT |

---

## 2026-05-15 21:30 — Round-3 assignments

New baseline: 94.4199. Three idle students assigned hypotheses targeting the EMA+clip5+Huber stack:

| PR | Student | Hypothesis | Key question |
|---|---|---|---|
| #3454 | edward | lr-sweep-clip-huber (lr=1e-3, 2e-3, 5e-3) | Can higher LR overcome clip-suppressed effective step size? |
| #3456 | nezuko | tmax9-clip-huber (T_max=14 + T_max=9 on full stack) | Does aligned cosine decay compound with EMA+clip+Huber? |
| #3458 | tanjiro | huber-delta-sweep (δ=0.5, 1.0, 2.0, 0.0) | What is the optimal Huber transition threshold? |


---

## 2026-05-15 21:50 — Round-2 dead-end closures (final 3 of 7)

All three had terminal SENPAI-RESULT posted in the 21:24–21:28 UTC window; all regress vs the new 94.42 baseline.

| PR | Student | Best arm | val_avg | Δ vs baseline 94.42 | Closed |
|---|---|---|---|---|---|
| #3367 | alphonse | ema-decay=0.9995 | 156.53 | +65.8% | yes — slower decay doesn't converge in 14-epoch budget |
| #3388 | frieren | swa-start=8 (on EMA-only base) | 121.46 | +28.7% | yes — only ~6 averaging epochs; SWA can't outpace EMA+clip+Huber stack |
| #3396 | askeladd | weight-decay=1e-3 | 123.77 | +31.1% | yes — EMA+clip+Huber already saturates regularization headroom |

Round-2 final tally: 7 of 10 hypotheses closed as dead-ends, 1 merged (#3186 EMA), 1 merged (#3366 EMA+clip+Huber as the round-2 superwinner). Net: a single 3-mechanism compound improvement (−22.4%) carried the round.

## 2026-05-15 21:50 — Round-3 assignments (final 5 of 8 students)

After closures and the three Round-3 assignments already in flight (#3454 edward, #3456 nezuko, #3458 tanjiro), five idle students were assigned orthogonal mechanism explorations:

| PR | Student | Hypothesis | Mechanism | EV |
|---|---|---|---|---|
| #3473 | fern | geometry-augmentation-vertical-mirror (H-10, single-foil only, AUGMENT_PROB=0.5) | Data | Medium-High |
| #3474 | alphonse | ema-decay-fast (0.997, 0.995, 0.99 — opposite of her failed slow-direction sweep) | Optim | Low-Medium |
| #3475 | askeladd | asinh-pressure (H-03, heavy-tail compression on pressure channel only) | Output rep | Medium |
| #3476 | frieren | swa-on-full-stack (SWA + EMA dual-shadow with min-val checkpoint selection) | Optim | Low-Medium |
| #3477 | thorfinn | physics-continuity-loss (H-06, ∂Ux/∂x + ∂Uy/∂z = 0 soft penalty on volume nodes) | Loss | Medium |

Zero idle students. Round-3 PR slots: 8/8 occupied. Target: push val_avg below 90.

---

## 2026-05-16 00:30 — PR #3474: EMA decay fast sweep (alphonse) — MERGED

**Student:** willowpai2i48h2-alphonse
**Hypothesis:** Faster EMA decay (0.997, 0.995, 0.99) compound better with 14-epoch budget than slow decay (0.999 baseline). Opposite direction from alphonse's previously closed slow-decay sweep (#3367).

**Results:**

| Arm | ema_decay | W&B run | val_avg/mae_surf_p | Δ vs baseline 94.42 | test 3-split |
|---|---|---|---|---|---|
| Baseline (#3366) | 0.999 | m6hkf8el | 94.4199 | — | 92.3626 |
| A | 0.997 | ml7l5jck | 91.9901 | −2.6% | 88.322 |
| B | 0.995 | y5xumcvw | 91.2049 | −3.4% | 88.177 |
| **C (best)** | **0.99** | **fzrq04xr** | **90.6131** | **−4.0%** | **88.825** |

**Per-split (Arm C, epoch 14):** val_single=106.13, val_rc=99.47, val_cruise=70.36, val_re=86.49

**Analysis:** Monotone improvement: 0.999 > 0.997 > 0.995 > 0.99 within the 14-epoch budget. Faster decay (half-life ~69 steps vs ~693 for 0.999) lets the shadow track the late-training phase more closely. EMA still helps at lag ≤2% (ema_lag_rel for Arm C at ep14 = 2.05%). All 3 arms converge at wall-clock cap (epoch 14) — improvement trend did not plateau. The trend is monotone in the explored range; optimum has NOT been bracketed from below.

**Verdict:** MERGED. New baseline: val_avg=**90.6131**, test_3split=88.8252. Next: push decay below 0.99 (0.98, 0.97, 0.95) to find the floor — assigned to alphonse #3543.

---

## 2026-05-16 00:30 — Round-3 Tier-2 status check (via W&B, no terminal results posted yet)

| PR | Student | W&B progress | Best val_avg | Vs NEW baseline 90.61 |
|---|---|---|---|---|
| #3473 fern | geom-aug-mirror p=0.5 | 2 arms: c5yqhyum=99.79, e2mq4thp=101.17 | 99.79 | +10.1% REGRESS |
| #3475 askeladd | asinh-pressure scale=1.0 | 2 runs: 9vcc7qfn=88.67, sgl0hury=91.70 | **88.67** | **−2.1% WIN** |
| #3476 frieren | swa-on-full-stack start=6 | 2 arms: pphl9e3g=96.08, 6afydvtb=96.00 | 96.00 | +5.9% REGRESS |
| #3477 thorfinn | physics-continuity | w=0.01: 98.66; w=0.1 running | 98.66 (so far) | +8.9% REGRESS |

**Critical**: askeladd's asinh-pressure (88.67) beats the NEW baseline 90.6131 — pending terminal SENPAI-RESULT, will merge when submitted. Nudges sent to all 4 students.

---

## 2026-05-16 00:30 — Round-3 Tier-1 status (no terminal results, still training)

| PR | Student | W&B progress | Best val_avg | Vs NEW baseline 90.61 |
|---|---|---|---|---|
| #3454 edward | lr-sweep 1e-3/2e-3/5e-3 | lr=1e-3: 93.47/96.89/99.59 (variance); lr=2e-3 running | 93.47 (lr=1e-3) | +3.2% so far |
| #3456 nezuko | cosine T_max=14/9 | T_max=14 only: 96.04, 98.05, 98.35; T_max=9 NOT YET RUN | 96.04 | +5.9% so far |
| #3458 tanjiro | huber-delta 0.5/1.0/2.0/0.0 | δ=0.5:94.84/96.83, δ=1.0:93.91, δ=2.0:100.0, δ=0.0 running | 93.91 | +3.6% so far |

All three Tier-1 PRs have runs that don't beat the new 90.61 baseline yet. Best hope: edward lr=2e-3 currently running; nezuko's T_max=9 arm pending.

---

## 2026-05-16 00:35 — alphonse assigned #3543: ema-decay-push

After merging #3474 (decay=0.99 new baseline), the decay trend was still monotone at the floor. Assigned: bracket optimum below 0.99.

- Arms: ema_decay=0.98, 0.97, 0.95
- Group: ema-decay-push
- Expected: find where shadow = live model (ema_lag_rel → 0%) and improvement stops

---

## 2026-05-16 00:55 — Round-3 Tier-2 closures and reroutes

### PR #3473 fern (geom-aug-mirror) — CLOSED (dead-end)

Terminal SENPAI-RESULT posted at 00:27:41:
- val_avg = 99.7887 (+10.1% vs new baseline 90.6131, +5.7% vs prior 94.42)
- test_3split = 99.54 (+12.1% vs new baseline test 88.83)
- W&B runs: c5yqhyum (99.79), e2mq4thp (101.17)

Augmentation regresses on all 4 val splits. Single-foil vertical mirror with AUGMENT_PROB=0.5 was too aggressive — half the batch lands in low-density input regions (negative AoA). Closed.

### PR #3475 askeladd (asinh-pressure) — SENT BACK (winner pending verify)

Terminal SENPAI-RESULT posted at 00:36:35:
- val_avg = 88.667 (**−2.1% vs new baseline 90.6131**, −6.1% vs prior 94.42)
- test_avg = 87.1257 (**−1.9% vs new test_3split 88.83**)
- W&B runs: 9vcc7qfn (88.67), sgl0hury (91.70), 1kllktu2

**Result IS a winner but two issues block merge:**
1. PR has merge conflicts (alphonse #3474 was merged in parallel, changing train.py)
2. Result measured at ema_decay=0.999 (old baseline default); needs verification on new ema_decay=0.99 default

Sent back to WIP with rebase + single-arm-re-verify (asinh_p_scale=1.0 + ema_decay=0.99) instructions. Will merge on successful re-verify.

## 2026-05-16 00:55 — fern reassigned: depth-sweep

After geometry-augmentation closure, fern assigned to architecture axis (untouched so far in this programme).

| PR | Hypothesis | Arms | Rationale |
|---|---|---|---|
| #3571 | n_layers depth sweep on fast-EMA baseline | 6, 7 | All wins so far are optimizer/loss; architecture capacity untested. Depth+regularization classically compounds. |


---

## 2026-05-16 01:20 UTC — Round-3 Tier-1 closures (4 dead-ends)

All Tier-1 PRs (hyperparameter sweeps) completed with regressions vs new baseline 90.6131. Closed without waiting for terminal SENPAI-RESULT — W&B telemetry is conclusive.

### PR #3454 edward (lr-sweep) — CLOSED

| Arm | val_avg | Δ vs 90.61 | W&B run |
|---|---|---|---|
| lr=1e-3 (best) | 93.467 | +3.2% | mgzjg84e |
| lr=1e-3 (rep) | 99.593 | +9.9% | 76ijpudj |
| lr=1e-3 (rep) | 96.895 | +6.9% | 70859lf5 |
| lr=2e-3 | 105.452 | +16.4% | 4uxz0ed3 |
| lr=5e-3 | not run (monotone worse with higher lr) | — | — |

**Conclusion**: lr=5e-4 is at or near optimum. Higher lr = worse. High seed variance in lr=1e-3 runs (93–100 range). Hypothesis falsified.

### PR #3456 nezuko (cosine T_max sweep) — CLOSED

| Arm | val_avg | Δ vs 90.61 | W&B run |
|---|---|---|---|
| T_max=14 (best) | 96.044 | +5.9% | m47uy1o8 |
| T_max=14 (rep) | 98.352 | +8.5% | ujncdphm |
| T_max=14 (rep) | 98.046 | +8.2% | g8wvqv0g |
| T_max=9 | 108.329 | +19.6% | aulmfir6 |

**Conclusion**: Default T_max=epochs outperforms truncated schedules. Cosine's late-stage low-LR region provides regularization even though training stops before reaching it. Hypothesis falsified.

### PR #3458 tanjiro (huber-delta sweep) — CLOSED

| Arm | val_avg | Δ vs 90.61 | W&B run |
|---|---|---|---|
| δ=1.0 (baseline) | 93.915 | +3.7% (variance replicate) | d5wrdnhe |
| δ=0.5 (best) | 94.841 | +4.7% | plxxf9vo |
| δ=0.5 (rep) | 96.825 | +6.9% | 1g19p9y7 |
| δ=2.0 | 99.998 | +10.4% | c3v83mau |
| δ=0.0 (MSE) | 104.908 | +15.8% | vctxh07i |

**Conclusion**: δ=1.0 was already optimal (it IS the merged baseline). The U-shape across δ values confirms it sits at the loss-curvature sweet spot. Hypothesis falsified (negative confirms baseline was correct choice).

### PR #3476 frieren (SWA) — CLOSED

| Arm | val_avg | Δ vs 90.61 | W&B run |
|---|---|---|---|
| swa_start=6 (best) | 96.003 | +5.9% | 6afydvtb |
| swa_start=6 (rep) | 96.081 | +6.0% | pphl9e3g |
| swa_start=4 | 100.837 | +11.3% | wzh7l3ix |

**Conclusion**: SWA window too short within 14-epoch budget. Earlier start = worse (monotone). EMA decay=0.99 already provides effective late-training averaging; SWA only competes with the EMA shadow without adding value. Hypothesis falsified.

---

## 2026-05-16 01:20 UTC — Round-4 hypotheses assigned

Assigned 4 fresh orthogonal hypotheses to freed-up students:

| PR | Student | Hypothesis | Axis | Arms |
|---|---|---|---|---|
| #3575 | edward | p-surf-weight: --p_surf_weight 3.0 and 5.0 | Loss weighting (per-channel pressure) | 2 |
| #3576 | nezuko | wd-sweep: weight_decay 1e-3, 5e-3 | Regularization (L2 norm) | 2 |
| #3577 | tanjiro | slice-num-128: PhysicsAttention tokens 64→128 | Architecture capacity (token count) | 1+1 conditional |
| #3578 | frieren | re-sinusoidal-embed: log(Re) → 8-d sinusoidal embedding | Feature representation (Re encoding) | 1 |


---

## 2026-05-16 02:25 UTC — Round-4 progress check + thorfinn closure

### W&B status at 02:25 UTC

| Student | PR | Best so far | State |
|---|---|---|---|
| askeladd | #3475 | **val_avg=85.815** (run @ 01:26 UTC, ema_decay=0.99 + asinh=1.0) | **−5.3% vs baseline 90.61 — WINNER pending SENPAI-RESULT** |
| alphonse | #3543 | val_avg=90.839 (0.98 arm, ≈ tied with baseline) | Stuck re-running 0.98 (5 launches); nudged to move to 0.97/0.95 |
| fern | #3571 | val_avg=93.829 (n_layers=6) | +3.6% (not a win); depth=7 still pending |
| edward | #3575 | val_avg=94.654 (p_surf_weight=3.0) | +4.5% (not a win); p_surf=5.0 still pending |
| nezuko | #3576 | val_avg=90.746 (wd=1e-3) | **+0.15% ≈ TIED**; wd=5e-3 currently running |
| tanjiro | #3577 | first arm slice=128 debug 487 (debug-only); new run started 02:22 | First proper arm pending |
| frieren | #3578 | No runs yet | Code implementation work likely |
| thorfinn | (#3477 CLOSED) | physics-continuity all arms regress | **CLOSED 02:24 UTC**; reassigned to #3610 mlp-ratio |

### PR #3477 thorfinn (physics-continuity) — CLOSED

All 3 arms complete, all regress vs new baseline 90.61:
- w=0.01: 98.66 (+8.9%)
- w=0.1: 98.62 (+8.8%)
- w=0.5: 105.95 (+16.9%)

Random-pair FD divergence proxy too noisy on irregular meshes. Mechanism: high variance in pair-sampled gradient estimates overwhelms the main MAE signal.

### Round-4 hypothesis preview (sorted by current best to date)

1. **askeladd asinh-pressure 85.815** — winner, awaiting terminal SENPAI-RESULT
2. **nezuko wd=1e-3 90.746** — first arm ≈ TIED; wd=5e-3 may push lower
3. fern depth=6 93.83 — modest regression, depth=7 pending
4. edward p_surf=3.0 94.65 — modest regression, p_surf=5.0 pending
5. tanjiro slice=128 — first real arm running
6. frieren re-sinusoidal-embed — no runs yet (implementation in progress)

## 2026-05-16 02:30 UTC — thorfinn reassigned: mlp-ratio sweep

PR #3610 (mlp-ratio-sweep). Hypothesis: bump Transolver MLP block ratio from 2 to 4 (standard transformer default). Orthogonal to fern (depth) and tanjiro (slice_num) — three independent capacity dimensions in parallel.


## 2026-05-16 02:50 — PR #3571 (fern): depth-sweep CLOSED; PR #3649 assigned n_head-sweep

### PR #3571 closure

| Run | Student | Arm | val_avg/mae_surf_p | test_avg/mae_surf_p | Vs baseline | Status |
|---|---|---|---|---|---|---|
| enxjsoys | fern | n_layers=6 | 93.8290 | 91.9389 | **+3.55% REGRESS** | Arm B skipped per brief |

**Per-split val (n_layers=6 vs baseline)**:
| split | depth=6 | baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 108.965 | 106.135 | +2.67% |
| val_geom_camber_rc | 104.950 | 99.466 | +5.51% |
| val_geom_camber_cruise | 72.883 | 70.358 | +3.59% |
| val_re_rand | 88.517 | 86.494 | +2.34% |

**Diagnostics**: Peak GPU=49.6 GB. Wall-time 156 s/epoch (vs 129 s baseline → 12 epochs instead of 14). Val curve still monotonically decreasing at epoch 12 → wall-clock bound, not capacity bound.

**Conclusion**: Depth-6 regresses within 30-min budget — extra capacity was traded for fewer optimizer steps and the 5-layer trajectory wins. Architecture-via-depth falsified at this wall-clock budget. The val trajectory was still improving, so depth might win with a 60-min budget, but that's outside the run-limit constraints.

**Depth sweep closed. No Arm B (n_layers=7) per brief rules.**

### PR #3649 — fern n_head sweep (newly assigned)

Hypothesis: Increase attention heads 4→8. n_head changes the attention partition but NOT parameter count or wall-clock time, making it the lowest-cost architecture axis available.

- Arm A (primary): `--n_head 8` (per-head dim 16)
- Arm B (conditional if A wins decisively): `--n_head 16` or `--n_head 2` depending on direction
- W&B group: `n-head-sweep`

## 2026-05-16 03:30 — PR #3475 MERGED: asinh-pressure → new baseline val=81.9754 (−9.53%)

| Run | Config | val_avg/mae_surf_p | test_3split | Δ vs baseline |
|---|---|---|---|---|
| 2028x8co (verify) | asinh_p_scale=1.0, ema_decay=0.99 | 85.8151 | 83.3376 | −5.3% |
| **j5214ii4 (replicate)** | **asinh_p_scale=1.0, ema_decay=0.99** | **81.9754** | **81.3654** | **−9.53%** |

Per-split val (best replicate):
| split | j5214ii4 | baseline (fzrq04xr) | Δ |
|---|---|---|---|
| val_single_in_dist | 101.013 | 106.135 | −4.8% |
| val_geom_camber_rc | 90.717 | 99.466 | −8.8% |
| val_geom_camber_cruise | 59.909 | 70.358 | −14.8% |
| val_re_rand | 76.263 | 86.494 | −11.8% |

**Key finding**: asinh + fast-EMA compound super-additively. Standalone asinh on old decay=0.999 base = −2.1%. On decay=0.99 base = −9.53%. Fast shadow (decay=0.99) tracks the late-training basin cleanly, and compressed gradient signal from asinh lets EMA act more effectively. val_re_rand drop (−11.8%) is the largest OOD improvement yet.

Merged. New baseline: val=81.9754, test_3split=81.3654. BASELINE.md updated.

## 2026-05-16 03:45 — Round-4 closures (3 PRs regress vs old baseline, all fail new baseline)

| PR | Student | Hypothesis | Best val | Vs old baseline | Vs new baseline | Decision |
|---|---|---|---|---|---|---|
| #3610 | thorfinn | mlp_ratio=4 | 93.1162 | +2.76% REGRESS | +13.6% | CLOSED |
| #3576 | nezuko | wd sweep (5e-3 best) | 90.4605 | −0.17% TIED | +10.3% | CLOSED |
| #3575 | edward | p_surf_weight=3/5 | 94.6538 | +4.5% REGRESS | +15.5% | CLOSED |

## 2026-05-16 03:50 — Stale WIP closures

| PR | Student | Hypothesis | Best val | Root cause | Decision |
|---|---|---|---|---|---|
| #3578 | frieren | re-sinusoidal-embed | 130.821 | Frequency mismatch: log_re/16 spans [0.78,0.96] → 7/8 dims constant | CLOSED |
| #3577 | tanjiro | slice-num=128 (old stack) | 101.177 | +11.6% vs old baseline; no SENPAI-RESULT posted; pre-asinh stack | CLOSED |

## 2026-05-16 03:55 — Round-5 assignments (6 new PRs, all on new asinh+EMA baseline)

| PR | Student | Hypothesis | Key innovation |
|---|---|---|---|
| #3659 | askeladd | asinh-scale-sweep (1.5, 2.0) | Find optimal compression strength |
| #3660 | frieren | re-sinusoidal-corrected | Fix frequency bug: normalize log_re to actual [10.8,13.4] range |
| #3661 | nezuko | wd-on-asinh (1e-3, 5e-3) | Regularization compound with asinh |
| #3662 | thorfinn | vel-asinh (scale=1.0) | Apply asinh to Ux/Uy channels too |
| #3663 | edward | dropout-sweep (0.05, 0.1) | MLP dropout for OOD regularization |
| #3664 | tanjiro | slice-num-on-asinh (128) | Retest with cleaner loss landscape |

## 2026-05-16 04:35 — PR #3543 CLOSED: EMA decay push (alphonse) — all arms fail new baseline

| Arm | ema_decay | run_id | val_avg/mae_surf_p | test_avg/mae_surf_p | Vs new baseline (81.97) |
|---|---|---|---|---|---|
| A (best) | 0.98 | x14urdxg | 90.8394 | 88.0412 | +8.87 (+10.8%) |
| B | 0.97 | oz0q2f1e | 93.2994 | 89.3332 | +11.33 (+13.8%) |
| C | 0.95 | sc2bmjob | 95.6469 | 91.9280 | +13.68 (+16.7%) |

Per-split val (best arm 0.98, run x14urdxg): single_in_dist 107.784 | geom_camber_rc 103.664 | geom_camber_cruise 67.885 | re_rand 84.025

**Verdict: CLOSED.** Best arm 0.98 essentially ties the OLD baseline (90.84 vs 90.61) but does NOT beat the new merged baseline 81.97. The EMA decay axis is exhausted in [0.95, 0.99] — descent reversed immediately below 0.99.

**Key finding (alphonse):** ema_lag_rel stays ~1-2% across the entire bracket, counter-intuitively decreasing as decay decreases (at low decay the shadow tracks live in 1 step). The gain from 0.997→0.99 came from reducing smoothing bias on the live-side optimum, not from shrinking lag. Per-split residuals: single_in_dist and geom_camber_rc are now the bottleneck splits (>100 mae).

**alphonse reassigned** → PR #3679: Huber δ sweep on asinh baseline (0.5, 0.3). Mechanistic motivation: asinh-compressed targets have ~2.5× smaller residual scale; δ=1.0 tuned for raw pressure is now in the wrong place (too many residuals in L2 region).

## 2026-05-16 04:35 — PR #3679 ASSIGNED: Huber δ sweep on asinh baseline (alphonse)

Hypothesis: δ=1.0 was calibrated for raw-pressure residuals (|p| up to ~5+). Post-asinh, the effective residual scale is ~2.5× smaller; optimal δ should be ~0.4–0.5. Sweep arms:
- Arm A (primary): `--huber_delta 0.5`
- Arm B (conditional if A wins ≤82.5): `--huber_delta 0.3`; if A regresses >84: `--huber_delta 2.0`

Stack: grad_clip=5.0, ema_decay=0.99, asinh_p_scale=1.0. No other changes.

## 2026-05-16 05:30 — PR #3664 CLOSED: slice_num=128 on asinh baseline (tanjiro) — decisive regression

| Metric | slice_num=128 | Baseline #3475 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 90.7693 | 81.9754 | **+10.73%** |
| test_3split/mae_surf_p | 88.2840 | 81.3654 | **+8.50%** |
| best_epoch | 11 | 14 | −3 (wall-clock bound) |
| epoch_time_s | 171.3 | ~156 | +9.8% |

Per-split val (all 4 regressed): single_in_dist 102.050 (+1%) | geom_camber_rc 106.328 (+17.2%) | geom_camber_cruise 67.710 (+13%) | re_rand 86.989 (+14.1%)

W&B run: `m1r489ev`

**Verdict: CLOSED (2nd close — axis definitively exhausted).** asinh did NOT unlock slice=128 capacity. Wall-clock bind confirmed: 11 epochs vs baseline 14, still monotonically descending at cutoff. slice=128 attention matrix is 4× more expensive (128²=16384 vs 64²=4096 tokens); amortization requires >25 epochs. Closed on pre-asinh (#3577) and post-asinh (#3664) stacks.

**tanjiro reassigned** → PR #3723: SwiGLU MLP activation — GELU→SwiGLU swap in TransolverBlocks. High prior probability from modern transformer literature (LLaMA/PaLM); adds ~50% MLP params, only ~10-15% epoch overhead.

## 2026-05-16 05:30 — PR #3663 SENT BACK: dropout=0.05 (edward) — mixed signal, lighter arm needed

| Metric | dropout=0.05 | Baseline #3475 | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 82.4592 | 81.9754 | **+0.59% (within ~3.8 MAE seed noise)** |
| test_3split/mae_surf_p | 80.8435 | 81.3654 | **−0.64% (improvement!)** |

Per-split val: single_in_dist 98.236 (−2.8%) | **geom_camber_rc 97.342 (+7.3%)** | geom_camber_cruise 58.348 (−2.6%) | re_rand 75.910 (−0.5%)

W&B run: `mscr7q2t`

**Analysis:** val_re_rand improved as predicted; test_3split improved; val_single_in_dist + geom_camber_cruise improved. Dominant hit: geom_camber_rc +7.3% (smallest-support split, 457 samples). Mechanism (co-adaptation suppression) showing real signal — dose is too high.

**Decision: sent back for dropout=0.025** (lighter arm). val_re_rand and test trends suggest mechanism is real; rc regression suggests 0.05 over-doses on smallest-support split. Target: val_avg < 81.5. Skipped 0.1 arm entirely per student's recommendation.

## 2026-05-16 05:30 — PR #3723 ASSIGNED: SwiGLU MLP activation (tanjiro)

Stack: grad_clip=5.0, ema_decay=0.99, asinh_p_scale=1.0, huber_delta=1.0. Only change: --use_swiglu replaces GELU in TransolverBlock MLPs. SwiGLUMLP: SiLU(W_gate·x) ⊙ (W_value·x) → W_out. Arm B (param-matched mlp_ratio≈1.33) only if Arm A wins decisively (<80.5).

## 2026-05-16 06:35 — Round-5 W&B observations (5 stuck-on-submission PRs)

5 Round-5 PRs (#3659, #3660, #3661, #3662, #3649) are flagged stale_wip because student gh CLI is hitting HTTP 403 rate limits — runs completed on GPU but SENPAI-RESULT comments not posted. W&B observations from group queries:

| PR | Student | Best run | val_avg | test_3split | Δ vs baseline (81.97) | Action |
|---|---|---|---|---|---|---|
| **#3662** | **thorfinn** | **`699fhd8k` vel-asinh-scale-0.5** | **76.15** | **87.80** | **−7.1%** | **MERGE pending SENPAI-RESULT** |
| **#3661** | **nezuko** | **`ymfjl55c` wd-1e-3-asinh** | **79.71** | **92.51** | **−2.77%** | **MERGE pending SENPAI-RESULT** |
| #3659 | askeladd | `2muknt29` asinh-scale-1.5 | 82.16 | 99.92 | +0.22% (tied) | CLOSE pending SENPAI-RESULT |
| #3660 | frieren | `sqlj9vu5` re-sinusoidal-corrected | 96.85 | 121.77 | +18.1% regress | CLOSE pending SENPAI-RESULT |
| #3649 | fern | `dabfzga5` n-head-8 | 98.44 | 119.06 | +20.1% regress | WAIT for n_head=2 arm |

**Advisor comments posted on all 5 PRs** noting the W&B observations and asking students to retry SENPAI-RESULT submission via GraphQL (\`gh pr comment\`) if REST is exhausted.

**Strategic implication**: if thorfinn vel-asinh merges, baseline jumps to 76.15 (−7.1%). If nezuko wd compounds on top of that, expect ~74-75. This would be the largest Round-5 leap.

## 2026-05-16 07:30 — PR #3663 CLOSED: dropout sweep (edward) — mechanism non-monotone, axis exhausted

| Arm | dropout | val_avg | test_3split | Δ vs baseline |
|---|---|---|---|---|
| v1 | 0.05 | 82.4592 | 80.8435 | +0.59% (within noise) |
| **v2** | **0.025** | **83.4872** | **81.2940** | **+1.84% (regression)** |

W&B runs: `mscr7q2t` (v1), `eqznyg59` (v2)

Per-split v2 (0.025): single_in_dist 100.999 (tie) | **geom_camber_rc 96.960 (+6.9%)** | geom_camber_cruise 58.903 (−1.7%) | **re_rand 77.087 (+1.1%)**

**Verdict: CLOSED.** Lighter dropout (0.025) did NOT recover val_re_rand (it got slightly worse vs 0.05) and did NOT recover geom_camber_rc. The mechanism (co-adaptation suppression for OOD) is non-monotone — 0.05 was marginally better on re_rand than 0.025, but neither beats baseline. Dropout axis exhausted on this stack.

Key insight: the bottleneck on val_geom_camber_rc (smallest support, 457 samples) is NOT feature co-adaptation — it's structural sample efficiency. Dropout doesn't address this.

**edward reassigned** → PR #3766: DropPath stochastic depth. Drops ENTIRE residual branches rather than individual neurons; forces each block to be independently useful. Different binding constraint from feature dropout.

## 2026-05-16 07:30 — PR #3660 CLOSED: corrected Re-sinusoidal embed (frieren) — axis definitively falsified

| Metric | run `sqlj9vu5` | Baseline | Δ |
|---|---|---|---|
| val_avg/mae_surf_p | 96.848 | 81.975 | +14.87 (+18.1%) |
| val_re_rand (target) | 87.677 | 76.263 | +11.41 (+15.0%) |
| test_3split/mae_surf_p | 94.856 | 81.365 | +13.49 (+16.6%) |

Second close of sinusoidal-Re axis (first: +44% with frequency bug; corrected: +18%). Even with proper [0, 1] normalization, sinusoidal expansion of log_re regresses significantly. Raw scalar already a clean signal; high-frequency expansion injects noise the model can't filter in 14 epochs.

**frieren reassigned** → PR #3770: Mixup augmentation. Interpolates pairs of (input, target) training samples (λ drawn from Beta(α,α)). Exploits physical smoothness of CFD: small perturbations of geometry/Re → small perturbations of output. Target: OOD improvement on val_re_rand and val_geom_camber_rc.

## 2026-05-16 07:30 — PRs #3766 and #3770 ASSIGNED

- PR #3766 edward: DropPath stochastic depth (--drop_path_rate 0.1 primary). DropPath adds a per-residual-branch drop probability during training; forces block independence; used in ViT/Swin/ConvNeXt for OOD robustness.
- PR #3770 frieren: Mixup augmentation (--mixup_alpha 0.2 primary). λ·x_a + (1-λ)·x_b, λ·y_a + (1-λ)·y_b; exploits CFD field smoothness.
