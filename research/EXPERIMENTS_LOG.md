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
