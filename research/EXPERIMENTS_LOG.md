# SENPAI Research Results

Track: `charlie-pai2i-24h-r1`
Advisor branch: `icml-appendix-charlie-pai2i-24h-r1`

## 2026-05-15 13:34 — PR #3130: Wider Transolver: n_hidden 128->192, n_head 4->6 — MERGED

- **Student branch**: `charliepai2i24h1-edward/wider-h192-h6`
- **Hypothesis**: Baseline at n_hidden=128 / n_head=4 (~0.65M params) is under-capacity for 74K–242K-node meshes across 3 domains; widening to n_hidden=192 / n_head=6 (dim_head=32 unchanged) should give monotonic capacity gains.
- **Verdict**: MERGED. First measured reference on this advisor branch.

### Results

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p (primary, best)** | **166.5037** (epoch 8) |
| test_avg/mae_surf_p | NaN (cruise test pressure overflowed; other 3 splits avg 166.58) |
| n_params | 1,447,521 (1.45 M) |
| peak_memory_gb | 63.0 / 96 |
| epochs completed | 9 of 50 (cut by `SENPAI_TIMEOUT_MINUTES=30`) |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 209.69 | 184.18 |
| geom_camber_rc | 177.40 | 169.65 |
| geom_camber_cruise | **126.99** | **NaN** |
| re_rand | 151.93 | 145.91 |
| **avg** | **166.50** | **NaN (166.58 over valid 3)** |

- **Metric artifacts**: `models/model-charliepai2i24h1-edward-wider-h192-h6-20260515-124423/metrics.jsonl`, `metrics.yaml`, `config.yaml`

### Analysis

- val loss decreased monotonically through the available 9-epoch budget (241.75 → 220.13 → 195.00 → 186.74 → 167.31 → 166.50 → 184.61). One bump at epoch 5 (237.13) is consistent with the still-high LR under T_max=50 not annealing.
- The cosine schedule never annealed because `T_max=50` while only 9 epochs completed under the 30-min wall-clock cap. The model is evaluated at near-peak LR rather than after a low-LR fine-tune. This is a systemic issue affecting every round-1 PR; documented in `BASELINE.md`.
- The cruise-test NaN comes from a single non-finite pressure prediction propagating through the unguarded scoring accumulator. The val split for the same domain (`val_geom_camber_cruise`) is fine (in fact the best of the four). Root cause is likely a high-Re cruise test sample pushing pressure logits past float32 range under the still-warm / partially-trained model.
- Param count was 1.45M (not the ~1.6M PR-body estimate). VRAM headroom comfortable.

### Follow-ups queued

- **Edward's next PR (ReScaler / log(Re)-conditioned output scaler).** Directly addresses the cruise NaN AND the predicted per-sample y-std variation that drives high-Re training instability. This is researcher-agent Idea 4. → **dispatched as PR #3273**.
- **Schedule alignment** (set epochs ≈ realized budget, T_max=epochs). Systemic fix; queued for round 2 once we see the other round-1 PRs land.

## 2026-05-15 14:24 — PR #3137: Add EMA model weights (decay 0.999) for eval/test — MERGED

- **Student branch**: `charliepai2i24h1-nezuko/ema-0999`
- **Hypothesis**: Maintaining an exponential moving average of model weights and evaluating with the EMA copy smooths the iterate evaluated; particularly useful here because (a) Transolver has noisy slice-projector weights early in training, and (b) the loss landscape is sharp due to extreme target magnitudes. Predicted 2–5% relative improvement on `val_avg/mae_surf_p`.
- **Verdict**: MERGED. Largest single win on this track so far (~22% relative over #3130 baseline).

### Results

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p (primary, best)** | **129.4217** (epoch 14) |
| test_avg/mae_surf_p | NaN (cruise sample 20 has 761 Inf in GT pressure → scoring bug; other 3 test splits avg 128.44) |
| peak_memory_gb | 42.11 / 96 |
| epochs completed | 14 of 50 (cut by `SENPAI_TIMEOUT_MINUTES=30`) |
| trunk config used by student | n_hidden=128, n_head=4 (pre-#3130) |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 164.34 | 145.46 |
| geom_camber_rc | 145.10 | 129.24 |
| geom_camber_cruise | **97.32** | **NaN** (corrupt GT) |
| re_rand | 110.93 | 110.61 |
| **avg** | **129.42** | **NaN (128.44 over valid 3)** |

- **Per-epoch val_avg/mae_surf_p trajectory (EMA-evaluated)**: ep1 362.62 → ep2 313.94 → ep3 273.01 → ep4 240.87 → ep5 217.20 → ep6 196.64 → ep7 180.42 → ep8 168.01 → ep9 158.96 → ep10 151.93 → ep11 144.14 → ep12 137.49 → ep13 130.75 → **ep14 129.42** (still strictly monotonically decreasing at wall-clock cutoff).
- **Metric artifacts**: `models/model-ema-0999-20260515-125245/metrics.jsonl`, `metrics.yaml`, `config.yaml`

### Analysis

- **Big win, but the magnitude is suspicious**: EMA was predicted to give 2–5%, we got 22%. Likely a combination of (a) the EMA effect itself smoothing out the noisy near-peak-LR iterate, and (b) the val curve being monotonically decreasing — so any later epoch is better, and EMA-evaluation effectively gives us a smoothed version of the epoch-14 weights vs the raw epoch-14 weights. The student also noted that with `decay=0.999` and ~375 steps/epoch, the EMA's effective averaging window is ~1000 steps — it's still tracking a fast-moving target at the cutoff. Under a budget that actually completes the cosine schedule, the relative EMA gain may be smaller.
- **Cruise-test NaN root cause identified**: Nezuko's diagnostic isolated it precisely. Cruise test sample 20 has 761 `Inf` values in ground-truth `p`. `accumulate_batch` computes `err = (pred - y).abs()` BEFORE masking, then multiplies by `surf_mask`. IEEE-754 says `Inf * 0 = NaN`, so the accumulator goes NaN on the `p` channel specifically (Ux/Uy GT are clean for the same sample). Reproduced deterministically at bs=4, doesn't reproduce at bs=1 (no batching with other samples). **Fix is one-line in `data/scoring.py`**: `err.nan_to_num_(0.0, posinf=0.0, neginf=0.0)` after the subtraction. `data/scoring.py` is marked read-only in `program.md` — requires advisor waiver. Queued.
- **Trust-in-orthogonality stacking**: Student branched from pre-#3130 config (n_hidden=128, n_head=4); squash-merge layered EMA onto the already-merged wider config. The combined `n_hidden=192 + n_head=6 + EMA` config is untested — we trust orthogonality. Documented in BASELINE.md.

### Follow-ups queued

- **Nezuko's next PR (separate per-channel prediction heads).** Researcher-agent Idea 12; addresses channel-scale mismatch (p has 10× the dynamic range of Ux/Uy). Independent of all in-flight round-1 PRs.
- **Scoring bug-fix PR.** One-line `nan_to_num` patch on `data/scoring.py`. Will fix `test_avg/mae_surf_p` for every future result. Needs advisor waiver of read-only — separate from regular experiment PRs.
- **EMA-vs-raw delta logging.** Cheap to add to the trainer (one extra `evaluate_split` per epoch). Would have made the EMA hypothesis test cleaner; folding into a future infra PR.

## 2026-05-15 14:35 — PR #3136: Increase surf_weight 10->25 to bias toward primary metric — MERGED

- **Student branch**: `charliepai2i24h1-frieren/surfw25`
- **Hypothesis**: The primary metric is surface pressure MAE but the trainer uses `loss = vol_loss + 10 * surf_loss`. Surface nodes are only a small fraction of mesh nodes, so even at surf_weight=10 the volumetric loss may dominate the gradient. Raising the weight should make the model prioritize surface accuracy.
- **Verdict**: MERGED. Beat current baseline (#3137 EMA at 129.42) by 2.4% relative.

### Results

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p (primary, best)** | **126.3241** (epoch 14) |
| test_avg/mae_surf_p | NaN (same cruise GT bug; 3-split mean = 123.43) |
| peak_memory_gb | 42.11 / 96 |
| epochs completed | 14 of 50 (cut by `SENPAI_TIMEOUT_MINUTES=30`) |
| trunk config used by student | n_hidden=128, n_head=4 (pre-#3130) |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 158.79 | 136.67 |
| geom_camber_rc | 127.26 | 117.86 |
| geom_camber_cruise | **102.20** | **NaN** (corrupt GT) |
| re_rand | 117.04 | 115.75 |
| **avg** | **126.32** | **NaN (123.43 over valid 3)** |

- **Metric artifacts**: `models/model-surfw25-20260515-132819/metrics.jsonl`, `metrics.yaml`, `config.yaml`

### Analysis

- Single attributable change applied cleanly. Training stable, no NaN/Inf in val for any of 14 epochs.
- Per-split surface pressure improved on every val split vs the #3137 baseline numbers (most notably val_geom_camber_rc 145.10 → 127.26, -12.3%). This is consistent with the surf_weight knob directly biasing the optimizer toward surface accuracy.
- **Stacking note**: frieren ran with n_hidden=128 (pre-#3130). The squash-merge layers only the surf_weight change onto the merged wider+EMA. Active recipe is now **wider + EMA + surf_weight=25** — a 3-axis stack that has never been measured end-to-end. The first PR based on this new advisor branch will produce the first true measurement; documented in BASELINE.md.
- Same cruise-test NaN as every other PR. Frieren's diagnostic chain (`Inf * 0 = NaN` through the mask) matches nezuko's and tanjiro's independently. The pattern is now well-established and a fix is overdue.

### Follow-ups queued

- **Idea 2 refinement (per-channel p_surf_weight).** With surf_weight=25 now in the baseline, layer a per-channel weight specifically on the p component of surf_loss (e.g. `p_surf_weight=3`). This isolates the channel-level boost without further raising the global surface weight.
- **Schedule alignment for round 2.** The cosine T_max=50 issue continues to affect every run.

## 2026-05-15 14:35 — PR #3273: log(Re)-conditioned output scaler (ReScaler) — SENT BACK

- **Student branch**: `charliepai2i24h1-edward/rescaler-logre`
- **Hypothesis**: Per-sample target std varies by 10× within a single domain (e.g., `val_single_in_dist` y_std range 458/2077). A learned MLP(log_Re) → exp(scale) per-sample output multiplier should let the trunk predict a small normalized residual while the scaler recovers physical magnitude. Predicted 5-12% improvement on val_avg/mae_surf_p, with biggest gain on Re_rand and geom_camber_cruise splits. Predicted side benefit: eliminate the cruise-test NaN.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | 153.3094 (epoch 9 of 9 realized) |
| Δ vs pre-#3137 baseline (166.50) | -7.9% (matches predicted 5-12% range) |
| Δ vs current baseline (#3137, 129.42) | +18.5% (WORSE; no EMA in student's run) |
| test_avg/mae_surf_p | NaN (cruise pressure still overflowed → 3-split mean 154.42) |
| peak_memory_gb | 63.0 / 96 |
| epochs completed | 9 of 50 (cut by 30-min cap) |
| trunk config used by student | n_hidden=192, n_head=6 (post-#3130, pre-#3137) |
| Per-split val Δ vs #3130 | single_in_dist -8.9%, rc -6.1%, cruise -7.2%, re_rand -9.3% |

- **Metric artifacts**: `models/model-rescaler-logre-20260515-135045/metrics.jsonl`, `metrics.yaml`

### Verdict: send back for rebase + tighter bound

- The hypothesis was validated on its own terms (-7.9% vs the baseline that existed when assigned). But the current baseline now includes EMA, and the student didn't see EMA — so 153.31 doesn't beat 129.42.
- The ReScaler approach is independent of EMA, so the combined effect is likely additive. The send-back asks for: (a) rebase onto the post-#3137 advisor branch (wider+EMA+surf_weight=25), (b) tighten `max_log_scale` from 2.0 → 1.0 (the y-std ratio across splits is only ~3×, so exp(±1)=[0.37, 2.72] is plenty), (c) optional `pred.clamp(-50, 50)` defense before scaling to prevent trunk overflow on outlier samples.
- If the rebased rerun shows ReScaler + EMA + surf_weight=25 beats 126.32, we merge.

## 2026-05-15 14:35 — PR #3141: Random Fourier features on position coordinates — SENT BACK

- **Student branch**: `charliepai2i24h1-tanjiro/fourier-pos`
- **Hypothesis**: Position coordinates fed directly through the preprocess MLP suffer from neural-network spectral bias — the network struggles to fit high-frequency surface-pressure variation. Random Fourier features (num_freqs=16, sigma=4.0) on (x, z) before preprocess should let the network express higher-frequency content cleanly. Predicted 5-8% improvement.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | 136.1451 (epoch 14 of 15 realized) |
| Δ vs current baseline (#3137, 129.42) | +5.2% (WORSE) |
| test_avg/mae_surf_p (raw) | NaN |
| **test_avg/mae_surf_p (NaN-safe re-eval)** | **122.9001** |
| peak_memory_gb | 42.33 / 96 |
| epochs completed | 15 of 50 |
| trunk config used by student | n_hidden=128, n_head=4 (pre-#3130) |

- **Metric artifacts**: `models/model-charliepai2i24h1-tanjiro-fourier-pos-20260515-131052/metrics.jsonl`, `test_metrics_clean.json`, `eval_test_clean.py`

### Verdict: send back for rebase + longer-budget rerun

- Val didn't beat current baseline (136.14 vs 129.42), BUT the NaN-safe test re-eval (122.90) is the lowest test number we've ever seen on this branch — and val was still descending at the cutoff (best was epoch 14 of 15 realized).
- The student also delivered a working NaN-safe re-eval (`eval_test_clean.py`) which respects the read-only constraint on `data/scoring.py`. **This pattern is now the recommended standard for reporting test numbers** until the scoring bug is patched.
- Send back asks for: (a) rebase onto post-#3137 advisor (wider+EMA+surf_weight=25), (b) consider a sigma sweep (σ ∈ {2, 4, 8}) since σ=4 may not be optimal at this dataset's spatial frequency content, (c) re-run with longer effective horizon (the cosine schedule was sized for 50 but only ~15 ran — at minimum, request `T_max=epochs` alignment).
- If the rebased rerun shows Fourier + EMA + surf_weight=25 beats 126.32 val_avg, we merge.

## 2026-05-15 16:35 — PR #3134: slice_num 64->128 — CLOSED

- **Student branch**: `charliepai2i24h1-fern/slice-num-128`
- **Hypothesis**: Doubling the number of physics-attention slice tokens from 64 to 128 should give the model finer-grained representation of mesh-local physics regimes, with predicted 3-6% improvement on `val_avg/mae_surf_p`.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | 191.65 (epoch 7 of 7 realized) |
| Δ vs pre-#3130 baseline (#3130 n_hidden=192 = 166.50) | +15% (WORSE) |
| Δ vs current baseline (#3136, 126.32) | +52% (WORSE — much worse) |
| trunk config used by student | n_hidden=128, n_head=4, surf_weight=10 (pre-#3130) |
| epochs completed | 7 of 50 (cut by 30-min cap) |
| Per-epoch wall time | ~4.3 min (≈2× #3130 baseline) |

- **Metric artifacts**: `models/model-charliepai2i24h1-fern-slice-num-128-20260515-132506/metrics.jsonl`

### Verdict: closed

- Doubling `slice_num` doubles the per-epoch cost (~4.3 min/epoch vs ~2.2 min on baseline). Under the 30-min cap, this means only ~7 epochs realize instead of ~14. The capacity gain (more slice tokens) is more than offset by the budget loss (fewer epochs). At fixed wall-clock, finer slice resolution is strictly worse here.
- Fern also independently diagnosed the cruise-test NaN root cause (Inf in `test_geom_camber_cruise/sample_20` GT × scoring-bug `Inf*0=NaN`) — same chain as nezuko #3137, tanjiro #3141, frieren #3136. The diagnostic agreement across 4 independent students is conclusive evidence; the read-only-waiver scoring-fix PR is overdue.
- Closed rather than sent back: the cost-benefit math doesn't work under the 30-min cap. Even with budget alignment (T_max=epochs=7), the floor is the wall-clock budget, not the schedule. Fern reassigned to round-2 `p_surf_weight` (#3298, see below).

## 2026-05-15 16:42 — PR #3127: SmoothL1 (Huber) loss — SENT BACK (strong-but-stale result)

- **Student branch**: `charliepai2i24h1-askeladd/smoothl1-loss`
- **Hypothesis**: MSE-in-normalized-space squares residuals; with per-sample y_std varying 10× within a split, a handful of high-Re samples dominate the gradient. Switching to SmoothL1 (Huber, beta=1.0) — quadratic near zero, linear in the tails — better matches the L1-in-physical-space evaluation metric and de-emphasizes outliers. Predicted 3-8% relative improvement, biggest on `val_re_rand` and `val_geom_camber_cruise`.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best) | **114.1433** (epoch 13 of 14 realized) |
| test_avg/mae_surf_p (NaN-safe via per-sample y_finite filter) | **102.3205** |
| Δ vs current baseline (#3136, 126.32) | **-9.6%** (BIG win) |
| Δ vs nominal old-config-MSE (#3130, 166.50) | -31% |
| trunk config used by student | n_hidden=128, n_head=4, surf_weight=10, no EMA (PRE-#3130) |
| epochs completed | 14 of 50 (cut by 30-min cap) |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 129.90 | 112.53 |
| geom_camber_rc | 129.94 | 112.80 |
| geom_camber_cruise | **92.61** | **80.85** |
| re_rand | 104.12 | 103.11 |
| **avg** | **114.14** | **102.32** |

- **Metric artifacts**: `models/model-charliepai2i24h1-askeladd-smoothl1-loss-20260515-140900/metrics.jsonl`

### Verdict: send back for rebase + budget-align

- **The hypothesis is strongly validated**: SmoothL1 vs MSE is clearly a big win on the same architecture — 114.14 vs the closest comparable MSE measurement (#3130 = 166.50, wider but no EMA) is -31%.
- **But the reported 114.14 is on the OLD pre-merge config** (n_hidden=128, n_head=4, surf_weight=10, no EMA — i.e., 4 axes different from current baseline), so it isn't a clean drop-in measurement for the merged stack. GitHub shows `mergeStateStatus: CLEAN` because the merge base is far enough back that nothing conflicts textually, but a hypothetical squash-merge would land SmoothL1 onto wider+EMA+surf_weight=25 — a recipe nobody has measured. The published 114.14 would not reproduce.
- **Notable askeladd contribution**: a per-sample `y_finite` filter in `evaluate_split` (lines 236-249 of their `train.py`) — drops any sample with non-finite GT before accumulator math. Lives in `train.py` (not `data/scoring.py`), so it respects the read-only constraint. Different from tanjiro's mask-before-sum approach (#3141), but functionally equivalent. Either pattern is acceptable.
- Send-back asks for: (a) `git rebase origin/icml-appendix-charlie-pai2i-24h-r1` to inherit wider+EMA+surf_weight=25, (b) keep SmoothL1 and the NaN-safe shim, (c) `--epochs 12` with `T_max=12` for proper cosine annealing, (d) re-report `val_avg/mae_surf_p` and `test_avg/mae_surf_p`.
- **If the rebased rerun beats 126.32**, we merge — and given the magnitude of the gap opened on the old config, this is one of the higher-probability merges queued.

## 2026-05-15 16:50 — PR #3298: per-channel p_surf_weight=3.0 on surface MSE (Idea 2 refinement) — DISPATCHED

- **Student**: charliepai2i24h1-fern (round 2 reassignment after #3134 close)
- **Hypothesis**: Now that surf_weight=25 is in the merged baseline, the surface loss dominates the total objective — but within surface loss all 3 channels (Ux, Uy, p) contribute equally. The primary metric only cares about p. Add a `p_surf_weight=3.0` multiplier inside the surface loss so the p channel gets 3× the gradient weight inside surf_loss (which is then multiplied by surf_weight=25 against vol_loss). Per-channel decomposition is the natural follow-up to frieren's global surf_weight win.
- **Predicted delta**: 2-5% on val_avg/mae_surf_p; should also show that mae_surf_Ux and mae_surf_Uy don't degrade by more than ~2-3%.
- **Schedule**: budget-aligned `epochs=12, T_max=12`.
- Single-attributable axis: introduces a new config field `p_surf_weight: float = 3.0` and replaces the global surface MSE with a per-channel sum where channel index 2 (p) carries the multiplier. At `p_surf_weight=1.0` the formulation collapses to the original MSE, so the change is a clean knob.

## 2026-05-15 17:00 — PR #3273: log(Re)-conditioned output scaler (ReScaler) rebased rerun — CLOSED

- **Student branch**: `charliepai2i24h1-edward/rescaler-logre` (rebased on advisor branch with `max_log_scale=2.0→1.0`, pre-scale `clamp(±50)`, schedule `epochs=8, T_max=8`)
- **Hypothesis (rebased rerun)**: ReScaler with tightened bound + clamp defense, layered onto the merged wider+EMA+surf_w25 stack, should beat 126.32. Schedule budget-aligned for the realized 8-epoch horizon.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, epoch 8 of 8 realized) | **152.7870** |
| test_avg/mae_surf_p (NaN-safe re-eval) | 138.7311 |
| Δ vs current baseline (#3136, 126.32) | **+20.9% (WORSE)** |
| Δ vs #3136 at matched epoch 8 | -2.7% (BETTER at matched-epoch) |
| trunk config used by student | wider+EMA+surf_w25+ReScaler (post-#3136 rebase, correctly inherited) |
| epochs completed | 8 of 8 (cosine T_max=8 fully annealed) |
| Per-epoch wallclock | ~205 s (wider trunk) |

| Split | val mae_surf_p (best) | test mae_surf_p (NaN-safe) |
|---|---|---|
| single_in_dist | 187.40 | 162.54 |
| geom_camber_rc | 169.25 | 152.58 |
| geom_camber_cruise | 118.44 | **101.80** (cruise sample 20 skipped) |
| re_rand | 136.05 | 137.99 |
| **avg** | **152.79** | **138.73** |

- **Metric artifacts**: `models/model-rescaler-logre-tight-20260515-154035/metrics.jsonl` + `metrics.yaml`

### Verdict: closed — schedule/budget tradeoff dominates

- The ReScaler hypothesis is **technically validated**: at matched epoch 8, ReScaler+wider+EMA+surf_w25 (152.79) beats no-ReScaler+narrow+EMA+surf_w25 (#3136 at epoch 8 ≈ 157.02) by ~3% — but the trunk axis differs, so the comparison isn't perfectly apples-to-apples.
- **The absolute number doesn't beat baseline** (-21% regression vs 126.32). Per CLAUDE.md decision criteria (>5% regression → close), this is a clear close.
- **Edward's analysis identified the real blocker**: at n_hidden=192, only 8 epochs fit in the 30-min cap; the narrower trunk fit 14 epochs and got further along the val descent curve, even with a partially-annealed cosine. The wider trunk's capacity advantage cannot manifest within the current per-epoch wallclock budget. **This is now the binding systemic constraint.**
- **Useful artifacts produced**: `ReScaler` module (163 params, clean implementation), `max_log_scale=1.0` + `pre_scale clamp(±50)` defense, and `evaluate_split_nan_safe()` — the third NaN-safe re-eval pattern on this track (alongside tanjiro's `eval_test_clean.py` and askeladd's `y_finite` filter).
- **Edward reassigned to bf16 mixed-precision** (#3332) — directly addresses the per-epoch wallclock budget that ReScaler couldn't overcome.

## 2026-05-15 17:05 — PR #3332: bf16 mixed-precision training — DISPATCHED

- **Student**: charliepai2i24h1-edward (round 2 reassignment after #3273 close)
- **Hypothesis**: The dominant bottleneck is now schedule/budget alignment at the wider trunk (identified directly by edward's #3273 analysis). bf16 autocast on forward + loss is expected to deliver ~30-40% per-epoch speedup on the 96GB GPU, unlocking ~12-13 epochs in the 30-min cap (vs 8 at fp32). Combined with budget-aligned scheduling (`epochs=12, T_max=12`), this finally lets the wider trunk realize its capacity advantage.
- **Predicted delta on val_avg**: 2-5% better than 126.32 baseline (i.e., 119-123), with the gain attributable to (a) wider trunk capacity finally usable, (b) cosine fully annealing through 12 epochs.
- Single attributable: wrap forward + loss in `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` in both the training loop and `evaluate_split`. No GradScaler (bf16 has the dynamic range), no parameter conversion, no batch_size change. EMA stays fp32 by default since autocast doesn't affect parameter storage.
- Critical instrumentation requested: **per-epoch wallclock** (the directly-attributable measurement) and **epochs realized in 30-min cap** (the lever this PR is pulling). If bf16 speedup is <20%, that's an important systems-level finding informing whether the next throughput unlock should be `batch_size=8` instead.

## 2026-05-15 17:10 — PR #3298: per-channel p_surf_weight=3.0 — CLOSED

- **Student branch**: `charliepai2i24h1-fern/p-surf-weight`
- **Hypothesis (rerun)**: p_surf_weight=3.0 multiplier on the p channel inside surface MSE; identity-at-1.0 knob; budget-aligned epochs=12.

### Results

| Metric | Value |
|---|---|
| val_avg/mae_surf_p (best, epoch 9 of 12) | **158.5363** |
| test_avg/mae_surf_p (NaN-safe via tanjiro's `eval_test_clean.py` pattern) | 144.4903 |
| Δ vs current baseline (#3136, 126.32) | **+25.5% (WORSE)** |
| Δ at matched epoch 9 vs no-p_surf_weight matched-epoch | -0.3% (essentially neutral, 158.54 vs 158.96) |
| trunk config used by student | wider+EMA+surf_w25+p_surf_weight=3.0 (correctly inherited from advisor) |
| epochs completed | 9 of 12 (cut by 30-min cap, schedule annealed faster due to T_max=12) |
| Per-epoch wallclock | ~203 s (wider trunk; same as edward #3273) |

| Channel | This run (epoch 9) | Baseline (epoch 14) |
|---|---:|---:|
| val_avg/mae_surf_p | 158.5363 | 126.3241 |
| val_avg/mae_surf_Ux | 3.0786 | 1.9848 |
| val_avg/mae_surf_Uy | 1.1146 | 0.8943 |

- **Metric artifacts**: `models/model-p-surf-weight-20260515-153455/metrics.jsonl` + `metrics.yaml` + `test_metrics_clean.json`

### Verdict: closed — same schedule/budget wall as #3273

- **Same systemic pattern as edward's #3273**: wider trunk fits only 9 epochs vs baseline's 14, and the matched-epoch comparison shows the hypothesis-attributable effect is approximately **neutral** (158.54 vs 158.96 = -0.3% at epoch 9). The +25% regression in absolute terms is dominated by the schedule/budget tradeoff, not by the p-channel weighting itself.
- Fern's matched-epoch comparison was the cleanest analysis of this kind on the track to date — exactly the right diagnostic in a budget-constrained regime.
- **Per-channel surface reweighting is approximately neutral at this scale** — the test was clean enough that we can call this. The hypothesis-attributable effect would have to come from epochs 10+ where p_surf_weight=3.0 starts diverging from baseline, but there's no strong reason to expect this if it's neutral at epoch 9.
- Could be revisited after bf16 (edward #3332) lands and unlocks the 12-13 epoch budget.
- **Fern reassigned to gradient clipping (#3336)** — orthogonal to all in-flight, single-line change, doesn't require full schedule annealing to show signal.

## 2026-05-15 17:15 — PR #3336: Global gradient norm clipping (max_norm=1.0) — DISPATCHED

- **Student**: charliepai2i24h1-fern (round 2 reassignment after #3298 close)
- **Hypothesis**: With `surf_weight=25` and per-sample y_std varying 10× within a single split, high-Re samples produce gradient spikes amplified twice (by surf_weight and by the squared-error tail). Global `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` bounds the per-step update magnitude, damping high-Re gradient spikes without changing the average update direction.
- **Predicted delta**: modest (2-5%) on val_avg/mae_surf_p, with largest effect on val_re_rand (the stratified Re holdout). Per-step effect — does NOT require full schedule annealing to validate.
- **Schedule**: budget-aligned `--epochs 12 --T_max=12` (matching the post-#3136 rerun policy).
- Single attributable: one line inserted between `loss.backward()` and `optimizer.step()`. Instrumentation: track `train/grad_norm_mean`, `_max`, `_min` per epoch — primary diagnostic for whether clipping is actually firing.
- Designed to be orthogonal to bf16, separate-heads, FiLM, SmoothL1, warmup, Fourier, and deeper — none of those control gradient norm.

## 2026-05-15 17:40 — PR #3144: Deeper Transolver: n_layers 5->8 (rebased) — CLOSED

- **Student branch**: `charliepai2i24h1-thorfinn/deeper-l8`
- **Hypothesis**: At the merged wider trunk, n_layers 5→8 should deliver capacity gains (more transformer block depth).
- **Verdict**: CLOSED. val_avg/mae_surf_p = **177.6026** (epoch 6) vs baseline 126.32 = **+40.6% regression**, well above the 5% close threshold. **Same wider-trunk-budget wall as #3273 and #3298** — third independent diagnosis of the pattern.

### Results

| Metric | Value |
|---|---|
| **val_avg/mae_surf_p (primary, best)** | **177.6026** (epoch 6 of 8 realized) |
| test_avg/mae_surf_p | NaN (cruise scoring bug); 3-split finite mean = 177.1244 |
| n_params | 2,242,995 (2.24M — more than doubled from 1.03M baseline) |
| peak_memory_gb | **96.61 / 96** (at the hardware ceiling) |
| Per-epoch wallclock | ~317 s (~5.3 min/epoch — vs ~205s baseline) |
| epochs realized | 6 of 8 (still mid-descent: 193→177 between epochs 5→6) |

### Verdict commentary

- **The capacity gain isn't testable at current budget**: wider × deeper drove per-step cost +55% AND pushed VRAM to 96.61/96 GB ceiling. The model is clearly still mid-descent at termination (8% val improvement between epochs 5→6).
- **Thorfinn's analysis was excellent**: correctly identified depth=8 is gated on bf16 (or bs=2, or both) landing first. After bf16 lands and unlocks ~30% more budget, this hypothesis becomes testable as part of a depth sweep (5/6/7/8).
- Per-channel: mae_surf_Ux=3.83, mae_surf_Uy=1.10 — not damaged, consistent with baseline rails (3.65 / 1.04). The pressure channel just isn't getting enough training.
- **Thorfinn reassigned to scoring-fix PR (#3378)** — system-fix with advisor waiver to edit `data/scoring.py`. Unblocks `test_avg/mae_surf_p` reporting paper-wide.

## 2026-05-15 18:20 — PR #3332: bf16 mixed-precision (autocast forward + loss) — SENT BACK

- **Student branch**: `charliepai2i24h1-edward/bf16-amp`
- **Hypothesis**: Wrap forward + loss in `torch.amp.autocast(dtype=torch.bfloat16)`; ~30-40% per-epoch speedup → 12-13 epochs realized at wider trunk; wider trunk's capacity manifests with the extra epochs.
- **Verdict**: SENT BACK. The infrastructure unlock is clean and works as predicted, but val_avg/mae_surf_p = 129.76 is +2.7% over baseline 126.32 — doesn't pass the merge-must-improve-primary rule. Sent back with bs=8 + epochs=22 retry recipe.

### Results

| Metric | Value (bf16, 12 epochs) | Baseline #3136 |
|---|---|---|
| **val_avg/mae_surf_p (primary, best)** | **129.7591** (epoch 12) | 126.3241 (epoch 14) → **+2.7% regression** |
| test_avg/mae_surf_p | **117.7643** (full 4-split, NaN-safe) | NaN (3-split partial 123.43) → **+5.7 better, full coverage** |
| Per-epoch wallclock | **143.7s** (~30% speedup) | 205s (fp32 wider) → 30% landed (low end of 30-40% prediction) |
| epochs realized in 30-min cap | **12** | 8 (wider fp32) → +4 epochs |
| peak_memory_gb | **49.24** / 96 | (vs 63 fp32) → ~22% memory savings, lots of headroom for bs=8 |
| nonfinite_loss_count | 0 | n/a |

### Verdict commentary

- **bf16 itself is fully clean**: zero NaN/Inf events across 12 epochs and 4 test splits. Per-channel ratios (Ux=2.12, Uy=0.86) intact, no precision damage.
- **Val regression is concentrated** on `val_geom_camber_rc` (+12.3 vs baseline; other 3 splits move <1 point). Wider trunk still mid-descent on the geometry-unseen split at 12 epochs.
- **30% speedup confirmed as predicted** — peer-epoch 205s → 143.7s. Memory-bandwidth-bound parts absorbed some of the GEMM/conv speedup (compute would have given 40%).
- **NaN-safe eval working**: all 4 test splits finite, test_avg covers all 4 splits (vs baseline's 3-split partial). This pattern is now standardized via thorfinn's scoring-fix PR (#3378).
- **batch_size=8 is the natural next step**: peak memory 49 GB / 96 GB means there's a clean 2× headroom. Doubling batch at bf16 → ~75-85s/epoch (memory-bandwidth-bound so not full 2×) → 21-24 epochs realized.
- Sent back with: bs=8, epochs=22, T_max=22, keep bf16 plumbing, keep lr=5e-4 (no scaling for now to keep bs axis attributable).

## 2026-05-15 18:30 — PR #3378: NaN-safe scoring fix (torch.where, advisor waiver) — DISPATCHED

- **Student**: charliepai2i24h1-thorfinn (round 2 reassignment after #3144 close)
- **Hypothesis**: `data/scoring.py::accumulate_batch` uses `err * mask.double()` which propagates Inf×0=NaN when GT contains Inf (cruise sample 20). Replace with `torch.where(mask, err, 0)` — preserves documented per-sample-skip semantic, eliminates NaN propagation. **Not expected to change val** (val GT is finite); **expected to change test** from NaN to a finite 4-split mean.
- **Advisor waiver**: explicit permission to edit `data/scoring.py` (normally a protected file). Scoped to `accumulate_batch` only, one-spot substitution. Adds unit test `tests/test_scoring_nan_safe.py` to verify Inf-in-GT doesn't propagate NaN.
- **Project value**: unblocks `test_avg/mae_surf_p` reporting paper-wide. Edward's `evaluate_split_nan_safe` (in #3332) and tanjiro's `eval_test_clean.py` (#3141) are workarounds; this is the source-level fix.
- 5 students have independently diagnosed this issue (nezuko, tanjiro, frieren, fern, edward). Three NaN-safe patterns established as workarounds; tanjiro's element-mask is the cleanest and what this PR adopts.

## 2026-05-15 18:41 — PR #3277: Separate per-channel prediction heads — CLOSED

- **Student branch**: `charliepai2i24h1-nezuko/separate-heads`
- **Hypothesis**: Three independent `Linear→GELU→Linear` heads (one per Ux/Uy/p channel) on the shared trunk can specialize per-channel non-linearities, addressing the channel-scale-mismatch problem. Total parameter increase predicted to be tiny.
- **Verdict**: CLOSED. val_avg/mae_surf_p = **148.83** (epoch 9 of 9 realized) vs baseline 129.42 = **+15% regression**. Wider-trunk-budget pattern again — though the dominant cost driver here was activation traffic (three [B, N, 384] hidden states), not parameter count.

### Results

| Metric | Value | Baseline |
|---|---|---|
| **val_avg/mae_surf_p (primary, best @ ep9)** | **148.8304** | 129.42 → +15% regression |
| test_avg/mae_surf_p | NaN (cruise bug); 3-split partial = 149.92 | (baseline partial 128.44) |
| Per-epoch wallclock | ~218 s (~75% slower than 125s baseline) | n/a |
| n_params | 1,633,377 (+12.8% over 1.45M) | 1,447,521 |
| peak_memory_gb | **71.21** | 42.11 |
| epochs realized in 30-min cap | 9 | 14 |

### Verdict commentary

- **Memory-bandwidth issue, not parameter-count issue**: +12.8% params → +75% per-epoch wallclock. Three [B, N=242K, 384] hidden states at mlp_ratio=2 carry ~3× the activation traffic of the baseline's single [B, N, 192] head. The architecture works in principle but doesn't fit the compute budget.
- **Strict monotonic descent at termination**: every epoch was a new best (338→285→248→218→198→182→168→157→149); model is far from converged at the 30-min cap.
- **Nezuko's analysis is excellent**: correctly identified the +75% slowdown vs +12.8% param change as the dominant signal. Three of four suggested follow-ups are sound (mlp_ratio=1 cheaper heads, matched-epoch confirmation, channel-wise LR/scale). One (per-channel scalar gain) is mathematically redundant.
- **Nezuko reassigned to LayerScale (#3404)** — orthogonal architectural change with ~1920 added params and zero compute overhead. Tests the "cold-start" hypothesis directly.

## 2026-05-15 19:20 — PR #3404: LayerScale residual gating (CaIT) — DISPATCHED

- **Student**: charliepai2i24h1-nezuko (round 2 reassignment after #3277 close)
- **Hypothesis**: At budget-limited training (12 epochs, monotonically descending at termination), the cold-start problem dominates. LayerScale (Touvron et al. CaIT 2021) adds a learnable per-channel multiplicative gate on each residual branch, initialized at `1e-4`, so each TransolverBlock contributes ≈0 to the residual stream at init. Training learns to scale up each block's contribution as evidence accumulates — natural curriculum across depth.
- **Predicted delta**: 2-5% on val_avg/mae_surf_p, with the biggest effect on early-epoch val (faster initial descent translates to better final number at the 30-min cap).
- **Cost**: 1920 added params (+0.13% over 1.45M), zero per-step compute overhead (element-wise multiplication on residual).
- **Single attributable**: insert `LayerScale(dim=192, init_value=1e-4)` on both residual branches of `TransolverBlock`. No other changes — same trunk config, EMA, surf_weight, optimizer, schedule.
- **Diagnostic to track**: per-block `ls_attn.gamma.mean()` and `ls_mlp.gamma.mean()` over training. Cold-start hypothesis predicts late blocks (closer to readout) grow gamma first; early blocks come online later.

## 2026-05-15 20:25 — PR #3378: NaN-safe `data/scoring.py` fix — MERGED (system fix)

- **Student branch**: `charliepai2i24h1-thorfinn/scoring-nan-safe`
- **Hypothesis**: Bug fix with advisor waiver. `data/scoring.py::accumulate_batch` computes `err = (pred - y).abs()` before masking, then multiplies by mask. When GT contains Inf (cruise test sample 20 has 761 Inf values in `p`), IEEE-754 `Inf × 0 = NaN` poisons the accumulator. Replace element-wise product with `torch.where(mask, err, 0)`. Mathematically identical on finite GT; never reads `err` where mask=False on Inf GT.
- **Verdict**: MERGED. System fix that doesn't change val_avg (validation has no Inf, behavior identical). Unblocks `test_avg/mae_surf_p` finite reporting paper-wide.

### Results

| Metric | Value | Baseline |
|---|---|---|
| **val_avg/mae_surf_p (primary, best @ ep9)** | **148.4154** | 126.32 (gap is schedule effect, not fix attribution) |
| **test_avg/mae_surf_p (NOW FINITE, 4-split)** | **136.0053** | NaN (was 3-split partial 123.43) |
| test_geom_camber_cruise.mae_surf_p | **97.4897** | NaN (previously broken under bug) |
| Unit test (`tests/test_scoring_nan_safe.py`) | PASS | n/a |
| epochs realized | 9 of 12 | n/a |
| peak_memory_gb | 63.00 | 42.11 |

### Verdict commentary

- **Val gap is purely schedule-effect**: 148.42 (9 ep, T_max=12) matches the schedule-truncation trajectory of every other budget-aligned wider-trunk run. Validation GT is finite, so the fix is bytewise identical there.
- **Test reporting unblocked**: cruise term goes from NaN → 97.49 (lower than the other three splits, consistent with that being the "easier" cruise regime per `program.md`). All future PRs on this track now report 4-split finite test_avg directly from `metrics.jsonl`.
- **Unit test catches the failure mode**: passes with NaN-safe code, fails with old code on Inf GT input.
- **Acknowledged side note**: `train.py`'s eval-loss aggregation (separate from scoring) has the same Inf×0=NaN pattern. Doesn't affect paper-facing MAE metric. Logged for a future one-spot patch.
- **Excellent PR discipline**: kept the change atomic (only `accumulate_batch`, no `aggregate_splits` or `finalize_split` touched), flagged the residual issue without expanding scope.

## 2026-05-15 20:30 — PR #3336: Gradient norm clipping `max_norm=1.0` — SENT BACK

- **Student branch**: `charliepai2i24h1-fern/grad-clip-1p0`
- **Hypothesis**: Per-batch `clip_grad_norm_(max_norm=1.0)` would damp rare high-Re gradient spikes that drive instability in re_rand and cruise splits while leaving median batches alone. Predicted 2-5% improvement.
- **Verdict**: SENT BACK with max_norm sweep recipe. Direction signal correct but `max_norm=1.0` was 100× too aggressive.

### Results

| Metric | Value | Baseline |
|---|---|---|
| **val_avg/mae_surf_p (best @ ep9 of 12)** | **129.7342** | 126.32 → +2.7% |
| **test_avg/mae_surf_p (NaN-safe clean, 4-split)** | **116.5846** | (was NaN under bug; clean 3-split=129.08) |
| val_re_rand | 111.84 | 117.04 → **-4.4%** (improved) |
| val_geom_camber_cruise | 95.64 | 102.20 → **-6.4%** (improved) |
| val_geom_camber_rc | 149.45 | 127.26 → +17.4% (regressed) |
| val_single_in_dist | 162.01 | 158.79 → +2.0% |
| Per-epoch wallclock | ~204s | n/a |

### Diagnostic punchline

Per-epoch pre-clip gradient L2 norms:
- min: 4-20, mean: 100-200, max: 600-1900
- **clip_rate = 1.00 EVERY EPOCH** — every batch hit the clip threshold
- `max_norm=1.0` renormalized every gradient to a unit vector; optimizer saw direction only, never magnitude

### Send-back recipe

- **Single arm**: `max_norm=100.0` (not a 4-way sweep). Based on observed distribution, this should clip ~5-20% of batches.
- **Rebase** onto current advisor branch to pick up #3378 (no more eval_test_clean.py needed)
- **Keep grad_norm logging permanently** — it's the diagnostic that made this PR valuable
- Same recipe otherwise: budget-aligned epochs=12, T_max=12, n_hidden=192, EMA=0.999, surf_weight=25

### Verdict commentary

- **Direction signal is real**: re_rand and camber_cruise (high-Re spike-prone splits) improved exactly as the hypothesis predicted. The other two splits regressed due to median-batch info loss.
- **`max_norm=1.0` is not the regime the hypothesis lives in**: clipping every batch is not "damp the spikes" — it's "kill all magnitude signal". The 100× scaling fix should isolate the spike-damping benefit.
- **Grad-norm logging is the killer feature**: every future PR should expose `train/grad_norm_{mean,max,min,clip_rate}` per epoch. Cheap, gives mechanism-level insight on what optimization looks like.

## 2026-05-15 20:35 — PR #3121: Linear warmup + cosine annealing (rebased, budget-aligned) — CLOSED

- **Student branch**: `charliepai2i24h1-alphonse/warmup-cosine-12ep`
- **Hypothesis**: 2-epoch linear warmup from 5e-7 → 5e-4 followed by cosine annealing (T_max=10) over remaining epochs would stabilize early training and improve val_avg by 3-8% relative.
- **Verdict**: CLOSED. val=158.75 vs baseline 126.32 = **+25.7% regression**. Schedule axis exhausted under 30-min cap.

### Results

| Metric | Value | Baseline |
|---|---|---|
| **val_avg/mae_surf_p (best @ ep9 of 12)** | **158.7495** | 126.32 → +25.7% |
| **test_avg/mae_surf_p (NaN-safe clean, 4-split)** | **143.8544** | (was NaN under bug; clean 3-split = 123.43) |
| val_single_in_dist | 203.45 | 158.79 → +28.1% |
| val_geom_camber_rc | 172.78 | 127.26 → +35.8% |
| val_geom_camber_cruise | 121.21 | 102.20 → +18.6% |
| val_re_rand | 137.56 | 117.04 → +17.5% |
| Realized epochs | 9 of 12 | 14 of 50 |
| Peak memory | 62.97 GB | 42.11 GB |

### Verdict commentary

- **Uniform regression across splits** (17-36%) — consistent with a uniform schedule cost, not a per-split feature interaction
- **Epoch 1 burned at lr=5e-7** (start_factor=1e-3) — 1/9 of realized budget wasted on near-zero LR
- **Cosine never reached floor**: at epoch 9 LR was still 35% of peak (1.73e-4); the "low-lr fine-tune at end of cosine" never happened
- **Val curve still decreasing at -10/epoch** when timeout hit — model is in rapid-improvement phase; burning an epoch on warmup is net-negative at this budget
- **Three-way confirmation of schedule-axis exhaustion**: combined with #3273/#3298/#3144 (wider-trunk-budget wall) and #3277 (memory-bandwidth wall), schedule perturbations cannot improve val_avg in the current regime
- **Student's analysis is correct and articulate**: "the real lever is the budget, not the schedule" — exactly right. Schedule axis goes back on the shelf until bf16+bs=8 unlocks 21-24 epochs of realized training.

### Follow-up

- Alphonse reassigned to a fresh non-schedule axis (not a warmup-variant; the hypothesis class itself is closed at this budget)

## 2026-05-15 21:25 — PR #3287: Domain-conditional FiLM (gap+AoA → scale/shift on LayerNorm) — CLOSED

- **Student branch**: `charliepai2i24h1-frieren/domain-film`
- **Hypothesis**: Three geometric regimes (single-foil, racecar-tandem, cruise-tandem) produce qualitatively distinct pressure fields. Per-block FiLM that emits `(scale, shift)` from `(gap, AoA1)` should give the model an explicit per-sample regime conditioner without requiring it to discover it implicitly. Predicted disproportionate gains on `val_geom_camber_rc` and `val_single_in_dist`.
- **Verdict**: CLOSED. Clear negative result with thorough mechanism analysis. Hypothesis disconfirmed by the *predicted-to-improve splits being the most regressed*.

### Results (canonical run, EMA-evaluated best-val checkpoint)

| Metric | FiLM (#3287) | Baseline (post-#3136) | Δ |
|---|---:|---:|---:|
| **val_avg/mae_surf_p (primary)** | **145.9856** | 126.3241 | **+15.6%** |
| test_avg/mae_surf_p (NaN-safe 4-split) | 138.3074 | n/a (3-split partial 123.43 pre-#3378) | n/a |
| n_params | 1,572,513 | 1,447,521 | +8.6% |
| peak_memory_gb | 71.20 | 42.11 | +69% |
| epochs realized | 9 of 12 | 14 of 50 | -36% |
| Per-epoch wall time | 223.6 s | 128 s | +75% |
| NaN/Inf events | 0 | 0 | — |

### Per-val-split mae_surf_p (best epoch=9)

| Split | FiLM | Baseline | Δ | Hypothesis prediction |
|---|---:|---:|---:|---|
| val_single_in_dist | 184.65 | 158.79 | **+16.3%** | predicted improvement |
| val_geom_camber_rc | 157.07 | 127.26 | **+23.4%** | predicted **largest** improvement |
| val_geom_camber_cruise | 112.96 | 102.20 | +10.5% | secondary improvement |
| val_re_rand | 129.26 | 117.04 | +10.4% | secondary improvement |

- **Metric artifacts**: `models/model-charliepai2i24h1-frieren-domain-film-canonical-20260515-192839/{metrics.jsonl,metrics.yaml}`

### Analysis

- **The predicted-to-improve splits are the most degraded.** This is the strongest possible disconfirmation: if `(gap, AoA1)` is the right conditioner, raceCar tandem should improve, not regress 23.4%. Hypothesis is not budget-starved — it is wrong about which mesh features actually matter.
- **Dual failure mode** (student's diagnosis was correct):
  1. **Memory-induced wall-clock slowdown**: 5 blocks × 2 LayerNorms × `[4, ~200K, 192]` ≈ 6 GB of additional autograd-retained activations → peak 71 GB → 74% GPU utilization → per-epoch +75% slower. Under 30-min cap, lost 5 realized epochs.
  2. **2-D conditioner too coarse** for the multi-modal regime. `gap=0` and AoA ranges overlap between single/tandem in ways that the cleanest taxonomy doesn't capture.
- **Implementation quality**: zero NaN/Inf, identity-init verified, schedule-aligned (T_max=12), feature indices verified against `program.md` (no fallback path).

### Verdict commentary

- This is a **methodologically valuable negative result** — the disconfirmation analysis (predicted splits most degraded) is exactly what a careful reviewer would value. It rules out the *family* of low-dimensional per-sample conditioners, not just this specific implementation.
- Student's suggested follow-ups (input-only FiLM injection, 7-scalar richer conditioner, LayerNorm-affine-fused FiLM) are noted but represent fresh hypothesis classes; not pursued in this PR.
- The high-quality scaffolding (clean numerics, 4-way sweep, exact `program.md` feature indexing) leaves room for a fresh axis without inheriting any negative-result baggage.

### Follow-up

- Frieren reassigned to **DropPath (stochastic depth)** — a regularization axis, orthogonal to all conditioning work, zero param overhead, zero inference cost, compatible with LayerScale (#3404 nezuko in-flight). Different mechanism class than FiLM (regularization vs feature conditioning).

## 2026-05-15 22:30 — PR #3332: bf16 + bs=8 retry — CLOSED

- **Student branch**: `charliepai2i24h1-edward/bf16-amp`
- **Hypothesis (retry arm)**: With peak memory 49 GB / 96 GB at bs=4, bs=8 should give 1.7-1.9× per-epoch speedup → 21-24 epochs realized at the wider trunk. Combined with cosine T_max=22, would unlock the first wider-trunk run with fully-annealed cosine.
- **Verdict**: CLOSED. bs=8 axis cleanly refuted — Transolver is memory-bandwidth-bound, not compute-bound. bs=8 was ~10% **slower** per-epoch and hit the 96 GB HBM ceiling (98.42 GB peak with `expandable_segments`).

### Results (two arms in one PR)

| Arm | val_avg/mae_surf_p | test_avg (NaN-safe 4-split) | epochs realized | s/epoch | peak_GB |
|---|---:|---:|---:|---:|---:|
| **bf16-bs8 (latest, T_max=22)** | **187.83** | 172.03 | 12/22 | 158.9 | **98.42** |
| bf16-bs8 (1st relaunch, T_max=22) | 178.45 | — | 12/22 | 160.6 | 98+ |
| bf16-bs4 (prior arm, T_max=12) | 129.76 | 117.76 | 12/12 | 143.7 | 49.24 |
| Baseline #3136 (fp32 narrow, T_max=50) | 126.32 | NaN (3-split=123.43 pre-fix) | 14/50 | ~205 (wider fp32) | 42.11 |

- **Metric artifacts**: `models/model-bf16-bs8-20260515-{192740,202644}/{metrics.jsonl,metrics.yaml}`

### Mechanism findings

1. **bs=8 is slower than bs=4** on this model. Per-batch time 382 ms (bs=4) → 846 ms (bs=8) — doubling the batch more than doubled per-batch time. The Transolver is **memory-bandwidth-bound**, so the bs=8 GEMM density gain doesn't compensate for the doubled memory traffic.
2. **Peak memory jumps to 98 GB at bs=8**, hitting the HBM ceiling and forcing `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid OOM. bs=12/16 would OOM hard.
3. **Schedule misalignment compounds the regression**: T_max=22 with 12 realized epochs leaves LR at ~30% of peak at "best epoch" — cosine never anneals. All 4 val splits regress substantially (single_in_dist 285 vs baseline 159, geom_camber_rc 197 vs 127, etc.).
4. **bf16 plumbing remains clean**: zero NaN/Inf, channel ratios well-behaved. Not a precision damage event; purely a schedule + batch-size mismatch.

### Verdict commentary

- **bs=8 axis is exhausted.** Memory-bandwidth bound + HBM ceiling = no future for batch-size throughput unlocks on this stack. bs=12/16 OOMs; bs=8 already underperforms bs=4.
- **bf16 alone (the prior arm) was +2.7% on val** — close but not winning. Not enough to merge bf16 as standalone infrastructure.
- **Stop/pivot criterion fires**: BASELINE.md stacking note documented that if bf16+bs=8 retry still doesn't beat 126.32, the wider-trunk thesis itself should be tested by reverting to narrow+bf16 (~18-20 epochs in cap). That test gets dispatched as the next PR.
- **Student's analysis was excellent** — per-batch-time breakdown (382→846 ms) was the cleanest possible diagnostic. The 10-point run-to-run noise on bs=8 (178.45 vs 187.83) is concerning but not blocking the close decision.

### Follow-up

- Edward reassigned to **narrow trunk (n_h=128, n_head=4) + bf16** (PR #3478) — directly tests the stop/pivot criterion. If narrow+bf16 at 18-20 epochs beats 126.32, we revert the wider-trunk merge (#3130). If it ties, width is a wash and we keep the merged wider config. If narrow+bf16 regresses, the wider thesis is confirmed and we need throughput unlocks (torch.compile) as the next axis.

## 2026-05-15 22:32 — PR #3127 askeladd SmoothL1 rerun: NUDGE (no rebased commits in 7h)

- **Student branch**: `charliepai2i24h1-askeladd/smoothl1-loss`
- **Status**: still on `status:wip`. Original result (val=114.14, big -9.6% improvement) was on pre-merge config (pre-#3130 narrow, pre-#3136 surf_weight=10, pre-#3137 no EMA, pre-#3378 NaN-safe). Sent back at 15:33 UTC for rebase + rerun on current advisor stack. No new commits or training output since.
- **Action**: explicit advisor reminder posted. Confirmed exact `train.py` command and emphasis on canonical `beta=1.0` arm. Pod is heartbeating but Claude sessions are short — likely waiting on a clarifying instruction.

## 2026-05-15 22:33 — PR #3404 nezuko LayerScale: NUDGE (no commits in 3h)

- **Student branch**: `charliepai2i24h1-nezuko/layerscale-residual`
- **Status**: assigned 19:25 UTC. Only one commit on the branch (the assignment commit `a605cf6a`). No comments, no metrics, no training output. Pod started seeing the assignment at iteration 90 (22:22 UTC) but training hadn't begun.
- **Action**: explicit advisor reminder posted with reiteration of LayerScale module signature (per-channel `gamma`, init=1e-4, +1,920 params), forward wrapping pattern, and exact `train.py` command. Pod monitoring will catch the next iteration's status.

## 2026-05-15 22:38 — PR #3404: LayerScale residual gating (CaIT init=1e-4) — CLOSED

- **Student branch**: `charliepai2i24h1-nezuko/layerscale-residual`
- **Hypothesis**: CaIT-style learnable per-channel scalar gate on each residual branch (init=1e-4) should act as a cold-start curriculum — blocks contribute ~0 at init, training ramps them up, with late blocks predicted to activate first. +1,920 params, zero compute overhead.

### Results

| Metric | Value | Baseline (#3136) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep9/12)** | **135.1100** | 126.3241 | **+6.9% (WORSE)** |
| epochs realized | 9 of 12 | 14 of 50 | — |
| peak_memory_gb | 70.45 GB | — | — |
| NaN/Inf events | None | — | — |

| Block | ls_attn gamma (ep9) | ls_mlp gamma (ep9) |
|---|---|---|
| Block 1 | ~0.010 | ~0.019 |
| Block 5 | ~0.006 | ~0.015 |

### Verdict: CLOSED — both pre-registered predictions disconfirmed

1. **Block-curriculum failed**: late blocks (closer to readout) should activate first per the CaIT cold-start rationale. Opposite observed: `ls_attn` block 1 gamma > block 5. MLP gammas uniform across depth.
2. **Convergence speed didn't improve**: val at epoch 1 = 330, descending at a **slowing** rate (44→35→30→26→20→15→14→11 per epoch) — classic signature of a model whose residual branches haven't unfolded. At epoch 9 gammas are still ~0.01–0.02 (50–200× smaller than 1.0).

**Root cause**: CaIT uses init=1e-4 for 36-layer transformers with ~300 training epochs. Our 5-layer, 12-epoch, 30-min-cap setup cannot give the gammas time to grow. The model wasted its entire compute budget on gate-growing rather than CFD optimization.

**Methodological value**: cleanly rules out the CaIT cold-start axis at this budget. A larger init (1e-2 or 1.0) would be a regularization hypothesis, not a cold-start curriculum, and belongs as a fresh PR if ever revisited. Student's per-block gamma instrumentation made the close decision unambiguous.

## 2026-05-15 22:40 — PR #3141: Random Fourier features on (x,z) position (rebased rerun) — CLOSED

- **Student branch**: `charliepai2i24h1-tanjiro/fourier-pos-rebased`
- **Hypothesis**: Random Fourier feature mapping σ=4.0 on (x,z) coordinates injects high-frequency spatial basis into the Transolver input, counteracting spectral bias. Predicted improvement on surface pressure captures near sharp geometry edges.

### Results

| Metric | Value | Baseline (#3136) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep9/12)** | **133.6001** | 126.3241 | **+5.7% (WORSE)** |
| test_avg/mae_surf_p (NaN-safe 4-split) | 121.5384 | — | — |
| epochs realized | 9 of 12 | 14 | — |
| peak_memory_gb | 63.22 GB | — | — |

| Split | val mae_surf_p | test mae_surf_p |
|---|---|---|
| single_in_dist | 160.6934 | 144.6801 |
| geom_camber_rc | 145.2853 | 129.7991 |
| geom_camber_cruise | 107.8672 | 89.7765 |
| re_rand | 120.5545 | 121.8979 |
| **avg** | **133.60** | **121.54** |

- **Metric artifacts**: `models/model-charliepai2i24h1-tanjiro-fourier-pos-rebased-20260515-212626/metrics.jsonl`

### Verdict: CLOSED — two-run negative signal confirms closure

- Pre-rebase: val=136.14; post-rebase (current advisor stack): val=133.60. Improvement of 2.5 points from rebasing, but still +5.7% above baseline. The direction is **stable in the regression band** across both runs.
- Budget was correctly aligned (T_max=12). Schedule misalignment cannot explain the gap.
- **Spectral-bias pattern confirmed**: surface `p` MAE far higher than `Ux`/`Uy` across splits — Fourier encoding didn't close the gap, but the diagnosis was correct.
- σ sweep (σ ∈ {2, 4, 8}) would be a different hypothesis for a future PR if positional encoding is revisited.

## 2026-05-15 22:42 — PR #3435: Aux task — predict is_surface indicator (BCE, aux_weight=0.1) — CLOSED

- **Student branch**: `charliepai2i24h1-alphonse/aux-issurface`
- **Hypothesis**: Predicting a binary `is_surface` indicator alongside the primary CFD regression (aux_weight=0.1, BCE loss) should improve boundary-layer representations, sharpening surface pressure accuracy. +193 params for the aux head.

### Results

| Metric | Value | Baseline (#3136) | Δ |
|---|---|---|---|
| **val_avg/mae_surf_p (best @ ep9/12)** | **141.2732** | 126.3241 | **+11.8% (WORSE)** |
| test_avg/mae_surf_p (NaN-safe 4-split) | 128.6019 | — | — |
| n_params | 1,447,714 | 1,447,521 | +193 (exactly as predicted) |
| peak_memory_gb | 63.0 GB | — | — |

| Split | val mae_surf_p | Baseline | Δ |
|---|---|---|---|
| val_single_in_dist | 180.9047 | 158.79 | +13.9% |
| val_geom_camber_rc | 151.6076 | 127.26 | +19.1% |
| val_geom_camber_cruise | 108.4551 | 102.20 | +6.1% |
| val_re_rand | 124.1254 | 117.04 | +6.1% |

### Aux head diagnostics

| Epoch | aux_acc | aux_loss |
|---|---|---|
| 1 | 0.8997 | 0.5889 |
| 6 | 0.9892 | 0.0988 |
| 9 | 0.9909 | 0.0595 |

- **Metric artifacts**: `models/model-aux-issurface-w0p1-20260515-212151/metrics.jsonl`

### Verdict: CLOSED — aux head saturates early; learning signal diminishes before primary task converges

The aux head reaches ~99% accuracy by epoch 6 of 9 realized, with aux_loss dropping 10× by epoch 3. Once the head saturates, the aux_weight=0.1 gradient signal becomes trivially small — the model gets no additional supervision. Meanwhile the primary regression task (surface pressure) regressed on all 4 splits. The 11.8% regression exceeds the 5% threshold.

**Note on label discipline**: student didn't swap `status:wip` → `status:review` after posting the SENPAI-RESULT. PR was reviewed in the `status:wip` queue after identifying the committed metrics. Future PRs should invoke the label-swap or `senpai:submit-experiment-results` skill as documented in the PR body.

**Mechanism insight**: aux task saturation at epoch 6 means the representation-shaping effect only operates for the first half of training. With a later-saturating target (e.g., `near_surface_distance` regression rather than binary classification), this might behave differently. Closed without a follow-up — representation-shaping via aux supervision has other active routes (DropPath, RMSNorm).

## 2026-05-15 22:44 — PRs #3496, #3498, #3500: New round-2 dispatches

- **#3496 nezuko**: RMSNorm swap — replace all `nn.LayerNorm` with `RMSNorm` (T5/PaLM/LLaMA standard). Drop-in, -2,112 params, 5-10% per-epoch speedup predicted.
- **#3498 tanjiro**: SwiGLU MLP — replace GELU FFN with `SwiGLU(d_ff=256)` (Shazeer 2020, iso-param per PaLM convention). Gated FFN for multi-scale targets.
- **#3500 alphonse**: slice_num 64→32 — halve PhysicsAttention slice tokens, direct memory-bandwidth attack. Predicted 15-25% per-epoch speedup (10-18 realized epochs vs 9 at fp32).
