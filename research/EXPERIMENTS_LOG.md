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
